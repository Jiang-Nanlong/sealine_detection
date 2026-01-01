import numpy as np
import cv2
from skimage.transform import radon
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


class FullDebugMuSCoWERT:
    def __init__(self, scales=[1, 2, 3]):
        self.scales = scales
        # 投票参数
        self.delta_alpha = 1.0
        self.delta_y = 5.0

    def _suppress_texture(self, binary_edges, window_size=15):
        """
        [辅助函数] 纹理抑制
        作用：消除密集的波浪纹理。
        原理：如果一个像素垂直方向的邻居太多，说明它是波浪（面状纹理），不是海天线（孤立线条）。
        """
        kernel = np.ones((window_size, 1), np.float32)
        density = cv2.filter2D(binary_edges.astype(np.float32), -1, kernel)
        is_texture = density > 4
        clean = binary_edges.copy()
        clean[is_texture] = 0
        return clean

    def _get_feature_map(self, gray_img, scale):
        """
        [核心函数] 特征提取管线
        输入：灰度图, 尺度 s
        输出：(加权后的特征图, 中值滤波后的图)
        """
        # --- 步骤 1: 多尺度中值滤波 (Multi-scale Median Filter) ---
        # 目的：论文的核心思想。
        # s=1 (k=11): 保留细节，但会有波浪噪点。
        # s=3 (k=31): 强力抹平波浪，但可能导致海天线变宽或位置轻微偏移。
        ksize = 10 * scale + 1
        blurred = cv2.medianBlur(gray_img, ksize)

        # --- 步骤 2: 梯度计算 (Gradient Calculation) ---
        # 目的：融合导师的建议。海天线是水平边缘，所以我们重点看 Y 方向梯度。
        # CV_64F: 使用浮点数防止负数梯度截断。
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        abs_grad_y = np.abs(grad_y)
        abs_grad_x = np.abs(grad_x)

        # --- 步骤 3: 自适应阈值 (Adaptive Thresholding) ---
        # 目的：替代 Canny 的固定阈值。
        # 逻辑：统计全图梯度的均值和标准差。只有显著高于背景梯度的点才被保留。
        mean_g = np.mean(abs_grad_y)
        std_g = np.std(abs_grad_y)
        thresh = mean_g + 0.5 * std_g  # 0.5 是经验系数，越小保留的边缘越多
        binary_edges = (abs_grad_y > thresh).astype(np.uint8)

        # --- 步骤 4: 骨架化 (Skeletonization) ---
        # 目的：模拟 Canny 的细化效果。
        # 梯度产生的边缘通常很宽（渐变），骨架化将其抽成 1 像素宽的细线，
        # 这样才能准确计算“长度”，否则一根粗线会被算成面积很大的块。
        binary_edges = skeletonize(binary_edges).astype(np.uint8)

        # --- 步骤 5: 纹理抑制 (Texture Suppression) ---
        # 目的：杀掉下方密集的波浪。这是解决“乱线干扰”的关键。
        clean_edges = self._suppress_texture(binary_edges)

        # --- 步骤 6: 形态学连接 (Morphological Closing) ---
        # 目的：把断断续续的海天线“焊”起来。
        # 使用横向长条核 (25, 1)，只进行水平连接。
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        closed_edges = cv2.morphologyEx(clean_edges, cv2.MORPH_CLOSE, kernel)

        # --- 步骤 7: 几何加权 (Geometric Weighting) ---
        # 目的：给每一条线打分。
        label_img = label(closed_edges, connectivity=2)
        regions = regionprops(label_img)
        weighted_map = np.zeros_like(closed_edges, dtype=np.float32)

        h, w = gray_img.shape
        for props in regions:
            length = props.major_axis_length

            # 过滤 A: 长度不够的不要 (过滤碎浪)
            if length < w / 5: continue

            # 过滤 B: 梯度方向不对的不要 (过滤斜线/杂乱纹理)
            coords = props.coords
            sum_gy = np.sum(abs_grad_y[coords[:, 0], coords[:, 1]])
            sum_gx = np.sum(abs_grad_x[coords[:, 0], coords[:, 1]])
            # 垂直梯度占比：海天线主要是 Gy，占比应很高
            score = sum_gy / (sum_gy + sum_gx + 1e-6)
            if score < 0.6: continue

            # 过滤 C: 形状不够直的不要 (过滤卷曲的波浪)
            if props.eccentricity < 0.95: continue

            # 最终权重：长度的平方 (放大长线优势) * 梯度方向分
            weight = (length ** 2) * score
            weighted_map[label_img == props.label] = weight

        return weighted_map, blurred

    def _radon_candidates(self, weighted_map, h_center, scale_idx):
        """提取候选线，不设固定数量限制，使用相对阈值"""
        theta = np.linspace(80., 100., 180, endpoint=False)
        sinogram = radon(weighted_map, theta=theta, circle=False)

        max_score = np.max(sinogram)
        # 相对阈值：只保留能量大于最大值 30% 的峰值
        score_thresh = 0.3 * max_score

        num_potential = 50
        flat_indices = np.argpartition(sinogram.ravel(), -num_potential)[-num_potential:]
        row_indices, col_indices = np.unravel_index(flat_indices, sinogram.shape)

        candidates = []
        for r, c in zip(row_indices, col_indices):
            score = sinogram[r, c]
            if score < score_thresh: continue

            angle_deg = theta[c]
            rho = r - (sinogram.shape[0] // 2)
            angle_rad = np.deg2rad(angle_deg)
            if np.abs(np.sin(angle_rad)) < 1e-4: continue

            y_offset = rho / np.sin(angle_rad)
            Y_img = y_offset + h_center
            alpha = angle_deg - 90

            # 位置约束 (去掉极上极下)
            h = weighted_map.shape[0]
            if Y_img < h * 0.15 or Y_img > h * 0.85: continue

            candidates.append({'Y': Y_img, 'alpha': alpha, 'score': score, 'scale': scale_idx})

        return candidates

    def detect(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 预处理：增强对比度
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        h_center = gray.shape[0] // 2

        all_candidates = []
        debug_info = {}  # 存储中间结果用于画图

        for s in self.scales:
            # 获取特征图 和 模糊后的图
            w_map, blurred = self._get_feature_map(gray, s)

            # 存起来
            debug_info[s] = {'map': w_map, 'blurred': blurred}

            candidates = self._radon_candidates(w_map, h_center, s)
            all_candidates.extend(candidates)

        if not all_candidates:
            return None, [], debug_info

        # 简单的投票逻辑
        best = max(all_candidates, key=lambda x: x['score'])
        final_result = (best['Y'], best['alpha'])

        return final_result, all_candidates, debug_info

    def visualize_all(self, image, final_result, all_candidates, debug_info):
        """
        绘制详细的分析图：
        1. 原始图像 + 最终结果
        2. 三个尺度的 中值滤波图 (并排)
        3. 三个尺度的 特征权重图 (并排)
        """
        # --- 1. 准备画布 ---
        fig = plt.figure(figsize=(18, 12))

        # 定义网格布局: 3行 (Result, Blurred, FeatureMap)
        gs = fig.add_gridspec(3, 3)

        # --- 2. 绘制最终结果图 (占据第一行) ---
        ax_res = fig.add_subplot(gs[0, :])

        res_img = image.copy()
        h, w = image.shape[:2]
        cx = w / 2

        # 定义颜色
        colors = {1: (0, 255, 255), 2: (0, 255, 0), 3: (0, 165, 255)}  # Cyan, Green, Orange

        # 画所有候选
        all_candidates.sort(key=lambda x: x['scale'], reverse=True)
        for cand in all_candidates:
            Y, a = cand['Y'], cand['alpha']
            color = colors.get(cand['scale'], (255, 255, 255))
            # 转换 BGR -> RGB 用于 matplotlib 显示
            # 这里直接画在 res_img (BGR) 上，后面再统一转
            t = np.tan(np.deg2rad(a))
            y1, y2 = int(Y - t * cx), int(Y + t * (w - cx))
            cv2.line(res_img, (0, y1), (w, y2), color, 1)  # BGR颜色

        # 画最终结果
        if final_result:
            Y, a = final_result
            t = np.tan(np.deg2rad(a))
            y1, y2 = int(Y - t * cx), int(Y + t * (w - cx))
            cv2.line(res_img, (0, y1), (w, y2), (0, 0, 255), 3)  # Red

        ax_res.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
        ax_res.set_title("Final Result (Red) & All Candidates (Cyan=S1, Green=S2, Orange=S3)")
        ax_res.axis('off')

        # --- 3. 绘制中间过程 ---
        for i, s in enumerate(self.scales):
            # 第一排小图：中值滤波结果
            ax_blur = fig.add_subplot(gs[1, i])
            blurred = debug_info[s]['blurred']
            ax_blur.imshow(blurred, cmap='gray')
            ax_blur.set_title(f"Scale {s}: Median Filter\n(Kernel {10 * s + 1}x{10 * s + 1})")
            ax_blur.axis('off')

            # 第二排小图：特征权重图
            ax_map = fig.add_subplot(gs[2, i])
            w_map = debug_info[s]['map']
            # 使用 log 显示让弱特征也能被看见
            ax_map.imshow(np.log1p(w_map), cmap='inferno')
            ax_map.set_title(f"Scale {s}: Weighted Feature Map\n(Input to Radon)")
            ax_map.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    img_path = r"D:\dataset\Hashmani's Dataset\MU-SID\DSC_0170_3.JPG"
    img = cv2.imread(img_path)
    detector = FullDebugMuSCoWERT(scales=[1, 2, 3])
    final_res, candidates, debug_info = detector.detect(img)

    detector.visualize_all(img, final_res, candidates, debug_info)