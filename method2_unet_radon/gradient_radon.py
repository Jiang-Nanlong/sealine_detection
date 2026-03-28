import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.transform import radon
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端，弹出独立窗口
import matplotlib.pyplot as plt


class TextureSuppressedMuSCoWERT:
    def __init__(self, scales=[1, 2, 3], full_scan=True):
        self.scales = scales
        self.full_scan = full_scan

        # 检查 GPU 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Radon Transform running on: {self.device}")

        # 定义角度范围 (numpy 用于逻辑，torch 用于计算)
        if self.full_scan:
            self.theta = np.linspace(0., 180., 180, endpoint=False)
        else:
            # 聚焦扫描：只看 85-95 度 (垂直线附近)
            self.theta = np.linspace(85., 95., 180, endpoint=False)

    def _suppress_texture(self, binary_edges, window_size=15):
        """纹理抑制：杀掉密集区域"""
        kernel = np.ones((window_size, 1), np.float32)
        density = cv2.filter2D(binary_edges.astype(np.float32), -1, kernel)
        is_texture = density > 4
        clean = binary_edges.copy()
        clean[is_texture] = 0
        return clean

    def _get_feature_map(self, gray_img, scale):
        # 1. 中值滤波
        ksize = 10 * scale + 1
        blurred = cv2.medianBlur(gray_img, ksize)

        # --- 核心修改 A: 梯度计算与差分 ---
        # 目的：利用波浪 Gx 很大的特点，自我抵消
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)

        abs_grad_y = np.abs(grad_y)
        abs_grad_x = np.abs(grad_x)

        # 差分逻辑：Pure_Vertical_Edge = Gy - 1.5 * Gx
        # 如果一个边缘也是斜的(Gx大)，它的得分会迅速变成负数
        diff_grad = abs_grad_y - 1.5 * abs_grad_x
        diff_grad[diff_grad < 0] = 0  # 负数归零

        # --- 核心修改 B: 梯度截断 (Clamping) ---
        # 目的：解决"贫富差距"。
        # 强行把海浪的高梯度削顶，让弱海天线也能由一席之地
        # 假设海天线梯度约为 10-20，我们将上限设为 20。
        # 这样，海浪(150)变成20，海天线(15)保持15，大家站在了同一起跑线！
        clamped_grad = np.clip(diff_grad, 0, 20)

        # 归一化方便后续处理
        norm_grad = cv2.normalize(clamped_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # --- 核心修改 C: 极低阈值 ---
        # 因为我们已经压制了波浪，现在可以大胆地把阈值设低，去捞那条弱海天线
        # 使用 OTSU 的一半作为阈值，或者直接固定一个小值
        ret, _ = cv2.threshold(norm_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_edges = (norm_grad > (ret * 0.4)).astype(np.uint8)  # 阈值非常低

        # --- 核心修改 D: 强力形态学开运算 ---
        # 物理擦除短线。只有连续长度超过 40 的横线才能活下来
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        clean_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_OPEN, kernel_open)

        # 骨架化 (变细)
        clean_edges = skeletonize(clean_edges).astype(np.uint8)

        # 再次纹理抑制
        clean_edges = self._suppress_texture(clean_edges)

        # 连接断线
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        closed_edges = cv2.morphologyEx(clean_edges, cv2.MORPH_CLOSE, kernel_connect)

        # 连通域加权
        label_img = label(closed_edges, connectivity=2)
        regions = regionprops(label_img)
        weighted_map = np.zeros_like(closed_edges, dtype=np.float32)

        h, w = gray_img.shape
        for props in regions:
            length = props.major_axis_length

            # 此时剩下的基本都是横线了，我们只卡长度
            if length < w / 4: continue

            # 偏心率再次确认
            if props.eccentricity < 0.95: continue

            # 权重：只看长度，不再看梯度强度(因为已经被Clamp了)
            weight = length ** 2
            weighted_map[label_img == props.label] = weight

        return weighted_map, blurred

    def _radon_gpu(self, image, theta):
        """
        基于 PyTorch Grid Sample 的 GPU 加速 Radon 变换
        :param image: numpy array (H, W)
        :param theta: numpy array (Angles,)
        :return: sinogram numpy array (Diagonal, Angles)
        """
        # 1. 准备数据
        h, w = image.shape
        diagonal = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
        pad_h = (diagonal - h) // 2
        pad_w = (diagonal - w) // 2

        # 转为 Tensor 并移动到 GPU
        img_tensor = torch.from_numpy(image).float().to(self.device)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # 填充为正方形 (Diagonal, Diagonal) 以允许任意旋转不丢失
        # F.pad 顺序是 (left, right, top, bottom)
        # 注意：padding 可能会少 1 像素如果 diff 是奇数，这里简单处理
        pad_left = pad_w
        pad_right = diagonal - w - pad_left
        pad_top = pad_h
        pad_bottom = diagonal - h - pad_top

        img_padded = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom))

        # 2. 循环处理每个角度 (循环比Batch更省显存，适合大图)
        sinogram_cols = []

        # 将角度转为弧度
        theta_rad = torch.deg2rad(torch.from_numpy(theta).float().to(self.device))

        N, C, H_new, W_new = img_padded.shape

        # 我们使用 affine_grid 和 grid_sample 进行旋转
        # grid_sample 是反向采样。如果我们想获得旋转 theta 的投影，
        # 实际上我们需要对采样网格旋转 theta，这样图像看起来就像旋转了 -theta。
        # Radon 标准定义是逆时针旋转图像。
        # 为了得到角度 theta 的投影（即旋转 theta 后垂直向下投影），
        # 我们需要让图像旋转 -theta。
        # 在 grid_sample 中，旋转网格 theta 对应图像旋转 -theta。
        # 所以这里的 Grid 角度就是 theta。

        for angle in theta_rad:
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            # 构建旋转矩阵 (2, 3)
            # PyTorch affine grid 的坐标是 (x, y)
            rot_mat = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0]
            ], device=self.device).unsqueeze(0)  # (1, 2, 3)

            # 生成网格
            grid = F.affine_grid(rot_mat, img_padded.size(), align_corners=False)

            # 采样 (旋转图像)
            rotated_img = F.grid_sample(img_padded, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

            # 垂直求和 (沿着 Height 维度 sum，即 dim=2)
            # 结果形状 (1, 1, 1, W_new) -> (1, W_new)
            # 注意：标准的 Radon 结果通常是 (Distance, Angle)
            # 这里的 W_new 就是 Distance 轴
            projection = torch.sum(rotated_img, dim=2).squeeze()

            sinogram_cols.append(projection)

        # 3. 堆叠结果
        # sinogram_cols 是一个 list，每个元素是 (Diagonal,)
        # stack 后变成 (Angles, Diagonal)
        sinogram_tensor = torch.stack(sinogram_cols, dim=0)

        # 转置为 (Diagonal, Angles) 以匹配 skimage 格式
        sinogram_tensor = sinogram_tensor.t()

        return sinogram_tensor.cpu().numpy()

    def _radon_candidates(self, weighted_map, h_center, scale_idx):
        # 1. 角度范围 (垂直于海天线的法线角度)
        if self.full_scan:
            theta = np.linspace(0., 180., 180, endpoint=False)  # 0-180度
        else:
            theta = np.linspace(85., 95., 180, endpoint=False)  # 局部扫描
        # 替换为gpu版本的radon
        # sinogram = radon(weighted_map, theta=theta, circle=False)
        sinogram = self._radon_gpu(weighted_map, theta)

        max_score = np.max(sinogram)
        if max_score == 0: return [], sinogram

        score_thresh = 0.3 * max_score

        num_potential = 20
        flat_indices = np.argpartition(sinogram.ravel(), -num_potential)[-num_potential:]
        row_indices, col_indices = np.unravel_index(flat_indices, sinogram.shape)

        candidates = []
        center_rho = sinogram.shape[0] // 2

        for r, c in zip(row_indices, col_indices):
            score = sinogram[r, c]
            if score < score_thresh: continue

            angle_deg = theta[c]
            rho = r - center_rho

            angle_rad = np.deg2rad(angle_deg)
            if np.abs(np.sin(angle_rad)) < 1e-4: continue

            #CPU
            # # --- 修正 1: Y轴位置 ---
            # # 图像坐标系Y向下，数学坐标系Y向上，所以由于 rho 产生的偏移量方向相反
            # y_offset = rho / np.sin(angle_rad)
            # Y_img = h_center - y_offset
            #
            # # --- 修正 2: 斜率方向 ---
            # # 同样因为Y轴反转，斜率 dy/dx 也变成了 -dy/dx
            # # 原本是 (angle_deg - 90)，现在要取反，变成 (90 - angle_deg)
            # alpha = 90 - angle_deg

            #GPU
            # --- 修正 1: Y轴位置 ---
            # GPU版本导致 rho 的方向定义反了，所以这里改成【加号】
            # 这样原本算出是负偏移(去上面)的，现在变成正偏移(去下面)
            y_offset = rho / np.sin(angle_rad)
            Y_img = h_center + y_offset  # <--- 改这里：由减变加

            # --- 修正 2: 斜率方向 ---
            # 旋转方向不同导致角度定义反转
            # 改成 angle_deg - 90 即可修正“左高右低”的问题
            alpha = angle_deg - 90  # <--- 改这里：交换顺序

            # 边界检查
            h_img = weighted_map.shape[0]
            if not self.full_scan:
                if Y_img < h_img * 0.05 or Y_img > h_img * 0.95: continue

            candidates.append({'Y': Y_img, 'alpha': alpha, 'score': score, 'scale': scale_idx})

        return candidates, sinogram

    def detect(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        h_center = gray.shape[0] // 2

        all_candidates = []
        debug_info = {}
        collected_sinograms = []  # 初始化列表

        for s in self.scales:
            w_map, blurred = self._get_feature_map(gray, s)
            debug_info[s] = {'map': w_map, 'blurred': blurred}

            #candidates, sinogram = self._radon_candidates(w_map, h_center, s)
            #使用gpu版本的radon
            candidates, sinogram = self._radon_candidates(w_map, h_center, s)

            all_candidates.extend(candidates)
            collected_sinograms.append(sinogram)  # 无论有没有候选点，正弦图都要存下来

        # === 修复点在这里 ===
        if not all_candidates:
            # 即使没有检测到线，也要返回 collected_sinograms (虽然可能里面全是0，但CNN需要这个输入)
            return None, [], debug_info, collected_sinograms

        best = max(all_candidates, key=lambda x: x['score'])
        final_result = (best['Y'], best['alpha'])

        return final_result, all_candidates, debug_info, collected_sinograms

    def visualize_all(self, image, final_result, all_candidates, debug_info):
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3)

        ax_res = fig.add_subplot(gs[0, :])
        res_img = image.copy()
        h, w = image.shape[:2]
        cx = w / 2

        colors = {1: (0, 255, 255), 2: (0, 255, 0), 3: (0, 165, 255)}

        all_candidates.sort(key=lambda x: x['scale'], reverse=True)
        for cand in all_candidates:
            Y, a = cand['Y'], cand['alpha']
            color = colors.get(cand['scale'], (255, 255, 255))
            t = np.tan(np.deg2rad(a))
            y1, y2 = int(Y - t * cx), int(Y + t * (w - cx))
            cv2.line(res_img, (0, y1), (w, y2), color, 1)

        if final_result:
            Y, a = final_result
            t = np.tan(np.deg2rad(a))
            y1, y2 = int(Y - t * cx), int(Y + t * (w - cx))
            cv2.line(res_img, (0, y1), (w, y2), (0, 0, 255), 3)

        ax_res.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
        ax_res.set_title("result")
        ax_res.axis('off')

        for i, s in enumerate(self.scales):
            ax_blur = fig.add_subplot(gs[1, i])
            blurred = debug_info[s]['blurred']
            ax_blur.imshow(blurred, cmap='gray')
            ax_blur.set_title(f"Scale {s}: Median Filter")
            ax_blur.axis('off')

            ax_map = fig.add_subplot(gs[2, i])
            w_map = debug_info[s]['map']
            # 不用 log 显示，直接看原始值，看看波浪是不是真的黑了
            ax_map.imshow(w_map, cmap='gray')
            ax_map.set_title(f"Scale {s}: Weight Map")
            ax_map.axis('off')

        plt.tight_layout()
        plt.show()

    def get_sinograms(self, image):
        """
        核心修改：返回所有尺度的正弦图列表
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 预处理
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        sinograms = []

        for s in self.scales:
            w_map = self._get_feature_map(gray, s)
            # 执行 Radon 变换
            # circle=False 保证对角线区域也被计算
            sinogram = radon(w_map, theta=self.theta, circle=False)
            sinograms.append(sinogram)

        return sinograms

if __name__ == "__main__":

    img_path = r"test4/smd_frames/VIS_Onshore__MVI_1578_VIS__000454.jpg"
    img = cv2.imread(img_path)
    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3])
    import time

    t0 = time.time()
    final_res, candidates, debug_info, collected_sinograms = detector.detect(img)
    t1 = time.time()

    print(f"Total Inference Time: {t1 - t0:.4f} seconds")
    detector.visualize_all(img, final_res, candidates, debug_info)