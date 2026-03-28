#!/usr/bin/env python3
"""
海天线检测系统 — Jetson Xavier NX 优化版 PyQt5 GUI

优化点：
  - 异步推理流水线：捕获与推理解耦，显示帧率可达 30fps
  - 支持视频文件输入
  - 摄像头使用 GStreamer 硬件加速
  - 更轻量的显示更新（跳过不必要的 QPixmap 创建）
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QGroupBox,
    QMessageBox, QSplitter, QFrame,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont

# ---- 项目路径 ----
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "method1_linea_entropy" / "LINEA"))
sys.path.insert(0, str(_PROJECT_ROOT / "method2_unet_radon"))

# ---- 权重路径 ----
WEIGHTS_DIR = _PROJECT_ROOT / "weights"
LINEA_CONFIG = str(_PROJECT_ROOT / "method1_linea_entropy" / "configs" / "linea_entropy_b_enhanced_musid.py")
LINEA_WEIGHTS = str(WEIGHTS_DIR / "linea_entropy_b_enhanced_best.pth")
LINEA_N_CONFIG = str(_PROJECT_ROOT / "method1_linea_entropy" / "configs" / "linea_entropy_n_musid.py")
LINEA_N_WEIGHTS = str(WEIGHTS_DIR / "linea_entropy_n_best.pth")
UNET_WEIGHTS = str(WEIGHTS_DIR / "rghnet_best_c2.pth")
DCE_WEIGHTS = str(WEIGHTS_DIR / "Epoch99.pth")
UNET_TRT_ENGINE = str(WEIGHTS_DIR / "unet_seg_288x512.engine")


# ======================================================================
#  模型加载线程
# ======================================================================

class ModelLoadThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector

    def run(self):
        try:
            self.detector.load()
            self.finished.emit(True, "模型加载完成")
        except Exception as e:
            self.finished.emit(False, f"模型加载失败: {e}")


# ======================================================================
#  主窗口
# ======================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("海天线检测系统")
        self.setMinimumSize(960, 640)

        self._detector = None
        self._pipeline = None  # AsyncDetectorPipeline
        self._camera = None
        self._video_cap = None
        self._camera_timer = QTimer()
        self._camera_timer.timeout.connect(self._on_timer_tick)
        self._is_running = False
        self._load_thread = None
        self._frame_times = []

        self._init_ui()
        self.statusBar().showMessage("就绪 — 请选择方法并加载模型")

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # ---- 顶部控制栏 ----
        ctrl_group = QGroupBox("控制面板")
        ctrl_layout = QHBoxLayout(ctrl_group)

        ctrl_layout.addWidget(QLabel("检测方法:"))
        self._method_combo = QComboBox()
        self._method_combo.addItem("方法二: UNet-Seg TRT (推荐, ~160fps)", "unet_trt")
        self._method_combo.addItem("方法二: UNet-Seg PyTorch FP16 (~20fps)", "unet_seg")
        self._method_combo.addItem("方法一: LINEA Entropy Transformer (~2fps)", "linea")
        self._method_combo.addItem("方法一: LINEA-N 轻量版 (~5fps)", "linea_n")
        self._method_combo.setMinimumWidth(340)
        ctrl_layout.addWidget(self._method_combo)

        self._btn_load = QPushButton("加载模型")
        self._btn_load.clicked.connect(self._on_load_model)
        self._btn_load.setFixedWidth(100)
        ctrl_layout.addWidget(self._btn_load)

        ctrl_layout.addSpacing(20)

        ctrl_layout.addWidget(QLabel("输入源:"))
        self._input_combo = QComboBox()
        self._input_combo.addItem("上传图片", "image")
        self._input_combo.addItem("摄像头", "camera")
        self._input_combo.addItem("视频文件", "video")
        self._input_combo.setMinimumWidth(120)
        ctrl_layout.addWidget(self._input_combo)

        self._btn_run = QPushButton("开始检测")
        self._btn_run.clicked.connect(self._on_run)
        self._btn_run.setFixedWidth(100)
        self._btn_run.setEnabled(False)
        ctrl_layout.addWidget(self._btn_run)

        self._btn_stop = QPushButton("停止")
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_stop.setFixedWidth(60)
        self._btn_stop.setEnabled(False)
        ctrl_layout.addWidget(self._btn_stop)

        ctrl_layout.addStretch()
        main_layout.addWidget(ctrl_group)

        # ---- 显示区 ----
        display_splitter = QSplitter(Qt.Horizontal)

        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(4, 4, 4, 4)
        lbl = QLabel("原始图像")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFont(QFont("", 10, QFont.Bold))
        left_layout.addWidget(lbl)
        self._label_original = QLabel()
        self._label_original.setAlignment(Qt.AlignCenter)
        self._label_original.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        self._label_original.setMinimumSize(400, 300)
        left_layout.addWidget(self._label_original, 1)
        display_splitter.addWidget(left_frame)

        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(4, 4, 4, 4)
        lbl = QLabel("检测结果")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFont(QFont("", 10, QFont.Bold))
        right_layout.addWidget(lbl)
        self._label_result = QLabel()
        self._label_result.setAlignment(Qt.AlignCenter)
        self._label_result.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        self._label_result.setMinimumSize(400, 300)
        right_layout.addWidget(self._label_result, 1)
        display_splitter.addWidget(right_frame)

        main_layout.addWidget(display_splitter, 1)

        # ---- 底部信息栏 ----
        info_group = QGroupBox("性能指标")
        info_layout = QHBoxLayout(info_group)

        self._info_labels = {}
        info_items = [
            ("method", "方法", "—"),
            ("infer_ms", "推理耗时", "— ms"),
            ("fps", "帧率", "— FPS"),
            ("display_fps", "显示帧率", "— FPS"),
            ("status", "状态", "未检测"),
        ]
        for key, title, default in info_items:
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(8, 4, 8, 4)
            t = QLabel(title)
            t.setAlignment(Qt.AlignCenter)
            t.setFont(QFont("", 9))
            t.setStyleSheet("color: #888;")
            fl.addWidget(t)
            v = QLabel(default)
            v.setAlignment(Qt.AlignCenter)
            v.setFont(QFont("", 12, QFont.Bold))
            fl.addWidget(v)
            self._info_labels[key] = v
            info_layout.addWidget(frame)

        main_layout.addWidget(info_group)

    # ----- 模型加载 -----

    def _on_load_model(self):
        method = self._method_combo.currentData()

        self._on_stop()
        if self._detector is not None:
            self._detector.unload()
            self._detector = None

        self._btn_load.setEnabled(False)
        self._btn_run.setEnabled(False)
        self.statusBar().showMessage("正在加载模型，请稍候...")
        QApplication.processEvents()

        try:
            from inference_engine_fast import (
                LINEADetectorFast, UNetSegDetectorFast, UNetSegDetectorTRT
            )

            if method == "linea":
                self._detector = LINEADetectorFast(LINEA_CONFIG, LINEA_WEIGHTS)
            elif method == "linea_n":
                self._detector = LINEADetectorFast(LINEA_N_CONFIG, LINEA_N_WEIGHTS)
            elif method == "unet_trt":
                self._detector = UNetSegDetectorTRT(UNET_TRT_ENGINE)
            else:
                self._detector = UNetSegDetectorFast(UNET_WEIGHTS, DCE_WEIGHTS)

            self._load_thread = ModelLoadThread(self._detector)
            self._load_thread.finished.connect(self._on_model_loaded)
            self._load_thread.start()
        except Exception as e:
            self._btn_load.setEnabled(True)
            QMessageBox.critical(self, "错误", f"创建检测器失败:\n{e}")

    def _on_model_loaded(self, success, msg):
        self._btn_load.setEnabled(True)
        if success:
            self._btn_run.setEnabled(True)
            self.statusBar().showMessage(f"模型加载完成 — {msg}")
            self._info_labels["status"].setText("就绪")
        else:
            self._detector = None
            self.statusBar().showMessage(f"加载失败: {msg}")
            QMessageBox.warning(self, "加载失败", msg)

    # ----- 检测操作 -----

    def _on_run(self):
        if self._detector is None:
            QMessageBox.warning(self, "提示", "请先加载模型")
            return

        input_mode = self._input_combo.currentData()
        if input_mode == "image":
            self._run_image()
        elif input_mode == "camera":
            self._run_camera()
        elif input_mode == "video":
            self._run_video()

    def _run_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片",
            str(_PROJECT_ROOT / "sample_images"),
            "Images (*.png *.PNG *.jpg *.JPG *.jpeg *.JPEG *.bmp *.BMP *.tiff *.TIFF)"
        )
        if not path:
            return

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            QMessageBox.warning(self, "错误", f"无法读取图片:\n{path}")
            return

        self._show_image(self._label_original, img_bgr)
        self.statusBar().showMessage("正在检测...")
        QApplication.processEvents()

        try:
            result_img, info = self._detector.detect(img_bgr)
            self._show_image(self._label_result, result_img)
            self._update_info(info)
            self.statusBar().showMessage("检测完成")
        except Exception as e:
            self.statusBar().showMessage(f"检测失败: {e}")
            QMessageBox.critical(self, "检测失败", str(e))

    def _run_camera(self):
        if self._is_running:
            return

        # 尝试 GStreamer 硬件加速管道
        gst_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )
        self._camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not self._camera.isOpened():
            # 回退到 V4L2
            self._camera = cv2.VideoCapture(0)
            if not self._camera.isOpened():
                QMessageBox.warning(self, "错误", "无法打开摄像头")
                self._camera = None
                return
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._start_async_pipeline()
        self.statusBar().showMessage("摄像头已启动 (异步流水线)")

    def _run_video(self):
        if self._is_running:
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频",
            str(_PROJECT_ROOT),
            "Videos (*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI)"
        )
        if not path:
            return

        self._video_cap = cv2.VideoCapture(path)
        if not self._video_cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开视频:\n{path}")
            self._video_cap = None
            return

        self._start_async_pipeline()
        self.statusBar().showMessage(f"视频播放: {Path(path).name}")

    def _start_async_pipeline(self):
        """启动异步推理流水线。"""
        from inference_engine_fast import AsyncDetectorPipeline

        self._pipeline = AsyncDetectorPipeline(self._detector)
        self._pipeline.start()
        self._is_running = True
        self._btn_stop.setEnabled(True)
        self._btn_run.setEnabled(False)
        self._frame_times = []
        self._camera_timer.start(16)  # ~60Hz 显示刷新

    def _on_timer_tick(self):
        """定时器回调：捕获帧 + 提交推理 + 显示结果。"""
        if not self._is_running:
            return

        t_tick = time.time()

        # 1) 捕获帧
        cap = self._camera if self._camera is not None else self._video_cap
        if cap is None:
            return

        ret, frame = cap.read()
        if not ret:
            if self._video_cap is not None:
                # 视频结束，循环播放
                self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            return

        # 2) 显示原始帧
        self._show_image(self._label_original, frame)

        # 3) 提交给异步推理
        self._pipeline.submit_frame(frame)

        # 4) 获取最新推理结果并显示
        result_img, info = self._pipeline.get_result()
        if result_img is not None:
            self._show_image(self._label_result, result_img)
            self._update_info(info)

        # 5) 计算显示帧率
        self._frame_times.append(t_tick)
        # 保留最近 60 帧
        if len(self._frame_times) > 60:
            self._frame_times = self._frame_times[-60:]
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                display_fps = (len(self._frame_times) - 1) / dt
                self._info_labels["display_fps"].setText(f"{display_fps:.1f} FPS")

    def _on_stop(self):
        self._camera_timer.stop()
        self._is_running = False

        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

        if self._camera is not None:
            self._camera.release()
            self._camera = None

        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None

        self._btn_stop.setEnabled(False)
        if self._detector is not None:
            self._btn_run.setEnabled(True)
        self.statusBar().showMessage("已停止")

    # ----- 显示 -----

    def _show_image(self, label, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label_size = label.size()
        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.FastTransformation)
        label.setPixmap(scaled)

    def _update_info(self, info):
        method = info.get("method", "—")
        infer_ms = info.get("infer_ms", 0)
        fps = info.get("fps", 0)
        detected = info.get("detected", False)

        self._info_labels["method"].setText(method)
        self._info_labels["infer_ms"].setText(f"{infer_ms:.1f} ms")
        self._info_labels["fps"].setText(f"{fps:.1f} FPS")
        self._info_labels["status"].setText("已检测到海天线" if detected else "未检测到")

        color = "#4CAF50" if detected else "#FF5722"
        self._info_labels["status"].setStyleSheet(f"color: {color};")

    def closeEvent(self, event):
        self._on_stop()
        if self._detector is not None:
            self._detector.unload()
        event.accept()


# ======================================================================
#  入口
# ======================================================================

def main():
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        app.setFont(QFont("Noto Sans CJK SC", 9))
    except Exception:
        pass

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
