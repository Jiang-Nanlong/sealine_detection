#!/usr/bin/env python3
"""
海天线检测系统 — PyQt5 GUI 应用

功能：
  - 两种检测方法切换（LINEA / UNet+Radon+ResNet-34）
  - 摄像头实时检测 / 上传图片检测
  - 性能指标显示
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
    QStatusBar, QMessageBox, QSplitter, QFrame,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont

# ---- 项目路径 ----
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "method1_linea_entropy" / "LINEA"))
sys.path.insert(0, str(_PROJECT_ROOT / "method2_unet_radon"))

# ---- 权重路径定义 ----
WEIGHTS_DIR = _PROJECT_ROOT / "weights"

# 方法一：LINEA
LINEA_CONFIG = str(_PROJECT_ROOT / "method1_linea_entropy" / "configs" / "linea_entropy_b_enhanced_musid.py")
LINEA_WEIGHTS = str(WEIGHTS_DIR / "linea_entropy_b_enhanced_best.pth")

# 方法二：UNet + Radon + ResNet-34（4 通道）
UNET_WEIGHTS = str(WEIGHTS_DIR / "rghnet_best_c2.pth")
DCE_WEIGHTS = str(WEIGHTS_DIR / "Epoch99.pth")
CNN_WEIGHTS = str(WEIGHTS_DIR / "best_fusion_cnn_4ch.pth")


# ======================================================================
#  模型加载线程（避免 GUI 卡死）
# ======================================================================

class ModelLoadThread(QThread):
    finished = pyqtSignal(bool, str)  # success, message

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
        self._camera = None
        self._camera_timer = QTimer()
        self._camera_timer.timeout.connect(self._on_camera_frame)
        self._is_camera_running = False
        self._load_thread = None

        self._init_ui()
        self.statusBar().showMessage("就绪 — 请选择方法并加载模型")

    # ----- UI 构建 -----

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # ---- 顶部控制栏 ----
        ctrl_group = QGroupBox("控制面板")
        ctrl_layout = QHBoxLayout(ctrl_group)

        # 方法选择
        ctrl_layout.addWidget(QLabel("检测方法:"))
        self._method_combo = QComboBox()
        self._method_combo.addItem("ESRLE-L Entropy-Enhanced Transformer", "linea")
        self._method_combo.addItem("UNet + Radon + ResNet-34", "unet_radon")
        self._method_combo.setMinimumWidth(280)
        ctrl_layout.addWidget(self._method_combo)

        # 加载模型按钮
        self._btn_load = QPushButton("加载模型")
        self._btn_load.clicked.connect(self._on_load_model)
        self._btn_load.setFixedWidth(100)
        ctrl_layout.addWidget(self._btn_load)

        ctrl_layout.addSpacing(20)

        # 输入选择
        ctrl_layout.addWidget(QLabel("输入源:"))
        self._input_combo = QComboBox()
        self._input_combo.addItem("上传图片", "image")
        self._input_combo.addItem("摄像头", "camera")
        self._input_combo.setMinimumWidth(120)
        ctrl_layout.addWidget(self._input_combo)

        # 操作按钮
        self._btn_run = QPushButton("开始检测")
        self._btn_run.clicked.connect(self._on_run)
        self._btn_run.setFixedWidth(100)
        self._btn_run.setEnabled(False)
        ctrl_layout.addWidget(self._btn_run)

        self._btn_stop = QPushButton("停止")
        self._btn_stop.clicked.connect(self._on_stop_camera)
        self._btn_stop.setFixedWidth(60)
        self._btn_stop.setEnabled(False)
        ctrl_layout.addWidget(self._btn_stop)

        ctrl_layout.addStretch()
        main_layout.addWidget(ctrl_group)

        # ---- 中间显示区 ----
        display_splitter = QSplitter(Qt.Horizontal)

        # 原始图像
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_label = QLabel("原始图像")
        left_label.setAlignment(Qt.AlignCenter)
        left_label.setFont(QFont("", 10, QFont.Bold))
        left_layout.addWidget(left_label)
        self._label_original = QLabel()
        self._label_original.setAlignment(Qt.AlignCenter)
        self._label_original.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        self._label_original.setMinimumSize(400, 300)
        left_layout.addWidget(self._label_original, 1)
        display_splitter.addWidget(left_frame)

        # 检测结果
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_label = QLabel("检测结果")
        right_label.setAlignment(Qt.AlignCenter)
        right_label.setFont(QFont("", 10, QFont.Bold))
        right_layout.addWidget(right_label)
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

        # 卸载当前模型
        if self._detector is not None:
            self._on_stop_camera()
            self._detector.unload()
            self._detector = None

        self._btn_load.setEnabled(False)
        self._btn_run.setEnabled(False)
        self.statusBar().showMessage("正在加载模型，请稍候...")
        QApplication.processEvents()

        try:
            if method == "linea":
                from inference_engine import LINEADetector
                self._detector = LINEADetector(LINEA_CONFIG, LINEA_WEIGHTS)
            else:
                from inference_engine import UNetRadonDetector
                self._detector = UNetRadonDetector(UNET_WEIGHTS, DCE_WEIGHTS, CNN_WEIGHTS)

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
        else:
            self._run_camera()

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
        if self._is_camera_running:
            return

        self._camera = cv2.VideoCapture(0)
        if not self._camera.isOpened():
            QMessageBox.warning(self, "错误", "无法打开摄像头\n请检查摄像头连接")
            self._camera = None
            return

        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._is_camera_running = True
        self._btn_stop.setEnabled(True)
        self._btn_run.setEnabled(False)
        self._camera_timer.start(30)  # ~33 FPS capture
        self.statusBar().showMessage("摄像头已启动")

    def _on_camera_frame(self):
        if not self._is_camera_running or self._camera is None:
            return

        ret, frame = self._camera.read()
        if not ret:
            return

        self._show_image(self._label_original, frame)

        try:
            result_img, info = self._detector.detect(frame)
            self._show_image(self._label_result, result_img)
            self._update_info(info)
        except Exception as e:
            self._info_labels["status"].setText(f"错误: {e}")

    def _on_stop_camera(self):
        self._camera_timer.stop()
        self._is_camera_running = False
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        self._btn_stop.setEnabled(False)
        if self._detector is not None:
            self._btn_run.setEnabled(True)
        self.statusBar().showMessage("摄像头已停止")

    # ----- 显示辅助 -----

    def _show_image(self, label, img_bgr):
        """将 BGR 图像显示在 QLabel 上。"""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # 自适应缩放
        label_size = label.size()
        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def _update_info(self, info):
        """更新性能指标显示。"""
        method = info.get("method", "—")
        infer_ms = info.get("infer_ms", 0)
        fps = info.get("fps", 0)
        detected = info.get("detected", False)

        self._info_labels["method"].setText(method)
        self._info_labels["infer_ms"].setText(f"{infer_ms:.1f} ms")
        self._info_labels["fps"].setText(f"{fps:.1f} FPS")
        self._info_labels["status"].setText("已检测到海天线" if detected else "未检测到")

        # 颜色提示
        color = "#4CAF50" if detected else "#FF5722"
        self._info_labels["status"].setStyleSheet(f"color: {color};")

    # ----- 窗口关闭 -----

    def closeEvent(self, event):
        self._on_stop_camera()
        if self._detector is not None:
            self._detector.unload()
        event.accept()


# ======================================================================
#  入口
# ======================================================================

def main():
    # 设置环境变量（Jetson 无显示器时可用 SSH X11 forwarding）
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"

    app = QApplication(sys.argv)

    # 设置全局样式
    app.setStyle("Fusion")
    app.setFont(QFont("Noto Sans CJK SC", 9))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
