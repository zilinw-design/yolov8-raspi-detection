"""
Phase 3a — 屏幕图形测距（USB 摄像头近距离验证）

功能：摄像头对着电脑屏幕显示的测试图形（test_shapes.html），
      检测图形的像素大小，用 pinhole 模型计算距离。

为什么先做屏幕测试：
  - 测试图形像素尺寸精确已知（HTML 中定义）
  - 物理尺寸可用尺子量屏幕上的图形，精确到 mm
  - 近距离（30-80cm）检测更容易，先验证算法
  - 验证通过后再切换到 A4 纸 1.5m 场景

原理：
  1. 屏幕显示已知像素尺寸的黑色图形（正方形/圆/三角）
  2. 摄像头拍到图形 → 轮廓检测 → 测量图形在画面中占的像素
  3. 用尺子量屏幕上图形的物理尺寸（mm）→ 得到 H_real
  4. D = (f × H_real) / h_px

标定：
  摄像头放已知距离 → 屏幕显示测试图 → 按 c → 反算 f

测量：
  任意距离 → 按 m → 计算距离

技术栈：
  cv2.VideoCapture(8) + findContours + minAreaRect + pinhole 公式

运行：
  python src/phase3/screen_measure.py
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 配置
USB_CAMERA_INDEX = 8
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
MIN_SHAPE_AREA = 5000  # 图形最小面积 px²（50cm处400px方≈20000）
FOCAL_FILE = "focal_length.txt"

# CLAHE
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """CLAHE + 灰度 + 高斯模糊。"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    l = clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def detect_shape(frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float], str]:
    """
    检测画面中的黑色图形。

    策略：
      1. 对灰度图二值化（黑色=图形，白色=背景）
      2. findContours 找所有轮廓
      3. 筛选最大轮廓 → minAreaRect
      4. 判断形状类型（正方形/圆形/三角形）

    返回: (found, box_points, pixel_size, shape_type)
      pixel_size: 正方形=边长, 圆形=直径, 三角=底边
    """
    gray = preprocess(frame)

    # 自适应二值化：黑图形在白色背景上
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None, ""

    # 取最大轮廓
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < MIN_SHAPE_AREA:
        return False, None, None, ""

    # 最小外接旋转矩形
    rect = cv2.minAreaRect(best)
    box = np.intp(cv2.boxPoints(rect))
    w, h = rect[1]
    pixel_size = max(w, h)

    # 形状判断
    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.02 * peri, True)
    n_corners = len(approx)

    # 圆形度 = 4π×面积/周长²，完美圆=1
    circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0

    if circularity > 0.85:
        shape_type = "circle"
    elif 3 <= n_corners <= 5:
        shape_type = "triangle" if n_corners == 3 else "square"
    else:
        shape_type = f"polygon({n_corners}pts)"

    return True, box, pixel_size, shape_type


def open_camera() -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(USB_CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("无法打开 /dev/video%d", USB_CAMERA_INDEX)
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("摄像头就绪: %dx%d MJPG", actual_w, actual_h)
    return cap


def load_focal() -> Optional[float]:
    try:
        with open(FOCAL_FILE, "r") as f:
            return float(f.readline().strip())
    except (FileNotFoundError, ValueError):
        return None


def save_focal(f: float) -> None:
    with open(FOCAL_FILE, "w") as fh:
        fh.write(f"{f:.1f}\n")
    logger.info("焦距已保存: f=%.1f px → %s", f, FOCAL_FILE)


def main():
    parser = argparse.ArgumentParser(description="屏幕图形测距")
    parser.add_argument("--camera", type=int, default=USB_CAMERA_INDEX)
    args = parser.parse_args()

    cap = open_camera()
    if cap is None:
        sys.exit(1)

    focal_px = load_focal()
    logger.info("焦距: %s", f"{focal_px:.0f} px" if focal_px else "未标定")

    msg = ""
    distance_mm: Optional[float] = None
    box: Optional[np.ndarray] = None
    shape_type = ""
    pixel_size: Optional[float] = None

    print("\n操作: c=标定  m=测量  q=退出")
    print("标定: 摄像头放已知距离 → 屏幕显示图形 → 按c → 输入 '距离mm,物理尺寸mm'")
    print("      例: 摄像头距屏幕500mm, 屏幕上的正方形实测105mm → 输入 '500,105'\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 每帧都检测（显示实时检测效果）
        found, current_box, current_size, current_type = detect_shape(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            try:
                inp = input("输入 '距离mm,图形物理尺寸mm': ").strip()
                parts = inp.split(",")
                known_d = float(parts[0].strip())
                known_h = float(parts[1].strip())
            except (ValueError, EOFError, IndexError):
                logger.warning("格式错误，例: 500,105")
                continue

            if found and current_size and current_size > 0:
                focal_px = (known_d * current_size) / known_h
                save_focal(focal_px)
                msg = f"标定: D={known_d}mm, 物理={known_h}mm, 像素={current_size:.0f}px → f={focal_px:.0f}px"
                box, pixel_size, shape_type = current_box, current_size, current_type
                distance_mm = known_d
                logger.info(msg)
            else:
                msg = "标定失败: 未检测到图形"

        elif key == ord("m"):
            if focal_px is None:
                msg = "请先标定焦距 (按 c)"
            elif found and current_size and current_size > 0:
                # 用标定时输入的物理尺寸（需再次输入，或复用上次的）
                try:
                    known_h = float(input("输入屏幕上图形的物理尺寸(mm)(用尺子量): ").strip())
                except (ValueError, EOFError):
                    known_h = 105  # fallback
                distance_mm = (focal_px * known_h) / current_size
                box, pixel_size, shape_type = current_box, current_size, current_type
                msg = f"测量: {shape_type} {current_size:.0f}px, 物理{known_h}mm → D={distance_mm:.0f}mm({distance_mm/10:.1f}cm)"
            else:
                msg = "未检测到图形"

        elif key == ord("q"):
            break

        # 绘制
        if box is not None:
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        y = 30
        for line in [
            f"f = {focal_px:.0f} px" if focal_px else "f = 未标定",
            f"类型: {shape_type} | 像素: {pixel_size:.0f} px" if pixel_size else "",
            f"D = {distance_mm:.0f} mm ({distance_mm/10:.1f} cm)" if distance_mm else "",
            msg if msg else "",
            "c:标定 | m:测量 | q:退出",
        ]:
            if line:
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25

        cv2.imshow("Screen Shape Measurement", cv2.resize(frame, (960, 540)))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
