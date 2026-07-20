"""
Phase 3a — 简单 Pinhole 单目测距

功能：用 4K USB Camera 拍摄 A4 纸（带黑色边框线），检测 A4 纸像素高度，
      用相似三角形公式 D = f × H / h 计算距离。

原理（Pinhole 相机模型）：
      实物高度 H(297mm)
      |←──── 距离 D(mm) ────→|
      相机 ────────────────────┼
      焦距 f(px)               |
      ┌───┐                    |
      │   │ 像素高 h(px)      |
      └───┘                    |

      D = (f × H) / h
      f 通过标定获得：已知距离放 A4 纸，反算 f

技术栈：
  - cv2.VideoCapture(8) → 4K USB Camera, MJPG 1920×1080
  - cv2.findContours → 检测 A4 纸黑边框
  - CLAHE 预处理（Phase 2b）→ 暗光下增强边框对比度
  - NumPy → 数值计算

运行方式（在树莓派上，不需要 venv 也可运行）：
    python src/phase3/pinhole_measure.py

操作：
    1. 按 'c' → 标定焦距：把 A4 纸放在已知距离（如 150cm）
    2. 输入实际距离(mm) → 程序自动反算 f
    3. 按 'm' → 测量：任意距离放 A4 纸，自动计算距离
    4. 按 'q' → 退出
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

# ======================================================================
# 配置常量
# ======================================================================
USB_CAMERA_INDEX = 8          # /dev/video8 → 4K USB Camera
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
A4_REAL_HEIGHT_MM = 297       # A4 纸高度（纵向，含黑边框）
A4_REAL_WIDTH_MM = 210        # A4 纸宽度

# CLAHE 参数（Phase 2b 验证通过）
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# 检测参数
CANNY_LOW = 50
CANNY_HIGH = 150
MIN_CONTOUR_AREA = 50000     # A4 纸最小面积（1920×1080下约20万px²）
FOCAL_FILE = "focal_length.txt"  # 焦距保存文件


# ======================================================================
# 图像预处理
# ======================================================================
def preprocess(frame: np.ndarray) -> np.ndarray:
    """CLAHE 增强 + 灰度化 + 高斯模糊，为轮廓检测准备。"""
    # CLAHE（仅 L 通道）
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur


# ======================================================================
# A4 纸检测
# ======================================================================
def find_a4_paper(frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
    """
    检测画面中的 A4 纸（黑色边框 + 白色内部）。

    策略：
      1. 对灰度图做 Canny 边缘检测 → 得到黑色边框的边
      2. findContours 找所有轮廓
      3. 筛选面积 > MIN_CONTOUR_AREA 且形状接近四边形的轮廓
      4. 用 minAreaRect 获取最小外接旋转矩形
      5. 返回矩形顶点和像素高度

    返回:
        (found, box_points, height_px)
        box_points: 4个角点 (N,1,2) 或 None
        height_px: 矩形短边像素高度（A4宽）或长边（A4高），取决于摆放方向
    """
    processed = preprocess(frame)
    edges = cv2.Canny(processed, CANNY_LOW, CANNY_HIGH)

    # 膨胀连接断边
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_height = 0.0
    best_box = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        # 周长+多边形近似 → 判断是否接近四边形
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # A4 纸黑边框有 4 个角点 ±2
        if len(approx) < 4 or len(approx) > 8:
            continue

        # 最小外接旋转矩形（比 boundingRect 更适合倾斜的 A4）
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # 矩形的宽和高（宽<高 → 宽=短边, 高=长边）
        w, h = rect[1]
        if w > h:
            w, h = h, w

        # A4 纸在 1920×1080 下，短边应该 > 100px
        if w < 100:
            continue

        # 取最大面积的那个（如果有多个候选）
        if area > best_height:
            best_height = area

        best_box = box
        best_height_px = h  # 长边 = A4 纸高度（纵向摆放时）

    if best_box is not None:
        return True, best_box, best_height_px

    return False, None, None


# ======================================================================
# 焦距标定
# ======================================================================
def calibrate_focal(frame: np.ndarray, known_distance_mm: float) -> Optional[float]:
    """
    用已知距离的 A4 纸标定焦距。

    公式：f = (D × h) / H
      其中 D = known_distance_mm, h = 检测到的 A4 像素高, H = 297mm
    """
    found, box, height_px = find_a4_paper(frame)
    if not found:
        logger.warning("未检测到 A4 纸，标定失败。请确认：")
        logger.warning("  1. A4 纸在画面中清晰可见")
        logger.warning("  2. 黑边框完整（线宽 2cm）")
        logger.warning("  3. 光线充足（暗光下开启 CLAHE）")
        return None

    f = (known_distance_mm * height_px) / A4_REAL_HEIGHT_MM
    logger.info("标定成功: 距离=%dmm, A4高=%dpx → 焦距 f=%.1f px",
                known_distance_mm, int(height_px), f)

    # 保存焦距
    with open(FOCAL_FILE, "w") as fh:
        fh.write(f"{f:.1f}\n")
    logger.info("焦距已保存到 %s", FOCAL_FILE)

    return f


def load_focal() -> Optional[float]:
    """读取已保存的焦距值。"""
    try:
        with open(FOCAL_FILE, "r") as fh:
            return float(fh.readline().strip())
    except (FileNotFoundError, ValueError):
        return None


# ======================================================================
# 距离测量
# ======================================================================
def measure_distance(frame: np.ndarray, focal_px: float) -> Tuple[bool, Optional[float], Optional[float], Optional[np.ndarray]]:
    """
    用已标定的焦距测量 A4 纸距离。

    返回:
        (found, distance_mm, height_px, box_points)
    """
    found, box, height_px = find_a4_paper(frame)
    if not found:
        return False, None, None, None

    D = (focal_px * A4_REAL_HEIGHT_MM) / height_px
    return True, D, height_px, box


# ======================================================================
# 摄像头初始化
# ======================================================================
def open_camera() -> Optional[cv2.VideoCapture]:
    """打开 4K USB Camera。"""
    cap = cv2.VideoCapture(USB_CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("无法打开 /dev/video%d (4K USB Camera)", USB_CAMERA_INDEX)
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 确认实际分辨率
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("4K USB Camera 就绪: %dx%d MJPG", actual_w, actual_h)
    return cap


# ======================================================================
# 画面绘制
# ======================================================================
def draw_overlay(
    frame: np.ndarray,
    box: Optional[np.ndarray],
    distance_mm: Optional[float],
    focal_px: Optional[float],
    calib_msg: str = "",
) -> np.ndarray:
    """在画面上叠加检测框、距离、状态信息。"""
    if box is not None:
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    # 状态栏（左上）
    y = 30
    status = []
    if focal_px:
        status.append(f"f = {focal_px:.0f} px")
    else:
        status.append("f = 未标定 (按 c 标定)")
    if distance_mm:
        status.append(f"D = {distance_mm:.0f} mm ({distance_mm/10:.1f} cm)")
    if calib_msg:
        status.append(calib_msg)

    for line in status:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 28

    # 操作提示（底部）
    hints = [
        "c: 标定焦距 | m: 测量距离 | q: 退出",
        "标定: A4纸放已知距离 → 按c → 输入距离(mm)",
    ]
    y = frame.shape[0] - 50
    for line in hints:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20

    return frame


# ======================================================================
# 主循环
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Pinhole 单目测距 — 4K USB Camera")
    parser.add_argument("--camera", type=int, default=USB_CAMERA_INDEX, help="摄像头索引")
    parser.add_argument("--width", type=int, default=CAPTURE_WIDTH)
    parser.add_argument("--height", type=int, default=CAPTURE_HEIGHT)
    args = parser.parse_args()

    cap = open_camera()
    if cap is None:
        sys.exit(1)

    focal_px = load_focal()
    if focal_px:
        logger.info("已加载焦距: f = %.1f px", focal_px)
    else:
        logger.info("未标定焦距。请把 A4 纸放在已知距离处，按 c 标定。")

    calib_msg = ""
    distance_mm: Optional[float] = None
    box_points: Optional[np.ndarray] = None

    logger.info("操作: c=标定 m=测量 q=退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("帧读取失败")
            continue

        key = cv2.waitKey(1) & 0xFF

        # ---- c: 标定焦距 ----
        if key == ord("c"):
            # 从终端读取已知距离
            try:
                user_input = input("请输入 A4 纸到摄像头的实际距离(mm): ").strip()
                known_d = float(user_input)
            except (ValueError, EOFError):
                logger.warning("输入无效")
                continue

            f = calibrate_focal(frame, known_d)
            if f:
                focal_px = f
                calib_msg = f"标定完成 f={f:.0f}px"
                distance_mm = known_d
                # 重新检测一次获取 box
                found, box_points, _ = find_a4_paper(frame)
            else:
                calib_msg = "标定失败"

        # ---- m: 测量距离 ----
        elif key == ord("m"):
            if focal_px is None:
                calib_msg = "请先标定焦距 (按 c)"
                logger.warning(calib_msg)
            else:
                found, D, h_px, box = measure_distance(frame, focal_px)
                if found:
                    distance_mm = D
                    box_points = box
                    calib_msg = f"测量: A4高={h_px:.0f}px → D={D:.0f}mm"
                else:
                    calib_msg = "未检测到 A4 纸"
                    distance_mm = None
                    box_points = None

        # ---- q: 退出 ----
        elif key == ord("q"):
            break

        # 绘制叠加信息
        frame = draw_overlay(frame, box_points, distance_mm, focal_px, calib_msg)
        cv2.imshow("Pinhole Distance Measurement", cv2.resize(frame, (960, 540)))

    cap.release()
    cv2.destroyAllWindows()
    logger.info("结束")


if __name__ == "__main__":
    main()
