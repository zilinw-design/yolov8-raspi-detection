"""
Task 1 — YOLOv8 实时目标检测（摄像头）

功能：加载 YOLOv8n 预训练模型，读取摄像头视频流，
      在画面上绘制检测框、物体类别标签和置信度分数。

运行方式：
    python src/task1_basic/detect_camera.py              # 默认：YOLOv8n, 摄像头0
    python src/task1_basic/detect_camera.py --model yolov8s.pt --source 1
    python src/task1_basic/detect_camera.py --source test.mp4   # 视频文件
    python src/task1_basic/detect_camera.py --source image.jpg  # 单张图片

退出：按 ESC 键
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# 配置常量（便于后续任务2调整）
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_SOURCE = "0"  # 默认摄像头
DEFAULT_CONF = 0.25   # 置信度阈值
DEFAULT_IMGSZ = 640   # 推理分辨率


# ---------------------------------------------------------------------------
# 摄像头抽象层：Windows 用 cv2.VideoCapture，树莓派用 Picamera2
# ---------------------------------------------------------------------------
class CameraSource:
    """Windows: cv2.VideoCapture。树莓派：Picamera2（部署时切换）。"""

    def __init__(self, source: str, width: int = 640, height: int = 480):
        self.source = source
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        # 尝试将 source 解析为数字（摄像头索引）
        try:
            camera_id = int(self.source)
        except ValueError:
            camera_id = None

        if camera_id is not None:
            self._cap = cv2.VideoCapture(camera_id)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def read(self) -> Tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def is_image(self) -> bool:
        """判断 source 是否为静态图片（非摄像头/视频流）。"""
        return not self.source.isdigit() and Path(self.source).suffix.lower() in (
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        )

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


# ---------------------------------------------------------------------------
# 检测绘图
# ---------------------------------------------------------------------------
def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    class_ids: List[int],
    confidences: List[float],
    class_names: dict,
) -> np.ndarray:
    """在帧上手动绘制检测框、类别和置信度（供任务2自定义样式练习用）。"""
    # 颜色池（BGR）
    color_palette = [
        (56, 56, 255),   # 蓝
        (0, 255, 127),   # 翠绿
        (0, 215, 255),   # 金
        (204, 0, 204),   # 紫
        (255, 128, 0),   # 橙
        (128, 0, 128),   # 深紫
        (0, 255, 255),   # 黄
        (0, 128, 255),   # 橙红
        (42, 42, 165),   # 棕
        (200, 200, 200), # 浅灰
    ]

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        cls_id = class_ids[i]
        conf = confidences[i]
        class_name = class_names.get(cls_id, f"#{cls_id}")
        label = f"{class_name} {conf:.2f}"

        color = color_palette[cls_id % len(color_palette)]

        # 框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 标签背景
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)

        # 标签文字（白字）
        cv2.putText(
            frame, label, (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
        )

    return frame


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="YOLOv8 实时目标检测")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型文件路径")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="摄像头索引 / 视频路径 / 图片路径")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="推理分辨率")
    args = parser.parse_args()

    # ---- 加载模型 ----
    model_path = args.model
    # 自动下载模型（如本地不存在）
    print(f"[INFO] 加载模型: {model_path}")
    model = YOLO(model_path)
    class_names = model.names  # {0: 'person', 1: 'bicycle', ...}
    print(f"[INFO] 模型类别数: {len(class_names)}")

    # ---- 打开视频源 ----
    cam = CameraSource(args.source)
    if not cam.open():
        print(f"[ERROR] 无法打开视频源: {args.source}")
        return

    # ---- 图片模式（单帧检测） ----
    if cam.is_image():
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] 读取图片失败")
            return
        results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        annotated = results.plot()  # YOLO 内置绘图
        cv2.imshow("YOLOv8 Detection", annotated)
        print("[INFO] 按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ---- 视频流模式 ----
    print(f"[INFO] 开始检测, 按 ESC 退出")
    fps_counter = []
    frame_count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[INFO] 视频流结束")
            break

        t_start = time.perf_counter()

        # YOLO 推理
        results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

        # 绘制结果
        annotated = results.plot()

        # FPS 计算
        t_elapsed = time.perf_counter() - t_start
        fps_counter.append(1.0 / t_elapsed if t_elapsed > 0 else 0)
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        fps_avg = sum(fps_counter) / len(fps_counter)
        frame_count += 1

        # 在画面左上角显示 FPS
        cv2.putText(
            annotated, f"FPS: {fps_avg:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        cv2.imshow("YOLOv8 Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 检测结束, 共处理 {frame_count} 帧, 平均 FPS: {fps_avg:.1f}")


if __name__ == "__main__":
    main()
