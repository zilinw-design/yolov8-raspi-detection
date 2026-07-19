"""
Phase 2 — YOLOv8 实时目标检测（树莓派5 + IMX219 CSI 摄像头）

功能：
  - 加载 YOLOv8 预训练模型，通过 picamera2 读取 CSI 摄像头
  - 不同类别使用不同固定颜色检测框（人=蓝, 瓶子=绿, 手机=红...）
  - 支持类别过滤（--classes person bottle）
  - 支持框粗细/字体大小自定义（--box-thickness, --font-scale）
  - MJPEG 流推送，Windows 浏览器查看实时画面
  - 画面显示各类别目标数量统计

YOLO 工作原理简述：
  1. 输入图像被缩放到 imgsz×imgsz，划分为 S×S 网格
  2. 每个网格预测 B 个边界框 + 类别概率
  3. NMS（非极大值抑制）：去掉 IoU 重叠过高的重复框，保留置信度最高的
  4. 输出：(x1,y1,x2,y2, confidence, class_id)

运行方式（在树莓派5上）：
    python src/task1_basic/detect_pi.py
    python src/task1_basic/detect_pi.py --classes person bottle cell phone
    python src/task1_basic/detect_pi.py --conf 0.5 --imgsz 320 --box-thickness 3
    python src/task1_basic/detect_pi.py --model yolov8s.pt

Windows 查看：
    浏览器打开 http://<树莓派IP>:8080

退出：Ctrl+C
"""

import argparse
import io
import logging
import socketserver
import time
from collections import Counter
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Condition
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------- 硬件相关 ----------
try:
    from picamera2 import Picamera2

    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

# ---------- YOLO ----------
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ======================================================================
# 配置常量
# ======================================================================
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 320
DEFAULT_PORT = 8080
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_BOX_THICKNESS = 2
DEFAULT_FONT_SCALE = 0.5
JPEG_QUALITY = 75

# ======================================================================
# 类别颜色映射
# ======================================================================
# 每个 COCO 类别 ID 固定对应一种颜色 (B, G, R)
# 电赛常见物体优先分配高辨识度颜色
_CLASS_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    # (B, G, R) 格式 — 用于 cv2 绘图，再经 BGR→RGB 转换后在浏览器正确显示
    0:  (255, 56, 56),    # person       — 浏览器端蓝色
    39: (56, 255, 56),    # bottle       — 浏览器端绿色
    67: (56, 56, 255),    # cell phone   — 浏览器端红色
    73: (0, 215, 255),    # book         — 浏览器端金色
    41: (56, 255, 255),   # cup          — 浏览器端黄色
    11: (255, 56, 255),   # stop sign    — 浏览器端品红
    9:  (255, 128, 0),    # traffic light— 浏览器端橙色
    47: (255, 255, 56),   # banana       — 浏览器端青色
    46: (128, 56, 255),   # apple        — 浏览器端粉紫
    62: (56, 128, 255),   # tv           — 浏览器端橙红
    63: (128, 255, 56),   # laptop       — 浏览器端翠绿
    64: (255, 56, 128),   # mouse        — 浏览器端玫红
    65: (56, 255, 128),   # remote       — 浏览器端薄荷绿
    72: (128, 56, 128),   # refrigerator — 浏览器端紫
    76: (255, 128, 128),  # scissors     — 浏览器端浅红
}

# 备用调色板（未在映射表中的类别自动分配）
_FALLBACK_PALETTE: List[Tuple[int, int, int]] = [
    # (B, G, R) — cv2 绘图用，经 BGR→RGB 转换后浏览器端正确显示
    (255, 56, 56),   # 浏览器端: 蓝
    (56, 255, 56),   # 浏览器端: 绿
    (56, 56, 255),   # 浏览器端: 红
    (0, 215, 255),   # 浏览器端: 金
    (56, 255, 255),  # 浏览器端: 黄
    (255, 56, 255),  # 浏览器端: 品红
    (255, 128, 0),   # 浏览器端: 橙
    (255, 255, 56),  # 浏览器端: 青
    (128, 56, 255),  # 浏览器端: 紫
    (56, 128, 255),  # 浏览器端: 橙红
    (128, 255, 56),  # 浏览器端: 翠绿
    (255, 56, 128),  # 浏览器端: 玫红
    (56, 255, 128),  # 浏览器端: 薄荷绿
    (128, 128, 255), # 浏览器端: 浅紫
    (255, 128, 128), # 浏览器端: 浅红
    (128, 255, 128), # 浏览器端: 浅绿
    (128, 128, 128), # 浏览器端: 灰
    (255, 200, 100), # 浏览器端: 暖橙
    (100, 200, 255), # 浏览器端: 天蓝
    (200, 100, 255), # 浏览器端: 薰衣草
]


def get_color(class_id: int) -> Tuple[int, int, int]:
    """根据类别 ID 返回固定颜色 (B, G, R)。

    优先查 _CLASS_COLOR_MAP，未找到则用调色板自动分配。
    同一 class_id 始终返回相同颜色。
    """
    if class_id in _CLASS_COLOR_MAP:
        return _CLASS_COLOR_MAP[class_id]
    return _FALLBACK_PALETTE[class_id % len(_FALLBACK_PALETTE)]


# ======================================================================
# 自定义检测框绘图
# ======================================================================
def draw_boxes(
    frame: np.ndarray,
    boxes: np.ndarray,
    class_ids: List[int],
    confidences: List[float],
    class_names: Dict[int, str],
    allowed_classes: Optional[List[str]] = None,
    box_thickness: int = DEFAULT_BOX_THICKNESS,
    font_scale: float = DEFAULT_FONT_SCALE,
) -> np.ndarray:
    """在画面上手动绘制检测框、类别标签和置信度。

    为什么不直接用 results.plot()？
      - plot() 每次随机生成颜色，同一类别可能颜色不同 → 画面混乱
      - plot() 样式不可配置
      - 手绘可以固定颜色映射、自定义粗细/字体、支持类别过滤

    参数:
        frame: BGR numpy array（摄像头原始帧）
        boxes: shape (N,4) → [[x1,y1,x2,y2], ...]
        class_ids: 每个框的类别 ID 列表
        confidences: 每个框的置信度列表
        class_names: {0: 'person', 1: 'bicycle', ...}
        allowed_classes: 如果指定，只绘制在此列表中的类别
        box_thickness: 检测框线条粗细
        font_scale: 标签字体大小

    返回:
        绘制后的 BGR numpy array
    """
    for i in range(len(boxes)):
        cls_id = class_ids[i]
        class_name = class_names.get(cls_id, f"#{cls_id}")

        # ---- 类别过滤：不在允许列表中的跳过 ----
        if allowed_classes is not None and class_name not in allowed_classes:
            continue

        conf = confidences[i]
        x1, y1, x2, y2 = map(int, boxes[i])
        color = get_color(cls_id)
        label = f"{class_name} {conf:.2f}"

        # ---- 绘制检测框 ----
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        # ---- 绘制标签背景 ----
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_thickness
        )
        cv2.rectangle(
            frame,
            (x1, y1 - th - baseline - 4),
            (x1 + tw, y1),
            color,
            -1,  # 填充
        )

        # ---- 绘制标签文字（白色） ----
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # 白色文字
            box_thickness,
            cv2.LINE_AA,
        )

    return frame


def collect_class_names(raw: str | None) -> Optional[List[str]]:
    """解析 --classes 参数：'person,bottle,cell phone' → ['person', 'bottle', 'cell phone']"""
    if raw is None:
        return None
    return [name.strip() for name in raw.split(",") if name.strip()]


# ======================================================================
# CSI 摄像头封装（picamera2）
# ======================================================================
class CSICamera:
    """树莓派5 CSI 摄像头，BGR888 格式，与 OpenCV 直接兼容。"""

    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        self.width = width
        self.height = height
        self._cam: Picamera2 | None = None

    def open(self) -> bool:
        if not PICAMERA_AVAILABLE:
            logger.error("picamera2 未安装！请在树莓派上运行: pip install picamera2")
            return False
        try:
            self._cam = Picamera2()
            config = self._cam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"},
                controls={"FrameDurationLimits": (33333, 33333)},
            )
            self._cam.configure(config)
            self._cam.start()
            time.sleep(0.5)
            logger.info("CSI 摄像头就绪: %dx%d BGR888", self.width, self.height)
            return True
        except Exception as exc:
            logger.error("摄像头初始化失败: %s", exc)
            return False

    def read(self) -> Tuple[bool, np.ndarray | None]:
        if self._cam is None:
            return False, None
        try:
            frame = self._cam.capture_array()
            return True, frame
        except Exception as exc:
            logger.warning("帧捕获失败: %s", exc)
            return False, None

    def release(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam = None


# ======================================================================
# MJPEG HTTP 流服务器
# ======================================================================
class MJPEGHandler(BaseHTTPRequestHandler):
    """多客户端 MJPEG 流。"""

    jpeg_frame: bytes = b""
    condition: Condition = Condition()

    def do_GET(self) -> None:
        if self.path == "/":
            self._serve_html()
        elif self.path == "/stream":
            self._serve_stream()
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        html = (
            "<!DOCTYPE html><html><head>"
            "<title>YOLOv8 实时检测</title>"
            "<style>body{margin:0;background:#000;display:flex;"
            "justify-content:center;align-items:center;min-height:100vh}"
            "img{max-width:100vw;max-height:100vh}</style>"
            "</head><body>"
            '<img src="/stream" alt="YOLOv8 Stream" />'
            "</body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        while True:
            with MJPEGHandler.condition:
                MJPEGHandler.condition.wait()
                frame = MJPEGHandler.jpeg_frame
            try:
                self.wfile.write(b"--FRAME\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, format, *args) -> None:
        pass


class MJPEGServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# ======================================================================
# 主循环
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 实时检测 — 树莓派5 CSI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型文件")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="推理分辨率")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP 端口")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="摄像头高度")
    # ---- Phase 2 新增参数 ----
    parser.add_argument(
        "--classes",
        default=None,
        help="只检测的类别，英文逗号分隔，如 'person,bottle,cell phone'",
    )
    parser.add_argument(
        "--box-thickness",
        type=int,
        default=DEFAULT_BOX_THICKNESS,
        help=f"检测框线条粗细（默认 {DEFAULT_BOX_THICKNESS}）",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=DEFAULT_FONT_SCALE,
        help=f"标签字体大小（默认 {DEFAULT_FONT_SCALE}）",
    )
    args = parser.parse_args()

    # ---- 解析类别过滤列表 ----
    allowed_classes = collect_class_names(args.classes)
    if allowed_classes:
        logger.info("类别过滤: %s", allowed_classes)
    else:
        logger.info("类别过滤: 关闭（检测全部 %d 类）", 80)

    # ---- 加载模型 ----
    logger.info("加载模型: %s", args.model)
    model = YOLO(args.model)
    logger.info("模型类别数: %d, 推理分辨率: %d", len(model.names), args.imgsz)

    # ---- 打开摄像头 ----
    cam = CSICamera(args.width, args.height)
    if not cam.open():
        return

    # ---- 启动 HTTP 服务器 ----
    server = MJPEGServer(("0.0.0.0", args.port), MJPEGHandler)
    logger.info("MJPEG 服务已启动: http://0.0.0.0:%d", args.port)
    logger.info("请在 Windows 浏览器打开: http://<树莓派IP>:%d", args.port)

    from threading import Thread
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # ---- 检测循环 ----
    fps_window = []
    frame_count = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()

            # ---- YOLO 推理 ----
            # results[0] 包含：boxes（边界框）、names（类别映射）
            results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

            # ---- 自定义绘图（替代 results.plot()） ----
            if results.boxes is not None and len(results.boxes) > 0:
                boxes_xyxy = results.boxes.xyxy.cpu().numpy()
                cls_ids = results.boxes.cls.int().cpu().tolist()
                confs = results.boxes.conf.float().cpu().tolist()

                # 类别数量统计
                cls_counter = Counter()
                for cid in cls_ids:
                    name = model.names.get(cid, f"#{cid}")
                    if allowed_classes is None or name in allowed_classes:
                        cls_counter[name] += 1

                annotated = draw_boxes(
                    frame,
                    boxes_xyxy,
                    cls_ids,
                    confs,
                    model.names,
                    allowed_classes=allowed_classes,
                    box_thickness=args.box_thickness,
                    font_scale=args.font_scale,
                )
            else:
                annotated = frame
                cls_counter = Counter()

            # ---- FPS ----
            elapsed = time.perf_counter() - t0
            fps_window.append(1.0 / elapsed if elapsed > 0 else 0)
            if len(fps_window) > 30:
                fps_window.pop(0)
            frame_count += 1
            fps_avg = sum(fps_window) / len(fps_window)

            # ---- 叠加信息到画面 ----
            cv2.putText(
                annotated, f"FPS: {fps_avg:.1f}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.putText(
                annotated, f"Model: {Path(args.model).stem} | imgsz: {args.imgsz}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
            )

            # ---- 各类别目标数量统计 ----
            y_offset = 74
            if cls_counter:
                for name, count in sorted(cls_counter.items(), key=lambda x: -x[1]):
                    cv2.putText(
                        annotated, f"{name}: {count}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
                    )
                    y_offset += 16

            # ---- 编码 JPEG ----
            # picamera2 BGR888 经实测直接传入 PIL 颜色正确，不翻转通道
            pil_img = Image.fromarray(annotated)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=JPEG_QUALITY)

            # ---- 更新共享帧 ----
            with MJPEGHandler.condition:
                MJPEGHandler.jpeg_frame = buf.getvalue()
                MJPEGHandler.condition.notify_all()

    except KeyboardInterrupt:
        logger.info("收到退出信号")
    finally:
        cam.release()
        server.shutdown()
        logger.info(
            "检测结束, 共 %d 帧, 平均 FPS: %.1f",
            frame_count,
            fps_avg if fps_window else 0,
        )


if __name__ == "__main__":
    main()


import argparse
import io
import logging
import socketserver
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Condition
from typing import Tuple

import numpy as np
from PIL import Image

# ---------- 硬件相关 ----------
try:
    from picamera2 import Picamera2

    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

# ---------- YOLO ----------
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ======================================================================
# 配置常量
# ======================================================================
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 320  # 树莓派上用小分辨率保证速度
DEFAULT_PORT = 8080
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
JPEG_QUALITY = 75


# ======================================================================
# CSI 摄像头封装（picamera2）
# ======================================================================
class CSICamera:
    """树莓派5 CSI 摄像头，BGR888 格式，与 OpenCV 直接兼容。"""

    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        self.width = width
        self.height = height
        self._cam: Picamera2 | None = None

    def open(self) -> bool:
        if not PICAMERA_AVAILABLE:
            logger.error("picamera2 未安装！请在树莓派上运行: pip install picamera2")
            return False
        try:
            self._cam = Picamera2()
            config = self._cam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"},
                controls={"FrameDurationLimits": (33333, 33333)},  # ~30fps
            )
            self._cam.configure(config)
            self._cam.start()
            time.sleep(0.5)  # 等待 AE/AWB 稳定
            logger.info("CSI 摄像头就绪: %dx%d BGR888", self.width, self.height)
            return True
        except Exception as exc:
            logger.error("摄像头初始化失败: %s", exc)
            return False

    def read(self) -> Tuple[bool, np.ndarray | None]:
        """返回 (成功, BGR numpy array)。"""
        if self._cam is None:
            return False, None
        try:
            frame = self._cam.capture_array()
            return True, frame
        except Exception as exc:
            logger.warning("帧捕获失败: %s", exc)
            return False, None

    def release(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam = None


# ======================================================================
# MJPEG HTTP 流服务器
# ======================================================================
class MJPEGHandler(BaseHTTPRequestHandler):
    """多客户端 MJPEG 流，每个请求独立接收同一帧。"""

    # 类变量，由主线程更新
    jpeg_frame: bytes = b""
    condition: Condition = Condition()

    def do_GET(self) -> None:
        if self.path == "/":
            self._serve_html()
        elif self.path == "/stream":
            self._serve_stream()
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        html = (
            "<!DOCTYPE html><html><head>"
            "<title>YOLOv8 实时检测</title>"
            "<style>body{margin:0;background:#000;display:flex;"
            "justify-content:center;align-items:center;min-height:100vh}"
            "img{max-width:100vw;max-height:100vh}</style>"
            "</head><body>"
            '<img src="/stream" alt="YOLOv8 Stream" />'
            "</body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        while True:
            with MJPEGHandler.condition:
                MJPEGHandler.condition.wait()
                frame = MJPEGHandler.jpeg_frame
            try:
                self.wfile.write(b"--FRAME\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, format, *args) -> None:
        pass  # 禁用 HTTP 请求日志


class MJPEGServer(socketserver.ThreadingMixIn, HTTPServer):
    """多线程 MJPEG 服务器，支持多客户端同时查看。"""
    allow_reuse_address = True
    daemon_threads = True


# ======================================================================
# 主循环
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 实时检测 — 树莓派5 CSI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型文件")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="推理分辨率")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP 端口")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="摄像头高度")
    args = parser.parse_args()

    # ---- 加载模型 ----
    logger.info("加载模型: %s", args.model)
    model = YOLO(args.model)
    logger.info("模型类别数: %d, 推理分辨率: %d", len(model.names), args.imgsz)

    # ---- 打开摄像头 ----
    cam = CSICamera(args.width, args.height)
    if not cam.open():
        return

    # ---- 启动 HTTP 服务器 ----
    server = MJPEGServer(("0.0.0.0", args.port), MJPEGHandler)
    logger.info("MJPEG 服务已启动: http://0.0.0.0:%d", args.port)
    logger.info("请在 Windows 浏览器打开: http://<树莓派IP>:%d", args.port)

    from threading import Thread
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # ---- 检测循环 ----
    fps_window = []
    frame_count = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()

            # YOLO 推理
            results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

            # 绘制检测结果
            annotated = results.plot()

            # FPS 计算
            elapsed = time.perf_counter() - t0
            fps_window.append(1.0 / elapsed if elapsed > 0 else 0)
            if len(fps_window) > 30:
                fps_window.pop(0)
            frame_count += 1

            # FPS 叠加到画面
            fps_avg = sum(fps_window) / len(fps_window)
            cv2.putText(
                annotated, f"FPS: {fps_avg:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            cv2.putText(
                annotated, f"Model: {Path(args.model).stem}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

            # BGR → RGB → JPEG
            rgb = np.ascontiguousarray(annotated[:, :, ::-1])
            pil_img = Image.fromarray(rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=JPEG_QUALITY)

            # 更新共享帧
            with MJPEGHandler.condition:
                MJPEGHandler.jpeg_frame = buf.getvalue()
                MJPEGHandler.condition.notify_all()

    except KeyboardInterrupt:
        logger.info("收到退出信号")
    finally:
        cam.release()
        server.shutdown()
        logger.info("检测结束, 共 %d 帧, 平均 FPS: %.1f",
                    frame_count, fps_avg if fps_window else 0)


if __name__ == "__main__":
    import cv2  # 仅在绘图时用到

    main()
