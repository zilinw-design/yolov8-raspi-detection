"""
Task 1 — YOLOv8 实时目标检测（树莓派5 + IMX219 CSI 摄像头）

功能：加载 YOLOv8n 预训练模型，通过 picamera2 读取 CSI 摄像头，
      将检测结果（画框 + 类别 + 置信度）以 MJPEG 流推送，
      在 Windows 浏览器中查看实时画面。

运行方式（在树莓派5上）：
    python src/task1_basic/detect_pi.py
    python src/task1_basic/detect_pi.py --model yolov8s.pt --conf 0.5 --port 8080

Windows 查看：
    浏览器打开 http://<树莓派IP>:8080

退出：Ctrl+C
"""

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
