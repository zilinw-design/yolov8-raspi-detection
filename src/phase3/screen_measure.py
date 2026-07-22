"""
Phase 3a — 屏幕图形测距（MJPEG 网页版，SSH 无桌面可用）

功能：摄像头对着电脑屏幕显示的测试图形（test_shapes.html），
      检测图形像素大小，用 pinhole 模型计算距离。

显示：Windows 浏览器 → http://<pi-ip>:8081 → 实时画面
控制：SSH 终端 c=标定 m=测量 q=退出

原理：D = (焦距 f × 物体物理尺寸 H_real) / 画面像素尺寸 h_px

运行：
  python src/phase3/screen_measure.py
"""

import argparse
import io
import logging
import queue
import re
import socketserver
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Condition
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 抑制 MJPG 解码警告（USB 摄像头正常现象）
cv2.setLogLevel(0)

CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
MIN_SHAPE_AREA = 5000
FOCAL_FILE = "focal_length.txt"
DEFAULT_PORT = 8081


# ═══════════════════════════════════════════════════
# 图像处理 — 文档扫描管线
# 参考：OpenCV 官方 Document Scanner (LearnOpenCV / Scanbot SDK)
#   1. CLAHE + GaussianBlur + Canny → 找所有边缘
#   2. findContours → approxPolyDP → 找到4边形 = 纸面/屏幕区域
#   3. 透视矫正 → 正视图（背景自动裁掉）
#   4. 在正视图内用 OTSU 检测黑色图形
# ═══════════════════════════════════════════════════
A4_RATIO = 297.0 / 210.0  # ≈ 1.414, A4 纸宽高比


def find_document_region(gray: np.ndarray) -> Optional[np.ndarray]:
    """
    在灰度图中找到最大四边形区域（A4纸/屏幕）。
    返回 4 个角点 (4,1,2) 或 None。
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    # 膨胀连接断边
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 取前 10 大轮廓，依次检查是否接近四边形
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    h, w = gray.shape
    min_area = (w * h) * 0.05  # 四边形至少占画面 5%

    for c in ranked:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            xs, ys = pts[:, 0], pts[:, 1]
            ww = max(xs) - min(xs)
            hh = max(ys) - min(ys)
            if min(ww, hh) <= 0:
                continue
            ratio = max(ww, hh) / min(ww, hh)
            if not (0.9 < ratio < 2.5):
                continue

            # 角点内缩验证：4角点向重心缩 → 检查是否全黑（防误识别窗户/显示器边框）
            centroid = pts.mean(axis=0)
            vectors = centroid - pts  # (4,2) 角点→重心方向
            dists = np.linalg.norm(vectors, axis=1)  # (4,)
            shrink = np.minimum(dists * 0.05, 20)     # 缩 5% 边长，最多 20px
            # 归一化方向向量（处理零距离）
            scaled = np.zeros_like(vectors)
            nonzero = dists > 0
            scaled[nonzero] = vectors[nonzero] / dists[nonzero, np.newaxis]
            sample_pts = pts + scaled * shrink[:, np.newaxis]
            sample_pts = np.clip(sample_pts.astype(int), 0, [w - 1, h - 1])

            all_dark = True
            for sx, sy in sample_pts:
                if gray[sy, sx] > 80:  # 灰度>80=不是黑色
                    all_dark = False
                    break
            if all_dark:
                return approx.astype(np.float32)
    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """四角点排序：左上→右上→右下→左下（透视变换的前置步骤）"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上（和最小）
    rect[2] = pts[np.argmax(s)]  # 右下（和最大）
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect


def four_point_transform(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """透视矫正 → 正视图。"""
    rect = order_points(corners.reshape(4, 2))
    (tl, tr, br, bl) = rect

    # 计算新图像宽高（取顶部和底部中较长的一边）
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    max_w = max(int(width_top), int(width_bot))
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_h = max(int(height_left), int(height_right))

    dst = np.array([
        [0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, M, (max_w, max_h))


def detect_shape(frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float], str]:
    """
    文档扫描管线：
    1. CLAHE + Canny → 找纸面/屏幕的四边框
    2. 透视矫正 → 正视图
    3. OTSU → 检测纸面上的黑色图形
    4. 分类形状类型 + 测量像素尺寸
    """
    # Step 1: CLAHE 增强暗部
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    gray = cv2.cvtColor(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)

    # Step 2: 找四边形区域
    doc_corners = find_document_region(gray)
    if doc_corners is None:
        return False, None, None, ""

    # Step 3: 透视矫正
    warped = four_point_transform(frame, doc_corners)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Step 4: 矫正后的正视图 → OTSU 找黑色图形
    _, binary = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None, ""

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < MIN_SHAPE_AREA:
        return False, None, None, ""

    # Step 5: 形状分类 + 尺寸
    rect = cv2.minAreaRect(best)
    box = np.intp(cv2.boxPoints(rect))
    rw, rh = rect[1]
    pixel_size = max(rw, rh)

    peri = cv2.arcLength(best, True)
    circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
    approx = cv2.approxPolyDP(best, 0.02 * peri, True)
    if circularity > 0.85 and len(approx) > 6:
        stype = "circle"
    elif len(approx) == 3:
        stype = "triangle"
    elif 4 <= len(approx) <= 5:
        aspect = min(rw, rh) / max(rw, rh) if max(rw, rh) > 0 else 0
        rect_ratio = area / (rw * rh) if rw * rh > 0 else 0
        stype = "square" if (aspect > 0.85 and rect_ratio > 0.85) else "polygon"
    else:
        stype = "polygon"

    # 返回在原图上画的框（用矫正前的纸面区域框）
    doc_box = np.intp(cv2.boxPoints(cv2.minAreaRect(doc_corners.reshape(4, 1, 2))))
    return True, doc_box, pixel_size, stype


# ═══════════════════════════════════════════════════
# 摄像头
# ═══════════════════════════════════════════════════
def find_usb_camera() -> Optional[int]:
    """自动发现 4K USB Camera 的设备节点。"""
    try:
        out = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True, stderr=subprocess.STDOUT)
        in_usb = False
        for line in out.split("\n"):
            if "4K USB Camera" in line:
                in_usb = True
                continue
            if in_usb:
                match = re.search(r"/dev/video(\d+)", line)
                if match:
                    logger.info("发现 4K USB Camera: /dev/video%s", match.group(1))
                    return int(match.group(1))
    except Exception:
        pass
    return None


def open_camera() -> Optional[cv2.VideoCapture]:
    dev = find_usb_camera()
    if dev is None:
        logger.error("未找到 4K USB Camera")
        return None
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        logger.error("无法打开 /dev/video%d", dev)
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    # 初始化相机参数（参照 4K USB Camera 手册 camera_init.sh + 审查补全）
    dev_path = f"/dev/video{dev}"
    subprocess.run(["v4l2-ctl", "-d", dev_path,
        "--set-ctrl", "brightness=10,contrast=35,saturation=80,sharpness=2"],
        capture_output=True)
    subprocess.run(["v4l2-ctl", "-d", dev_path,
        "--set-ctrl", "backlight_compensation=36,white_balance_automatic=0,white_balance_temperature=5000"],
        capture_output=True)
    # 对焦：先开自动对焦让它对准，1秒后锁定当前值
    subprocess.run(["v4l2-ctl", "-d", dev_path,
        "--set-ctrl", "focus_automatic_continuous=1"],
        capture_output=True)
    time.sleep(1.0)
    result = subprocess.run(["v4l2-ctl", "-d", dev_path, "-C", "focus_absolute"],
                            capture_output=True, text=True)
    try:
        focus_val = int(result.stdout.strip().split(":")[-1].strip())
        subprocess.run(["v4l2-ctl", "-d", dev_path,
            "--set-ctrl", f"focus_automatic_continuous=0,focus_absolute={focus_val}"],
            capture_output=True)
        logger.info("对焦锁定: focus_absolute=%d", focus_val)
    except Exception:
        logger.warning("对焦锁定失败，保持自动对焦")

    logger.info("摄像头就绪: %dx%d MJPG (brightness=20, contrast=50, saturation=80)",
                CAPTURE_WIDTH, CAPTURE_HEIGHT)
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


# ═══════════════════════════════════════════════════
# MJPEG HTTP
# ═══════════════════════════════════════════════════
class MJPEGHandler(BaseHTTPRequestHandler):
    jpeg_frame: bytes = b""
    condition: Condition = Condition()

    def do_GET(self) -> None:
        if self.path == "/":
            html = ("<!DOCTYPE html><html><head><title>Phase 3a</title>"
                    "<style>body{margin:0;background:#000;display:flex;"
                    "justify-content:center;align-items:center;min-height:100vh}"
                    "img{max-width:100vw;max-height:100vh}</style>"
                    "</head><body><img src='/stream'></body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html.encode())
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            while True:
                with MJPEGHandler.condition:
                    MJPEGHandler.condition.wait(timeout=1.0)
                    frame = MJPEGHandler.jpeg_frame
                if not frame:
                    continue
                try:
                    self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
        else:
            self.send_error(404)

    def log_message(self, format, *args) -> None:
        pass


class MJPEGServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# ═══════════════════════════════════════════════════
# 终端输入（后台线程，queue 通信，避免线程冲突）
# ═══════════════════════════════════════════════════
_cmd_queue: queue.Queue = queue.Queue()


def _input_worker() -> None:
    """后台线程：阻塞等待终端输入，结果放入队列。"""
    while True:
        try:
            cmd = input().strip()
            _cmd_queue.put(cmd)
        except (EOFError, KeyboardInterrupt):
            _cmd_queue.put("q")
            break


def poll_input() -> str:
    """主线程：非阻塞取出队列中的命令，无命令返回 ''。"""
    try:
        return _cmd_queue.get_nowait()
    except queue.Empty:
        return ""


def get_pi_ip() -> str:
    """自动获取树莓派局域网 IP。"""
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True)
        return out.strip().split()[0]
    except Exception:
        return "未知IP"


# ═══════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════
def main() -> None:
    p = argparse.ArgumentParser(description="屏幕图形测距")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = p.parse_args()

    cap = open_camera()
    if cap is None:
        sys.exit(1)

    focal_px = load_focal()

    server = MJPEGServer(("0.0.0.0", args.port), MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=_input_worker, daemon=True).start()

    pi_ip = get_pi_ip()
    print(f"""
╔══════════════════════════════════════════════╗
║  Phase 3a — 屏幕图形测距                   ║
║  浏览器 → http://{pi_ip}:{args.port}              ║
║                                            ║
║  终端输入命令 + 回车:                      ║
║    c → 标定焦距                            ║
║    m → 测量距离                            ║
║    q → 退出                                ║
╚══════════════════════════════════════════════╝
""")
    logger.info("浏览器 → http://%s:%d", pi_ip, args.port)
    logger.info("焦距: %s", f"{focal_px:.0f} px" if focal_px else "未标定")

    distance_mm = None
    last_known_h = 105.0
    overlay = ""
    waiting_calib = False
    frame_count = 0
    found, box, psize, stype = False, None, None, ""

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame_count += 1

        # 每隔 3 帧做一次检测（主瓶颈：CLAHE LAB转换 @1920×1080）
        if frame_count % 3 == 0:
            found, box, psize, stype = detect_shape(frame)
        # 有框就画（无论是新检测还是复用上次）
        if found and box is not None and psize:
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        y = 35
        for text in [
            f"f={focal_px:.0f}px" if focal_px else "f=未标定",
            f"{stype} {psize:.0f}px" if found else "未检测到图形",
            f"D={distance_mm:.0f}mm ({distance_mm/10:.1f}cm)" if distance_mm else "",
            overlay if overlay else "",
        ]:
            if text:
                cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y += 28

        # JPEG → MJPEG
        rgb = np.ascontiguousarray(cv2.resize(frame, (960, 540))[:, :, ::-1])
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=75)
        with MJPEGHandler.condition:
            MJPEGHandler.jpeg_frame = buf.getvalue()
            MJPEGHandler.condition.notify_all()

        # 处理输入
        line = poll_input()
        if not line:
            pass  # 无输入，继续循环
        elif waiting_calib:
            # 上一条是 c 命令，本条当作标定数据
            waiting_calib = False
            if line == "q":
                logger.info("退出")
                cap.release()
                print("已退出")
                sys.exit(0)
            if line in ("c", "m"):
                overlay = f"需要标定数据(例:500,105)，收到 '{line}' 已忽略，重新输入 c"
                logger.warning(overlay)
                continue
            try:
                parts = line.split(",")
                known_d = float(parts[0].strip())
                known_h = float(parts[1].strip())
                last_known_h = known_h
                found2, _, sz, _ = detect_shape(frame)
                if found2 and sz and sz > 0:
                    focal_px = (known_d * sz) / known_h
                    save_focal(focal_px)
                    distance_mm = known_d
                    overlay = f"标定完成: f={focal_px:.0f}px (D={known_d}mm, H={known_h}mm)"
                    logger.info(overlay)
                else:
                    overlay = "标定失败: 未检测到图形"
            except (ValueError, IndexError):
                overlay = "标定失败: 格式错误, 重新输入 c"
                logger.warning("格式错误")
        elif line == "c":
            logger.info(">>> 请输入距离和物体尺寸 (例: 500,105)")
            overlay = "标定: 输入 '距离mm,物体尺寸mm' (例: 500,105)"
            waiting_calib = True
        elif line == "m":
            if focal_px is None:
                logger.warning("请先标定 (输入 c)")
                overlay = "请先标定"
            else:
                found2, _, sz, stype2 = detect_shape(frame)
                if found2 and sz and sz > 0:
                    distance_mm = (focal_px * last_known_h) / sz
                    overlay = f"{stype2} {sz:.0f}px → D={distance_mm:.0f}mm ({distance_mm/10:.1f}cm)"
                    logger.info(overlay)
                else:
                    overlay = "未检测到图形"
                    distance_mm = None
        elif line == "q":
            logger.info("退出")
            break

    cap.release()
    print("已退出")
    sys.exit(0)


if __name__ == "__main__":
    main()
