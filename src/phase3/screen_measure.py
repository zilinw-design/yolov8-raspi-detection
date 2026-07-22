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
MIN_SHAPE_AREA = 15000
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
    # OTSU 二值化 → 直接找黑色边框区域（比 Canny 更适合厚边框合成图像）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    h, w = gray.shape
    min_area = (w * h) * 0.05
    dbg_quad = sum(1 for c in ranked if len(cv2.approxPolyDP(c, 0.02*cv2.arcLength(c,True), True))==4)
    logger.info("DEBUG: 总轮廓=%d 四边形候选=%d 最大面积=%.0f min=%.0f",
                len(contours), dbg_quad, cv2.contourArea(ranked[0]) if ranked else 0, min_area)
    for c in ranked:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            xs, ys = pts[:, 0], pts[:, 1]
            ww, hh = max(xs) - min(xs), max(ys) - min(ys)
            if min(ww, hh) <= 0:
                continue
            ratio = max(ww, hh) / min(ww, hh)
            if not (0.9 < ratio < 2.5):
                continue
            # 角点内缩黑边框验证
            centroid = pts.mean(axis=0)
            vectors = centroid - pts
            dists = np.linalg.norm(vectors, axis=1)
            shrink = np.minimum(dists * 0.05, 20)
            scaled = np.zeros_like(vectors)
            nonzero = dists > 0
            scaled[nonzero] = vectors[nonzero] / dists[nonzero, np.newaxis]
            sample_pts = pts + scaled * shrink[:, np.newaxis]
            sample_pts = np.clip(sample_pts.astype(int), 0, [w - 1, h - 1])
            all_dark = all(gray[sy, sx] <= 80 for sx, sy in sample_pts)
            if not all_dark:
                continue

            # [验证3] 边框中点内缩验证：四边中点向重心缩→检查是否全黑
            ordered = order_points(pts)
            midpoints = []
            for k in range(4):
                p1 = ordered[k]
                p2 = ordered[(k + 1) % 4]
                midpoints.append(((p1 + p2) / 2.0))
            midpoints = np.array(midpoints)
            mp_vectors = centroid - midpoints
            mp_dists = np.linalg.norm(mp_vectors, axis=1)
            mp_shrink = np.minimum(mp_dists * 0.03, 15)  # 边中点缩进更保守
            mp_scaled = np.zeros_like(mp_vectors)
            mp_nz = mp_dists > 0
            mp_scaled[mp_nz] = mp_vectors[mp_nz] / mp_dists[mp_nz, np.newaxis]
            mp_samples = np.clip((midpoints + mp_scaled * mp_shrink[:, np.newaxis]).astype(int), 0, [w-1, h-1])
            edges_dark = all(gray[sy, sx] <= 80 for sx, sy in mp_samples)
            if edges_dark:
                return approx.astype(np.float32)
    return None


def classify_contour(contour: np.ndarray, area: float = 0) -> dict:
    """独占通道分类——每个图形只依赖一个凸包不变量。"""
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if area <= 0:
        area = cv2.contourArea(contour)

    peri_hull = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.03 * peri_hull, True)
    circularity = (4 * np.pi * hull_area) / (peri_hull * peri_hull) if peri_hull > 0 else 0

    rect = cv2.minAreaRect(hull)
    rw, rh = rect[1]
    size = max(rw, rh)
    aspect = min(rw, rh) / max(rw, rh) if max(rw, rh) > 0 else 0
    rect_ratio = hull_area / (rw * rh) if rw * rh > 0 else 0

    # Layer 1: 圆 — circularity 独占（正方理论上限 0.785）
    if circularity > 0.82:
        stype = "circle"
    # Layer 2: 三角 — 顶点数独占（凸包不改变拓扑）
    elif len(approx) == 3:
        stype = "triangle"
    # Layer 3: 正方 — 几何兜底 + 松凸包安全网
    elif (aspect > 0.80 and rect_ratio > 0.80) or len(approx) <= 5:
        stype = "square"
    else:
        stype = "polygon"

    box = np.intp(cv2.boxPoints(rect))
    return {"type": stype, "size_px": size, "contour": contour, "box": box, "area": area}



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


def detect_shape(frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], list, Optional[np.ndarray]]:
    """返回 (found, doc_box, pixel_size, shape_type, warped_img)。"""
    # 统一缩放到最大 1200px（消除高 DPI 厚边框双边缘问题）
    h, w = frame.shape[:2]
    if max(h, w) > 1200:
        scale = 1200.0 / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # CLAHE
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    gray = cv2.cvtColor(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)

    doc_corners = find_document_region(gray)
    if doc_corners is None:
        return False, None, None, "", None

    warped = four_point_transform(frame, doc_corners)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # 裁掉边框区（外圈 20%），只在中心区域找图形
    h2, w2 = warped_gray.shape
    margin = int(min(h2, w2) * 0.20)
    center = warped_gray[margin:h2-margin, margin:w2-margin]
    _, binary = cv2.threshold(center, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, [], warped

    # 对所有合格轮廓分类，polygon 尝试凹点分割
    shapes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_SHAPE_AREA:
            continue
        stp = classify_contour(c, area)
        if not stp:
            continue
        # polygon → 尝试凹点分割 → 子轮廓重分类
        if stp["type"] == "polygon":
            subs = split_by_convexity_defects(c)
            if len(subs) >= 2:
                for sc in subs:
                    sa = cv2.contourArea(sc)
                    if sa < MIN_SHAPE_AREA:
                        continue
                    sstp = classify_contour(sc, sa)
                    if sstp:
                        shapes.append(sstp)
                continue  # 成功分割=跳过原 polygon
        shapes.append(stp)

    if not shapes:
        return False, None, [], warped

    # 对正方形尝试数字识别（用连通域法提取孔洞）
    for s in shapes:
        if s["type"] == "square":
            digit_region = extract_digit_from_square(binary, s["contour"], hierarchy, 0)
            if digit_region is not None:
                digit, conf = recognize_digit(digit_region)
                if conf > 0.3:
                    s["digit"] = digit
                    s["digit_conf"] = round(conf, 2)
                    s["digit_conf"] = round(conf, 2)
                    s["digit_conf"] = round(conf, 2)

    # 取最大图形的外接框作为整体检测框
    largest = max(shapes, key=lambda s: s["size_px"])
    box = np.intp(cv2.boxPoints(cv2.minAreaRect(largest["contour"])))
    return True, box, shapes, warped


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

        # 每隔 3 帧做一次检测
        if frame_count % 3 == 0:
            found, doc_box, shapes, warped = detect_shape(frame)
        # 画文档边界 + 每个图形
        if found and doc_box is not None:
            cv2.drawContours(frame, [doc_box], 0, (0, 255, 0), 2)
        if found and shapes:
            min_sq = None
            for s in shapes:
                clr = (0,255,255) if s['type']=='circle' else (0,0,255) if s['type']=='triangle' else (255,0,0)
                cv2.drawContours(frame, [s['box']], 0, clr, 2)
                lbl = f"{s['type']} {s['size_px']:.0f}px"
                if s.get('digit') is not None: lbl += f" [{s['digit']}]"
                cv2.putText(frame, lbl, (int(s['box'][0][0]), int(s['box'][0][1])-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1)
                if s['type']=='square': min_sq = s if (min_sq is None or s['area']<min_sq['area']) else min_sq

        y = 35
        summary = f"f={focal_px:.0f}px" if focal_px else "f=未标定"
        if found and shapes:
            cat = set(s['type'] for s in shapes)
            summary += f" | {len(shapes)} shapes: " + ", ".join(
                f"{s['type']} {s['size_px']:.0f}px" + (f"[{s['digit']}]" if s.get('digit') is not None else "")
                for s in shapes[:6])
            sqs = [s for s in shapes if s['type']=='square']
            if len(sqs)>1: summary += f" | min_sq={min(sqs,key=lambda s:s['area'])['size_px']:.0f}px"
        if distance_mm: summary += f" | D={distance_mm:.0f}mm"
        cv2.putText(frame, summary, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

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
                found2, _, shapes2, _ = detect_shape(frame)
                if found2 and shapes2:
                    sz = max(s['size_px'] for s in shapes2)
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
                found2, _, shapes2, _ = detect_shape(frame)
                if found2 and shapes2:
                    sz = max(s['size_px'] for s in shapes2)
                    distance_mm = (focal_px * last_known_h) / sz
                    overlay = f"{len(shapes2)} shapes, max {sz:.0f}px → D={distance_mm:.0f}mm ({distance_mm/10:.1f}cm)"
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
