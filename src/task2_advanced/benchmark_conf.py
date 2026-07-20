"""
实验二：置信度阈值扫描

测什么：固定一张测试图，conf 从 0.1 扫到 0.7，统计检测数量变化曲线。
需要什么：测试图片文件（--image），或摄像头对着固定场景（--camera）。
为什么用图片：100% 可复现，同一张图每次跑结果完全一致。

用法：
    python src/task2_advanced/benchmark_conf.py --image tests/test_images/simple_01.jpg
    python src/task2_advanced/benchmark_conf.py --camera  # 用摄像头（场景需固定）
    python src/task2_advanced/benchmark_conf.py --image test.jpg --model yolov8s.pt --imgsz 640

输出解读：
    conf=0.1 检测数 = 基准 100%
    conf 越高 → vs_baseline 越低 → 被过滤掉的框越多
    推荐：保留 70-80% 检测的最高 conf 值（假阳性被过滤，真阳性保留）
"""

import argparse
import csv
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_IMGSZ = 320
DEFAULT_CONFS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
WARMUP_FRAMES = 10
TIMED_FRAMES = 100


# ======================================================================
# 暗光预处理（与 detect_pi.py 共用逻辑，文献: Zuiderveld 1994, Poynton 1998）
# ======================================================================
def preprocess_frame(
    frame: np.ndarray,
    enable_clahe: bool = False,
    gamma: float = 1.0,
) -> np.ndarray:
    """CLAHE + Gamma 预处理，增强暗部细节。"""
    if not enable_clahe and gamma == 1.0:
        return frame
    if enable_clahe:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if gamma != 1.0:
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        frame = cv2.LUT(frame, table)
    return frame


def load_image(path: str, imgsz: int) -> np.ndarray:
    """加载图片并缩放到指定尺寸（保持比例，padding 填充）。"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    h, w = img.shape[:2]
    scale = imgsz / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    # 填充到 imgsz × imgsz
    h2, w2 = img.shape[:2]
    canvas = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    canvas[:h2, :w2] = img
    # 还原为原始尺寸（YOLO 内部会处理缩放）
    return cv2.imread(path)


def benchmark_conf_scan(
    model: YOLO,
    image: np.ndarray,
    imgsz: int,
    confs: List[float],
    class_names: dict,
) -> List[dict]:
    """
    固定图片 + 模型 + 分辨率，扫描不同置信度阈值。
    对每个 conf 跑 TIMED_FRAMES 次（同一张图重复推理），取检测数中位数。
    """
    results = []
    baseline_count: Optional[float] = None

    for conf in confs:
        det_counts = []
        times_ms = []

        # Warmup
        for _ in range(WARMUP_FRAMES):
            model(image, conf=conf, imgsz=imgsz, verbose=False)

        # Timed
        for _ in range(TIMED_FRAMES):
            t0 = time.perf_counter()
            result = model(image, conf=conf, imgsz=imgsz, verbose=False)[0]
            times_ms.append((time.perf_counter() - t0) * 1000)
            if result.boxes is not None:
                det_counts.append(len(result.boxes))
            else:
                det_counts.append(0)

        median_dets = statistics.median(det_counts)
        avg_ms = statistics.mean(times_ms)

        if conf == 0.25:
            baseline_count = median_dets

        relative = ""
        if baseline_count and baseline_count > 0:
            relative = f"{(median_dets / baseline_count) * 100:.0f}%"
        elif baseline_count == 0:
            relative = "—"

        results.append({
            "conf": f"{conf:.1f}",
            "dets_per_frame": f"{median_dets:.0f}",
            "vs_baseline(conf=0.25)": relative or "—",
            "avg_ms": f"{avg_ms:.1f}",
            "fps": f"{1000.0 / avg_ms:.1f}" if avg_ms > 0 else "—",
        })

        logger.info("  conf=%.1f → %.0f 检测 (%s), %.1f ms",
                    conf, median_dets, relative, avg_ms)

    return results


def main():
    parser = argparse.ArgumentParser(description="实验二：置信度扫描")
    parser.add_argument("--image", help="测试图片路径（推荐，可复现）")
    parser.add_argument("--camera", action="store_true", help="使用摄像头（场景需固定）")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--clahe", action="store_true", help="CLAHE 暗光增强")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma 校正，暗光推荐 0.7")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    if not args.image and not args.camera:
        logger.error("请指定 --image <图片路径> 或 --camera")
        return

    # ---- 获取测试帧 ----
    if args.image:
        logger.info("加载图片: %s", args.image)
        frame = load_image(args.image, args.imgsz)
    else:
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            config = cam.create_video_configuration(
                main={"size": (640, 480), "format": "BGR888"},
                controls={"FrameDurationLimits": (33333, 33333)},
            )
            cam.configure(config)
            cam.start()
            time.sleep(0.5)
            frame = cam.capture_array()
            cam.stop()
            logger.info("从摄像头捕获一帧")
        except ImportError:
            logger.error("picamera2 不可用（不在树莓派上？）")
            return

    # ---- 加载模型 ----
    logger.info("加载: %s, imgsz=%d", args.model, args.imgsz)
    if args.clahe:
        logger.info("CLAHE: 启用 (clipLimit=2.0, tileGridSize=8x8)")
    if args.gamma != 1.0:
        logger.info("Gamma: %.1f", args.gamma)
    model = YOLO(args.model)
    logger.info("类别数: %d", len(model.names))

    # ---- 预处理 ----
    frame = preprocess_frame(frame, args.clahe, args.gamma)

    # ---- 扫描 ----
    logger.info("扫描 conf: %s", [f"{c:.1f}" for c in DEFAULT_CONFS])
    results = benchmark_conf_scan(model, frame, args.imgsz, DEFAULT_CONFS, model.names)

    # ---- 输出 ----
    h = ["conf", "dets_per_frame", "vs_baseline(conf=0.25)", "avg_ms", "fps"]
    widths = {k: max(len(k), max(len(r[k]) for r in results)) for k in h}
    print("\n" + "=" * 65)
    print("  实验二：置信度扫描")
    if args.image:
        print(f"  图片: {args.image}")
    print("=" * 65)
    print(" | ".join(f"{k:<{widths[k]}}" for k in h))
    print("-|-".join("-" * widths[k] for k in h))
    for r in results:
        print(" | ".join(f"{r[k]:<{widths[k]}}" for k in h))

    print("\n解读：")
    print("  vs_baseline 列 = 以 conf=0.25 为基准的检测保留率")
    print("  保留 70-80% 的最高 conf = 假阳性少且真阳性不丢的最优点")

    # CSV
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem if args.image else "camera"
    csv_path = out / f"conf_scan_{stem}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=h)
        w.writeheader()
        w.writerows(results)
    logger.info("CSV: %s", csv_path)


if __name__ == "__main__":
    main()
