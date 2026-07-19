"""
实验一：模型 × 分辨率 FPS 对比

测什么：纯推理速度。3 模型 × 4 分辨率 = 12 组数据。
需要什么：摄像头打开即可，画面内容不影响结果。
参照基线：ncnn YOLOv8n@640 = 20 FPS（RPi 5）

运行：
    python src/task2_advanced/benchmark_fps.py
    python src/task2_advanced/benchmark_fps.py --models yolov8n.pt yolov8s.pt --resolutions 640 320
"""

import argparse
import csv
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS = ["yolov8n.pt", "yolov8s.pt"]
DEFAULT_RESOLUTIONS = [640, 480, 320, 256]
WARMUP_FRAMES = 10
TIMED_FRAMES = 100

# ncnn 基线（Qengineering, RPi 5 实测）
NCNN_BASELINE: Dict[str, Dict[int, float]] = {
    "yolov8n.pt": {640: 20.0},
    "yolov8s.pt": {640: 11.0},
}


class _Camera:
    def __init__(self):
        self._cam: Picamera2 | None = None

    def open(self) -> bool:
        if not PICAMERA_AVAILABLE:
            return False
        try:
            self._cam = Picamera2()
            config = self._cam.create_video_configuration(
                main={"size": (640, 480), "format": "BGR888"},
                controls={"FrameDurationLimits": (33333, 33333)},
            )
            self._cam.configure(config)
            self._cam.start()
            time.sleep(0.5)
            return True
        except Exception as exc:
            logger.error("摄像头失败: %s", exc)
            return False

    def read(self) -> np.ndarray | None:
        try:
            return self._cam.capture_array() if self._cam else None
        except Exception:
            return None

    def release(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam = None


def benchmark_one(model: YOLO, frame: np.ndarray, imgsz: int) -> Tuple[float, float]:
    """返回 (avg_ms, fps)。warmup 10 帧 + timed 100 帧。"""
    for _ in range(WARMUP_FRAMES):
        model(frame, imgsz=imgsz, verbose=False)

    times = []
    for _ in range(TIMED_FRAMES):
        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = statistics.mean(times)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    return avg_ms, fps


def main():
    parser = argparse.ArgumentParser(description="实验一：FPS 对比")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--resolutions", nargs="+", type=int, default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    cam = _Camera()
    if not cam.open():
        return
    frame = cam.read()
    if frame is None:
        cam.release()
        return

    results = []
    for model_name in args.models:
        logger.info("加载: %s", model_name)
        yolo = YOLO(model_name)
        for imgsz in args.resolutions:
            fresh = cam.read()
            if fresh is not None:
                frame = fresh
            avg_ms, fps = benchmark_one(yolo, frame, imgsz)
            ncnn = NCNN_BASELINE.get(model_name, {}).get(imgsz)
            results.append({
                "model": Path(model_name).stem, "imgsz": str(imgsz),
                "avg_ms": f"{avg_ms:.1f}", "avg_fps": f"{fps:.1f}",
                "ncnn_ref_fps": f"{ncnn:.1f}" if ncnn else "—",
            })
            logger.info("  %s @ %d → %.1f ms = %.1f FPS", Path(model_name).stem, imgsz, avg_ms, fps)

    cam.release()

    # 输出表格
    h = ["model", "imgsz", "avg_ms", "avg_fps", "ncnn_ref_fps"]
    widths = {k: max(len(k), max(len(r[k]) for r in results)) for k in h}
    print("\n" + "=" * 70)
    print("  实验一：模型 × 分辨率 FPS 对比")
    print("=" * 70)
    print(" | ".join(f"{k:<{widths[k]}}" for k in h))
    print("-|-".join("-" * widths[k] for k in h))
    for r in results:
        print(" | ".join(f"{r[k]:<{widths[k]}}" for k in h))

    # 结论
    best = max(results, key=lambda r: float(r["avg_fps"]))
    usable = [r for r in results if float(r["avg_fps"]) >= 8]
    print(f"\n最高 FPS: {best['model']} @ {best['imgsz']} → {best['avg_fps']} FPS")
    if usable:
        rec = max(usable, key=lambda r: int(r["imgsz"]))
        print(f"实时推荐 (≥8 FPS, 最高分辨率): {rec['model']} @ {rec['imgsz']} → {rec['avg_fps']} FPS")
    else:
        print("无组合达到 8 FPS，建议降低分辨率或换更轻量模型")

    # CSV
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"fps_benchmark_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=h)
        w.writeheader()
        w.writerows(results)
    logger.info("CSV: %s", csv_path)


if __name__ == "__main__":
    main()
