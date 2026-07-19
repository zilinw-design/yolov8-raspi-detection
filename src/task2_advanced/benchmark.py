"""
Phase 2b — YOLO 性能对比实验

功能：
  在树莓派5上定量测试 YOLO 模型的推理速度，包括：
  - 3 个模型 (yolov8n, yolov8s, yolov8m) × 4 个分辨率 (640, 480, 320, 256)
  - 7 档置信度扫描 (0.1 → 0.7)，分析假阳性过滤效果
  - 输出终端格式化表格 + CSV 文件

测量方法（参照 yolo_comparison 和 Ultralytics 官方 benchmark）：
  - Warmup: 模型加载后先跑 10 帧，跳过初始化开销
  - Timed runs: 正式计 100 帧推理时间
  - 只计推理时间，不含摄像头采集和 MJPEG 编码

Qengineering ncnn FPS 基线（RPi 5，纯推理）：
  YOLOv8n @640 = 20.0 FPS, YOLOv8s @640 = 11.0 FPS
  这是我们 PyTorch 版本的速度上限参照。

运行方式（在树莓派5上）：
    python src/task2_advanced/benchmark.py
    python src/task2_advanced/benchmark.py --models yolov8n yolov8s --resolutions 640 320
    python src/task2_advanced/benchmark.py --conf-scan-only

输出：
    终端：格式化对比表 + 结论建议
    outputs/benchmark_YYYYMMDD_HHMMSS.csv：可用 Excel 画图
"""

import argparse
import csv
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------- 硬件 ----------
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
# 配置
# ======================================================================
DEFAULT_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
DEFAULT_RESOLUTIONS = [640, 480, 320, 256]
DEFAULT_CONFS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
WARMUP_FRAMES = 10
TIMED_FRAMES = 100
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Qengineering ncnn 基线（RPi 5，仅供参照）
NCNN_BASELINE: Dict[str, Dict[int, float]] = {
    "yolov8n.pt": {640: 20.0},
    "yolov8s.pt": {640: 11.0},
    # yolov8m 没有公开 ncnn 数据
}


# ======================================================================
# 摄像头辅助
# ======================================================================
class _Camera:
    """最小化摄像头封装，仅用于 benchmark 提供固定输入帧。"""

    def __init__(self):
        self._cam: Picamera2 | None = None

    def open(self) -> bool:
        if not PICAMERA_AVAILABLE:
            return False
        try:
            self._cam = Picamera2()
            config = self._cam.create_video_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "BGR888"},
                controls={"FrameDurationLimits": (33333, 33333)},
            )
            self._cam.configure(config)
            self._cam.start()
            time.sleep(0.5)
            return True
        except Exception as exc:
            logger.error("摄像头初始化失败: %s", exc)
            return False

    def read(self) -> np.ndarray | None:
        if self._cam is None:
            return None
        try:
            return self._cam.capture_array()
        except Exception:
            return None

    def release(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam = None


# ======================================================================
# Benchmark 核心
# ======================================================================
def benchmark_single(
    model: YOLO,
    frame: np.ndarray,
    imgsz: int,
    conf: float,
) -> Tuple[float, int]:
    """
    对单个模型 + 分辨率 + 置信度组合跑一次 Timed Run。

    返回:
        (avg_inference_ms, detection_count)
        avg_inference_ms: 每帧平均推理时间（毫秒）
        detection_count: 该帧检测到的目标数
    """
    # Warmup: 跳过初始化开销
    for _ in range(WARMUP_FRAMES):
        model(frame, conf=conf, imgsz=imgsz, verbose=False)

    # Timed runs
    times_ms = []
    det_count = 0
    for _ in range(TIMED_FRAMES):
        t0 = time.perf_counter()
        results = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        times_ms.append(elapsed)
        if results.boxes is not None:
            det_count += len(results.boxes)

    avg_ms = statistics.mean(times_ms)
    return avg_ms, det_count


def format_table(
    rows: List[dict],
    title: str = "",
) -> str:
    """将结果列表格式化为终端表格。"""
    if not rows:
        return "(无数据)"

    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}

    lines = []
    if title:
        lines.append(f"\n{'=' * (sum(col_widths.values()) + 3 * len(headers) - 1)}")
        lines.append(f"  {title}")
        lines.append(f"{'=' * (sum(col_widths.values()) + 3 * len(headers) - 1)}")

    # 表头
    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    lines.append(header_line)
    lines.append("-|-".join("-" * col_widths[h] for h in headers))

    # 数据行
    for row in rows:
        line = " | ".join(f"{str(row.get(h, '')):<{col_widths[h]}}" for h in headers)
        lines.append(line)

    return "\n".join(lines)


# ======================================================================
# 实验一：分辨率 + 模型 FPS 对比
# ======================================================================
def run_resolution_benchmark(
    models_list: List[str],
    resolutions: List[int],
    conf: float,
    output_dir: Path,
) -> List[dict]:
    """跑模型 × 分辨率全组合，返回结果列表。"""
    logger.info("=" * 60)
    logger.info("实验一：模型 × 分辨率 FPS 对比")
    logger.info("Warmup: %d 帧, Timed: %d 帧/组合", WARMUP_FRAMES, TIMED_FRAMES)

    cam = _Camera()
    if not cam.open():
        logger.error("无法打开摄像头，终止")
        return []

    frame = cam.read()
    if frame is None:
        logger.error("无法读取帧")
        cam.release()
        return []

    results = []
    total = len(models_list) * len(resolutions)

    for model_name in models_list:
        logger.info("加载模型: %s", model_name)
        yolo = YOLO(model_name)

        for imgsz in resolutions:
            logger.info("  测试: %s @ imgsz=%d", Path(model_name).stem, imgsz)

            # 每次测试前重新读一帧（避免帧缓存影响）
            fresh_frame = cam.read()
            if fresh_frame is not None:
                frame = fresh_frame

            avg_ms, det_count = benchmark_single(yolo, frame, imgsz, conf)
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0

            # 查找 ncnn 基线
            ncnn_ref = NCNN_BASELINE.get(model_name, {}).get(imgsz, None)

            results.append({
                "model": Path(model_name).stem,
                "imgsz": imgsz,
                "avg_ms": f"{avg_ms:.1f}",
                "min_fps": f"{1000.0 / (avg_ms * 1.15):.1f}",  # 估算 ~15% 波动
                "avg_fps": f"{fps:.1f}",
                "dets_per_frame": f"{det_count / TIMED_FRAMES:.1f}",
                "ncnn_ref_fps": f"{ncnn_ref:.1f}" if ncnn_ref else "—",
            })

            logger.info("    avg %.1f ms → %.1f FPS, 每帧 %.1f 个检测",
                        avg_ms, fps, det_count / TIMED_FRAMES)

    cam.release()

    # 保存 CSV
    csv_path = output_dir / f"benchmark_resolution_{datetime.now():%Y%m%d_%H%M%S}.csv"
    _save_csv(results, csv_path)
    logger.info("结果已保存: %s", csv_path)

    return results


# ======================================================================
# 实验二：置信度扫描
# ======================================================================
def run_confidence_scan(
    model_name: str,
    imgsz: int,
    confs: List[float],
    output_dir: Path,
) -> List[dict]:
    """固定模型 + 分辨率，扫描不同置信度阈值。"""
    logger.info("=" * 60)
    logger.info("实验二：置信度扫描 — %s @ imgsz=%d", Path(model_name).stem, imgsz)
    logger.info("扫描范围: conf = %.1f → %.1f（步长 0.1）", confs[0], confs[-1])

    cam = _Camera()
    if not cam.open():
        logger.error("无法打开摄像头，终止")
        return []

    frame = cam.read()
    if frame is None:
        logger.error("无法读取帧")
        cam.release()
        return []

    yolo = YOLO(model_name)
    results = []

    baseline_count = None  # conf=0.25 时的检测数作为参照

    for conf in confs:
        avg_ms, det_count = benchmark_single(yolo, frame, imgsz, conf)
        dets_per_frame = det_count / TIMED_FRAMES

        if conf == 0.25:
            baseline_count = dets_per_frame

        # 相对检测率（以 conf=0.25 为 100%）
        relative = ""
        if baseline_count and baseline_count > 0:
            pct = (dets_per_frame / baseline_count) * 100
            relative = f"{pct:.0f}%"

        results.append({
            "conf": f"{conf:.1f}",
            "dets_per_frame": f"{dets_per_frame:.1f}",
            "vs_baseline": relative or "—",
            "avg_ms": f"{avg_ms:.1f}",
            "fps": f"{1000.0 / avg_ms:.1f}" if avg_ms > 0 else "—",
        })

        logger.info("  conf=%.1f → %.1f 检测/帧 (%s), %.1f ms",
                    conf, dets_per_frame, relative, avg_ms)

    cam.release()
    yolo = None  # 释放模型内存

    # 保存 CSV
    csv_path = output_dir / f"benchmark_conf_{datetime.now():%Y%m%d_%H%M%S}.csv"
    _save_csv(results, csv_path)
    logger.info("结果已保存: %s", csv_path)

    return results


# ======================================================================
# 辅助
# ======================================================================
def _save_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _print_conclusions(
    res_results: List[dict],
    conf_results: List[dict],
) -> None:
    """根据数据自动生成结论建议。"""
    print("\n" + "=" * 60)
    print("结论与建议")
    print("=" * 60)

    # 找 FPS > 10 的最高分辨率组合
    fast = [r for r in res_results if float(r["avg_fps"]) > 10]
    if fast:
        best = min(fast, key=lambda r: float(r["avg_ms"]))
        print(f"• 主打推荐：{best['model']} @ imgsz={best['imgsz']}"
              f" → {best['avg_fps']} FPS, {best['avg_ms']}ms/帧")
    else:
        # 找 FPS 最高的
        best = max(res_results, key=lambda r: float(r["avg_fps"]))
        print(f"• 最高 FPS：{best['model']} @ imgsz={best['imgsz']}"
              f" → {best['avg_fps']} FPS（所有组合中最高）")

    # 找 FPS > 5 且精度最高的组合
    usable = [r for r in res_results if float(r["avg_fps"]) > 5]
    if usable:
        best_quality = max(usable, key=lambda r: int(r["imgsz"]))
        print(f"• 平衡推荐：{best_quality['model']} @ imgsz={best_quality['imgsz']}"
              f" → {best_quality['avg_fps']} FPS（质量优先，速度可用）")

    # 与 ncnn 基线对比
    ncnn_entries = [r for r in res_results if r.get("ncnn_ref_fps") != "—"]
    if ncnn_entries:
        for entry in ncnn_entries:
            pytorch_fps = float(entry["avg_fps"])
            ncnn_fps = float(entry["ncnn_ref_fps"])
            ratio = ncnn_fps / pytorch_fps if pytorch_fps > 0 else 0
            print(f"• {entry['model']} @ {entry['imgsz']}: "
                  f"PyTorch {pytorch_fps:.1f} FPS vs ncnn(C++) {ncnn_fps:.1f} FPS "
                  f"→ ncnn 快 {ratio:.1f}×")

    # 置信度建议
    if conf_results:
        print("\n置信度阈值选择：")
        print("  conf 越高 → 检测框越少（假阳性被过滤）→ 画面越干净")
        print("  conf 越低 → 检测框越多（可能包含假阳性）→ 不容易漏检")
        print("  根据 vs_baseline 列选择：保留约 70-80% 检测的 conf 值通常是最优平衡点")

    print("\n提示：将 CSV 文件导入 Excel 可绘制 FPS 曲线图。")
    print("=" * 60)


# ======================================================================
# 主入口
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO 性能对比实验 — 树莓派5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/task2_advanced/benchmark.py                           # 全量测试
  python src/task2_advanced/benchmark.py --models yolov8n yolov8s  # 只测两个模型
  python src/task2_advanced/benchmark.py --resolutions 640 320     # 只测两档分辨率
  python src/task2_advanced/benchmark.py --conf-scan-only          # 只做置信度扫描
  python src/task2_advanced/benchmark.py --no-scan                 # 跳过置信度扫描
        """,
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="测试的模型列表（默认: yolov8n yolov8s yolov8m）",
    )
    parser.add_argument(
        "--resolutions", nargs="+", type=int, default=DEFAULT_RESOLUTIONS,
        help="测试的分辨率列表（默认: 640 480 320 256）",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="实验一的置信度阈值（默认: 0.25）",
    )
    parser.add_argument(
        "--conf-scan-model", default="yolov8n.pt",
        help="实验二使用的模型（默认: yolov8n.pt）",
    )
    parser.add_argument(
        "--conf-scan-imgsz", type=int, default=320,
        help="实验二使用的分辨率（默认: 320）",
    )
    parser.add_argument(
        "--conf-scan-only", action="store_true",
        help="只跑置信度扫描，跳过实验一",
    )
    parser.add_argument(
        "--no-scan", action="store_true",
        help="跳过置信度扫描，只跑实验一",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="CSV 输出目录（默认: outputs）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # ---- 实验一：分辨率 + 模型对比 ----
    res_results = []
    if not args.conf_scan_only:
        res_results = run_resolution_benchmark(
            models_list=args.models,
            resolutions=args.resolutions,
            conf=args.conf,
            output_dir=output_dir,
        )
        if res_results:
            print(format_table(res_results, title="模型 × 分辨率 FPS 对比"))

    # ---- 实验二：置信度扫描 ----
    conf_results = []
    if not args.no_scan:
        conf_results = run_confidence_scan(
            model_name=args.conf_scan_model,
            imgsz=args.conf_scan_imgsz,
            confs=DEFAULT_CONFS,
            output_dir=output_dir,
        )
        if conf_results:
            # 简化显示：只保留关键列
            display = [{k: r[k] for k in ["conf", "dets_per_frame", "vs_baseline", "fps"]}
                       for r in conf_results]
            print(format_table(display, title="置信度扫描"))

    # ---- 结论 ----
    if res_results or conf_results:
        _print_conclusions(res_results, conf_results)
    else:
        logger.warning("没有收集到任何数据。请确认：摄像头已连接、虚拟环境已激活")


if __name__ == "__main__":
    main()
