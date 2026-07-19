"""
实验三：光照鲁棒性测试

测什么：同一场景在不同光照下，YOLO 检测置信度的变化。
需要什么：摄像头对着固定场景（手机显示测试图或实物摆放），
         按提示依次切换光照条件（正常光 → 暗光 → 强光）。

为什么测这个：
  YOLO 基于语义理解而非像素颜色，理论上比 HSV 颜色检测更鲁棒。
  本实验量化验证：光照变化时同一物体的置信度波动有多大。

用法：
    python src/task2_advanced/benchmark_lighting.py
    python src/task2_advanced/benchmark_lighting.py --model yolov8s.pt --imgsz 320

流程：
    1. 摄像头对着测试场景，保持不动
    2. 程序提示"正常光照"→ 调整灯光，按 Enter
    3. 程序提示"暗光"→ 拉窗帘/关灯，按 Enter
    4. 程序提示"强光"→ 开灯/用手电筒照，按 Enter
    5. 输出对比表
"""

import argparse
import csv
import logging
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_IMGSZ = 320
DEFAULT_CONF = 0.25
N_FRAMES = 30  # 每种光照拍 30 帧取平均


def capture_stats(
    model: YOLO,
    cam: Picamera2,
    imgsz: int,
    conf: float,
    label: str,
) -> List[dict]:
    """
    在当前光照下连续采集 N_FRAMES 帧，统计每帧检测结果。
    返回每帧的 {class_name, count, avg_confidence} 列表。
    """
    logger.info("  [%s] 采集 %d 帧...", label, N_FRAMES)
    all_dets: List[dict] = []

    for i in range(N_FRAMES):
        frame = cam.capture_array()
        result = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]

        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.int().cpu().tolist()
            confs = result.boxes.conf.float().cpu().tolist()
            for cid, cfs in zip(cls_ids, confs):
                all_dets.append({
                    "class": model.names.get(cid, f"#{cid}"),
                    "confidence": round(float(cfs), 3),
                })

    return all_dets


def summarize(lighting_name: str, dets: List[dict]) -> dict:
    """按类别聚合：出现次数、平均置信度。"""
    by_class: Dict[str, List[float]] = {}
    for d in dets:
        by_class.setdefault(d["class"], []).append(d["confidence"])

    summary = {"lighting": lighting_name, "total_dets": len(dets)}
    for cls_name, confs in sorted(by_class.items()):
        summary[f"{cls_name}_count"] = len(confs)
        summary[f"{cls_name}_conf_avg"] = round(statistics.mean(confs), 3)
    return summary


def main():
    parser = argparse.ArgumentParser(description="实验三：光照鲁棒性")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    if not PICAMERA_AVAILABLE:
        logger.error("picamera2 不可用")
        return

    # ---- 摄像头初始化 ----
    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": (640, 480), "format": "BGR888"},
        controls={"FrameDurationLimits": (33333, 33333)},
    )
    cam.configure(config)
    cam.start()
    time.sleep(0.5)

    # ---- 加载模型 ----
    logger.info("加载: %s", args.model)
    model = YOLO(args.model)

    # ---- 交互式光照切换 ----
    conditions = [
        ("normal", "正常光照 — 调整好灯光后按 Enter"),
        ("dim",    "暗光     — 拉窗帘/关灯后按 Enter"),
        ("bright", "强光     — 开灯/手电筒照射后按 Enter"),
    ]

    all_summaries = []

    for cond_key, instruction in conditions:
        input(f"\n>>> {instruction}")
        dets = capture_stats(model, cam, args.imgsz, args.conf, cond_key)
        s = summarize(cond_key, dets)
        all_summaries.append(s)
        logger.info("  检出: %d 个目标", s["total_dets"])

    cam.stop()

    # ---- 输出对比表 ----
    if not all_summaries:
        logger.warning("无数据")
        return

    # 收集所有出现过的类别
    all_classes = set()
    for s in all_summaries:
        for k in s:
            if k.endswith("_count"):
                all_classes.add(k.replace("_count", ""))
    all_classes = sorted(all_classes)

    # 打印每类对比
    for cls_name in all_classes:
        print(f"\n{'─' * 50}")
        print(f"  类别: {cls_name}")
        print(f"  {'光照':<8} {'出现次数':<10} {'平均置信度':<12} {'波动'}")
        confs_across = []
        for s in all_summaries:
            cnt = s.get(f"{cls_name}_count", 0)
            avg = s.get(f"{cls_name}_conf_avg", 0)
            print(f"  {s['lighting']:<8} {cnt:<10} {avg:<12.3f}", end="")
            if avg > 0:
                confs_across.append(avg)
                print("", end="")
            print()
        if len(confs_across) >= 2:
            delta = max(confs_across) - min(confs_across)
            print(f"  → 最大波动: {delta:.3f} " +
                  ("✅ 稳定" if delta < 0.15 else "⚠️ 有一定波动" if delta < 0.3 else "❌ 波动较大"))

    # CSV
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"lighting_{datetime.now():%Y%m%d_%H%M%S}.csv"
    # 动态列名
    all_keys = ["lighting", "total_dets"]
    for c in all_classes:
        all_keys.append(f"{c}_count")
        all_keys.append(f"{c}_conf_avg")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(all_summaries)
    logger.info("CSV: %s", csv_path)


if __name__ == "__main__":
    main()
