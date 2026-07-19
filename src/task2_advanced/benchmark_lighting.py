"""
实验三：光照鲁棒性测试

测什么：同一场景在正常光/暗光/微光下，YOLO 检测置信度的变化。
需要什么：三张同一场景、不同亮度的图片文件（已放在 tests/test_images/）。

为什么测这个：
  YOLO 基于语义理解而非像素颜色，理论上比 HSV 颜色检测更鲁棒。
  同一物体在三张图里的置信度波动 < 0.15 = 光照不敏感。

用法：
    python src/task2_advanced/benchmark_lighting.py \
        --images tests/test_images/正常室内灯光.jpg \
                tests/test_images/窗帘拉上一半.jpg \
                tests/test_images/只开小台灯.jpg
"""

import argparse
import csv
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_IMGSZ = 320
DEFAULT_CONF = 0.25
N_RUNS = 50


def analyze_image(
    model: YOLO, image_path: str, imgsz: int, conf: float, label: str,
) -> List[dict]:
    """对一张图跑 N_RUNS 次推理，收集所有检测结果的 {class, confidence}。"""
    img = cv2.imread(image_path)
    if img is None:
        logger.error("无法读取: %s", image_path)
        return []

    logger.info("  [%s] %s → 推理 %d 次", label, Path(image_path).name, N_RUNS)
    all_dets: List[dict] = []

    for _ in range(10):  # warmup
        model(img, conf=conf, imgsz=imgsz, verbose=False)

    for _ in range(N_RUNS):
        result = model(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.int().cpu().tolist()
            confs = result.boxes.conf.float().cpu().tolist()
            for cid, cfs in zip(cls_ids, confs):
                all_dets.append({
                    "class": model.names.get(cid, f"#{cid}"),
                    "confidence": round(float(cfs), 3),
                })
    return all_dets


def summarize(lighting: str, dets: List[dict]) -> dict:
    """按类别聚合：次数、平均/最低/最高置信度。"""
    by_class: Dict[str, List[float]] = {}
    for d in dets:
        by_class.setdefault(d["class"], []).append(d["confidence"])

    s = {"lighting": lighting, "total_dets": len(dets)}
    for cls_name, confs in sorted(by_class.items()):
        s[f"{cls_name}_count"] = len(confs)
        s[f"{cls_name}_conf_avg"] = round(statistics.mean(confs), 3)
        s[f"{cls_name}_conf_min"] = round(min(confs), 3)
        s[f"{cls_name}_conf_max"] = round(max(confs), 3)
    return s


def main():
    parser = argparse.ArgumentParser(description="实验三：光照鲁棒性")
    parser.add_argument(
        "--images", nargs=3, required=True,
        help="三张同一场景不同亮度的图片：正常光 暗光 微光",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    logger.info("加载: %s, imgsz=%d", args.model, args.imgsz)
    model = YOLO(args.model)

    labels = ["normal", "dim", "dark"]
    all_summaries = []

    for img_path, label in zip(args.images, labels):
        dets = analyze_image(model, img_path, args.imgsz, args.conf, label)
        s = summarize(label, dets)
        all_summaries.append(s)
        logger.info("  检出 %d 个目标", s["total_dets"])

    # ---- 收集类别 ----
    all_classes = set()
    for s in all_summaries:
        for k in s:
            if k.endswith("_count"):
                all_classes.add(k.replace("_count", ""))
    all_classes = sorted(all_classes)

    if not all_classes:
        logger.warning("未检测到任何目标")
        return

    # ---- 输出对比表 ----
    for cls_name in all_classes:
        print(f"\n{'─' * 55}")
        print(f"  类别: {cls_name}")
        print(f"  {'光照':<8} {'次数':<6} {'平均置信度':<10} {'最低':<8} {'最高':<8}")
        confs_across = []
        for s in all_summaries:
            cnt = s.get(f"{cls_name}_count", 0)
            avg = s.get(f"{cls_name}_conf_avg", 0)
            lo = s.get(f"{cls_name}_conf_min", 0)
            hi = s.get(f"{cls_name}_conf_max", 0)
            print(f"  {s['lighting']:<8} {cnt:<6} {avg:<10.3f} {lo:<8.3f} {hi:<8.3f}")
            if avg > 0:
                confs_across.append(avg)
        if len(confs_across) >= 2:
            delta = max(confs_across) - min(confs_across)
            verdict = ("✅ 稳定（<0.15）" if delta < 0.15
                       else "⚠️ 有波动" if delta < 0.3
                       else "❌ 波动较大")
            print(f"  → 置信度最大波动: {delta:.3f}  {verdict}")

    print(f"\n{'=' * 55}")
    print("判定标准：<0.15 稳定 | 0.15-0.30 有波动 | >0.30 光照敏感")

    # CSV
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"lighting_{datetime.now():%Y%m%d_%H%M%S}.csv"
    all_keys = ["lighting", "total_dets"]
    for c in all_classes:
        for suffix in ["_count", "_conf_avg", "_conf_min", "_conf_max"]:
            all_keys.append(f"{c}{suffix}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(all_summaries)
    logger.info("CSV: %s", csv_path)


if __name__ == "__main__":
    main()
