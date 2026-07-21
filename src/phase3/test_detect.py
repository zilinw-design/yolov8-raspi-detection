"""
Phase 3a — 离线图形检测测试

功能：读图片文件 → 文档扫描管线（找四边形→透视矫正→检测图形）→ 保存结果
不依赖树莓派、不依赖摄像头，Windows 本地直接跑。

用法：
    python src/phase3/test_detect.py tests/test_images/shapes/square.png
    python src/phase3/test_detect.py --dir tests/test_images/shapes/

输出：
    outputs/detect_<原文件名>.jpg  （画上检测框的结果图）
    终端打印：形状类型、像素尺寸、检测坐标
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 检测参数（与 screen_measure.py 保持一致）──
MIN_SHAPE_AREA = 5000


# ═══════════════════════════════════════════════════
# 检测管线（从 screen_measure.py 抽取，纯 OpenCV）
# ═══════════════════════════════════════════════════
def find_document_region(gray: np.ndarray) -> Optional[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    h, w = gray.shape
    min_area = (w * h) * 0.05
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
            if all_dark:
                return approx.astype(np.float32)
    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect


def four_point_transform(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    rect = order_points(corners.reshape(4, 2))
    (tl, tr, br, bl) = rect
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    width_left = np.linalg.norm(bl - tl)
    width_right = np.linalg.norm(br - tr)
    max_w = max(int(width_top), int(width_bot))
    max_h = max(int(width_left), int(width_right))
    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, M, (max_w, max_h))


def detect_shape(frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float], str, Optional[np.ndarray]]:
    """返回 (found, doc_box, pixel_size, shape_type, warped_img)。"""
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
    _, binary = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None, "", warped

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < MIN_SHAPE_AREA:
        return False, None, None, "", warped

    rect = cv2.minAreaRect(best)
    rw, rh = rect[1]
    pixel_size = max(rw, rh)

    peri = cv2.arcLength(best, True)
    circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
    approx = cv2.approxPolyDP(best, 0.02 * peri, True)
    if circularity > 0.85:
        stype = "circle"
    elif len(approx) == 3:
        stype = "triangle"
    elif 4 <= len(approx) <= 5:
        stype = "square"
    else:
        stype = "polygon"

    box = np.intp(cv2.boxPoints(rect))
    return True, box, pixel_size, stype, warped


# ═══════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════
def process_image(image_path: str, output_dir: str = "outputs") -> None:
    """处理单张图片：检测 → 画框 → 保存。"""
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error("无法读取: %s", image_path)
        return

    logger.info("处理: %s (%dx%d)", Path(image_path).name, frame.shape[1], frame.shape[0])

    found, box, psize, stype, warped = detect_shape(frame)

    if found:
        # 在原图上画检测框
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        centroid = box.mean(axis=0).astype(int)
        cv2.putText(frame, f"{stype} {psize:.0f}px",
                    (centroid[0] - 60, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        logger.info("  → %s, %.0f px", stype, psize)

        # 保存矫正图
        if warped is not None:
            warp_path = Path(output_dir) / f"warped_{Path(image_path).name}"
            cv2.imwrite(str(warp_path), warped)
            logger.info("  矫正图: %s", warp_path)
    else:
        cv2.putText(frame, "NOT FOUND", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        logger.info("  → 未检测到（四边形或图形）")

    out_path = Path(output_dir) / f"detect_{Path(image_path).name}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    logger.info("  结果: %s", out_path)


def main():
    p = argparse.ArgumentParser(description="离线图形检测测试")
    p.add_argument("image", nargs="?", help="单张图片路径")
    p.add_argument("--dir", help="批量处理目录下的图片")
    p.add_argument("--output", default="outputs", help="输出目录（默认 outputs）")
    args = p.parse_args()

    if args.dir:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for f in sorted(Path(args.dir).glob(ext)):
                process_image(str(f), args.output)
    elif args.image:
        process_image(args.image, args.output)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
