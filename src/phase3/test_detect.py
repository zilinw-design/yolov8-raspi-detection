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
try:
    from .digit_recognition import extract_digit_from_square, recognize_digit, generate_templates
except ImportError:
    from digit_recognition import extract_digit_from_square, recognize_digit

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 检测参数（与 screen_measure.py 保持一致）──
MIN_SHAPE_AREA = 15000  # ~122px, 覆盖多正方形中最小的 1.5in 正方形


# ═══════════════════════════════════════════════════
# 检测管线（从 screen_measure.py 抽取，纯 OpenCV）
# ═══════════════════════════════════════════════════
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


def split_by_convexity_defects(contour: np.ndarray, depth_ratio: float = 0.10) -> list:
    """
    凹点分割——对重叠/粘连的轮廓，用凸性缺陷检测寻找切割点。

    原理（方案 A）：
      凸包 = 补上重叠缺口的"理想形状"
      原始轮廓在重叠处 = 凹入缺口
      convexityDefects 返回每个凹入区域的 {起点, 终点, 最深处, 深度}
      深度大的缺陷 = 真正的重叠切口 → 从最深点切分轮廓

    参数：
      depth_ratio: 缺陷深度/轮廓短边阈值，低于此值的微小凹入忽略（0.10 = 10%）

    返回：子轮廓列表。如果无有效缺陷，返回仅含原轮廓的列表。
    """
    hull = cv2.convexHull(contour, returnPoints=False)  # 返回索引
    h, w = cv2.minAreaRect(contour)[1]
    min_side = min(h, w)

    try:
        defects = cv2.convexityDefects(contour, hull)
    except Exception:
        return [contour]

    if defects is None or len(defects) == 0:
        return [contour]

    # 筛选深度足够的缺陷
    defects = defects.reshape(-1, 4)
    deep_defects = []
    for s, e, f_idx, depth in defects:
        if depth > min_side * depth_ratio:
            far_pt = contour[int(f_idx)][0]
            deep_defects.append((int(far_pt[0]), int(far_pt[1]), float(depth)))

    if len(deep_defects) < 1:
        return [contour]

    # 取深度最大的 1-2 个缺陷点做切分
    deep_defects.sort(key=lambda x: x[2], reverse=True)
    split_pts = deep_defects[:2]  # 最多用 2 个最深缺陷

    # 用 cv2.watershed 风格的距离变换做精确切分
    # 简化版：用缺陷最深点连线做直线切割
    mask = np.zeros((int(min_side * 2), int(min_side * 2)), dtype=np.uint8)
    # 绘制轮廓到 mask
    x, y, cw, ch = cv2.boundingRect(contour)
    cv2.drawContours(mask, [contour - (x - 5, y - 5)], -1, 255, -1,
                     offset=(-(x - 5), -(y - 5)))

    # 距离变换 → 分水岭
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
    _, markers = cv2.connectedComponents(np.uint8(dist > 20))

    # 用缺陷点作为分水岭 seed
    seed_map = np.int32(markers)
    for spx, spy, _ in split_pts:
        sx, sy = spx - x + 5, spy - y + 5
        if 0 <= sx < mask.shape[1] and 0 <= sy < mask.shape[0]:
            seed_map[sy, sx] = seed_map.max() + 1

    cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), seed_map)

    # 提取分割后的连通区域
    sub_contours = []
    for label in range(2, seed_map.max() + 1):
        region = np.uint8(seed_map == label) * 255
        scnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for sc in scnts:
            if cv2.contourArea(sc) > 100:  # 最小面积过滤
                # 偏移回原坐标
                sc[:, 0, 0] += x - 5
                sc[:, 0, 1] += y - 5
                sub_contours.append(sc)

    return sub_contours if len(sub_contours) >= 2 else [contour]


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
# 主函数
# ═══════════════════════════════════════════════════
DATASET_DIR = "D:/Pi/sy2/ai_harness_framework/exam/dataset_nuedc_2025_v2"


def process_image(image_path: str, output_dir: str = "outputs") -> None:
    """处理单张图片：检测 → 画框 → 保存。"""
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error("无法读取: %s", image_path)
        return

    logger.info("处理: %s (%dx%d)", Path(image_path).name, frame.shape[1], frame.shape[0])

    found, box, shapes, warped = detect_shape(frame)

    if found and shapes:
        # 画文档边界
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        # 画每个检测到的图形
        min_square = None
        for s in shapes:
            cv2.drawContours(frame, [s["box"]], 0, (0, 0, 255), 2)
            label = f"{s['type']} {s['size_px']:.0f}px"
            if s.get("digit") is not None:
                label += f" [{s['digit']}]"
            cv2.putText(frame, label,
                        (int(s["box"][0][0]), int(s["box"][0][1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if s["type"] == "square":
                min_square = s if (min_square is None or s["area"] < min_square["area"]) else min_square

        # 汇总（含数字）
        summary = f"{len(shapes)} shapes: " + ", ".join(
            f"{s['type']} {s['size_px']:.0f}px" + (f"[{s['digit']}]" if s.get("digit") is not None else "")
            for s in shapes)
        if min_square and len([s for s in shapes if s["type"]=="square"]) > 1:
            summary += f" | min square={min_square['size_px']:.0f}px"
        logger.info("  → %s", summary)

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
