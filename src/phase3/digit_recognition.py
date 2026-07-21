"""
Step 7 — 印刷体数字识别

架构：模板匹配优先（同字体 100%） → 拓扑决策树兜底（字体未知时）。
"""

import cv2
import numpy as np

TEMPLATE_SIZE = (40, 60)


# ═══════════════════════════════════════════════════
# 模板生成
# ═══════════════════════════════════════════════════
def generate_templates() -> dict:
    """生成 0-9 标准印刷体模板（sans-serif, bold）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    templates = {}
    for digit in range(10):
        fig, ax = plt.subplots(figsize=(0.5, 0.75), dpi=100)
        ax.text(0.5, 0.5, str(digit), fontsize=36, ha='center', va='center',
                weight='bold', color='white', family='sans-serif')
        ax.set_facecolor('black')
        ax.axis('off')
        fig.subplots_adjust(0, 0, 1, 1)
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        plt.close(fig)
        templates[digit] = cv2.resize(gray, TEMPLATE_SIZE)
    return templates


# ═══════════════════════════════════════════════════
# 识别
# ═══════════════════════════════════════════════════
def recognize_digit(region: np.ndarray, templates: dict = None) -> tuple:
    """
    模板匹配优先（同字体 100%）+ 拓扑兜底。

    Returns: (digit, confidence)
    """
    if region is None or region.size == 0:
        return -1, 0.0

    # ── Primary: 模板匹配 ──
    if templates and len(templates) == 10:
        region_resized = cv2.resize(region, TEMPLATE_SIZE)
        best_d, best_s = -1, -1.0
        for digit, tmpl in templates.items():
            result = cv2.matchTemplate(region_resized, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_s:
                best_s = max_val
                best_d = digit
        if best_s > 0.6:
            return best_d, round(best_s, 2)

    # ── Fallback: 拓扑决策树 ──
    _, binary = cv2.threshold(region, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return -1, 0.0

    h = hierarchy[0]
    total = binary.shape[0] * binary.shape[1]

    # 统计有效孔洞（>总面积 5%）
    holes = []
    for i, hi in enumerate(h):
        if hi[3] != -1 and cv2.contourArea(contours[i]) > total * 0.05:
            holes.append(i)

    h_box, w_box = binary.shape[:2]
    aspect = h_box / w_box if w_box > 0 else 0

    # Layer 1: 孔洞数
    if len(holes) == 2:
        return 8, 0.95
    if len(holes) == 1:
        M = cv2.moments(contours[holes[0]])
        if M['m00'] > 0:
            cy = M['m01'] / M['m00']
            cx = M['m10'] / M['m00']
            if cy < h_box * 0.35 and cx < w_box * 0.40:
                return 4, 0.75
            if cy < h_box * 0.38:
                return 9, 0.85
            if cy > h_box * 0.62:
                return 6, 0.85
            return 0, 0.85
        return 0, 0.60

    # Layer 2: 0孔 → 宽高比判 1
    if aspect > 2.0:
        return 1, 0.90

    # Layer 3: 0孔投影特征 (2/3/4/5/7)
    norm = cv2.resize(binary, (40, 60)) > 127
    top_bar = np.any(norm[:20, :].mean(axis=1) > 0.55)
    left_dense = norm[:, :20].mean() > 0.25
    left_vert = norm[:, :20].mean(axis=0).max() > 0.40 and left_dense
    tr_low = norm[:20, 20:].mean() < 0.30

    if top_bar and left_vert and tr_low:
        return 4, 0.80
    if top_bar and not left_vert and norm[40:, :20].mean() < 0.15:
        return 7, 0.80
    if top_bar and norm[40:, :20].mean() > 0.20:
        return 5, 0.75

    right_proj = norm.mean(axis=0)[20:]
    peaks = sum(1 for i in range(1, len(right_proj)-1)
                if right_proj[i] > right_proj[i-1] and right_proj[i] > right_proj[i+1] and right_proj[i] > 0.30)
    if peaks >= 2:
        return 3, 0.75
    if norm[5:25, 15:35].mean() > 0.35 and norm[35:55, 5:25].mean() > 0.35:
        return 2, 0.75

    return -1, 0.0


# ═══════════════════════════════════════════════════
# 提取
# ═══════════════════════════════════════════════════
def extract_digit_from_square(binary: np.ndarray, square_contour: np.ndarray,
                               hierarchy: np.ndarray = None, contour_idx: int = 0) -> np.ndarray:
    """在正方形内部找黑色数字区域。返回白字黑底 ROI 或 None。"""
    x, y, w, h = cv2.boundingRect(square_contour)
    if min(w, h) < 30:
        return None
    roi = binary[y:y+h, x:x+w]
    if roi.size == 0:
        return None

    m = max(2, int(min(w, h) * 0.06))
    inner = roi[m:-m, m:-m] if m * 2 < min(h, w) else roi
    if inner.size == 0:
        return None

    black_mask = inner < 100
    if black_mask.sum() < 20:
        return None

    ys, xs = np.where(black_mask)
    if len(ys) < 10:
        return None

    digit_black = inner[ys.min():ys.max()+1, xs.min():xs.max()+1]
    return cv2.bitwise_not(digit_black)
