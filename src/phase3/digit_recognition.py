"""
Step 7 — 印刷体数字识别（Tesseract OCR）

替换原因：模板匹配依赖字体一致性，拓扑法对 0 孔数字不可靠。
Tesseract 是预训练的通用 OCR 引擎，印刷体数字准确率 ~99%。
"""

import cv2
import numpy as np
import pytesseract


# ═══════════════════════════════════════════════════
# 识别（Tesseract：--psm 10 = 单字符模式）
# ═══════════════════════════════════════════════════
def recognize_digit(region: np.ndarray, _templates=None) -> tuple:
    """
    用 Tesseract OCR 识别单个印刷体数字。

    Returns: (digit, confidence)
    """
    if region is None or region.size == 0:
        return -1, 0.0

    # 放大到至少 60px 高（Tesseract 对过小图片不准确）
    h, w = region.shape[:2]
    if h < 60:
        scale = 60.0 / h
        region = cv2.resize(region, (int(w * scale), 60))

    # 确保白字黑底（Tesseract 标准输入）
    if region.mean() > 127:
        region = cv2.bitwise_not(region)

    # Tesseract 单字符模式，仅识别数字
    try:
        text = pytesseract.image_to_string(
            region,
            config='--psm 10 -c tessedit_char_whitelist=0123456789'
        ).strip()
        if text.isdigit():
            return int(text), 0.99
    except Exception:
        pass

    return -1, 0.0


# ═══════════════════════════════════════════════════
# 提取
# ═══════════════════════════════════════════════════
def extract_digit_from_square(binary: np.ndarray, square_contour: np.ndarray,
                               hierarchy=None, contour_idx: int = 0) -> np.ndarray:
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
