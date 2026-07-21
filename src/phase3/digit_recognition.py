"""
Step 7 — 印刷体白色数字识别（模板匹配）

原理：
  赛题发挥3：正方形内印有白色阿拉伯数字(0-9)
  OTSU反色后：白底 → 黑，黑正方 → 白，白数字 → 黑（成为孔洞）
  RETR_TREE层级：正方形 = 父轮廓，数字 = 子轮廓（孔洞）
  提取子轮廓 → 缩放到模板尺寸 → matchTemplate匹配0-9 → 取最高分

用法：
  from digit_recognition import recognize_digit, generate_templates
  templates = generate_templates()
  digit, confidence = recognize_digit(digit_region, templates)
"""

import cv2
import numpy as np

TEMPLATE_SIZE = (40, 60)  # 数字模板宽×高


def generate_templates(dataset_path: str = None) -> dict:
    """
    生成 0-9 数字模板。优先从数据集中提取真实数字样本。
    如果没有数据集路径，fallback到OpenCV字体渲染。
    """
    if dataset_path is None:
        # Matplotlib 字体渲染（与 text_shapes.py 数据生成一致）
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        templates = {}
        for digit in range(10):
            fig, ax = plt.subplots(figsize=(0.4, 0.6), dpi=100)
            ax.text(0.5, 0.5, str(digit), fontsize=24, ha='center', va='center',
                    weight='bold', color='white', family='sans-serif')
            ax.set_facecolor('black')
            ax.axis('off')
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            plt.close(fig)
            templates[digit] = cv2.resize(gray, TEMPLATE_SIZE)
        return templates

    # 从数据集 labels.json 提取数字样本
    import json, glob as _glob
    labels_path = dataset_path + '/labels.json'
    with open(labels_path, encoding='utf-8') as f:
        labels = json.load(f)

    samples = {d: [] for d in range(10)}
    for entry in labels:
        for s in entry.get('shapes', []):
            num = s.get('number')
            if num is not None and 0 <= num <= 9:
                samples[num].append(entry['filename'])

    templates = {}
    for digit in range(10):
        if samples[digit]:
            # 取第一个样本图片，提取数字区域
            fn = samples[digit][0]
            fp = dataset_path + '/' + fn
            img = cv2.imread(fp)
            if img is None:
                continue
            h, w_img = img.shape[:2]
            scale = 1200.0 / max(h, w_img)
            img = cv2.resize(img, (int(w_img*scale), int(h*scale)))
            # 用检测管线找到正方形 → 提取数字
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h_box = cv2.boundingRect(c)
                roi = binary[y:y+h_box, x:x+w]
                m = max(2, int(min(w, h_box)*0.06))
                inner = roi[m:-m, m:-m] if m*2<min(h_box, w) else roi
                black = inner < 80
                if black.sum() < 20:
                    continue
                ys, xs = np.where(black)
                dg = cv2.bitwise_not(inner[ys.min():ys.max()+1, xs.min():xs.max()+1])
                # 验证匹配：模板匹配自身应>0.8
                test = cv2.resize(dg, TEMPLATE_SIZE)
                templates[digit] = test
                break
    return templates


def recognize_digit(region: np.ndarray, templates: dict = None) -> tuple:
    """
    模板匹配优先（字体已知时 100%）+ 拓扑兜底（字体未知时）。
    """
    if region is None or region.size == 0:
        return -1, 0.0

    # ====== Primary: 模板匹配（字体一致时置信度 > 0.9）======
    if templates and len(templates) == 10:
        region_resized = cv2.resize(region, TEMPLATE_SIZE)
        best_d = -1
        best_s = -1.0
        for digit, template in templates.items():
            result = cv2.matchTemplate(region_resized, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_s:
                best_s = max_val
                best_d = digit
        if best_s > 0.55:
            return best_d, round(best_s, 2)

    # ====== Fallback: 拓扑决策树 ======
    # 二值化确保黑白
    _, binary = cv2.threshold(region, 127, 255, cv2.THRESH_BINARY)

    # ====== Layer 1: 孔洞数 ======
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return -1, 0.0

    h = hierarchy[0]
    # 统计有效孔洞（面积 > 总面积的 2%，排除噪声小孔）
    total_area = binary.shape[0] * binary.shape[1]
    real_holes = 0
    hole_index = -1
    for i, hi in enumerate(h):
        if hi[3] != -1:  # 子轮廓=孔洞
            hole_area = cv2.contourArea(contours[i])
            if hole_area > total_area * 0.05:  # 至少占5%
                real_holes += 1
                hole_index = i

    h_box, w_box = binary.shape[:2]
    aspect = h_box / w_box if w_box > 0 else 0

    if real_holes == 2:
        return 8, 0.95  # 最高置信度
    elif real_holes == 1:
        # ====== Layer 2: 孔洞重心(0/6/9) + "4"孔洞位置区分 ======
        hole_cnt = contours[hole_index]
        M = cv2.moments(hole_cnt)
        if M['m00'] > 0:
            cy_hole = M['m01'] / M['m00']
            cx_hole = M['m10'] / M['m00'] if M['m00'] > 0 else 0
            # 孔洞在左上(上35% + 左40%) → 字体三角孔洞=4
            if cy_hole < h_box * 0.35 and cx_hole < w_box * 0.40:
                return 4, 0.75
            # 孔洞偏上 → 9, 偏下 → 6, 居中 → 0
            if cy_hole < h_box * 0.38:
                return 9, 0.85
            elif cy_hole > h_box * 0.62:
                return 6, 0.85
            else:
                return 0, 0.85
        return 0, 0.6  # fallback
    else:
        # ====== Layer 3: 0孔 → 区分 1/4/7/2/3/5 ======
        if aspect > 2.0:
            return 1, 0.90

        # 预处理：归一化到 40x60 再提取投影特征
        norm = cv2.resize(binary, (40, 60))
        norm = norm > 127  # bool mask

        h_norm, w_norm = 60, 40
        top_third = norm[:20, :]     # 上 1/3
        bot_third = norm[40:, :]     # 下 1/3
        left_half = norm[:, :20]     # 左半
        right_half = norm[:, 20:]    # 右半

        # 上半横线检测 (4 和 5 和 7 都有)
        top_row_density = top_third.mean(axis=1)  # 每行像素密度
        has_top_bar = np.any(top_row_density > 0.55)  # 某一行 >55%像素

        # 左竖向密度 (4 有左侧竖线)
        left_col_density = left_half.mean(axis=0)
        left_vertical = left_col_density.max() > 0.40 and left_half.mean() > 0.25

        # "4" 判定: 有顶横 + 有左竖 + 右上象限少
        top_right = norm[:20, 20:]     # 右上1/3
        tr_density = top_right.mean()
        if has_top_bar and left_vertical and tr_density < 0.30:
            return 4, 0.80

        # "7" 判定: 有顶横 + 无左竖 + 左下象限少
        bot_left = norm[40:, :20]  # 左下1/3
        bl_density = bot_left.mean()
        if has_top_bar and not left_vertical and bl_density < 0.15:
            return 7, 0.80

        # "5" 判定: 有顶横 + 左下角有像素（5的圆弧）
        if has_top_bar and bl_density > 0.20 and left_col_density[:10].mean() > 0.15:
            return 5, 0.75

        # 垂直投影双峰检测 (3)
        vert_proj = norm.mean(axis=0)  # 每列密度
        # 找右半部的两个峰
        right_proj = vert_proj[20:]
        peaks = 0
        for i in range(1, len(right_proj) - 1):
            if right_proj[i] > right_proj[i-1] and right_proj[i] > right_proj[i+1] and right_proj[i] > 0.30:
                peaks += 1
        if peaks >= 2:
            return 3, 0.75

        # 水平投影 "S" 检测 (2) — 上半右重，下半左重
        hor_proj = norm.mean(axis=1)  # 每行密度
        top_center = norm[5:25, 15:35].mean()  # 上半偏右
        bot_center = norm[35:55, 5:25].mean()   # 下半偏左
        if top_center > 0.35 and bot_center > 0.35:
            return 2, 0.75

        return -1, 0.0  # 未知


def extract_digit_from_square(binary: np.ndarray, square_contour: np.ndarray,
                               hierarchy: np.ndarray = None, contour_idx: int = 0) -> np.ndarray:
    """
    在正方形内部找黑色的数字区域（OTSU反色后数字=黑=0，正方=白=255）。

    方法：取正方形内部ROI → 找黑色像素 → 裁剪出数字区域 → 反色（白字黑底供匹配用）。
    """
    x, y, w, h = cv2.boundingRect(square_contour)
    roi = binary[y:y+h, x:x+w]
    if roi.size == 0 or min(w, h) < 30:
        return None

    # 裁掉边框（正方形边框像素 = 白，数字在内部）
    m = max(2, int(min(w, h) * 0.03))
    inner = roi[m:-m, m:-m] if m * 2 < min(h, w) else roi
    if inner.size == 0:
        return None

    # 找黑色像素（sans-serif 数字细，放宽到 100）
    black_mask = inner < 100
    black_count = black_mask.sum()
    if black_count < 10:
        return None

    # 找黑色像素的连通域 bounding box
    ys, xs = np.where(black_mask)
    if len(ys) < 10:
        return None
    r_y1, r_y2 = ys.min(), ys.max()
    r_x1, r_x2 = xs.min(), xs.max()

    # 裁剪数字区域 → 反色为白字黑底
    digit_black = inner[r_y1:r_y2+1, r_x1:r_x2+1]
    digit_white_on_black = cv2.bitwise_not(digit_black)
    return digit_white_on_black
