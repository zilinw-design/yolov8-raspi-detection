"""验证：数量正确率 + 最小正方形正确率"""
import sys, glob, json
sys.path.insert(0, '.')
from src.phase3.test_detect import detect_shape
import cv2, numpy as np

DS = 'D:/Pi/sy2/ai_harness_framework/exam/dataset_nuedc_2025_v2'
with open(f'{DS}/labels.json', encoding='utf-8') as f:
    gt_data = json.load(f)

gt_map = {}
for entry in gt_data:
    key = entry['filename'].replace('\\', '/')
    gt_map[key] = entry.get('shapes', [])

for cat in ['2_multi', '4_overlapping']:
    files = sorted(glob.glob(f'{DS}/{cat}/*.png'))
    count_match = 0
    min_match = 0
    min_errors = []
    total = len(files)

    for fp in files:
        img = cv2.imread(fp)
        h, w = img.shape[:2]
        s = 1200 / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)))
        found, box, shapes, warped = detect_shape(img)

        fname = fp.split('\\')[-1]
        key = cat + '/' + fname
        gt_shapes = gt_map.get(key, [])
        gt_sizes_px = [float(sh['size_inch']) * 300 * (1200/3507) for sh in gt_shapes]

        dt_squares = [sh for sh in (shapes or []) if sh['type'] == 'square']
        dt_sizes = [sh['size_px'] for sh in dt_squares]

        # 数量匹配
        if len(dt_squares) == len(gt_shapes):
            count_match += 1

        # 最小正方形匹配（DT min_sq vs GT min by pixel size）
        if dt_sizes and gt_sizes_px:
            dt_min = min(dt_sizes)
            gt_min = min(gt_sizes_px)
            # 容差：DT 像素尺寸基于 perspective warp（约 15% scale 波动）
            if abs(dt_min - gt_min) / gt_min < 0.25:
                min_match += 1
            else:
                min_errors.append(f'{fname}: DT={dt_min:.0f}px GT={gt_min:.0f}px')

    print(f'\n{cat} ({total} images):')
    print(f'  数量匹配: {count_match}/{total} ({count_match/total*100:.0f}%)')
    print(f'  最小正方正确: {min_match}/{total} ({min_match/total*100:.0f}%)')
    if min_errors:
        print(f'  偏差过大 (>25%):')
        for e in min_errors[:5]:
            print(f'    {e}')
