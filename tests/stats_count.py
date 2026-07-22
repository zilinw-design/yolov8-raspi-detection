"""对比 GT vs DT 的图形数量——逐张统计"""
import sys, glob, json
sys.path.insert(0, '.')
from src.phase3.test_detect import detect_shape
import cv2, numpy as np

DS = 'D:/Pi/sy2/ai_harness_framework/exam/dataset_nuedc_2025_v2'
with open(f'{DS}/labels.json', encoding='utf-8') as f:
    gt_data = json.load(f)

gt_map = {}
for entry in gt_data:
    # labels.json keys: "2_multi\\separated_001.png" — normalize
    key = entry['filename'].replace('\\', '/')
    gt_map[key] = len(entry.get('shapes', []))

for cat in ['2_multi', '4_overlapping']:
    files = sorted(glob.glob(f'{DS}/{cat}/*.png'))
    correct = 0
    over = 0
    under = 0
    total_gt = 0
    total_dt = 0

    for fp in files:
        img = cv2.imread(fp)
        h, w = img.shape[:2]
        s = 1200 / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)))
        found, box, shapes, warped = detect_shape(img)

        fname = fp.split('\\')[-1]
        fn_full = cat + '/' + fname
        gt_count = gt_map.get(fn_full, 0)
        dt_count = len(shapes) if found and shapes else 0
        total_gt += gt_count
        total_dt += dt_count

        if dt_count == gt_count:
            correct += 1
        elif dt_count > gt_count:
            over += 1
        else:
            under += 1

    n = len(files)
    print(f'\n{cat} ({n} images):')
    print(f'  GT图形总数: {total_gt}, DT图形总数: {total_dt}')
    print(f'  数量匹配: {correct}/{n} ({correct/n*100:.0f}%)')
    print(f'  检测偏多: {over}, 偏少: {under}')
