"""统计全量测试结果的正确率"""
import sys, glob
sys.path.insert(0, '.')
from src.phase3.test_detect import detect_shape
import cv2, numpy as np
from collections import Counter

DS = 'D:/Pi/sy2/ai_harness_framework/exam/dataset_nuedc_2025_v2'

for cat in ['1_single', '2_multi', '3_numbered', '4_overlapping']:
    files = sorted(glob.glob(f'{DS}/{cat}/*.png'))
    stats = Counter()
    total_shapes = 0
    for fp in files:
        img = cv2.imread(fp)
        h, w = img.shape[:2]
        s = 1200 / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)))
        found, box, shapes, warped = detect_shape(img)
        if found and shapes:
            for sh in shapes:
                stats[sh['type']] += 1
                total_shapes += 1
            stats['_detected'] += 1
        else:
            stats['_missed'] += 1

    n = len(files)
    print(f'\n{cat} ({n} images):')
    print(f'  检测成功: {stats["_detected"]}/{n} ({stats["_detected"]/n*100:.0f}%)')
    print(f'  图形总数: {total_shapes}')
    for tp in ['circle', 'triangle', 'square', 'polygon']:
        cnt = stats.get(tp, 0)
        if cnt > 0:
            print(f'    {tp}: {cnt} ({cnt/total_shapes*100:.0f}%)')
