"""对比检测结果 vs Ground Truth"""
import json, cv2, sys, glob
import numpy as np
sys.path.insert(0, '.')
from src.phase3.test_detect import detect_shape

DS = 'D:/Pi/sy2/ai_harness_framework/exam/dataset_nuedc_2025_v2'
with open(DS + '/labels.json', encoding='utf-8') as f:
    gt_data = json.load(f)

gt_map = {}
for entry in gt_data:
    gt_map[entry['filename']] = entry

categories = ['1_single', '2_multi', '3_numbered', '4_overlapping']
for cat in categories:
    print(f"\n{'='*50}")
    print(f"  {cat}")
    print(f"{'='*50}")
    files = sorted(glob.glob(f'{DS}/{cat}/*.png'))
    for fp in files[:5]:
        img = cv2.imread(fp)
        h, w_img = img.shape[:2]
        scale = 1200.0 / max(h, w_img)
        img = cv2.resize(img, (int(w_img*scale), int(h*scale)))
        found, box, shapes, warped = detect_shape(img)

        fn_key = fp.replace('\\', '/')
        gt_shapes = gt_map.get(fn_key, {}).get('shapes', [])

        fname = fp.split('\\')[-1]
        det = 'NOT FOUND'
        if found and shapes:
            parts = [str(s['type']) + ' ' + str(int(s['size_px'])) + 'px' for s in shapes]
            det = ', '.join(parts)
            squares = [s for s in shapes if s['type'] == 'square']
            if squares:
                ms = min(squares, key=lambda s: s['size_px'])
                det += ' | min_sq=' + str(int(ms['size_px'])) + 'px'

        gt_str = ''
        if gt_shapes:
            gt_parts = []
            for s in gt_shapes:
                sz_cm = float(s['size_inch']) * 2.54
                gt_parts.append(str(s['type']) + ' ' + f'{sz_cm:.1f}cm @{s["angle"]}deg')
            gt_str = str(len(gt_shapes)) + ' shapes: ' + ', '.join(gt_parts)

        print(f'  {fname}')
        print(f'    GT: {gt_str}')
        print(f'    DT: {det}')
