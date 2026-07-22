"""从 test_detect.py 提取检测函数，注入 screen_measure.py"""
import re

# 读取 test_detect.py
with open('src/phase3/test_detect.py', 'r', encoding='utf-8') as f:
    td = f.read()

# 读取 screen_measure.py
with open('src/phase3/screen_measure.py', 'r', encoding='utf-8') as f:
    sm = f.read()

# 从 test_detect.py 提取关键函数
funcs = ['find_document_region', 'classify_contour', 'order_points',
         'four_point_transform', 'detect_shape']
extracted = {}
for fn in funcs:
    pattern = rf'def {fn}\b.*?(?=\n(?:def |# ═|class |if __name__))'
    m = re.search(pattern, td, re.DOTALL)
    if m:
        extracted[fn] = m.group(0)
        print(f'Extracted {fn}: {len(m.group(0))} chars')

# 替换 screen_measure.py
# 1. MIN_SHAPE_AREA
sm = re.sub(r'MIN_SHAPE_AREA = \d+', 'MIN_SHAPE_AREA = 15000', sm)

# 2. 替换 find_document_region
if 'find_document_region' in extracted:
    sm = re.sub(r'def find_document_region\b.*?(?=\ndef )', extracted['find_document_region'], sm, flags=re.DOTALL)

# 3. 在 order_points 前插入 classify_contour
if 'classify_contour' in extracted:
    insert_pos = sm.find('def order_points')
    sm = sm[:insert_pos] + extracted['classify_contour'] + '\n\n' + sm[insert_pos:]

# 4. 替换 detect_shape
if 'detect_shape' in extracted:
    sm = re.sub(r'def detect_shape\b.*?(?=\n(?:# ═|def |class |if __name__|# ----))', extracted['detect_shape'], sm, flags=re.DOTALL)

with open('src/phase3/screen_measure.py', 'w', encoding='utf-8') as f:
    f.write(sm)

print('Done syncing screen_measure.py')
