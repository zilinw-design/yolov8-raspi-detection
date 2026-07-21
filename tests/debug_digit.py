import cv2, sys, numpy as np
sys.path.insert(0, 'src/phase3')
from test_detect import detect_shape
from digit_recognition import extract_digit_from_square, recognize_digit, generate_templates

img = cv2.imread('D:/Pi/sy2/ai_harness_framework/exam/dataset_nuedc_2025_v2/3_numbered/numbered_001.png')
h,w = img.shape[:2]; s = 1200/max(h,w); img = cv2.resize(img,(int(w*s),int(h*s)))
found, box, shapes, warped = detect_shape(img)

templates = generate_templates()
for s in shapes:
    if s['type'] == 'square':
        h2, w2 = warped.shape[:2]
        m = int(min(h2,w2)*0.20)
        center = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)[m:h2-m, m:w2-m]
        _, binary = cv2.threshold(center, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        digit_region = extract_digit_from_square(binary, s['contour'], None, 0)
        if digit_region is not None:
            # Debug: check hole area
            _, test_bin = cv2.threshold(digit_region, 127, 255, cv2.THRESH_BINARY)
            cnts, hier = cv2.findContours(test_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            total = test_bin.shape[0] * test_bin.shape[1]
            hole_info = 'no_hierarchy' if hier is None else ''
            if hier is not None:
                holes_data = [(i, cv2.contourArea(cnts[i])/total*100) for i, hi in enumerate(hier[0]) if hi[3] != -1]
                hole_info = str(holes_data) if holes_data else 'none'
            digit, conf = recognize_digit(digit_region, templates)
            print(f"square {s['size_px']:.0f}px: region={digit_region.shape} dt={digit} cf={conf:.2f} holes={hole_info}")
