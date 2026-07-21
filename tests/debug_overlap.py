"""Debug overlap_002 - check aspect/rect_ratio per shape"""
import cv2, sys
import numpy as np
sys.path.insert(0,'.')
from src.phase3.test_detect import detect_shape

img = cv2.imread('D:/Pi/sy2/ai_harness_framework/exam/dataset_nuedc_2025_v2/4_overlapping/overlap_002.png')
h,w = img.shape[:2]
s = 1200/max(h,w)
img = cv2.resize(img,(int(w*s),int(h*s)))
found, box, shapes, warped = detect_shape(img)

print('shape     px   aspect  rect_rt  hull_vx_tight  hull_vx_loose')
for shape in shapes:
    c = shape['contour']
    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull)
    peri = cv2.arcLength(hull, True)
    a_t = cv2.approxPolyDP(hull, 0.02*peri, True)
    a_l = cv2.approxPolyDP(hull, 0.04*peri, True)
    rw, rh = cv2.minAreaRect(hull)[1]
    asp = min(rw,rh)/max(rw,rh) if max(rw,rh)>0 else 0
    rr = ha/(rw*rh) if rw*rh>0 else 0
    print('%8s %4d  %.3f  %.3f    %d          %d' % (shape['type'], int(shape['size_px']), asp, rr, len(a_t), len(a_l)))
