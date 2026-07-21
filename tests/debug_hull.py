import cv2, numpy as np, sys
sys.path.insert(0,'.')
from src.phase3.test_detect import detect_shape

img = cv2.imread('dataset_nuedc_2025_v2/2_multi/separated_003.png')
h,w = img.shape[:2]
s = 1200/max(h,w)
img = cv2.resize(img,(int(w*s),int(h*s)))
found,box,shapes,warped = detect_shape(img)

print(f'Found: {len(shapes)} shapes')
for shape in shapes:
    c = shape['contour']
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    orig_area = cv2.contourArea(c)
    t = shape['type']
    sz = int(shape['size_px'])
    print(f'  {t} {sz}px  orig={int(orig_area)} hull={int(hull_area)}')
