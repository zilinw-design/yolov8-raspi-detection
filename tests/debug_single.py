import cv2, sys, numpy as np
sys.path.insert(0, '.')
from src.phase3.test_detect import detect_shape

img = cv2.imread('D:/Pi/sy2/ai_harness_framework/exam/dataset_nuedc_2025_v2/1_single/target_003.png')
h,w = img.shape[:2]; s = 1200/max(h,w); img = cv2.resize(img,(int(w*s),int(h*s)))
found, box, shapes, warped = detect_shape(img)
print('GT: triangle')
for s in shapes:
    c = s['contour']
    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull)
    peri = cv2.arcLength(hull, True)
    a_t = cv2.approxPolyDP(hull, 0.02*peri, True)
    a_l = cv2.approxPolyDP(hull, 0.04*peri, True)
    circ = (4*np.pi*ha)/(peri*peri) if peri>0 else 0
    rw,rh = cv2.minAreaRect(hull)[1]
    asp = min(rw,rh)/max(rw,rh) if max(rw,rh)>0 else 0
    rr = ha/(rw*rh) if rw*rh>0 else 0
    print(f'{s["type"]} {s["size_px"]:.0f}px circ={circ:.3f} tight_vx={len(a_t)} loose_vx={len(a_l)} asp={asp:.3f} rr={rr:.3f}')
