# 单目视觉测量 — 电赛备赛项目

> **项目轨道：Part B — 电赛单目测距**（进行中） | **Part A YOLO 学习** → `D:\Pi\sy3`

## 目标

对准 2025 全国大学生电子设计竞赛 C 题。树莓派5 + 单摄像头，测量 A4 纸目标物的距离 D 和图形尺寸 x。

## 技术栈

OpenCV (文档扫描/透视矫正/solvePnP) + 4K USB Camera + picamera2 (辅助)

## 阶段

| Phase | 内容 | 状态 | 代码 |
|---|---|---|---|
| 3a | 文档扫描管线、图形检测、离线测试 | 🔄 | `src/phase3/` |
| 3b | 焦距标定、pinhole 测距验证 | 待做 | — |
| 3c | solvePnP 测距、参考目标消误差 | 待做 | — |
| 4 | 重叠正方形、倾斜测量、赛题适配 | 待做 | — |

## 赛题要求

| 指标 | 要求 |
|---|---|
| 摄像头数量 | 仅 1 颗 |
| 测量距离 | 100cm-200cm |
| 测距误差 | ≤ 2cm |
| 测量耗时 | ≤ 5 秒 |
| 目标物 | A4 纸 + 黑色几何图形（圆/正方/三角） |

## 关键交付

- 文档扫描管线：Canny → 四边形检测 → 透视矫正 → OTSU 图形检测
- 角点内缩黑边框验证（防误识别窗户/显示器）
- 4K USB Camera 参数基准 + 对焦锁定
- 离线测试：`src/phase3/test_detect.py`

## 环境

- 树莓派5 + 4K USB Camera (MJPG 1920×1080) + IMX219 CSI (辅助)
- Python 3.8+, venv `--system-site-packages`

## 目录

```
src/phase3/               Phase 3 图形检测 + 测距
tests/test_shapes.html    测试图形生成器
tests/test_images/shape/  测试截图
docs/                     讲义
references/summarized/    参数记录
ai/governance/            项目治理
```
