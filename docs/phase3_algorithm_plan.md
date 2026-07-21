# Phase 3 识别算法 — 完整方案

> 2026-07-21 | 对应 2025 电赛 C 题

---

## 算法管线

```
输入图像（复杂背景，A4纸 100-200cm 外）
        │
   Step 1: A4纸粗定位
   Canny + findContours(RETR_TREE) + 面积排序
        │
   Step 2: A4纸精定位
   外轮廓(4角点) + 内轮廓(4角点) + 标记点
        │
   Step 3: 透视校正 + 方向归一化
   4点 Homography → 正面视图 (840×1188 或自适应)
   标记点强制底边朝下，防 180° 翻转
        │
   Step 4: 去除边框，提取图形区域
   内轮廓 4 点 → 二次 Homography → 纯内容
        │
   Step 5: 图形检测与分类
   OTSU 二值化 → findContours → 圆形度 + 顶点数 + 宽高比
        ├── 基本：1个图形(圆/三角/正方)
        ├── 发挥1：单个正方形
        ├── 发挥2：多个正方形(分离/重叠)
        └── 发挥3：带数字的正方形
        │
   Step 6: 重叠正方形分割
   白边切割 + RETR_TREE 层次分析
        │
   Step 7: 数字识别
   模板匹配 (cv2.matchTemplate, 10个印刷体模板)
        │
   Step 8: 测距
   方案A: 4点 PnP (外轮廓角点)
   方案B: 8点 PnP (外+内轮廓, 精度更高)
```

## 技术栈

| 组件 | 用途 | 来源 |
|---|---|---|
| OpenCV `findContours(RETR_TREE)` | 双层轮廓检测 | 标准库 |
| OpenCV `approxPolyDP` | 四边形/多边形逼近 | 标准库 |
| OpenCV `minAreaRect` | 旋转矩形外接 | 标准库 |
| OpenCV `solvePnP` | 位姿估计→测距 | 标准库 |
| OpenCV `getPerspectiveTransform` | 透视矫正 | 标准库 |
| OpenCV `matchTemplate` | 数字模板匹配 | 标准库 |
| 圆形度公式 `4πA/P²` | 圆 vs 多边形判别 | Otsu 1979 |
| OTSU 自适应阈值 | 二值化 | Otsu 1979 |
| Matplotlib + Shapely | 数据集生成 | text_shapes.py |

## 当前实现进度

| Step | 状态 | 备注 |
|---|---|---|
| 1 | ⚠️ RETR_EXTERNAL | 需升级为 RETR_TREE |
| 2 | ❌ 未实现 | 只有外轮廓，缺内轮廓+标记点 |
| 3 | ⚠️ 部分实现 | 有 Homography，缺方向归一化 |
| 4 | ❌ 未实现 | 矫正图中含边框 |
| 5 | ⚠️ 部分实现 | 圆形度+顶点数，缺宽高比+内角验证 |
| 6 | ❌ 未实现 | |
| 7 | ❌ 未实现 | |
| 8 | ❌ 未实现 | 有 pinhole 框架，无 PnP |

## 本轮改动

| 优先级 | 改动 | 文件 | 行数 |
|---|---|---|---|
| P0 | 形状分类加宽高比+顶点双重验证 | test_detect.py, screen_measure.py | ~5行 |
| P1 | RETR_EXTERNAL → RETR_TREE | find_document_region() | ~3行 |
| P2 | 生成数据集 | 运行 text_shapes.py | 0（已写好） |

## 测试方法

| 改动 | 怎么测 |
|---|---|
| 形状分类 | 跑 `test_detect.py --dir tests/test_images/shape/` → 三张图都正确分类 |
| RETR_TREE | 跑 `test_detect.py` 对数据集图片 → 内轮廓被正确发现 |
| 数据集 | `python tests/text_shapes.py` → 5个目录各10张 → 人工抽查 |
