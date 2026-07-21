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
| 1 | ✅ | OTSU + RETR_TREE + 四边形验证1-3 |
| 2 | ✅ | 外轮廓+角点内缩+边中点验证 |
| 3 | ✅ | Homography + 20% 裁剪 |
| 4 | ✅ | 矫正后裁边（等效去边框） |
| 5 | ✅ | 独占通道分类（圆circ/三角vx/正方geo） |
| 6 | ✅ | 凸包松弛+白边分离+凸包顶点兜底 |
| 7 | ✅ | 模板优先(0.6) + 拓扑决策树兜底 |
| 8 | ❌ | PnP 未实现（需物理器材） |

## 最终架构（2026-07-22）

```
classify_contour: 独占通道
  Layer 1: circularity > 0.82 → circle
  Layer 2: len(approx) == 3  → triangle  
  Layer 3: aspect>0.80 & rect>0.80 → square
  Fallback: len(approx) <= 5 → square（凸四边形安全网）

recognize_digit: 模板优先 + 拓扑兜底
  matchTemplate(CCOEFF) > 0.6 → digit
  else → hole count → centroid → projection → -1 if uncertain
```

## 已验证改动（2026-07-21 → 2026-07-22）

| 改动 | 状态 | 效果 |
|---|---|---|
| OTSU 替代 Canny 找 A4 边框 | ✅ | 100% 检测率（50+张） |
| 输入缩放至 1200px | ✅ | 统一高 DPI 处理 |
| RETR_EXTERNAL → RETR_TREE | ✅ | 为内轮廓预留 |
| 角点内缩黑边框验证 | ✅ | 排除非黑边框矩形 |
| 边框中点内缩验证（验证3） | ✅ | 排除边框不均匀的假A4 |
| 形状分类 + minAreaRect 旋转修正 | ✅ | 6/10→8/10 正确分类 |
| 矫正后 20% 裁边 | ✅ | 隔离边框干扰 |
| 数据集 5 类 50 张 | ✅ | 全覆盖赛题要求 |

## 测试结果

| 类别 | 检测率 | 单图形分类 |
|---|---|---|
| 1_single 基本图形 | 10/10 | 圆2 三角2 正方4 多边2 |
| 2_multi 分离正方 | 10/10 | 待 Step 5 |
| 3_numbered 带数字 | 10/10 | 待 Step 7 |
| 4_overlapping 重叠 | 10/10 | 待 Step 6 |
| 5_tilted 倾斜 | 20/20 | 待 Step 3完善 |
