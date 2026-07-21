# 文档扫描式目标检测算法 —— 原理讲义

> 2026-07-20 | Phase 3a

---

## 一、算法解决了什么问题

摄像头对着 A4 纸（或电脑屏幕），上面印有黑色几何图形。我们想找到这个图形，测量它的像素尺寸。

**难点**：摄像头看到的不仅是纸，还有桌子、墙壁、阴影、杂物。如果直接在整幅画面中找"黑色物体"，任何暗色背景都可能被误判为目标。

**解法**：先找到"纸"，再在纸面上找"图形"。纸有四条直边——这是背景杂物不具备的特征。

---

## 二、算法全景

```
输入帧 (1920×1080 BGR)
    │
    ▼
┌─────────────────────────────────────────────┐
│  第一步：CLAHE 增强                          │
│  在 LAB 色彩空间的 L 通道做对比度受限均衡     │
│  目的：暗光下也能看清纸的边缘                 │
├─────────────────────────────────────────────┤
│  第二步：Canny 边缘检测                       │
│  目的：找到所有"有明显亮度变化"的像素         │
├─────────────────────────────────────────────┤
│  第三步：找到四边形区域    ← 核心               │
│  findContours → 面积排序 → approxPolyDP(4点) │
│  → 宽高比验证 → 角点内缩黑边框验证            │
├─────────────────────────────────────────────┤
│  第四步：透视矫正                             │
│  四角点 → 正视图（背景自动裁掉）               │
├─────────────────────────────────────────────┤
│  第五步：检测图形                             │
│  OTSU 二值化 → findContours → 形状分类        │
│  → 测量像素尺寸                               │
└─────────────────────────────────────────────┘
    │
    ▼
输出: 图形类型 (circle/square/triangle) + 像素尺寸
```

---

## 三、各步骤详解

### 3.1 CLAHE 增强

```python
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)     # BGR → LAB
l, a, b = cv2.split(lab)                          # 拆出 L（亮度）通道
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l = clahe.apply(l)                                 # 只在 L 通道做均衡
enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
```

**为什么是 LAB 而不是直接灰度？**

RGB 转灰度会丢失色彩信息。LAB 把"亮度"和"颜色"分离——L 通道是纯亮度，A 和 B 是色彩。在 L 通道做 CLAHE 增强对比度，不影响色彩还原。

**为什么 tileGridSize=(8,8)？**

太大 → 局部细节不增强。太小 → 噪声也被放大。8×8 是 OpenCV 文档扫描场景的推荐值——A4 纸在 1920×1080 下大约占 800×1100 像素，8×8 分块后每块约 100×137 像素，恰好覆盖纸面局部纹理。

---

### 3.2 Canny 边缘检测

```python
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
```

**Canny 的两个阈值 50 和 150 分别做什么？**

Canny 是双阈值算法：
- 梯度 > 150：强边缘，直接保留
- 梯度在 50-150 之间：弱边缘，只有当它连接到强边缘时才保留
- 梯度 < 50：丢弃

50/150 这个比例（1:3）是 OpenCV 推荐的默认值，适合大多数自然图像。

**为什么后面加 dilate（膨胀）？**

Canny 输出的边缘线可能因为噪声或光照不均匀而断开。膨胀 1 次把断口连上，让 A4 纸的四边形成闭合轮廓。

---

### 3.3 找到四边形区域（核心）

这是整个算法最关键的步骤。分三个子步骤。

#### 3.3.1 轮廓检测 + 面积排序

```python
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ranked = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
```

取面积前 10 的轮廓。A4 纸几乎一定是最大的几个轮廓之一（因为纸面大，Canny 会在纸的边缘产生长轮廓线）。

**为什么 `RETR_EXTERNAL`？**

只取最外层轮廓，忽略内嵌轮廓（比如纸上的图形）。纸的边框是外轮廓，图形是内轮廓。

#### 3.3.2 四边形近似 + 宽高比验证

```python
peri = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 0.02 = 2% 轮廓周长
if len(approx) == 4:
    # 通过！
```

**`approxPolyDP` 怎么工作的？**

Douglas-Peucker 算法：从轮廓的两个最远点出发，递归地找距离直线最远的中间点，直到所有点到简化线的距离 < epsilon。

epsilon = `0.02 × 轮廓周长`。A4 纸轮廓周长约 2000px（在 1920×1080 下），epsilon ≈ 40px。这意味着 A4 纸的角点允许有 40px 的圆角偏差，能被容忍。

**宽高比验证**：

```python
ratio = max(width, height) / min(width, height)
if 0.9 < ratio < 2.5:  # 通过！
```

A4 纸宽高比 = 297/210 = 1.414。16:9 屏幕 = 1.778。容忍范围 0.9~2.5 能覆盖"纸面没有正对摄像头"时的透视变形。

#### 3.3.3 角点内缩黑边框验证（防误识别）

```python
centroid = pts.mean(axis=0)                    # 四个角点的几何中心
vectors = centroid - pts                       # 每个角点指向中心的方向
dists = np.linalg.norm(vectors, axis=1)        # 每个角点到中心的距离
shrink = np.minimum(dists * 0.05, 20)          # 缩进量 = 5%距离，上限20px
sample_pts = pts + (vectors / dists) * shrink  # 采样点坐标

# 检查四个采样点是否全是黑色（灰度 < 80）
for sx, sy in sample_pts:
    if gray[sy, sx] > 80:
        all_dark = False  # 有一个不够黑 → 不是黑边框
```

**为什么需要这一步？**

前面两步只验证了"有四条直边 + 宽高比合理"。但不保证这四条边是"黑色的"。

场景举例：窗户框、显示器外壳、桌面边缘——都有四条直边且宽高比合理，但它们不是目标。

角点内缩验证的逻辑：如果是 A4 纸的黑边框，角点向纸面中心稍微缩一点，采样到的应该还是黑边框（因为边框有 2cm 宽）。如果是窗户框，缩进去就是玻璃/墙壁——不够黑。

**shrink 的计算**：

`shrink = min(距离×5%, 20px)` —— 自适应。

- 纸面占画面大（角点离中心远 = 距离大）：shrink 取 20px 上限，恰好穿过 2cm 黑边框的像素宽度
- 纸面占画面小（角点离中心近 = 距离小）：shrink 取更小值，确保不会缩出边框之外

---

### 3.4 透视矫正

```python
warped = four_point_transform(frame, doc_corners)
```

把斜拍的 A4 纸（梯形）纠正为正视图（矩形）。四角点 → 4×4 透视变换矩阵 → warped。

**这时画面里只剩下 A4 纸本身**，桌子、墙壁全部被裁在外面。

---

### 3.5 在正视图内检测图形

```python
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
best = max(contours, key=cv2.contourArea)
```

现在画面是干净的：白纸 + 黑色图形。OTSU 在双峰直方图上完美工作。找最大轮廓 → 就是目标图形。

**形状分类**：

```python
circularity = (4π × area) / perimeter²
if circularity > 0.85 → "circle"
elif approxPolyDP 有 3 顶点 → "triangle"
elif approxPolyDP 有 4 顶点 → "square"
else → "polygon"
```

---

## 四、为什么这个方案比之前好

| 维度 | 之前（全图找黑色轮廓） | 现在（文档扫描管线） |
|---|---|---|
| 背景干扰 | 暗色家具/阴影影响严重 | 透视矫正自动裁掉 |
| 误检来源 | 任何大面积暗区域 | 只有"有四条直边+黑边框"的物体 |
| 虚假矩形 | 无判断 | 宽高比 + 角点内缩双重验证 |
| 光线适应 | 依赖相机参数调优 | CLAHE + OTSU 自动适应 |
| 调试复杂度 | 一个环节出问题全链条排查 | 每步输出独立（可单独验证） |

---

## 五、局限与后续

1. **需要四个边都可见**：如果 A4 纸一个角被遮挡，四边形检测失败
2. **黑边框必须存在**：如果目标物是纯白纸没有黑边框（比赛不会），角点内缩验证失效
3. **焦距尚未标定**：能检测到图形和像素尺寸，但还不能换算为物理距离——下一步任务
