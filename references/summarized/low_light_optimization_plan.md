# 暗光优化方案与文献参考

> 整理日期：2026-07-20
> 目的：记录暗光场景下 YOLO 检测优化的技术方案、文献依据与决策过程

---

## 一、背景问题

实验三数据（2026-07-19）：
- person 正常光 conf=0.864 → 暗光 0.855 → 微光 0.856，波动 0.009 ✅
- book 正常光 conf=0.281 → 暗光 未检出 → 微光 未检出 ❌
- cup 正常光 conf=0.289 → 暗光 未检出 → 微光 未检出 ❌

**核心矛盾**：小物体在暗光下因图像特征退化而丢失，但 person 等大物体几乎不受影响。同一画面中不同类别对暗光的响应差异极大。

---

## 二、三种预处理技术原理

### 2.1 HE — 直方图均衡化

**原理**：将图像灰度分布从"集中在某区间"拉伸为"均匀分布在全范围"。

```
s_k = (L-1) · Σ_{j=0}^{k} p_r(r_j)
```

- 一张整体偏暗的图：大部分像素灰度集中在 0-80 → HE 拉伸后分布到 0-255
- **优点**：简单，全局提亮效果显著
- **缺点**：全局操作，噪声与细节同等放大

**出处**：R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018, Chapter 3.3.

### 2.2 CLAHE — 对比度受限自适应直方图均衡化

**原理**：

```
1. 图像切为 tileGridSize 个小块（如 8×8 = 64块）
2. 每块独立做直方图均衡
3. clipLimit 限制单块的对比度放大上限（超出部分均匀重新分配）
4. 块与块之间用双线性插值平滑过渡
```

**clipLimit 的作用**：
- clipLimit=1.0 → 不做限制，等效于普通自适应直方图均衡
- clipLimit=2.0-4.0 → 中等限制，适合大多数场景（OpenCV 默认 40.0 已废弃，新默认 2.0）
- clipLimit 越大 → 对比度越强，但噪声也越多

**为什么优于 HE**：局部操作 + 对比度限制 = 细节增强的同时不放大平坦区域的噪声。

**出处**：K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," in *Graphics Gems IV*, P. S. Heckbert, Ed. San Diego: Academic Press, 1994, pp. 474–485.

原始用途：医学影像（低对比度 X 光片增强）。

### 2.3 Gamma 校正

**原理**：

```
V_out = V_in^γ
```

- γ < 1：提亮暗部（暗光场景用 0.5-0.8）
- γ > 1：压暗暗部（过曝修正）
- γ = 1：不变

**为什么有效**：人眼对暗部亮度变化比亮部敏感（Weber-Fechner 定律）。Gamma 编码利用这一特性，用有限的 8-bit 编码达到接近人眼感知均匀的灰度分布。

**关键结论**（Poynton 1998）：Gamma 校正本质是**感知编码**——不是为了修正显示器缺陷，而是让有限 bit 最大化人眼感知质量。不用 Gamma 需要 12bit/通道以上才能达到同等感知效果。

**出处**：C. Poynton, "The rehabilitation of gamma," in *Human Vision and Electronic Imaging III*, Proc. SPIE 3299, 1998, pp. 232–249.

---

## 三、YOLO 暗光增强 —— 近期文献证据

### 3.1 改进 CLAHE + Gamma + 中值滤波

**研究**：The Research on Low-Light Autonomous Driving Object Detection Method (2025, *ScienceDirect*)

**方法**：在 YOLOv5 基础上，使用改进 CLAHE（+中值滤波 + 自适应 Gamma）预处理暗光图像。

**结果**：
- BDD100K 数据集：+13.3% 精度，+1.7% mAP
- KITTI 数据集：+21.1% 精度
- 同时减少了 240,000 参数

**参考部分**：CLAHE 参数选择（clipLimit, tileGridSize）和 Gamma 自适应策略。

### 3.2 CLAHE 对 YOLOv9 在 PPE 检测中的效果

**研究**：Enhancing Low-Light PPE Violation Detection Using Multi-Contrast Image Processing and YOLOv9 (2025, *ETASR*)

**结果**：
- mAP@50: 0.915
- CLAHE 检测覆盖率：79.43%（高于 GAN 方法）
- 证明 CLAHE 在实时性要求下的性价比优于 GAN 增强

**参考部分**：CLAHE 与其他方法的对比数据（HE vs CLAHE vs GAN）。

### 3.3 CLAHE 对 YOLOv8 在车辆检测中的效果

**研究**：Improved Vehicle Traffic Detection Using YOLO Algorithm with Brightness and Contrast Adjustment in Low-light Conditions (2024, *IEEE*)

**结果**：
- 暗光下 +10-20% 检测率
- 亮度+对比度+CLAHE 组合效果最优

**参考部分**：预处理方法的选择依据——亮度+对比度调整不能替代 CLAHE，但可以作为补充。

### 3.4 YOLO 多版本 CLAHE 综合对比

**研究**：Multiple papers (2024-2025) on YOLOv5/v7/v8/v9/v11 + CLAHE for various low-light tasks

**综合结论**：
- CLAHE 在所有 YOLO 版本上均有正向效果
- 与 HE 对比：HE 的 mAP 有时更高（fire detection: 0.771 vs 0.759），但 CLAHE 的局部细节保留更好
- 与 Gamma 对比：Gamma 在整体亮度调节上更自然，但不增强局部对比度
- **CLAHE + Gamma 组合是最优方案**，兼顾局部对比度增强和全局亮度调节

**参考部分**：多篇论文的横向对比数据，用于确定最优参数组合。

---

## 四、置信度优化 —— 文献证据

### 4.1 逐类置信度阈值

**研究**：Han, Won, Koo, "A Strategic Approach to Enhancing the Practical Applicability of Vision-based Detection and Classification Models for Construction Tools — Sensitivity Analysis of Model Performance Depending on Confidence Threshold" (2025, *Korean Journal of Construction Engineering and Management*)

**方法**：
1. 用 YOLOv8 训练检测 4 种建筑工具
2. 对每类单独做 F1-score vs conf 的灵敏度分析
3. 根据 F1-score 曲线确定每类的最优阈值区间

**结果**：

| 类别组 | 最优 conf 区间 | 适用策略 |
|---|---|---|
| Group A（易检） | 0.7–0.8 | 高阈值：精准检测，低误报 |
| Group B（难检） | 0.2–0.3 | 低阈值：安全优先，保召回 |

**核心方法**（我们可复用的）：F1-score vs conf 曲线分析法——对每类单独画曲线找拐点。

**参考部分**：逐类 F1-score 灵敏度分析的方法论，直接复用到我们的 6 类检测目标。

### 4.2 置信度校准 — Temperature Scaling

**研究**：Ingrisch, Schilling, Chmielewski, Twieg, "Calibration of the Open-Vocabulary Model YOLO-World by Using Temperature Scaling" (2025, *ICAIIT 2025*)

**方法**：用 Temperature Scaling（温度缩放）对 YOLO-World 的置信度进行后校准。单一参数 T，缩放所有类别的 logits。

**结果**：
- 校准前：75% 的预测 conf < 0.1（模型过度不自信）
- 校准后：期望校准误差(ECE) 从 6.78% 降到 2.31%
- mAP 不变（校准不影响排序准确性）

**局限性**：
- 全局单参数，不区分类别
- 校准后仍有部分 bin 过自信
- 需要校准数据集

**参考部分**：置信度不可直接信任的理论依据——ECE 指标证明了原始 conf 值与实际正确率之间存在系统偏差。

### 4.3 环境因素对置信度的影响（观测性研究）

**研究**：Shams, "Securing CCTV Cameras Against Blind Spots" (2024, *DEF CON 32*)

**观测事实**：
- 同一人物在同一摄像头下，不同时间/光照时，YOLOv3 置信度波动范围 0.25–0.92
- 波动与光照、人物位置（距离/角度）、摄像头分辨率均有关
- 但**没有给出自动调节方案**

**参考部分**：环境因素影响置信度的定量证据——用来论证"为什么固定 conf 不够用"。

---

## 五、方案决策

### 5.1 选用的方案

**两步优化管线**：

```
摄像头帧
  │
  ├─→ CLAHE 预处理 (clipLimit=2.0, tileGridSize=8×8)
  │    依据：Zuiderveld 1994 + 多篇2024-2025 YOLO论文
  │    效果：增强暗部局部对比度，不放大噪声
  │    开销：O(n) 逐像素，对 FPS 影响 < 5%
  │
  ├─→ Gamma 校正 (γ=0.7，暗光场景)
  │    依据：Poynton 1998 + 自动驾驶论文 2025
  │    效果：整体提亮，补偿全局亮度不足
  │    开销：O(n) 查表法，几乎无开销
  │
  └─→ YOLO 推理（逐类 conf 阈值）
       依据：Han, Won, Koo 2025
       效果：person 用 conf=0.4（高精度），book/cup 用 conf=0.2（保召回）
```

### 5.2 不选用的方案及原因

| 方案 | 不选用原因 |
|---|---|
| 亮度自适应 conf（我之前提议的） | **无文献支持**。亮度变化影响的是图像质量，不是置信度的准确性 |
| Temperature Scaling 校准 | 需要校准数据集，且是全局单参数，不解决逐类差异 |
| GAN/Retinex 增强 | 树莓派上推理不起，实时性不达标 |
| HE（代替 CLAHE） | 全局操作放大噪声，多数文献 CLAHE 优于 HE |
| 训练暗光专用模型 | 不是本阶段目标，且树莓派无 GPU 训练能力 |

### 5.3 待验证假设

| 假设 | 验证方法 |
|---|---|
| CLAHE 预处理后 book/cup 在暗光下可被检出 | 用实验三的 3 张不同亮度图，加 CLAHE 后再跑 benchmark_conf |
| CLAHE 对 FPS 影响 < 5% | benchmark_fps 对比加/不加 CLAHE |
| 逐类 conf 阈值优于单一 conf | 固定图片，分别用单一 conf=0.3 和逐类 conf 跑，对比 F1-score |

---

## 六、参考文献总表

| 编号 | 出处 | 类型 | 参考部分 |
|---|---|---|---|
| [1] | K. Zuiderveld, "CLAHE," in *Graphics Gems IV*, Academic Press, 1994, pp. 474–485 | 原始论文 | CLAHE 原理与实现 |
| [2] | C. Poynton, "The rehabilitation of gamma," *Proc. SPIE 3299*, 1998, pp. 232–249 | 原始论文 | Gamma 校正的感知编码理论 |
| [3] | R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018, Ch.3 | 教科书 | HE 原理 |
| [4] | YOLO-LKSDS, *ScienceDirect*, 2025 | 近期论文 | CLAHE+Gamma+中值滤波 组合优化 |
| [5] | PPE Violation Detection with YOLOv9 + CLAHE, *ETASR*, 2025 | 近期论文 | CLAHE vs GAN 实时性对比 |
| [6] | Vehicle Detection YOLOv8 + CLAHE, *IEEE*, 2024 | 近期论文 | 暗光下 +10-20% 检测提升 |
| [7] | Han, Won, Koo, "Per-Class Confidence Threshold for YOLOv8," *KJCEM*, 2025 | 近期论文 | 逐类 conf 阈值方法论 |
| [8] | Ingrisch et al., "Temperature Scaling Calibration of YOLO-World," *ICAIIT*, 2025 | 近期论文 | 置信度校准方法论 |
| [9] | Shams, "Securing CCTV Cameras Against Blind Spots," *DEF CON 32*, 2024 | 观测报告 | 光照影响置信度的定量证据 |
| [10] | Fire/Smoke Detection YOLOv11 + HE/CLAHE, *ScienceDirect*, 2025 | 近期论文 | HE vs CLAHE 对比数据 |
