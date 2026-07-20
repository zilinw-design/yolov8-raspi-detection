# Phase 3 — 目标跟踪 + 单目测距：方案设计

> 日期：2026-07-20
> 状态：已修订，待审批

## 修订记录

| 日期 | 变更 |
|---|---|
| 2026-07-20 初版 | Pinhole 模型 + ByteTrack 跟踪方案 |
| 2026-07-20 修订 | 对准 2025 电赛 C 题，改用 solvePnP + A4 纸自标定 |
| 2026-07-20 修订 | 新增 4K USB Camera 代替 IMX219 做测量 |
| 2026-07-20 修订 | 无固定支架 → A4 纸自标定（每次开机一次）替代棋盘格标定 |
| 2026-07-20 修订 | 先做屏幕图形近距离测试(30-80cm)验证管线，再做A4纸1.5m距离 |

---

## 一、目标

在 Phase 2 检测基础上增加两项能力：

1. **目标 ID 持续跟踪**：同一物体在连续帧中保持相同 ID，不因短暂遮挡或漏检而丢失
2. **单目测距**：已知物体真实尺寸，通过 pinhole 相机模型估算距离，画面显示 "ID:1 person 1.5m"

**不需要新硬件、不需要新依赖。**

---

## 二、跟踪方案：ByteTrack（Ultralytics 内置）

### 2.1 为什么选 ByteTrack

Ultralytics 已内置两种跟踪器，一行代码调用：

```python
results = model.track(frame, persist=True, tracker="bytetrack.yaml")
# results[0].boxes.id  → 跟踪 ID
```

| 属性 | ByteTrack | BoT-SORT（默认） | SORT |
|---|---|---|---|
| ReID 模块 | ❌ 无（轻量） | ✅ 有 | ❌ 无 |
| 速度 | ★ 最快 | ★★ 中等 | ★ 最快 |
| ID 稳定性 | 好（关联所有检测框） | 更好 | 差（ID 切换多） |
| 低帧率适应性 | ★ 好 | 中等 | ★ 好 |
| 树莓派适用 | ★ 推荐 | 偏重 | 参考基准 |

**文献依据**：

- ByteTrack 论文：Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV 2022
- 边缘设备对比 (2025)：ByteTrack + YOLOv5 已在 NVIDIA Jetson 验证为实时可行的边缘追踪方案
- 2024 学位论文直接对比：ByteTrack 在速度上领先 BoT-SORT，无 ReID 模块更适配资源受限设备
- 2025 番茄产量评估：无 ReID 的跟踪器（SORT/ByteTrack）对帧率波动适应性更好

### 2.2 关键参数

| 参数 | 作用 | 推荐值 |
|---|---|---|
| `persist=True` | **必须**：跨帧保持 ID，否则每帧新建跟踪器 | True |
| `tracker="bytetrack.yaml"` | 选择 ByteTrack（默认是 BoT-SORT） | "bytetrack.yaml" |
| `conf` | 检测置信度阈值 | 0.25（正常）/ 0.2（暗光+逐类conf） |
| `track_high_thresh` | ByteTrack 高阈值（关联高质量检测） | 0.5 |
| `track_low_thresh` | ByteTrack 低阈值（关联低质量检测） | 0.1 |

### 2.3 实现方式

```python
# 改动 detect_pi.py 主循环中的一行：
# 之前：results = model(frame, ...)[0]
# 之后：results = model.track(frame, persist=True, tracker="bytetrack.yaml", ...)[0]

# 获取跟踪 ID：
track_ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else []
```

### 2.4 预期效果

- 最暗场景下偶发检测不稳定 → ByteTrack 用上一帧位置预测当前帧，漏检 1-2 帧不丢 ID
- 画面显示 "ID:1 person 0.85"、"ID:2 bottle 0.28"
- FPS 影响：ByteTrack 仅做卡尔曼滤波 + 匈牙利匹配，开销 < 5ms/帧

---

## 三、测距方案：Pinhole 相机模型

### 3.1 原理

```
                    实物高度 H (已知)
                    |←──── 距离 D ────→|
    相机 ───────────┼─────────────────┼
    焦距 f           │                 │
    传感器           │                 │
    ┌───┐           │                 │
    │   │ 像素高 h  │                 │
    └───┘           │                 │
                    │                 │

    相似三角形：D = (f × H) / h

    其中：
      f = 焦距（像素），需要标定
      H = 物体真实高度（米），预设值
      h = 检测框像素高度，YOLO 输出
```

**公式出处**：Han et al. (2025), "A lightweight distance estimation method using pinhole camera geometry model," *Measurement Science and Technology*.

**别用宽度**：YOLO 框的宽度随物体旋转剧烈变化，高度相对稳定。同类论文全部用高度。

### 3.2 文献验证

| 论文 | YOLO | 准确度 | 测量范围 | 应用 |
|---|---|---|---|---|
| Han et al. (2025) | YOLOv8 | MAE=0.085m, MAPE=1.94% | 1.1–7.5m | 机器人/自动驾驶 |
| Han et al. (2026) | YOLOv8 | 误差 5.87%（GPS） | 4.5–40m | UAV 嵌入式 |
| Wang et al. (2023) | YOLOv5s | <2cm（采摘阶段） | 近距离 | 农业采摘机器人 |

### 3.3 焦距标定

IMX219 官方参数：

| 参数 | 值 |
|---|---|
| 传感器尺寸 | 3.674mm × 2.760mm |
| 像素 | 3280 × 2464 |
| 像素大小 | 1.12µm |
| 镜头焦距（物理） | 3.04mm（Raspberry Pi Camera Module v2，同款传感器） |

**理论焦距（像素）**：f_pixel = (3.04mm / 3.674mm) × 3280 ≈ **2714 像素**（全分辨率时）

但由于我们实际使用 640×480 采集 + imgsz=320 推理，且树莓派摄像头有 binning/crop，实际焦距需实测标定：

```
标定方法：
1. 放一个已知高度 H_cm 的物体，用卷尺量出准确距离 D_cm
2. 拍一张照片，读取检测框像素高度 h
3. 反算：f_pixel = (D_cm × h) / H_cm
4. 重复 3-5 次取平均
```

### 3.4 电赛常见物体的已知尺寸预设

| 物体 | COCO 类 | 典型高度/宽度(cm) | 测距用途 |
|---|---|---|---|
| 人（站立） | person | 170 | 视觉跟随赛题 |
| 瓶子 | bottle | 20 | 分拣赛题 |
| 手机 | cell phone | 15 | 识别赛题 |
| 书本 | book | 25 | 分拣赛题 |
| 杯子 | cup | 10 | 分拣赛题 |
| 标志牌 | stop sign | 60 | 导航赛题 |

### 3.5 预期精度

以 Han et al. (2025) 的数据作为参照——在正确标定焦距且目标尺寸准确的前提下：
- 1-2m 范围：误差 < 5cm
- 2-5m 范围：误差 < 10cm
- >5m：像素高度 < 20px，误差急剧增大

---

## 四、实现方案

### 4.1 改动范围

| 文件 | 改动 | 行数 |
|---|---|---|
| `detect_pi.py` | 推理行改 `model.track()`；绘图加 ID 标签 | ~15行 |
| 新增 `src/phase3_tracking/distance.py` | 距离计算 + 物体尺寸常量 | ~30行 |
| 新增 `src/phase3_tracking/calibrate_focal.py` | 焦距标定脚本（跑一次） | ~20行 |

**不引入新依赖**：ByteTrack 是 Ultralytics 内置，不需要额外安装。

### 4.2 画面显示

```
┌─────────────────────────────────┐
│ FPS: 10.2  Model: yolov8n       │
│ person: 2  bottle: 1            │
│                                 │
│ ┌──────────┐    ┌──────────┐    │
│ │ ID:1     │    │ ID:3     │    │
│ │ person   │    │ bottle   │    │
│ │ 1.52m ←  │    │ 0.87m    │    │
│ └──────────┘    └──────────┘    │
│                                 │
│    ┌──────────┐                 │
│    │ ID:2     │                 │
│    │ person   │                 │
│    │ 2.31m    │                 │
│    └──────────┘                 │
└─────────────────────────────────┘
```

### 4.3 新增 CLI 参数

```bash
# 跟踪
python detect_pi.py --track                           # 启用跟踪（默认 ByteTrack）
python detect_pi.py --track --tracker bytetrack.yaml  # 指定跟踪器

# 测距
python detect_pi.py --distance                        # 启用测距

# 组合
python detect_pi.py --track --distance --clahe --gamma 0.7 --per-class-conf
```

---

## 五、验证计划

### 5.1 焦距标定验证

用卷尺 + 已知物体（水瓶 20cm 高），在 50cm/100cm/150cm 三个距离拍照，反算焦距。三次结果相差 < 5% 则标定成功。

### 5.2 跟踪验证

用实验二的测试图片序列模拟视频帧，确认同一物体 ID 跨帧不变。

### 5.3 测距验证

固定物体在已知距离（1m/2m/3m），对比算法估算距离与实际距离。

---

## 六、Plan B（如果 ByteTrack 太慢）

**备选**：SORT（更轻量的卡尔曼滤波+匈牙利匹配）。ID 切换更多，但树莓派上省 2-3ms/帧。

如果测距误差太大：
- 改用物体宽度的中位数而非高度（H 值更稳定但公式相同）
- Phase 4 考虑标定板精确标定（Zhang 标定法）

---

## 七、参考文献

| 编号 | 出处 | 参考部分 |
|---|---|---|
| [T1] | Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV 2022 | 跟踪算法原理 |
| [T2] | Ultralytics, `model.track()` API documentation | 内置跟踪接口 |
| [T3] | 2024 thesis: BoT-SORT vs ByteTrack + YOLOv8 comparison | 速度优先选 ByteTrack |
| [T4] | 2025 tomato yield estimation: ByteTrack + RT-DETR accuracy 95.5% | 跟踪计数验证 |
| [D1] | Han et al. (2025), "A lightweight distance estimation method using pinhole camera geometry model," *Meas. Sci. Technol.* | pinhole 测距公式 + YOLOv8 精度数据 |
| [D2] | Han et al. (2026), "Real-Time GPS Coordinate Estimation Using a Monocular Camera," *J. Electr. Comput. Eng.* | 嵌入式平台测距验证 |
| [D3] | Wang et al. (2023), "Monocular Vision Based Target Localization for Rose Picking Robot," *ICRSA* | 农业机器人近距离测距 |
| [D4] | Sony IMX219 datasheet | 传感器物理参数（焦距标定基准） |
