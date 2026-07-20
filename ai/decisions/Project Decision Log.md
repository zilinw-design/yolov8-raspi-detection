---
status: append-only
read_policy: before_each_operation
owner: user_or_lead_ai
update_policy: append_on_proposal
---

# Project Decision Log

## Purpose

记录项目各阶段的技术决策：选了什么方案、为什么、文献出处、否决了什么、为什么否决。

每次提出优化方案时必须追加记录。服务于：
- 当前项目后续阶段的方案回溯
- 其他项目的技术决策参考
- 电赛答辩时的技术选型说明

## Record Template

```markdown
## [日期] 决策标题

### 背景
（当前阶段、遇到的问题、已有数据）

### 候选方案

| 方案 | 原理 | 出处 | 优点 | 缺点 |
|---|---|---|---|---|
| A | ... | ... | ... | ... |
| B | ... | ... | ... | ... |

### 选用方案
（选了什么，为什么）

### 否决方案及原因
（否决了什么，为什么）

### 引用文件
- 本项目文件：
- 外部参考：
- 文献：
```

---

## Phase 1 — 基础环境搭建

### [2026-07-18] 摄像头采集方案：picamera2 vs cv2.VideoCapture

**背景**：树莓派5 + 亚博智能 IMX219 CSI 摄像头，需要获取视频流给 YOLO。

**候选方案**：

| 方案 | 原理 | 出处 | 优点 | 缺点 |
|---|---|---|---|---|
| A. cv2.VideoCapture(0) | OpenCV 通用摄像头 API | OpenCV 官方文档 | 跨平台，代码一致 | **Pi5 CSI 不支持**，只能用 USB 摄像头 |
| B. picamera2 | 树莓派官方 Python 库，PiSP pipeline | Raspberry Pi OS 预装 | BGR888 格式直接兼容 OpenCV，零拷贝 | 只能在 Pi 上跑，Windows 无法测试 |
| C. libcamera 命令行 + 管道 | 直接调命令行获取帧 | libcamera 官方 | 底层控制 | Python 集成复杂，性能差 |

**选用方案**：B — picamera2，BGR888 格式。

**否决方案及原因**：
- A：树莓派5的 CSI 摄像头不走 V4L2 默认路径，cv2.VideoCapture(0) 无法打开
- C：管道方式的 JPEG 编解码额外开销，且 Python 集成不优雅

**引用文件**：
- 本项目：`src/task1_basic/detect_pi.py` — CSICamera 类
- 外部参考：`D:\Pi\sy1\...\camera_stream.py` — picamera2 BGR888 先例
- 硬件验证：`rpicam-still -o test.jpg` — libcamera v0.7.1, PiSP pipeline

---

### [2026-07-18] 显示方案：MJPEG HTTP 流 vs cv2.imshow vs VNC

**背景**：树莓派无显示器，需要在 Windows 上实时查看检测画面。

**候选方案**：

| 方案 | 原理 | 出处 | 优点 | 缺点 |
|---|---|---|---|---|
| A. MJPEG HTTP 流 | Python http.server + multipart JPEG | 社区通用方案 | 浏览器即可查看，无需客户端 | 有 1-2 帧延迟 |
| B. VNC 远程桌面 | 树莓派系统自带 | Raspberry Pi OS | 桌面级体验 | cv2.imshow 在 VNC 下可能失败 |
| C. RTSP 流 | GStreamer/FFmpeg 推流 | 视频流标准协议 | 专业级低延迟 | 配置复杂，依赖重 |

**选用方案**：A — MJPEG HTTP 流。

**否决方案及原因**：
- B：cv2.imshow 通过 VNC 不稳定，且有额外图形开销
- C：GStreamer 配置复杂，对学习项目来说过度工程化

**引用文件**：
- 本项目：`src/task1_basic/detect_pi.py` — MJPEGHandler, MJPEGServer 类

---

### [2026-07-18] Python 虚拟环境方案：--system-site-packages

**背景**：picamera2 和 libcamera 预装在树莓派系统 Python 中，pip 无法安装。

**候选方案**：

| 方案 | 原理 | 出处 | 优点 | 缺点 |
|---|---|---|---|---|
| A. `--system-site-packages` | venv 继承系统包 | Python 官方文档 | picamera2 可用，依赖隔离兼顾 | 依赖列表不完整 |
| B. 直接用系统 Python | 不创建 venv | — | 最简单 | 依赖污染，无隔离 |
| C. pip install picamera2 到 venv | 尝试在 venv 中安装 | — | 完全隔离 | **picamera2 不在 PyPI 上**，无法安装 |

**选用方案**：A — `python3 -m venv --system-site-packages venv`。

**否决方案及原因**：
- B：无依赖隔离，后续项目冲突风险
- C：picamera2 是系统包，不在 PyPI

**引用文件**：
- 外部参考：`D:\Pi\sy1\...\camera_stream.py` 的注释：`from picamera2 import Picamera2`
- 社区讨论：Raspberry Pi Forums — "no module called libcamera" → 解决方案 `--system-site-packages`

---

## Phase 2 — 核心目标检测

### [2026-07-19] 绘图方案：自定义 draw_boxes vs YOLO 内置 results.plot()

**背景**：需要不同类别不同颜色、类别过滤、样式自定义。

**候选方案**：

| 方案 | 原理 | 出处 | 优点 | 缺点 |
|---|---|---|---|---|
| A. 自定义 draw_boxes() | 遍历 results.boxes，用 cv2.rectangle 手绘 | 本项目实现 | 颜色固定、支持过滤、样式可配 | 需要手写绘图逻辑 |
| B. results.plot() | YOLO 内置绘图 | Ultralytics API | 一行代码 | 颜色随机每帧变化、不支持过滤、不可配样式 |

**选用方案**：A — 自定义 draw_boxes()。

**否决方案及原因**：
- B：plot() 随机生成颜色、无类别过滤、无样式配置参数

**引用文件**：
- 本项目：`src/task1_basic/detect_pi.py` — draw_boxes(), get_color(), _CLASS_COLOR_MAP
- 参考架构：`yolo_benchmark_driver.py` (mhuzaifadev/ultralytics_yolo_comparison) — 自定义绘图的指标叠加模式

---

### [2026-07-19] 测试体系：三个独立实验 vs 单个综合脚本

**背景**：需要对 FPS、置信度、光照鲁棒性分别做定量测试。

**候选方案**：

| 方案 | 原理 | 出处 | 优点 | 缺点 |
|---|---|---|---|---|
| A. 三个独立脚本 | 每个实验一个脚本 | 本项目设计 | 各自独立、参数清晰、互不干扰 | 三份代码 |
| B. 一个综合 benchmark.py | 所有实验在一个脚本 | 初版实现 | 一个入口 | 摄像头冲突 bug、参数混乱 |

**选用方案**：A — benchmark_fps.py / benchmark_conf.py / benchmark_lighting.py。

**否决方案及原因**：
- B：实验一的摄像头未完全释放导致实验二报错（Camera in Configured state trying acquire），拆分后各自管理摄像头生命周期

**引用文件**：
- 本项目：`src/task2_advanced/benchmark_fps.py`, `benchmark_conf.py`, `benchmark_lighting.py`
- 参考方法：`yolo_benchmark_driver.py` (mhuzaifadev/ultralytics_yolo_comparison) — warmup + timed runs 模式
- 参考方法：`YoloV8-ncnn-Raspberry-Pi-4` (Qengineering) — 纯推理时间测量、FPS 基线数据

---

## Phase 2b — 性能调优

### [2026-07-19] FPS 测量方法：warmup + timed runs

**背景**：需要准确测量 YOLO 在树莓派上的推理速度。

**候选方案**：

| 方案 | 原理 | 出处 | 优点 | 缺点 |
|---|---|---|---|---|
| A. Warmup 10帧 + Timed 100帧 | 跳过首帧初始化开销 | yolo_comparison 项目 | 排除冷启动干扰，数据稳定 | 需要 warmup 帧数配置 |
| B. 简单跑100帧取平均 | 直接计时 | — | 简单 | 首帧 2-3 秒 JIT 编译严重拉低均值 |

**选用方案**：A — warmup 10 frames + timed 100 frames。

**否决方案及原因**：
- B：PyTorch 首次推理触发 JIT 编译，首帧耗时 2-3s，均摊到 100 帧会使 FPS 被低估约 30%

**引用文件**：
- 外部参考：`yolo_benchmark_driver.py` — `num_timed_runs=100, num_warmup_runs=10`
- 外部参考：Ultralytics 官方 `benchmarks.py` — `ProfileModels` 类的 `num_warmup_runs`

### [2026-07-19] 基线参照：Qengineering ncnn FPS 数据

**背景**：需要知道 PyTorch 版本的 FPS 在树莓派上的"天花板"是多少。

**关键数据**（Qengineering 实测，RPi 5 @ 2.9GHz）：

| 模型 | 框架 | imgsz=640 FPS |
|---|---|---|
| YOLOv8n | ncnn C++ | 20.0 |
| YOLOv8n | PyTorch (本项目) | 3.2 |
| YOLOv8s | ncnn C++ | 11.0 |
| YOLOv8s | PyTorch (本项目) | 1.1 |

**结论**：PyTorch 比 ncnn 慢 6-10 倍。Phase 4 的 ONNX/ncnn 导出有明确的 6× 提升空间。

**引用文件**：
- 外部参考：`D:\Pi\sy2\ai_harness_framework\references\raw\YoloV8-ncnn-Raspberry-Pi-4\README.md`
- 本项目：`src/task2_advanced/benchmark_fps.py` — `NCNN_BASELINE` 字典

---

### [2026-07-20] 暗光优化方案：CLAHE + Gamma + 逐类 conf

**背景**：实验三数据显示 book/cup 在暗光下丢失，person 稳定。

**候选方案**：

| 方案 | 原理 | 出处 | 优点 | 缺点 |
|---|---|---|---|---|
| A. CLAHE + Gamma + 逐类 conf | 三步管线：预处理增强 + 推理 + 逐类阈值 | 见文献 | 有充分文献支持，轻量可实时 | 逐类阈值需预设 |
| B. 亮度自适应 conf（按画面平均亮度自动调 conf） | 亮度低→降 conf | 自定义 | 自适应 | **无文献支持**，逻辑有问题 |
| C. HE 替代 CLAHE | 全局直方图均衡 | Gonzalez 2018 | 简单 | 放大平坦区域噪声 |
| D. GAN/Retinex 增强 | 深度学习增强 | 学术论文 | 效果极致 | **树莓派跑不动** |
| E. Temperature Scaling 校准 | 修正模型置信度系统偏置 | Ingrisch 2025 | 校准后置信度更准确 | 全局单参数，不解决逐类差异 |
| F. 训练暗光专用模型 | 在暗光数据集上 fine-tune | 通用做法 | 最根本 | 需 GPU + 标注数据，非现阶段目标 |

**选用方案**：A — CLAHE (Zuiderveld 1994) + Gamma (Poynton 1998) + 逐类 conf (Han, Won, Koo 2025)。

**否决方案及原因**：
- B：亮度自适应 conf 无文献支持，且逻辑有缺陷——亮度影响的是图像质量，不是置信度的准确性。暗光下检测差是因为模型从退化图像中提取到的特征本身就弱了，调 conf 不解决根本问题
- C：HE 全局操作，噪声放大。2024-2025 多篇论文 CLAHE 综合优于 HE
- D：GAN 推理开销大，树莓派无 GPU 加速，实时性不达标
- E：Temperature Scaling 是全局参数，不逐类区分。且需要校准数据集
- F：需要标注暗光数据 + GPU 训练，Phase 4 考虑

**引用文件**：
- 本项目：`references/summarized/low_light_optimization_plan.md` — 完整文献整理
- 文献 [1]：Zuiderveld (1994) — CLAHE 原始论文
- 文献 [2]：Poynton (1998) — Gamma 感知编码理论

### 验证结果（2026-07-20 树莓派5实测）

**暗光图片 A/B 对比**（`tests/test_images/窗帘拉上一半.jpg`）：

| conf | 优化前(基线) | 优化后(CLAHE+Gamma 0.7) | 变化 |
|---|---|---|---|
| 0.1 | 4 | 6 | +50% |
| 0.2 | 1 | 3 | +200% |
| 0.3-0.7 | 1 | 1 | — |
| 推理耗时 | 69.2ms | 69.4ms | +0.3% |

**实时摄像头测试**（`detect_pi.py --clahe --gamma 0.7 --per-class-conf`）：
- 最暗测试图（`只开小台灯.jpg`）：cup/bottle 可检出，置信度 0.2-0.3，偶有不稳定
- person 检测稳定不受影响
- 画面整体亮度明显提升，但未过度曝光

**Phase 2b 边界确认**：
- ✅ 目标达成：暗光下从小物体"未检出"→"可检出"（conf 0.2-0.3）
- ❌ 超出范围（留给 Phase 3/4）：置信度偏低（imgsz=320 物理限制）、偶发不稳定（需要跟踪补帧）

---

### [2026-07-20] 自适应亮度路由（已记录，延后实现）

**背景**：当前 CLAHE+Gamma 需要手动 `--clahe --gamma 0.7`。电赛场景光照不确定，需自动判断。

**文献依据**：FAIERDet (2025) — 模糊推理评估曝光 → 暗帧增强/亮帧跳过；YOLO-DER (2025) — Dynamic Enhancement Routing；共 6 篇 2024-2025 论文一致采用"选择性增强"策略。

**计划方案**：`--clahe auto` 模式。计算 LAB L 通道均值 → < 阈值自动开 CLAHE+Gamma → ≥ 阈值仅 CLAHE。改动量 ~20 行。

**延后原因**：先完成 Phase 3（跟踪+测距）和 Phase 4（ONNX导出），此优化属于锦上添花——当前手动 `--clahe --gamma 0.7` 已覆盖暗光场景。
- 文献 [4]：自动驾驶低光检测 (2025) — CLAHE+Gamma+中值滤波, +13.3%
- 文献 [5]：PPE检测 YOLOv9+CLAHE (2025) — CLAHE 实时性优于 GAN
- 文献 [6]：车辆检测 YOLOv8+CLAHE (2024) — 暗光 +10-20%
- 文献 [7]：Han, Won, Koo (2025) — 逐类 conf 阈值方法论
- 文献 [8]：Ingrisch et al. (2025) — Temperature Scaling 校准
- 文献 [9]：Shams (2024) — 光照+位置影响置信度的观测证据

---

## 流程规范

### 提出优化方案时必须执行

1. **先查文献**：搜索是否有相关的论文/GitHub 项目，确认方案有出处
2. **对比候选方案**：列出 ≥2 个方案，注明出处、优缺点
3. **明确否决原因**：否决的方案要说明为什么否决
4. **追加本记录**：在此文件中追加决策条目
5. **等待审批**：用户确认后再执行

此规范已写入 `AGENTS.md` → Coding Pre-Execution Protocol。
