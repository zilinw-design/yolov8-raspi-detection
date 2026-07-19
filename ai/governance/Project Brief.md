---
status: living
read_policy: before_every_task
owner: user
update_policy: user_or_authorized_ai
---

# Project Brief

## Purpose

Define the project goal, user context, success criteria, explicit non-goals, and key uncertainties.

## Project Identity

| Field | Value |
|---|---|
| Project name | 智能视觉识别与测距 — 电赛备赛项目 |
| One-sentence goal | 在 AI 辅助开发模式下，掌握 YOLO 目标检测、目标跟踪与单目测距的核心能力，能够将算法部署到树莓派5并针对赛题进行系统级优化 |
| Current stage | Phase 2 — YOLOv8 核心目标检测 |

## Background

这是一个**电赛备赛项目**，目标是为智能视觉分拣 / 视觉跟随类赛题做好技术储备。项目核心不是刷参数，而是建立一套**AI 辅助开发的方法论**：在 AI 工具协助下，学会分析问题、设计方案、优化代码，最终将算法高效部署到嵌入式平台（树莓派5）。

**开发环境**：Windows 11（代码编写 + AI 辅助），树莓派5（运行/部署/测试）

**已有参考资料**：
- `D:\Pi\sy2\ai_harness_framework\references\raw\opencv/` — OpenCV 4.14.0-pre 完整源码参考
- `D:\Pi\sy2\ai_harness_framework\references\raw\ultralytics/` — Ultralytics 8.4.100 完整源码参考
- `D:\Pi\sy1\ai_harness_framework\references\raw\rpi-object-detection/` — 树莓派目标检测全流程示例
- `D:\Pi\sy1\ai_harness_framework\src\raspi_vision\camera_stream.py` — Picamera2 BGR 摄像头封装参考

## Hardware

| 硬件 | 型号 | 关键参数 |
|---|---|---|
| 主控 | 树莓派5 | Broadcom BCM2712, 4核 Cortex-A76, 8GB RAM |
| 摄像头 | 亚博智能 Pi5 CSI Camera | Sony IMX219, 8MP, 77° 广角, 22-pin MIPI CSI-2 |
| 排线 | 22Pin-FPC 15cm | 标配 |

## Camera Configuration Reference

- `/boot/firmware/config.txt`：`dtoverlay=imx219`, `camera_auto_detect=1`
- 用户需在 `video` 组中：`sudo usermod -a -G video $USER`
- 验证：`rpicam-still -o test.jpg`（传感器 I2C 地址 0x10）
- Python venv 必须 `--system-site-packages` 以访问系统 `picamera2`/`libcamera`
- 当前内核 6.18.34+rpt ✅ 无回归风险

## Target Users

- 学习者本人（具备 Python 基础，初学计算机视觉，目标为电赛智能视觉类赛题）

## Learning Objectives（分级）

### 工具方法论（贯穿全程）
- 在 AI 辅助下分析问题、设计方案、审查代码
- 学会向 AI 提出精准的技术问题，验证 AI 输出而非盲信
- 建立"先理解原理 → 再动手实现 → 最后优化迭代"的开发习惯

### Phase 2：YOLOv8 目标检测（当前阶段）
- YOLO 工作原理、模型选择（n/s/m）、置信度与 NMS 调参
- 检测结果可视化：不同类别不同颜色、标签、置信度
- 类别过滤：只检测电赛常见物体（人、瓶子、标志、数字等）
- 样式自定义：框粗细、字体大小可配置

### Phase 2b：检测性能调优（独立任务，Phase 2 完成后进入）
- **颜色处理鲁棒性**：为什么 YOLO 语义检测在不同光照下比传统 HSV 颜色检测更鲁棒
- **减少误检**：置信度阈值、NMS IoU 调参对假阳性的影响；光照变化场景的专项测试
- **提高处理速度**：推理分辨率（imgsz 640→480→320）对 FPS 的量化对比；跳帧策略分析
- **模型对比**：YOLOv8n vs YOLOv8s vs YOLOv8m 在树莓派5上的速度/精度权衡

### Phase 3：目标跟踪 + 测距融合
- 跟踪算法（SORT / ByteTrack）：目标 ID 持续跟踪，遮挡处理
- 单目测距原理：已知物体真实尺寸，通过焦距与像素高度估算距离
- 多信息融合显示：画面上同时显示 ID、类别、距离

### Phase 4：电赛级优化与赛题适配
- 复杂环境鲁棒性测试（不同光照、角度、部分遮挡）
- 性能指标量化（准确率、帧率、响应时间）
- 提出并实现至少一种优化方案（模型轻量化 / 后处理改进 / 多传感器融合）

## Success Criteria

- [x] **Phase 1（基础环境）**：树莓派5 + IMX219 摄像头 + YOLOv8n 实时检测 + MJPEG 流到 Windows 浏览器 ✅
- [x] **Phase 2（核心检测）**：检测人/瓶子/标志等常见物体；不同类别不同颜色框；支持类别过滤；理解调参原理 ✅
- [x] **Phase 2b（性能调优）**：FPS量化对比(3模型×4分辨率)；置信度拐点分析(3场景×7档conf)；光照鲁棒性验证(person波动0.009) ✅
- [ ] **Phase 3（跟踪测距）**：ByteTrack 实现 ID 持续跟踪；单目测距误差 < 20%；画面显示 "ID:1 距离:45cm"
- [ ] **Phase 4（赛题适配）**：完成鲁棒性测试报告；量化性能指标；实现一种优化方案并对比前后效果

## Explicit Non-Goals

| Non-Goal | Reason | Reconsider Later |
|---|---|---|
| 训练自定义 YOLO 模型 | 先掌握预训练模型的使用和调优，训练需 GPU 和标注数据 | 是（赛题需要时） |
| 双目/结构光测距 | 先掌握单目测距原理，双目需额外硬件 | 是 |
| 构建完整电赛作品 | 本项目聚焦视觉算法模块，不包含机械结构/电路设计 | 是（后续整合） |
| Web/移动端应用 | 聚焦嵌入式部署 | 否 |

## Key Uncertainties

| Uncertainty | Why It Matters | Risk If Wrong | Verification Method | Status |
|---|---|---|---|---|
| 树莓派5 上 ByteTrack 的实时性能 | 跟踪算法增加计算量，可能拉低帧率 | 需降分辨率或换更轻量跟踪器 | 实际部署测试 | Open |
| 单目测距精度（IMX219 77° 镜头畸变） | 影响距离估算准确性 | 需标定或加修正因子 | 已知距离物体测试 | Open |
| 电赛具体赛题规则 | 不同赛题对检测目标、精度、实时性要求不同 | 需针对性调整算法 | 等赛题发布 | Open |

## Agent Rules

- Separate confirmed facts, assumptions, and open questions.
- 每次实现前先解释原理，再写代码。
- 代码修改后必须说明验证方法。
- Stop before implementing if an uncertainty affects safety, scope, data, architecture, or P0/P1 outcomes.
- Update this file only when project-level intent changes.
