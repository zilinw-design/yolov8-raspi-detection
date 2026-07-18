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
| Project name | YOLOv8 实时目标检测学习项目 |
| One-sentence goal | 使用 YOLOv8 预训练模型 + OpenCV 实现实时目标检测系统，从基础推理到树莓派5性能优化 |
| Current stage | Planning — 项目初始化，尚未开始编码 |

## Background

这是一个计算机视觉学习项目，旨在系统掌握 YOLOv8 模型在实时视频流中的目标检测应用。从加载预训练模型进行基础推理开始，逐步深入到参数调优、自定义检测逻辑，最终在树莓派5上完成性能优化部署。

**开发环境**：Windows 11（代码编写），树莓派5（运行/部署）

**已有参考资料**：
- `<repo_root>/ai_harness_framework/references/raw/opencv/` — OpenCV 4.14.0-pre 完整源码参考
- `<repo_root>/ai_harness_framework/references/raw/ultralytics/` — Ultralytics 8.4.100 完整源码参考
- `D:\Pi\sy1\ai_harness_framework\references\raw\rpi-object-detection/` — 树莓派目标检测全流程示例（YOLO/TFLite/颜色/形状跟踪）
- `D:\Pi\sy1\ai_harness_framework\references\raw\Install-OpenCV-Raspberry-Pi-64-bits/` — OpenCV 树莓派安装脚本集
- `D:\Pi\sy1\ai_harness_framework\src\raspi_vision\camera_stream.py` — Picamera2 + OpenCV BGR 摄像头封装（可复用模式）

## Hardware

| 硬件 | 型号 | 关键参数 |
|---|---|---|
| 主控 | 树莓派5 | Broadcom BCM2712, 4核 Cortex-A76 |
| 摄像头 | 亚博智能 Pi5 CSI Camera | Sony IMX219, 8MP, 77° 广角, 22-pin MIPI CSI-2 |
| 排线 | 22Pin-FPC 15cm | 标配 |

## Camera Configuration Reference

基于 [YahboomTechnology/Pi5-MIPI-CSI-Camera](https://github.com/YahboomTechnology/Pi5-MIPI-CSI-Camera) 和 [[Hackaday IMX219 bringup log]](https://hackaday.io/project/203704-gesturebot-ros2-computer-vision-mobile-platform/log/242459-bringing-up-an-imx219-camera-on-raspberry-pi-5-with-ubuntu-2404)：

- `/boot/firmware/config.txt` 配置：`dtoverlay=imx219`, `camera_auto_detect=1`
- 用户需在 `video` 组中：`sudo usermod -a -G video $USER`
- 验证命令：`dmesg \| grep imx219`, `i2cdetect -y 10`（传感器地址 0x10）, `rpicam-still -o test.jpg`
- Python 虚拟环境需 `--system-site-packages` 以访问系统预装的 `picamera2` 和 `libcamera`
- ⚠️ 内核 6.12.34+rpt 存在已知回归，IMX219 可能在 media graph 中缺失，部署前需先确认当前内核版本

## Target Users

- 学习者本人（具备 Python 基础，初学计算机视觉）

## Core Requirements

1. 加载 YOLOv8 预训练模型（n/s/m 系列），对图片和实时视频流执行目标检测
2. 在画面上绘制检测框、物体类别标签和置信度分数
3. 支持只检测特定类别（如人、瓶子、手机），并自定义检测框颜色/粗细/字体
4. 支持自定义物体检测（如特定颜色物体识别），统计检测目标数量
5. 在树莓派5上实现 8-15fps 以上的实时检测帧率（通过 ONNX 导出、分辨率调整等手段）

## Success Criteria

- [ ] **任务1**：能加载 YOLOv8n 预训练模型，对摄像头视频流实时检测，画框+标签+置信度正常显示
- [ ] **任务2**：实现类别过滤、检测框样式自定义、n/s/m 模型速度/精度对比
- [ ] **任务3**：实现自定义检测逻辑（颜色识别/特定标志）+ 目标数量统计
- [ ] **任务4**：模型导出为 ONNX，树莓派5 上达到 8-15fps 以上实时帧率

## Explicit Non-Goals

Non-goals are out of scope for the current project stage.

| Non-Goal | Reason | Risk If Done Accidentally | Reconsider Later |
|---|---|---|---|
| 训练自定义数据集 | 本阶段仅使用预训练模型 | 训练需要大量标注数据和GPU资源，偏离学习目标 | 是（后续阶段可扩展） |
| 构建 Web/移动端应用 | 聚焦桌面端和树莓派嵌入式部署 | 增加不必要的前端复杂度 | 是 |
| 多模型融合或模型修改 | 聚焦模型使用而非模型设计 | 需要深度学习架构知识 | 是 |
| 生产级部署和监控 | 学习项目，非生产系统 | 过度工程化 | 否 |

## Key Uncertainties

| Uncertainty | Why It Matters | Risk If Wrong | Verification Method | Status |
|---|---|---|---|---|
| 树莓派5 上 YOLOv8n ONNX 的实际推理速度 | 影响任务4的优化策略选择 | 如果 ONNX 速度仍不达标，需要尝试 TensorRT 或 NCNN | 实际部署测试 | Open |
| Python 版本兼容性（Python 3.8-3.13） | Ultralytics 要求 >=3.8，树莓派默认版本待确认 | 版本不兼容导致无法安装依赖 | 检查树莓派环境 | Open |
| 摄像头设备可用性 | 已确认：亚博智能 IMX219 CSI Camera for Pi 5 | 配置错误或无驱动时降级为视频文件检测 | 在树莓派5上执行 rpicam-still 拍摄测试照片 | Resolved — 硬件已确认，软件配置待实际验证 |
| 树莓派5内核版本兼容性 | 内核 6.12.34+rpt 存在 IMX219 回归 Bug | 摄像头无法被 rpicam-hello/libcamera 识别 | `uname -r` 检查内核版本，如有问题降级或等补丁 | Open — 部署前检查 |

## Agent Rules

- Separate confirmed facts, assumptions, and open questions.
- Use conservative assumptions only for low-risk work.
- Stop before implementing if an uncertainty affects safety, scope, data, architecture, or P0/P1 outcomes.
- Update this file only when project-level intent changes.
