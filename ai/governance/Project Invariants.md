---
status: scoped-living
read_policy: when_related
owner: user
update_policy: user_only_or_explicit_authorization
---

# Project Invariants

## Purpose

Define non-security technical constraints that agents must not change without explicit authorization.

## Technology Stack

| Area | Current Choice | Change Policy |
|---|---|---|
| 核心框架 | YOLOv8 (Ultralytics >= 8.0) | Explicit approval required |
| 模型 | YOLOv8n / YOLOv8s（轻量级预训练模型） | 可扩展到 m/l/x（任务2），需批准 |
| 模型格式 | .pt（PyTorch 原生）+ ONNX（可选部署优化） | 新增格式需批准 |
| 图像处理 | OpenCV (opencv-python >= 4.6.0) | Explicit approval required |
| 可视化 | Matplotlib >= 3.3.0 | 不可替换为其他绑图库 |
| 数值计算 | NumPy >= 1.23.0 | Explicit approval required |
| 深度学习框架 | PyTorch >= 1.8.0 | Explicit approval required |
| 语言 | Python >= 3.8 | Explicit approval required |
| 摄像头接口 | picamera2（树莓派5原生, BGR888格式） | 不可替换为 cv2.VideoCapture（树莓派CSI不支持） |

## Runtime Compatibility

| Area | Requirement |
|---|---|
| 开发环境 | Windows 11（当前） |
| 目标部署平台 | 树莓派5 (Raspberry Pi OS) |
| Python 版本 | >= 3.8, <= 3.13 |
| 摄像头接口 | 亚博智能 IMX219 CSI Camera (22-pin MIPI) — 已验证用于树莓派5 |

## Dependency Policy

- Do not add dependencies without approval.
- Do not upgrade core dependencies without approval.
- Do not replace existing libraries without approval.
- 本阶段核心依赖：`ultralytics`, `opencv-python`, `numpy`, `matplotlib`, `torch`
- 可选依赖（任务4）：`onnx`, `onnxruntime`

## Interface Contracts

Do not change without approval:
- 检测结果的数据结构（bounding box 格式：[x1, y1, x2, y2], confidence, class_id, class_name）
- 可视化输出规范（框颜色范围 BGR/RGB、字体类型、线宽默认值）

## Structure Contracts

Do not reorganize without approval:
- `src/` 目录下按任务分模块（`task1_basic/`, `task2_advanced/`, `task3_custom/`, `task4_optimize/`）
- 每个任务目录独立，互不依赖

## Build, Test, and Run

Do not change without approval:
- 运行方式：`python src/taskX_xxx/main.py` 从项目根目录执行
- 不使用 pipenv/poetry（使用 pip + requirements.txt）
- 模型文件存放：`models/` 目录（.pt 文件不提交 git，通过 `.gitignore` 排除）
- 树莓派上虚拟环境必须使用 `--system-site-packages`（picamera2/libcamera 仅在系统 Python 中可用）
- 源代码同时兼容 Windows（用视频文件/图片测试）和树莓派5（用摄像头实时检测）

## External Dependencies (sy1)

以下资源位于 `D:\Pi\sy1\`，作为只读参考资料，不复制到本项目中：

- `D:\Pi\sy1\ai_harness_framework\src\raspi_vision\camera_stream.py` — Picamera2 BGR 摄像头封装参考实现
- `D:\Pi\sy1\ai_harness_framework\references\raw\rpi-object-detection/` — 树莓派目标检测完整示例
- `D:\Pi\sy1\ai_harness_framework\references\raw\Install-OpenCV-Raspberry-Pi-64-bits/` — OpenCV 安装脚本集
