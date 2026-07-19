---
status: living
read_policy: before_each_operation
owner: user_or_lead_ai
update_policy: update_per_task
---

# Constraints And Priority

## Purpose

Define the current task scope, priority levels, and attention rules.

## Priority Levels

| Level | Meaning | Rule |
|---|---|---|
| P0 | Must not break | No change may damage this. |
| P1 | Must complete this round | Required for current task success. |
| P2 | Useful improvement | May do only if it does not expand scope or risk P0/P1. |
| P3 | Optional enhancement | Do not do unless explicitly approved. |

## Current Priorities — Phase 2：YOLOv8 核心目标检测

### P0: Must Not Break

- 树莓派5 环境（venv、picamera2、torch、ultralytics）不破坏
- IMX219 摄像头正常运行
- MJPEG 流正常推送到浏览器
- `detect_pi.py` 的现有默认行为不变（新增参数都有默认值，不加参数=原行为）

### P1: Must Complete

- **电赛常见物体检测**：人(person)、瓶子(bottle)、标志类(stop sign/traffic light)、手机(cell phone)、书本(book)、杯子(cup)
- **不同类别不同颜色检测框**：每个类别固定对应一种颜色，直观区分
- **类别过滤功能**：`--classes person bottle` 只检测指定类别
- **样式自定义**：框粗细(`--box-thickness`)、字体大小(`--font-scale`) 通过参数可调
- **原理理解**：代码注释说明 YOLO 推理流程、NMS 原理、置信度含义

### P2: Useful Improvements

- 画面同时显示各类别目标数量统计
- 保存带检测结果的截图

### P3: Optional Enhancements

- 小物体检测专项测试（不同 imgsz 对比）
- 误检/漏检分析
- GUI 控制面板

## Current Scope

| Field | Value |
|---|---|
| Task type | Implementation（实现 Phase 2 核心检测功能） |
| Allowed changes | 修改 `src/task1_basic/detect_pi.py`（新增参数+自定义绘图）；新建 `src/task2_advanced/` |
| Forbidden changes | 不得删除/破坏任务1已有功能；不得修改项目目录结构；不得修改治理文件（本文件除外） |
| Required confirmation | 安装新依赖前告知用户 |

## Attention Rules

- **编码前必须执行审批协议**：参见 `AGENTS.md` → Coding Pre-Execution Protocol。先出方案 → 用户审核 → 再写代码。
- **先讲原理，再写代码**：每实现一个功能前，先解释工作原理。
- 代码修改后说明如何验证（用什么命令、预期什么效果）。
- P2/P3 items must not drive architecture, data model, dependency, or workflow changes.

## Scope Expansion Rule

If completion requires expansion: stop → explain → identify risks → propose minimal expansion → wait for confirmation.
