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

## Current Priorities — Phase 1: 基础入门（任务1）

### P0: Must Not Break

- Python 环境正常运行，不破坏已有安装的包
- 不修改 `references/` 中的 OpenCV 和 Ultralytics 参考源码
- 摄像头权限：必须先确认用户同意再打开摄像头

### P1: Must Complete（任务1）

- 安装 ultralytics、opencv-python 依赖
- 下载 YOLOv8n.pt 预训练模型
- 实现图片目标检测（画框 + 类别标签 + 置信度）
- 实现摄像头实时视频流检测
- 代码放在 `src/task1_basic/` 目录下

### P2: Useful Improvements

- 检测结果显示 FPS 信息
- 保存检测结果截图功能
- 使用 Matplotlib 对比原图和检测结果

### P3: Optional Enhancements

- GUI 控制面板
- 录制检测视频

## Current Scope

| Field | Value |
|---|---|
| Task type | Implementation（实现任务1） |
| Allowed changes | 新建 `src/task1_basic/` 目录及代码文件；安装 Python 依赖；下载预训练模型 |
| Forbidden changes | 不得修改现有框架文件；不得修改 references/ 中的参考源码；不得删除任何文件 |
| Required confirmation | 安装依赖前告知用户；下载模型前告知用户文件大小和存放位置 |

## Attention Rules

- User emphasis is an attention signal, not automatic P0/P1 priority.
- P2/P3 items must not drive architecture, data model, dependency, or workflow changes.
- Local polish must not become global refactoring without approval.
- Do not introduce complexity only to appear more professional.
- 任务按顺序完成（1→2→3→4），当前只做任务1。

## Scope Expansion Rule

Agents must not expand scope during execution.
If completion requires expansion: stop → explain → identify risks → propose minimal expansion → wait for confirmation.
