# AGENTS.md

**项目轨道：Part B — 电赛单目视觉测量（进行中）**

YOLO 学习部分（Part A）已迁移至 `D:\Pi\sy3`。

## Required Reading

Before project work, read:

1. `ai/governance/File Registry.md`
2. `ai/governance/Project Brief.md`
3. `ai/governance/Security Boundary.md`

## 双轨路由

| 涉及内容 | 读什么 |
|---|---|
| 图形识别/测距/赛题 | `src/phase3/` |
| 项目治理/状态 | `ai/governance/` + `ai/state/` |
| 相机参数 | `references/summarized/camera_parameter_history.md` |
| 检测算法原理 | `docs/detection_algorithm_lecture.md` |
| YOLO 学习部分 | `D:\Pi\sy3`（独立项目，已归档） |

## Coding Pre-Execution Protocol

1. **方案说明** → 文献依据 → 预期效果 → 替代方案
2. **等待审批** → 用户确认后再写代码
3. **追加决策记录** → `ai/decisions/Project Decision Log.md`

## Key Directories

| 目录 | 内容 |
|---|---|
| `src/phase3/` | 图形检测 + 测距管线 |
| `tests/test_images/shape/` | 测试截图 |
| `tests/test_shapes.html` | 测试图形生成器 |
| `docs/` | 讲义 |
| `references/summarized/` | 相机参数 + 问题记录 |
