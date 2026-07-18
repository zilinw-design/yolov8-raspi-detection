# YOLOv8 实时目标检测学习项目

使用 YOLOv8 预训练模型 + OpenCV 实现实时目标检测系统。

## 技术栈

- **核心框架**: YOLOv8 (Ultralytics >= 8.0)
- **模型**: YOLOv8n / YOLOv8s（轻量级预训练模型）
- **图像处理**: OpenCV >= 4.6.0
- **深度学习**: PyTorch >= 1.8.0
- **辅助**: NumPy, Matplotlib

## 学习任务

| 任务 | 内容 | 预计产出 |
|---|---|---|
| 任务1 基础入门 | YOLOv8n 预训练模型实时检测，画框+标签+置信度 | `src/task1_basic/` |
| 任务2 进阶练习 | 类别过滤、样式自定义、n/s/m 模型对比 | `src/task2_advanced/` |
| 任务3 应用强化 | 自定义检测（颜色/标志）+ 目标计数统计 | `src/task3_custom/` |
| 任务4 性能优化 | 树莓派5 部署优化，ONNX 导出，目标 8-15fps | `src/task4_optimize/` |

## 环境要求

- Python >= 3.8
- `pip install ultralytics opencv-python numpy matplotlib`
- (任务4) `pip install onnx onnxruntime`

## 快速开始

```bash
# 安装依赖
pip install ultralytics opencv-python numpy matplotlib

# 运行任务1
python src/task1_basic/detect_camera.py
```

## 项目结构

```
yolo_learning_project/
  README.md
  AGENTS.md
  src/              # 源代码（按任务分目录）
  ai/governance/    # 项目治理文件
  ai/state/         # 当前状态
  outputs/          # AI 生成草稿
  references/       # 外部参考资料
  models/           # 预训练模型文件（.pt, .onnx）
```
