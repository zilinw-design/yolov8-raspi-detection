# 参考资料分析总结

> 来源：D:\Pi\sy2\ai_harness_framework\references\raw\
> 整理日期：2026-07-19

---

## 一、Qengineering/YoloV8-ncnn-Raspberry-Pi-4

### 概述
C++ 实现，使用腾讯 ncnn 推理框架，直接跑在裸树莓派 4/5 上（无加速卡）。

### 关键数据（RPi 5 @ 2.9GHz，纯推理时间，不含采集/绘图）

| 模型 | 输入尺寸 | RPi 5 FPS | 对比 RPi 4 |
|---|---|---|---|
| YOLOv8n | 640×640 | **20.0** | 3.1（6.5×提升） |
| YOLOv8s | 640×640 | **11.0** | 1.47（7.5×提升） |
| YOLOv5n | 640×640 | 13.6 | 1.6 |
| YOLOv6n | 640×640 | 15.8 | 2.7 |

### 对我们项目的启示
- **ncnn 比 PyTorch 快约 4×**：我们的 PyTorch 方案在 640 分辨率下预计只有 3-5 FPS
- **这就是为什么我们默认 `--imgsz 320`**：降低分辨率是弥补 PyTorch 开销的最直接方法
- Phase 4 如果考虑 ONNX/ncnn 导出，有望达到接近 20 FPS

---

## 二、ultralytics_yolo_comparison

### 概述
Python 工具，基于 Ultralytics 官方 API，支持 YOLOv5/v8/v11/v26 多模型对比。

### 架构设计（可借鉴）

```
yolo_benchmark.py          ← 核心类 YOLOBenchmark（模型加载、推理、指标收集）
yolo_benchmark_driver.py   ← CLI 入口 + benchmark_yolo() 便捷函数
```

### 关键设计模式

| 模式 | 实现 |
|---|---|
| **Warmup** | 跳过前若干帧后的正式计时 |
| **指标** | avg/min/max FPS, avg/min/max inference_ms, total_detections, avg_confidence |
| **输出** | JSON + 带指标叠加的 MP4 视频 + 日志 |
| **模式过滤** | mode=1 全部 / mode=2 车辆 / mode=3 行人 |
| **帧限制** | `--max-frames` 只跑前 N 帧用于快速测试 |

### 局限性
- 只支持 x 级大模型，不支持 n/s/m/l
- 依赖视频文件输入，不支持摄像头实时流

### 对我们项目的启示
- `benchmark.py` 可直接借鉴其指标收集方式：warmup + timed runs + 统计
- CLI 参数设计：`--versions` 的多模型概念可简化为 `--models yolov8n yolov8s yolov8m`
- JSON 输出格式值得参考

---

## 三、vehicle_detection_visualization

### 概述
D3.js 交互式可视化，对比 13 个 YOLOv8 模型 × 6 种分辨率 × 6 种设备 × 批次大小 + FP16/INT8 量化。

### 数据组织方式（JSON 嵌套结构）

```
设备 → 模型 → 输入尺寸 → 后端 → 量化 → 批次 → {mAP, FPS}
```

### 测试矩阵设计

| 维度 | 取值 |
|---|---|
| 模型 | YOLOv8n/s/m/l/x 的 6 种变体 |
| 分辨率 | 6 档（含 352×192 等非常见尺寸） |
| 设备 | Jetson AGX, Xavier NX, Nano, MX150, i7-9850H, **RPi 4B** |
| 批次 | 1, 2, 4, 8, 16, 32 |
| 量化 | FP32, FP16, INT8 |

### 设计经验

- **交互性不可替代**：不互动的静态表格无法支撑多维对比
- **两阶段过滤**：先按大类过滤（如只看大模型），再逐个选择
- **简单 vs 全面是根本矛盾**：可视化做得越好，就越需要取舍

### 对我们项目的启示
- 我们的 `benchmark.py` 维度少得多（模型 × 分辨率 × conf），不需要 D3 可视化，简洁的终端表格 + CSV 导出足够
- 但如果后续需要做详细的 Phase 4 性能报告，JSON 结构设计值得参考

---

## 四、MPQ-YOLO 论文

### 概述
混合精度量化：YOLOv5 骨干网络用 1-bit，检测头用 4-bit。

### 关键数据

| 指标 | 数值 |
|---|---|
| 计算量压缩 | **16.3×** |
| 模型大小压缩 | **14.2×** |
| VOC mAP | 74.7% |
| COCO mAP | 51.5% |

### 对我们项目的启示
- 这是学术前沿，短期不适用于我们的学习项目
- 但概念层面值得了解：**量化 = 用精度换速度**，不同位宽对不同模块的影响不同
- Phase 4 若考虑 INT8 量化，预期获得 20-40% 推理加速

---

## 五、综合结论：Phase 2b benchmark.py 设计建议

### 不要过度设计
参照资料中 vehicle_detection 的 6×6×13 矩阵是专业研究级，我们的学习项目不需要。

### 应该保留的核心设计

| 借鉴来源 | 借鉴内容 |
|---|---|
| Qengineering | RPi 5 基线数据：ncnn YOLOv8n@640 = 20 FPS，作为我们 PyTorch 方案的参照系 |
| yolo_comparison | warmup + timed runs 模式；JSON 输出格式；CLI 参数模式 |
| vehicle_detection | 分辨率作为关键变量；测试矩阵简化版（3 模型 × 4 分辨率 × 7 置信度 = 84 次推理） |

### 推荐的数据收集维度

```
3 模型 (n, s, m) × 4 分辨率 (640, 480, 320, 256) × conf 扫描 (单独测)
```

输出：终端表格 + CSV。不引入视频保存、不引入 Web 可视化。
