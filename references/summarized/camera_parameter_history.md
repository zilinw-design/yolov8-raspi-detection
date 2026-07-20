# 4K USB Camera 参数调优记录

> 摄像头：4K USB Camera (32e4:0415)，设备节点 /dev/videoX（自动发现）
> 参考手册：D:\Pi\sy2\ai_harness_framework\references\raw\摄像头模块.md
> 测试环境：树莓派5，1920×1080 MJPG

## 参数完整列表（数据手册提取）

### User Controls

| 参数 | 范围 | 默认 | 说明 |
|---|---|---|---|
| brightness | -64~64 | 0 | 整体亮度 |
| contrast | 0~95 | 0 | 对比度 |
| saturation | 0~255 | 66 | 饱和度 |
| sharpness | 0~7 | 0 | 锐化强度 |
| hue | -2000~2000 | 0 | 色调 |
| gamma | 64~300 | 100 | Gamma校正 |
| white_balance_automatic | bool | 1 | 自动白平衡 |
| white_balance_temperature | 2800~6500 | 4600 | 手动色温(自动关闭时生效) |
| backlight_compensation | 36~160 | 84 | 背光补偿(高=更强补偿) |
| power_line_frequency | 0~2 | 1(50Hz) | 电源频率防频闪 |

### Camera Controls

| 参数 | 范围 | 默认 | 说明 |
|---|---|---|---|
| auto_exposure | 0~3 | 3 | 0=全自动 1=全手动 2=快门优先 3=光圈优先 |
| exposure_time_absolute | 1~8188 | 156 | 曝光时间(×100µs)，自动模式时忽略 |
| focus_automatic_continuous | bool | 0 | 连续自动对焦 |
| focus_absolute | 0~1023 | 100 | 手动对焦值 |
| zoom_absolute | 0~60 | 0 | 数字变焦 |

---

## 调优历程

### 第1版（手册推荐值，camera_init.sh）
```
brightness=20, contrast=50, saturation=80, sharpness=5
focus_automatic_continuous=1
```
**效果**：画面高对比度，亮部过曝严重，暗部死黑，类似二值化效果。
**原因**：`backlight_compensation=84`（默认值偏高）在白色背景场景下过度补偿。

### 第2版（降低背光补偿）
```
brightness=20, contrast=50, saturation=80, sharpness=5
backlight_compensation=36, white_balance_automatic=1
```
**效果**：过曝略有好转，但对比度仍然偏高。

### 第3版（大幅降低对比度和锐化）
```
brightness=0, contrast=20, saturation=80, sharpness=2
backlight_compensation=36, white_balance_automatic=1
```
**效果**：暗部细节恢复，但整体偏暗，画面曝光不足。

### 第4版（取中 + 手动白平衡）
```
brightness=10, contrast=35, saturation=80, sharpness=2
backlight_compensation=36, white_balance_automatic=0, white_balance_temperature=5000
```
**效果**：亮暗平衡尚可，但画面偏红，白平衡手动值不适用所有场景。

### 第5版（当前——回到简洁配置）
```
brightness=10, contrast=35, saturation=80, sharpness=2
backlight_compensation=36, white_balance_automatic=1
```
**理由**：自动白平衡虽然有偏差但适应性强；手动色温(5000K)只在特定灯光下正确。

---

## 核心认知

这组参数只是为了**让摄像头画面看起来正常**——与检测算法无关。环境光线变化时参数总会有偏差，靠微调参数"适配所有场景"不可能。

**检测算法的责任是：从任何合理画质的输入中正确找到目标图形。** 这通过算法设计（ROI 区域、贴边过滤等）而非相机参数来保证。
