# 电赛 C 题 — 基本图形识别：技术栈与操作流程

> 2026-07-22 | Phase 3 完整文档

---

## 一、整体架构

```
摄像头 (4K USB Camera, 1920×1080, MJPG)
        │
        ▼
┌───────────────────────────────────────────┐
│              screen_measure.py            │
│                                           │
│  1. CSICamera → capture BGR frame         │
│  2. detect_shape() → 文档扫描管线         │
│  3. classify_contour() → 独占通道分类     │
│  4. MJPEGHandler → HTTP 视频流            │
│  5. /results endpoint → JSON 检测结果     │
└───────────────────────────────────────────┘
        │                    │
        ▼                    ▼
   浏览器:8081          浏览器:8081/results
   MJPEG 实时画面         JS 轮询 → 右侧面板
```

## 二、技术栈

| 组件 | 版本 | 用途 |
|---|---|---|
| Python | 3.11+ | 主语言 |
| OpenCV (`cv2`) | 4.10.0 | 图像处理全管线 |
| NumPy | 2.2+ | 数组运算 |
| PIL/Pillow | 12.2+ | JPEG 编码（MJPEG 流） |
| Tesseract OCR | 5.5.0 | 数字识别（Step 7） |
| pytesseract | 0.3.13 | Tesseract Python 绑定 |
| picamera2 | 0.3.36 | 树莓派 CSI 摄像头（备用） |
| 4K USB Camera | USB UVC | 主摄像头 (1920×1080, MJPG) |

## 三、检测管线（detect_shape）

### 3.1 完整流程

```
输入帧 (1920×1080 BGR)
    │
    ▼
[1] 缩放至最大 1200px (统一处理不同分辨率)
    │
    ▼
[2] CLAHE 增强 (LAB 色彩空间, L 通道)
    │  clipLimit=2.0, tileGridSize=8×8
    │  作用：暗光下增强 A4 纸边缘对比度
    │
    ▼
[3] find_document_region() → 找到 A4 纸四边框
    │
    ├── OTSU 二值化 (THRESH_BINARY_INV + THRESH_OTSU)
    │   将黑边框变成白色前景，白色纸面变成黑色背景
    │
    ├── findContours(RETR_TREE) → 取前 15 大轮廓
    │
    ├── approxPolyDP(epsilon=0.02*perimeter) → 筛选四边形
    │
    ├── 宽高比验证 (0.9 < ratio < 2.5)
    │   覆盖 A4(1.414) 和屏幕(1.78)
    │
    ├── 角点内缩黑边框验证
    │   4 角点向重心缩 5% 边长 → 采样 → 必须全黑
    │
    └── 边中点内缩验证
        4 边中点向重心缩 3% → 采样 → 必须全黑
    │
    ▼
[4] four_point_transform() → 透视矫正为正视图
    │  A4 纸在画面中是梯形 → 矫正为矩形
    │
    ▼
[5] 裁剪边框区 35%
    只保留纸面中心区域，排除黑边框干扰
    │
    ▼
[6] OTSU 二值化 + findContours → 找到图形轮廓
    │
    ▼
[7] classify_contour() → 独占通道分类
    │
    ├── 圆: circularity > 0.82
    ├── 三角: len(approx) == 3
    └── 正方: aspect > 0.80 && rect_ratio > 0.80
    │
    ▼
输出: [(type, size_px, digit?), ...]
```

### 3.2 关键参数

| 参数 | 值 | 位置 | 作用 |
|---|---|---|---|
| `MIN_SHAPE_AREA` | 15000 | test_detect.py:33 | 过滤 <122px 的碎片 |
| 缩放上限 | 1200px | detect_shape() | 统一不同分辨率 |
| CLAHE clipLimit | 2.0 | detect_shape() | 对比度增强上限 |
| 四边形宽高比 | 0.9-2.5 | find_document_region() | A4/屏幕宽容度 |
| 角点内缩 | 5% 边长, 上限20px | find_document_region() | 黑边框验证 |
| 边框裁剪 | 35% | detect_shape() | 排除黑边框干扰 |
| 圆 circularity | >0.82 | classify_contour() | 正方理论上限0.785 |
| 正方 aspect | >0.80 | classify_contour() | 宽高接近1 |
| 正方 rect_ratio | >0.80 | classify_contour() | 面积填满外接矩形 |

### 3.3 独占通道分类原理

```
任何轮廓 → 凸包平滑
    │
    ├── circularity > 0.82  →  circle  (正方最大 0.785, 永不跨过)
    │
    ├── len(approx) == 3    →  triangle (凸包不改变拓扑)
    │
    └── aspect>0.80 && rect>0.80 → square (圆和三角已被拦截)
             或 len(approx) <= 5  → square (凸四边形安全网)
```

**核心**: 每个图形只依赖一个凸包不变量，不对其他特征设门槛。

## 四、数字识别（Tesseract OCR）

```
正方形轮廓 → extract_digit_from_square()
    │
    ├── 取正方形 boundingRect
    ├── 裁掉 6% 边框
    ├── 找黑色像素 (binary < 100)
    ├── 裁剪数字区域 → 反色为白字黑底
    │
    ▼
recognize_digit() → Tesseract --psm 10
    │  psm 10 = 单字符模式
    │  whitelist = 0123456789
    │
    ▼
返回 (digit, confidence)
```

## 五、网页仪表板

```
MJPEGHandler:
  GET /              → dashboard.html (深色仪表板)
  GET /stream        → MJPEG 视频流
  GET /results       → JSON 检测结果 {shapes:[], minSq, msg}
  GET /test-images/* → 静态测试图

dashboard.html:
  ├── 左 2/3: 测试图下拉选择 + 大图预览 (白底模拟 A4 纸)
  ├── 右 1/3: 实时摄像头 MJPEG 流
  └── JS 轮询 /results, 每秒更新右侧结果面板
```

## 六、操作流程

### 6.1 树莓派部署

```bash
# 1. 安装系统依赖
sudo apt install tesseract-ocr -y

# 2. 克隆项目
cd ~/projects
git clone https://github.com/zilinw-design/yolov8-raspi-detection.git
cd yolov8-raspi-detection

# 3. 创建虚拟环境
python3 -m venv --system-site-packages venv
source venv/bin/activate

# 4. 安装 Python 依赖
pip install opencv-python numpy pillow pytesseract

# 5. 运行
python src/phase3/screen_measure.py
```

### 6.2 Windows 离线测试

```powershell
cd D:\Pi\yolo-raspi-detection

# 单张测试
python src/phase3/test_detect.py test_images/target_001.png

# 批量测试
python src/phase3/test_detect.py --dir D:/Pi/sy2/ai_harness_framework/exam/dataset_nuedc_2025_v2/1_single/

# 全量统计
python tests/stats.py          # 分类正确率
python tests/stats_count.py    # 数量匹配率
python tests/verify_min_square.py  # 最小正方形正确率
```

### 6.3 生成测试数据集

```powershell
cd D:\Pi\sy2\ai_harness_framework\exam
python text_shapes.py
# 输出: dataset_nuedc_2025_v2/
#   1_single/      基本图形(圆/三角/正方) + 单个正方形
#   2_multi/       彼此分离的正方形组合
#   3_numbered/    带白色数字编号的正方形
#   4_overlapping/ 局部重叠正方形
#   5_tilted/      倾斜30-60°透视模拟
```

## 七、测试结果（200张，50/类）

| 类别 | A4检测率 | 正方正确率 | 最小正方形 |
|---|---|---|---|
| 1_single | 100% | 100% | — |
| 2_multi | 100% | 95.5% | 94% |
| 4_overlapping | 100% | 99% | 96% |

## 八、未完成项

| 项目 | 状态 | 原因 |
|---|---|---|
| PnP 测距 (D) | ❌ | 需棋盘格标定 + 物理器材 |
| 尺寸测量 (x) | ❌ | 需像素→物理校准 |
| 倾斜30-60° | ⚠️ | 算法支持，缺实物验证 |
| 数字识别准确率 | ⚠️ | Tesseract 接入，待实物测试 |
