# 树莓派5 项目文件布局规划

> 日期：2026-07-19
> 设备：Raspberry Pi 5, welsey@10.161.238.102
> 原则：每个项目独立目录，各自 venv，互不干扰。历史项目归档不删除。

---

## 一、目标结构

```
/home/welsey/
  projects/                          ← 所有项目统一放这里
    raspi-vision/                    ← sy1 项目（已完成，归档保留）
    yolov8-raspi-detection/          ← 当前项目（活跃开发中）
      src/                           # 源代码
      models/                        # 预训练模型（.pt, .onnx，不入 git）
      venv/                          # 项目独立虚拟环境
      outputs/                       # 运行时生成输出
      ...
  references/                        ← （可选）共享参考资料
  tmp/                               ← pip 安装临时目录
```

---

## 二、各目录说明

### 2.1 `~/projects/raspi-vision/` — sy1 项目

| 属性 | 值 |
|---|---|
| 状态 | **归档保留，不删除，不修改** |
| 内容 | Flask 应用 + QR 码生成 + 摄像头测试 + OpenCV 参考 |
| 来源 | 从 `~/ai_harness_framework/` 移动而来 |

**操作步骤**（在树莓派上执行一次）：

```bash
# 确认 sy1 没有进程在跑
ps aux | grep python

# 移到 projects 下统一管理
mv ~/ai_harness_framework ~/projects/raspi-vision

# 如果 sy1 有自己的 venv，保留不动
ls ~/projects/raspi-vision/venv 2>/dev/null && echo "venv exists" || echo "no venv"
```

### 2.2 `~/projects/yolov8-raspi-detection/` — 当前项目

| 属性 | 值 |
|---|---|
| 状态 | **活跃开发中** |
| 虚拟环境 | `venv/`（已创建，`--system-site-packages`） |
| 模型文件 | `yolov8n.pt` 等由 Ultralytics 自动下载到项目根目录 |
| 同步方式 | `git pull origin main`（从 GitHub 拉取最新代码） |

**目录结构**（以 GitHub 仓库为准）：

```
yolov8-raspi-detection/
  src/task1_basic/
    detect_pi.py          ← Phase 1+2 实时检测 + MJPEG 流
    detect_camera.py      ← Windows 版（本地开发用，不在 Pi 上跑）
  src/task2_advanced/
    benchmark.py          ← Phase 2b 性能对比（待创建）
  tests/
  docs/
  outputs/                ← 运行时生成（截图、CSV），不入 git
  models/                 ← 本地模型缓存，不入 git
  venv/                   ← 虚拟环境，不入 git
```

### 2.3 共享参考资料

树莓派上已有两份参考源码（在 sy1 的 `references/raw/` 下，约 850MB）：

| 内容 | 路径 | 处理 |
|---|---|---|
| OpenCV 源码 | `~/projects/raspi-vision/references/raw/opencv/` | 保留，不删除 |
| Ultralytics 源码 | `~/projects/raspi-vision/references/raw/ultralytics/` | 保留，不删除 |
| rpi-object-detection | `~/projects/raspi-vision/references/raw/rpi-object-detection/` | 保留，当前项目直接有用 |

> 不复制到新项目——占用 SD 卡空间且不需要两份。sy1 归档后路径固定，代码中的引用路径无需修改。

---

## 三、清理项

以下文件和目录是历史操作残留，建议清理：

| 文件/目录 | 原因 | 操作 |
|---|---|---|
| `~/venv/` | Phase 1 部署时在 home 下误建的 venv（已被删除？） | `rm -rf ~/venv` |
| `~/test.jpg` | 摄像头测试照片，不需要保留 | `rm ~/test.jpg` |
| `~/tmp/` | pip 安装临时目录，可保留也可清 | `rm -rf ~/tmp` 或保留 |
| `~/proxy-setup.sh` | 代理配置脚本，系统级工具，保留 | 不动 |
| `~/.pip/pip.conf` | 已清掉 proxy 配置，保留 piwheels 索引 | 不动（当前只有 `extra-index-url`） |

---

## 四、磁盘空间预估

| 内容 | 约占用 |
|---|---|
| sy1 项目（含 opencv + ultralytics 参考源码） | ~850 MB |
| 当前项目（代码 + venv + 模型） | ~2 GB（venv 含 torch 约 1.8GB） |
| 系统 + 桌面环境 | ~5 GB |
| **总计** | **约 8 GB / 29 GB（27% 使用率）** |
| **剩余** | **约 21 GB** |

> 空间充足，不需要删 sy1。两个项目和平共存。

---

## 五、日常开发工作流

```
Windows (D:\Pi\yolo-raspi-detection)
  │
  │  写代码 → git commit → git push
  │
  ▼
树莓派 (~/projects/yolov8-raspi-detection)
  │
  │  git pull → 运行测试 → 反馈结果
  │
  ▼
Windows
  │
  │  根据结果调整代码
  │
  └── 循环
```

**树莓派上只做两件事**：pull 代码、跑测试。不直接修改代码。

**Windows 上做所有开发**：写代码、commit、push。SCP 传单文件用于快速验证。

---

## 六、紧急恢复

如果树莓派 SD 卡损坏或重装系统：

```bash
# 1. 系统初始化后
sudo apt update && sudo apt upgrade -y

# 2. 确认摄像头
sudo raspi-config  # → Interface Options → Camera → Enable
rpicam-still -o test.jpg

# 3. 克隆项目
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/zilinw-design/yolov8-raspi-detection.git
cd yolov8-raspi-detection

# 4. 创建 venv 并安装依赖
python3 -m venv --system-site-packages venv
source venv/bin/activate
TMPDIR=~/tmp pip install ultralytics torch

# 5. 运行
python src/task1_basic/detect_pi.py
```
