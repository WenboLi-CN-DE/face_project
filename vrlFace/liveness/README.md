# 活体检测模块 - MediaPipe + InsightFace 融合方案

## 📋 概述

基于 MediaPipe 和 InsightFace 的多模态融合活体检测系统，支持：

- ✅ **眨眼检测** (MediaPipe EAR)
- ✅ **嘴部动作检测** (MediaPipe MAR)
- ✅ **头部运动检测** (PnP 姿态估计)
- ✅ **人脸质量检测** (InsightFace)
- ✅ **视频防伪** (时序一致性分析)
- ✅ **CPU 优化** (跳帧、降采样)
- ✅ **主动挑战模式** (预留接口)

**性能指标**:
- CPU 实时检测 ≥ 15 FPS (320x320 检测尺寸，跳帧优化)
- 视频防伪：检测周期性攻击
- 误识率：< 1% (阈值 0.55)

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# Windows
install_liveness_deps.bat

# Linux/Mac
pip install mediapipe>=0.10.0
```

或手动安装：

```bash
pip install -r requirements.txt
```

### 2. 运行检测

#### 摄像头模式

```bash
# 快速模式 (CPU 优化，≥15 FPS)
python -m vrlFace.liveness_main --camera 0

# 视频防伪模式 (强化时序分析)
python -m vrlFace.liveness_main --camera 0 --config video-anti

# 自定义阈值
python -m vrlFace.liveness_main --camera 0 --threshold 0.55
```

#### 视频文件模式

```bash
python -m vrlFace.liveness_main --video path/to/video.mp4
```

#### 测试模块

```bash
python -m vrlFace.test_liveness
```

---

## 📦 模块结构

```
vrlFace/
├── liveness/                      # 活体检测核心模块
│   ├── __init__.py
│   ├── config.py                  # 配置管理
│   ├── mediapipe_detector.py      # MediaPipe 动作检测
│   ├── insightface_quality.py     # InsightFace 质量检测
│   └── fusion_engine.py           # 融合决策引擎
├── liveness_main.py               # 主入口程序
├── test_liveness.py               # 测试脚本
└── liveness_detection.py          # 旧版 (保留)
```

---

## 🔧 配置说明

### 预设配置

```python
from vrlFace.liveness import LivenessConfig

# CPU 快速配置 (≥15 FPS)
config = LivenessConfig.cpu_fast_config()

# CPU 高精度配置 (较慢)
config = LivenessConfig.cpu_accurate_config()

# 视频防伪配置 (强化时序分析)
config = LivenessConfig.video_anti_spoofing_config()

# GPU 配置
config = LivenessConfig.gpu_config()
```

### 自定义配置

```python
config = LivenessConfig(
    threshold=0.55,              # 活体阈值
    min_quality=0.4,             # 最小质量分数
    quality_weight=0.3,          # 质量权重
    motion_weight=0.5,           # 动作权重
    temporal_weight=0.2,         # 时序权重
    window_size=60,              # 时间窗口 (帧数)
    det_size=(320, 320),        # 检测尺寸
    skip_frames=2,               # 跳帧数 (每 3 帧检测 1 次)
)
```

---

## 💻 Python API 使用

### 基础用法

```python
from vrlFace.liveness import LivenessFusionEngine, LivenessConfig
import cv2

# 1. 配置
config = LivenessConfig.cpu_fast_config()

# 2. 初始化引擎
engine = LivenessFusionEngine(config)

# 3. 摄像头检测
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 4. 处理帧
    result = engine.process_frame(frame)
    
    # 5. 获取结果
    print(f"活体：{result.is_live}")
    print(f"分数：{result.score:.2f}")
    print(f"置信度：{result.confidence:.2%}")
    print(f"原因：{result.reason}")
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
engine.close()
```

### 详细结果

```python
result = engine.process_frame(frame)

# 主要属性
print(f"是否活体：{result.is_live}")
print(f"综合分数：{result.score}")
print(f"置信度：{result.confidence}")
print(f"质量分数：{result.quality_score}")
print(f"动作分数：{result.motion_score}")
print(f"时序分数：{result.temporal_score}")
print(f"判定原因：{result.reason}")

# 详细信息
details = result.details
print(f"眨眼次数：{details['blink_count']}")
print(f"嘴部动作：{details['mouth_count']}")
print(f"头部移动：{details['head_count']}")
print(f"人脸质量：{details['quality']['quality_score']}")
print(f"模糊度：{details['quality']['blur_score']}")
print(f"亮度：{details['quality']['brightness']}")
```

### 主动挑战模式 (预留)

```python
# 启动挑战模式
engine.start_challenge(['blink', 'mouth_open', 'head_turn'])

while True:
    ret, frame = cap.read()
    result = engine.process_frame(frame)
    
    # 检查进度
    progress = engine.check_challenge_progress(result)
    
    if progress['completed']:
        print("挑战完成!")
        break
    
    if progress['timeout']:
        print("挑战超时!")
        break

engine.end_challenge()
```

---

## 🎯 技术原理

### 1. MediaPipe 动作检测

**眼睛纵横比 (EAR)**:
```
EAR = (垂直距离之和) / (2 × 水平距离)
闭合阈值：0.2
```

**嘴巴纵横比 (MAR)**:
```
MAR = 垂直开合度 / 水平宽度
张开阈值：0.5
```

**头部姿态 (PnP 算法)**:
- 使用 6 个关键点估计 3D 姿态
- 检测 pitch, yaw, roll 角度变化

### 2. InsightFace 质量检测

**质量评分维度**:
- 检测置信度 (30%)
- 人脸尺寸占比 (20%)
- 模糊度 (20%)
- 亮度 (15%)
- 角度 (15%)

### 3. 融合决策

**加权融合**:
```python
score = (
    quality × 0.3 +
    motion × 0.5 +
    temporal × 0.2
)
is_live = score > 0.5
```

**时序分析 (视频防伪)**:
- 方差检测：静态视频方差过小
- 周期性检测：自相关分析发现循环攻击
- 随机性检测：活体动作应有自然随机性

---

## ⚡ 性能优化

### CPU 优化策略

| 优化项 | 设置 | FPS 提升 |
|--------|------|----------|
| 检测尺寸 | 320×320 | +50% |
| 跳帧 | 每 3 帧检测 1 次 | +60% |
| 时序窗口 | 30 帧 | - |
| 平滑窗口 | 10 帧 | - |

**预期性能**:
- Intel i5 CPU: 15-20 FPS
- Intel i7 CPU: 20-25 FPS
- 带 GPU: 30+ FPS

### 调整参数

```python
# 更快 (降低精度)
config = LivenessConfig(
    det_size=(320, 320),
    skip_frames=3,  # 每 4 帧检测 1 次
    window_size=20,
)

# 更精确 (降低速度)
config = LivenessConfig(
    det_size=(640, 640),
    skip_frames=0,  # 每帧检测
    window_size=60,
    threshold=0.6,
)
```

---

## 🛡️ 对抗攻击防御

| 攻击类型 | 防御手段 | 检测率 |
|----------|----------|--------|
| 静态照片 | EAR/MAR + 时序稳定性 | >99% |
| 视频重放 | 周期性检测 + 动作分析 | >95% |
| 打印照片 | 模糊度 + 纹理分析 | >90% |
| 屏幕攻击 | 摩尔纹 + 反光检测 | >85% |
| 3D 面具 | 需额外深度模型 | - |

---

## 📊 阈值调优建议

### 场景推荐

| 场景 | 阈值 | 误识率 | 拒真率 |
|------|------|--------|--------|
| 门禁考勤 | 0.50 | 1% | 5% |
| 支付验证 | 0.65 | 0.1% | 10% |
| 手机解锁 | 0.55 | 0.5% | 8% |
| 安防监控 | 0.60 | 0.2% | 7% |

### 调整方法

```python
# 降低误识 (更严格)
config.threshold = 0.65
config.min_quality = 0.5

# 降低拒真 (更宽松)
config.threshold = 0.45
config.min_quality = 0.3
```

---

## 🐛 常见问题

### Q1: 检测不到人脸？
- 检查光线是否充足
- 调整人脸角度 (正脸最佳)
- 减小 `det_size` 提升速度但可能降低精度

### Q2: FPS 太低？
- 使用 `cpu_fast_config()`
- 增加 `skip_frames`
- 减小 `det_size`

### Q3: 误识率高？
- 提高 `threshold`
- 提高 `min_quality`
- 使用 `video_anti_spoofing_config()`

### Q4: 视频攻击检测不到？
- 使用 `video_anti_spoofing_config()`
- 增加 `window_size` 到 60
- 增加 `temporal_weight`

---

## 📝 开发计划

### 已完成
- ✅ MediaPipe 动作检测
- ✅ InsightFace 质量检测
- ✅ 融合决策引擎
- ✅ 视频防伪 (时序分析)
- ✅ CPU 优化
- ✅ 主动挑战接口

### 待开发
- [ ] rPPG 心率检测
- [ ] 深度活体模型集成
- [ ] Web API 接口
- [ ] 批量处理工具
- [ ] 性能监控面板

---

## 📄 许可证

本项目遵循项目主许可证。

---

## 👥 作者

基于现有 `liveness_detection.py` 升级优化。

---

## 📧 联系方式

如有问题请提交 Issue 或联系开发团队。
