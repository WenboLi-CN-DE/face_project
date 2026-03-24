# vrlFace.liveness — 活体检测模块

**位置**: `vrlFace/liveness/`  
**职责**: 基于 MediaPipe 的视频活体检测与动作验证

## 结构

```
liveness/
├── __init__.py            # 导出核心类
├── config.py              # LivenessConfig（预设配置）
├── fusion_engine.py       # 多信号融合引擎（11KB）
│
├── mediapipe_detector.py  # MediaPipe 检测器（32KB，794 行）
├── fast_detector.py       # 轻量状态机检测器（26KB）
├── video_analyzer.py      # 视频逐段分析器（24KB）
├── head_action.py         # 头部动作事件检测（3.1KB）
├── insightface_quality.py # 质量评估（8.6KB）
│
├── async_processor.py     # 异步任务处理器（5.2KB）
├── callback.py            # HTTP 回调客户端（3.8KB）
├── video_rotation.py      # 视频旋转处理（5.1KB）
├── recorder.py            # CSV 录制器（22KB）
├── schemas.py             # Pydantic 数据模型（3.3KB）
├── utils.py               # 工具函数（1.6KB）
├── cli.py                 # 命令行工具（21KB）
│
├── benchmark_*.py         # 基准测试（3 个文件）
└── create_test_video.py   # 测试视频生成
```

## 核心类

| 类 | 文件 | 职责 |
|------|------|------|
| `LivenessFusionEngine` | `fusion_engine.py` | 活体检测主入口，融合多信号 |
| `LivenessConfig` | `config.py` | 配置预设（CPU/实时/视频防伪） |
| `MediaPipeLivenessDetector` | `mediapipe_detector.py` | MediaPipe 478 关键点检测 |
| `FastLivenessDetector` | `fast_detector.py` | 轻量状态机检测器 |
| `VideoLivenessAnalyzer` | `video_analyzer.py` | 视频逐段分析器 |
| `HeadActionDetector` | `head_action.py` | 头部动作事件检测 |
| `BenchmarkCalibrator` | `benchmark_calibrator.py` | 基准帧校准器（防替换攻击） |

## 配置预设

```python
from vrlFace.liveness import LivenessConfig

LivenessConfig.cpu_fast_config()           # CPU 快速模式
LivenessConfig.realtime_config()           # 实时摄像头模式
LivenessConfig.video_anti_spoofing_config()# 视频防伪模式
LivenessConfig.video_anti_spoofing_with_silent_config()  # 视频防伪+静默检测
LivenessConfig.video_fast_config()         # 视频快速模式
```

## 阈值自动调整

**最近新增功能**: 支持根据场景自动调整检测阈值

```python
# config.py 中实现
# - 根据动作类型自动调整灵敏度
# - 根据环境光线调整检测阈值
# - 根据人脸质量动态调整阈值
```

## 阈值配置

### 默认值
```python
ear_threshold = 0.20       # 眨眼（EAR < 0.20 判定为闭眼）
mar_threshold = 0.55       # 张嘴（MAR > 0.55 判定为张嘴）
yaw_threshold = 15°        # 左右转头
pitch_threshold = 15°      # 上下点头
window_size = 30           # 滑动窗口大小（帧）
```

### 调整阈值
```python
config = LivenessConfig.realtime_config()
config.ear_threshold = 0.25  # 调整眨眼灵敏度
config.mar_threshold = 0.60  # 调整张嘴灵敏度
```

## 使用示例

### 同步检测（单帧）
```python
from vrlFace.liveness import LivenessFusionEngine, LivenessConfig

config = LivenessConfig.realtime_config()
engine = LivenessFusionEngine(config)
result = engine.process_frame(frame)
print(f"活体：{result.is_live}, 分数：{result.score}")
```

### 视频分析（异步回调）
```python
from vrlFace.liveness import VideoLivenessAnalyzer, LivenessConfig

config = LivenessConfig.video_anti_spoofing_config()
analyzer = VideoLivenessAnalyzer(config)
result = await analyzer.analyze(
    video_path="video.mp4",
    required_actions=["blink", "nod"],
    callback_url="http://localhost:8080/callback"
)
```

### 命令行
```bash
# 摄像头检测
uv run python -m vrlFace.liveness.cli --camera 0

# 视频检测
uv run python -m vrlFace.liveness.cli --video path/to/video.mp4

# 视频 CSV 录制
uv run python -m vrlFace.liveness.recorder --video video.mp4
```

## API 路由

| 端点 | 方法 | 说明 |
|------|------|------|
| `/vrlMoveLiveness` | POST | 视频活体检测（异步回调） |

## 动作类型

| 动作 | 触发条件 |
|------|----------|
| `blink` | EAR < ear_threshold 持续 3 帧 |
| `mouth` | MAR > mar_threshold 持续 3 帧 |
| `nod` | Pitch 峰峰值变化 > pitch_threshold（优化：基线追踪→峰峰值检测） |
| `shake` | Yaw 峰峰值变化 > yaw_threshold（优化：基线追踪→峰峰值检测） |
| `turn_left` | Yaw < -yaw_threshold |
| `turn_right` | Yaw > yaw_threshold |

**头部动作检测优化**: 从基线追踪改为峰峰值检测，提高动作识别准确性

## 技术细节

**MediaPipe**: FaceLandmarker 提取 478 个面部关键点  
**EAR 计算**: 眼睛纵横比（Eye Aspect Ratio）  
**MAR 计算**: 嘴巴纵横比（Mouth Aspect Ratio）  
**头部姿态**: 欧拉角（Pitch/Yaw/Roll）  
**时序平滑**: 滑动窗口（默认 30 帧）  
**旋转处理**: 自动检测视频元数据并修正  

## 异步回调架构

```
客户端 → FastAPI → AsyncProcessor → VideoAnalyzer → 后台处理
                                              ↓
客户端 ← HTTP POST 回调 ← CallbackClient ← 完成通知
```

**回调签名**: HMAC-SHA256 签名验证  
**重试机制**: 可配置重试次数和间隔  

## 基准测试

| 文件 | 用途 |
|------|------|
| `benchmark_calibrator.py` | 基准帧校准（防替换攻击）：动态采集高质量正面人脸帧，提取 embedding/landmarks |
| `benchmark_demo.py` | 演示校准流程：实时显示基准采集状态、相似度、活体结果 |
| `benchmark_test.py` | 校准测试用例：验证基准采集、同一人验证、不同人验证 |

**基准校准流程**:
1. 采集阶段（2 秒）→ 收集 3-10 帧高质量正面人脸
2. 计算基准特征 → embedding 平均值 + landmarks 平均值
3. 校准阶段 → 每帧与基准比对，相似度低于阈值判定为替换攻击

## 复杂度热点

| 文件 | 行数 | 说明 |
|------|------|------|
| `mediapipe_detector.py` | 794 | MediaPipe 完整推理，478 关键点处理 |
| `cli.py` | 649 | 命令行参数解析，多模式支持 |
| `recorder.py` | 648 | CSV 录制，实时数据输出 |
| `fast_detector.py` | 625 | 状态机检测逻辑 |
| `video_analyzer.py` | 623 | 视频逐帧分析，动作分段 |
| `benchmark_demo.py` | 546 | 基准测试演示 |

## 故障排查

**动作未检测到**:
- 降低阈值：`ear_threshold = 0.25` → `0.20`
- 增加窗口大小：`window_size = 30` → `50`
- 使用阈值自动调整功能（最近新增）

**视频旋转错误**:
- 检查 `video_rotation.py` 日志
- 手动指定旋转角度

**回调失败**:
- 验证 `callback_url` 可达性
- 检查 HMAC 签名密钥配置

**基准校准失败**:
- 确保采集阶段有足够高质量正面人脸帧（3-10 帧）
- 检查光线条件和人脸角度

## 测试覆盖

| 测试文件 | 覆盖内容 |
|----------|----------|
| `tests/liveness/test_liveness.py` | 配置预设、检测器初始化、融合引擎、时序一致性 |
| `tests/liveness/test_benchmark.py` | 基准帧采集、同一人验证、不同人验证、阈值校准 |
| `tests/liveness/test_async_processor.py` | 异步任务处理、回调数据构建、错误处理 |
| `tests/liveness/test_callback.py` | HMAC 签名生成、回调发送、重试机制 |
| `tests/liveness/test_liveness_api.py` | FastAPI 路由、请求验证、错误响应 |
| `tests/liveness/test_fix.py` | 阈值配置修复验证 |

**手动测试脚本** (`scripts/manual_tests/`):
- `batch_test_videos.py` — 批量视频动作测试
- `test_frame_allocation.py` — 帧分配修复测试
- `debug_is_liveness.py` — 活体判定调试（逐帧分析）
- `remote_fetch_tests/` — 远程拉取视频测试

---

## 最近变更

- **静默检测集成**: 支持 AI 生成视频和手机播放攻击检测
- **阈值自动调整**: 支持根据场景自动调整检测阈值
- **基准帧校准**: 防替换攻击，动态采集基准帧
- **头部动作优化**: 从基线追踪改为峰峰值检测
- **持久化模型**: InsightFace 模型持久化，避免重复下载
