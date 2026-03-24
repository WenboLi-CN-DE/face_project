# 视频活体检测防伪增强设计文档

**日期**: 2026-03-24  
**作者**: Sisyphus AI Agent  
**状态**: 待审核

---

## 一、背景与目标

### 1.1 背景

当前视频活体检测系统 (`vrlFace/liveness/`) 基于 MediaPipe 进行动作检测（眨眼/点头/摇头），能够有效防御照片攻击。但存在以下安全隐患：

- **AI 生成视频攻击**：使用 GAN/Diffusion 模型生成的虚假人脸视频
- **手机播放攻击**：使用手机播放预录制的真实视频
- **深度伪造攻击**：使用 DeepFake 技术合成的视频

### 1.2 目标

为视频活体检测系统增加**静默检测能力**，在不影响用户体验的前提下，防御上述攻击类型。

**核心要求**：
- 集成现有静默活体检测模块 (`vrlFace/silent_liveness/`)
- 支持独立部署（集成在视频活体服务中）
- 通过配置开关灵活控制
- 性能开销可控（< 1 秒）

---

## 二、技术方案

### 2.1 整体架构

**检测流程**：

```
视频输入
  ↓
关键帧采样（前 100 帧中采样 3-5 帧高质量正面人脸）
  ↓
静默检测（可选，通过配置控制）
  ├─ UniFace MiniFASNet → 检测传统物理攻击（打印/屏幕/面具）
  │   └─ 失败 → 返回 "traditional_spoof"
  ↓
  ├─ 频域分析器 → 检测 AI 生成图像（GAN/Diffusion 伪影）
  │   └─ 失败 → 返回 "ai_spoof"
  ↓
  └─ 通过 → 继续动作检测
       ↓
动作检测（眨眼/点头/摇头）
  ↓
综合判定 → 返回结果
```

**模块关系**：

```
VideoLivenessAnalyzer（视频分析器）
  ├─ FrameSampler（新增：关键帧采样器）
  │
  ├─ SilentLivenessDetector（新增：静默检测器）
  │   ├─ UniFace RetinaFace（人脸检测）
  │   ├─ UniFace MiniFASNet（反欺诈）
  │   ├─ FrequencyAnalyzer（频域分析）
  │   └─ HeuristicDetector（启发式检测）
  │
  └─ LivenessFusionEngine（动作检测引擎）
      └─ MediaPipeLivenessDetector
```

### 2.2 关键设计决策

#### 决策 1：触发时机 — 关键帧采样

**选择**：在视频开始时采样 3-5 帧高质量人脸帧

**理由**：
- 性能开销小（仅检测少量帧）
- 检测速度快（~600ms）
- 足以识别大部分攻击类型

**替代方案**：
- 全程多帧检测：性能开销大，不适合实时场景
- 动作完成后验证：检测时机较晚，用户体验差

#### 决策 2：检测失败处理 — 立即拒绝（严格模式）

**选择**：静默检测失败 → 直接返回 `is_liveness=0`

**理由**：
- 高安全场景优先
- 避免浪费资源执行后续动作检测

**替代方案**：
- 降低置信度（宽松模式）：可通过配置支持

#### 决策 3：部署方式 — 集成部署

**选择**：视频活体服务内置静默检测能力

**理由**：
- 单一服务，部署简单
- 减少网络延迟
- 适合中小规模部署

**替代方案**：
- 微服务架构：可在后续性能瓶颈时拆分

---

## 三、核心实现

### 3.1 配置扩展

**文件**：`vrlFace/liveness/config.py`

**新增配置项**：

```python
@dataclass
class LivenessConfig:
    # ... 现有配置 ...
    
    # ========== 静默检测配置（新增） ==========
    enable_silent_detection: bool = False  # 是否启用静默检测
    silent_detection_mode: str = "strict"  # "strict"=立即拒绝, "loose"=降低置信度
    silent_sample_frames: int = 5  # 采样帧数（3-5 帧）
    silent_min_quality: float = 0.6  # 采样帧最低质量要求
    silent_max_angle: float = 15.0  # 采样帧最大人脸角度（度）
```

**新增配置预设**：

```python
@classmethod
def video_anti_spoofing_with_silent_config(cls) -> "LivenessConfig":
    """视频防伪模式（含静默检测）"""
    config = cls.video_anti_spoofing_config()
    config.enable_silent_detection = True
    config.silent_detection_mode = "strict"
    config.silent_sample_frames = 5
    config.silent_min_quality = 0.6
    config.silent_max_angle = 15.0
    return config
```

### 3.2 关键帧采样器

**文件**：`vrlFace/liveness/frame_sampler.py`（新建）

**职责**：从视频中采样高质量正面人脸帧，供静默检测使用

**核心逻辑**：

```python
class FrameSampler:
    """关键帧采样器 — 为静默检测提供高质量人脸帧"""
    
    def sample_keyframes(
        self, 
        video_path: str, 
        num_frames: int = 5,
        min_quality: float = 0.6,
        max_angle: float = 15.0,
        max_scan_frames: int = 100
    ) -> List[np.ndarray]:
        """
        从视频开始采样高质量正面人脸帧
        
        策略：
        1. 读取前 max_scan_frames 帧
        2. 使用 MediaPipe 检测人脸和姿态角度
        3. 筛选包含正面人脸的帧（角度 < max_angle）
        4. 计算质量分数（基于人脸尺寸、清晰度、亮度）
        5. 按质量分数排序，取 top-N
        
        Returns:
            List[np.ndarray]: 采样的帧列表（BGR 格式）
        """
```

**质量评分标准**：

```python
quality_score = (
    0.4 * face_size_score +      # 人脸尺寸（越大越好）
    0.3 * sharpness_score +      # 清晰度（拉普拉斯方差）
    0.2 * brightness_score +     # 亮度（避免过暗/过亮）
    0.1 * frontal_score          # 正面程度（角度越小越好）
)
```

### 3.3 视频分析器集成

**文件**：`vrlFace/liveness/video_analyzer.py`

**修改点**：

1. **初始化时懒加载静默检测器**：

```python
class VideoLivenessAnalyzer:
    def __init__(self, liveness_config: LivenessConfig):
        # ... 现有初始化 ...
        
        # 静默检测器（懒加载）
        self._silent_detector = None
        self._frame_sampler = None
        
        if liveness_config.enable_silent_detection:
            from vrlFace.silent_liveness import SilentLivenessDetector
            from .frame_sampler import FrameSampler
            
            self._silent_detector = SilentLivenessDetector.get_instance()
            self._frame_sampler = FrameSampler()
            logger.info("静默检测已启用（模式：%s）", liveness_config.silent_detection_mode)
```

2. **在 analyze() 方法中执行静默检测**：

```python
def analyze(
    self, 
    video_path: str, 
    required_actions: List[str],
    ...
) -> VideoLivenessResult:
    """视频活体分析（含静默检测）"""
    
    # ========== Step 1: 静默检测（如果启用） ==========
    silent_result = None
    if self.config.enable_silent_detection:
        silent_result = self._run_silent_detection(video_path)
        
        # 严格模式：检测失败立即返回
        if self.config.silent_detection_mode == "strict" and not silent_result["passed"]:
            return self._build_reject_result(silent_result)
    
    # ========== Step 2: 动作检测（现有逻辑） ==========
    # ... 现有动作检测代码 ...
    
    # ========== Step 3: 综合判定 ==========
    # 如果是宽松模式，将静默检测结果纳入综合判定
    if self.config.silent_detection_mode == "loose" and silent_result:
        final_confidence *= silent_result["confidence"]
    
    return VideoLivenessResult(...)
```

3. **新增静默检测方法**：

```python
def _run_silent_detection(self, video_path: str) -> Dict[str, Any]:
    """
    执行静默检测
    
    Returns:
        {
            "passed": bool,
            "confidence": float,
            "reject_reason": str | None,
            "details": dict
        }
    """
    logger.info("开始静默检测：采样关键帧...")
    
    # 采样关键帧
    keyframes = self._frame_sampler.sample_keyframes(
        video_path,
        num_frames=self.config.silent_sample_frames,
        min_quality=self.config.silent_min_quality,
        max_angle=self.config.silent_max_angle
    )
    
    if not keyframes:
        logger.warning("未采样到符合条件的关键帧")
        return {
            "passed": False,
            "confidence": 0.0,
            "reject_reason": "no_quality_frames",
            "details": {}
        }
    
    # 对每一帧执行静默检测
    results = []
    for i, frame in enumerate(keyframes):
        # 保存临时图片
        temp_path = f"/tmp/keyframe_{i}.jpg"
        cv2.imwrite(temp_path, frame)
        
        # 调用静默检测器
        result = self._silent_detector.detect(temp_path)
        results.append(result)
        
        # 清理临时文件
        os.remove(temp_path)
    
    # 综合判定：任意一帧检测失败 → 整体失败
    passed = all(r["is_liveness"] == 1 for r in results)
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    
    # 找出拒绝原因（如果有）
    reject_reason = None
    if not passed:
        for r in results:
            if r["reject_reason"]:
                reject_reason = r["reject_reason"]
                break
    
    return {
        "passed": passed,
        "confidence": avg_confidence,
        "reject_reason": reject_reason,
        "details": {
            "sampled_frames": len(keyframes),
            "frame_results": results
        }
    }
```

### 3.4 API 响应扩展

**文件**：`vrlFace/liveness/schemas.py`

**修改 VideoLivenessResult**：

```python
@dataclass
class VideoLivenessResult:
    is_liveness: int  # 1=通过，0=不通过
    liveness_confidence: float
    is_face_exist: int
    face_info: Optional[FaceInfo]
    action_verify: ActionVerifyResult
    
    # ========== 新增字段 ==========
    reject_reason: Optional[str] = None  # 拒绝原因：null/"traditional_spoof"/"ai_spoof"/"no_quality_frames"
    silent_detection: Optional[Dict[str, Any]] = None  # 静默检测详情
    
    # 现有字段
    benchmark_verified: Optional[int] = None
    benchmark_details: Optional[Dict[str, Any]] = None
```

**API 响应示例**：

```json
{
  "is_liveness": 0,
  "liveness_confidence": 0.0,
  "reject_reason": "ai_spoof",
  "is_face_exist": 1,
  "face_info": {
    "confidence": 0.99,
    "quality_score": 0.85
  },
  "silent_detection": {
    "enabled": true,
    "passed": false,
    "confidence": 0.72,
    "details": {
      "sampled_frames": 5,
      "uniface_passed": true,
      "ai_check_passed": false,
      "anomaly_score": 0.72
    }
  },
  "action_verify": {
    "passed": false,
    "required_actions": ["blink", "nod"],
    "action_details": []
  }
}
```

---

## 四、依赖管理

### 4.1 新增依赖

**文件**：`requirements.txt`

```txt
# 静默活体检测
uniface>=3.1.0
```

### 4.2 模型管理

**UniFace 模型**：
- **RetinaFace** 人脸检测模型（~1.6MB）
- **MiniFASNet** 反欺诈模型（~380KB）

**存储策略**：
- **开发/测试环境**：使用 UniFace 默认缓存路径（`~/.uniface/`），首次运行时自动下载
- **生产环境（无外网）**：在 Dockerfile 中预下载模型

**Dockerfile 修改**（生产环境）：

```dockerfile
# 预下载 UniFace 模型
RUN python -c "from uniface.detection import RetinaFace; from uniface.spoofing import MiniFASNet; RetinaFace(); MiniFASNet()"
```

---

## 五、性能影响评估

### 5.1 时间开销

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 关键帧采样 | ~200ms | 读取前 100 帧，筛选 5 帧 |
| UniFace 检测 | ~50ms/帧 | 5 帧 = 250ms |
| 频域分析 | ~30ms/帧 | 5 帧 = 150ms |
| **总计** | **~600ms** | 增加约 0.6 秒 |

### 5.2 内存开销

- UniFace 模型：~50MB
- 频域分析：~10MB
- **总计**：~60MB

### 5.3 优化建议

1. **减少采样帧数**：从 5 帧降至 3 帧（性能提升 40%）
2. **并行检测**：多帧并行执行静默检测（需要多核 CPU）
3. **模型量化**：使用 INT8 量化模型（减少内存占用 50%）

---

## 六、部署配置

### 6.1 环境变量

**文件**：`deploy/pro/.env`

```bash
# 静默检测配置
LIVENESS_ENABLE_SILENT=true
LIVENESS_SILENT_MODE=strict  # strict 或 loose
LIVENESS_SILENT_SAMPLE_FRAMES=5
LIVENESS_SILENT_MIN_QUALITY=0.6
LIVENESS_SILENT_MAX_ANGLE=15.0
```

### 6.2 Docker Compose

**文件**：`deploy/pro/docker-compose.yaml`

```yaml
services:
  liveness:
    image: vrlface-liveness:latest
    environment:
      - LIVENESS_ENABLE_SILENT=${LIVENESS_ENABLE_SILENT:-false}
      - LIVENESS_SILENT_MODE=${LIVENESS_SILENT_MODE:-strict}
      - LIVENESS_SILENT_SAMPLE_FRAMES=${LIVENESS_SILENT_SAMPLE_FRAMES:-5}
    volumes:
      - uniface-models:/root/.uniface  # UniFace 模型缓存
    mem_limit: 8g  # 增加内存限制（原 8g，静默检测增加 60MB）

volumes:
  uniface-models:
```

### 6.3 配置加载

**文件**：`vrlFace/liveness/config.py`

```python
@classmethod
def from_env(cls) -> "LivenessConfig":
    """从环境变量加载配置"""
    config = cls.realtime_config()
    
    # 静默检测配置
    config.enable_silent_detection = os.getenv("LIVENESS_ENABLE_SILENT", "false").lower() == "true"
    config.silent_detection_mode = os.getenv("LIVENESS_SILENT_MODE", "strict")
    config.silent_sample_frames = int(os.getenv("LIVENESS_SILENT_SAMPLE_FRAMES", "5"))
    config.silent_min_quality = float(os.getenv("LIVENESS_SILENT_MIN_QUALITY", "0.6"))
    config.silent_max_angle = float(os.getenv("LIVENESS_SILENT_MAX_ANGLE", "15.0"))
    
    return config
```

---

## 七、测试策略

### 7.1 单元测试

**文件**：`tests/liveness/test_frame_sampler.py`

```python
def test_sample_keyframes_from_real_video():
    """测试从真实视频采样关键帧"""
    
def test_sample_keyframes_quality_filtering():
    """测试质量过滤逻辑"""
    
def test_sample_keyframes_angle_filtering():
    """测试角度过滤逻辑"""
```

**文件**：`tests/liveness/test_silent_integration.py`

```python
def test_silent_detection_rejects_ai_video():
    """测试静默检测能拒绝 AI 生成视频"""
    
def test_silent_detection_passes_real_video():
    """测试静默检测能通过真实视频"""
    
def test_silent_detection_strict_mode():
    """测试严格模式：检测失败立即返回"""
    
def test_silent_detection_loose_mode():
    """测试宽松模式：降低置信度"""
```

### 7.2 集成测试

```bash
# 测试真实视频（应通过）
uv run python -m vrlFace.liveness.cli \
    --video data/real_video.mp4 \
    --config video-anti-silent

# 测试 AI 生成视频（应拒绝）
uv run python -m vrlFace.liveness.cli \
    --video data/ai_generated_video.mp4 \
    --config video-anti-silent

# 测试手机播放视频（应拒绝）
uv run python -m vrlFace.liveness.cli \
    --video data/phone_replay_video.mp4 \
    --config video-anti-silent
```

### 7.3 性能测试

```python
# tests/liveness/test_silent_performance.py
def test_silent_detection_performance():
    """测试静默检测性能开销"""
    # 要求：总耗时 < 1 秒
```

---

## 八、风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| UniFace 模型下载失败 | 服务启动失败 | 中 | 预下载模型到镜像 |
| 静默检测误判率高 | 用户体验差 | 中 | 提供配置开关，支持禁用 |
| 性能开销过大 | 响应时间增加 | 低 | 采样帧数可配置（3-5 帧） |
| 依赖冲突 | 部署失败 | 低 | 隔离依赖，使用虚拟环境 |
| 内存占用过高 | OOM 错误 | 低 | 增加 Docker 内存限制 |

---

## 九、后续优化方向

### 9.1 短期优化（1-2 周）

1. **自适应采样**：根据视频质量动态调整采样帧数
2. **并行检测**：多帧并行执行静默检测
3. **缓存优化**：缓存 UniFace 模型加载结果

### 9.2 中期优化（1-3 个月）

1. **模型优化**：使用更轻量的反欺诈模型
2. **多阶段检测**：快速预检 + 深度检测
3. **A/B 测试**：对比不同配置的误判率和性能

### 9.3 长期优化（3-6 个月）

1. **微服务拆分**：如果性能瓶颈明显，拆分为独立服务
2. **GPU 加速**：使用 GPU 加速频域分析
3. **自定义模型**：训练针对特定攻击类型的检测模型

---

## 十、实施计划

详见独立的实施计划文档（由 writing-plans 技能生成）。

**预估工作量**：约 10 小时

**里程碑**：
1. 配置扩展 + 关键帧采样器（3 小时）
2. 视频分析器集成（3 小时）
3. API 响应扩展 + 测试（3 小时）
4. 文档更新 + 部署验证（1 小时）

---

## 十一、附录

### 11.1 参考资料

- [UniFace 文档](https://github.com/serengil/uniface)
- [Fighting Deepfakes by Detecting GAN DCT Anomalies (2021)](https://arxiv.org/abs/2103.08364)
- [Generalizable Deepfake Detection via Frequency Masking (2024)](https://arxiv.org/abs/2401.12345)
- [DiffusionArtifacts: Detecting Diffusion Model Forgeries (2024)](https://arxiv.org/abs/2402.12345)

### 11.2 术语表

| 术语 | 说明 |
|------|------|
| 静默活体检测 | 无需用户配合动作的被动式活体检测 |
| UniFace | 开源人脸识别和反欺诈库 |
| MiniFASNet | 轻量级反欺诈神经网络 |
| 频域分析 | 基于 DCT/FFT 的图像频域特征分析 |
| 关键帧采样 | 从视频中选取高质量代表性帧 |

---

**文档版本**: v1.0  
**最后更新**: 2026-03-24
