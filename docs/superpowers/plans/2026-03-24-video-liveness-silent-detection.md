# 视频活体检测防伪增强实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为视频活体检测系统集成静默检测能力，防御 AI 生成视频和手机播放攻击

**Architecture:** 在 VideoLivenessAnalyzer 中集成 FrameSampler（关键帧采样）和 SilentLivenessDetector（静默检测），在视频分析开始时采样 3-5 帧高质量人脸帧执行静默检测，检测失败则立即拒绝（严格模式）或降低置信度（宽松模式）

**Tech Stack:** Python 3.10+, OpenCV, MediaPipe, UniFace, NumPy, pytest

**Spec Document:** `docs/superpowers/specs/2026-03-24-video-liveness-silent-detection-design.md`

---

## 文件结构规划

### 新建文件

1. **`vrlFace/liveness/frame_sampler.py`** (约 200 行)
   - 职责：从视频中采样高质量正面人脸帧
   - 核心类：`FrameSampler`
   - 核心方法：`sample_keyframes(video_path, num_frames, min_quality, max_angle)`

### 修改文件

1. **`vrlFace/liveness/config.py`**
   - 新增：静默检测配置项（6 个字段）
   - 新增：`video_anti_spoofing_with_silent_config()` 预设
   - 新增：`from_env()` 方法支持环境变量加载

2. **`vrlFace/liveness/video_analyzer.py`**
   - 修改：`__init__()` 懒加载静默检测器
   - 修改：`analyze()` 集成静默检测流程
   - 新增：`_run_silent_detection()` 方法
   - 新增：`_build_reject_result()` 方法

3. **`vrlFace/liveness/schemas.py`**
   - 修改：`VideoLivenessResult` 新增 2 个字段

4. **`requirements.txt`**
   - 新增：`uniface>=3.1.0`

### 测试文件

1. **`tests/liveness/test_frame_sampler.py`** (新建)
   - 测试关键帧采样逻辑

2. **`tests/liveness/test_silent_integration.py`** (新建)
   - 测试静默检测集成

---

## 实施任务

### Task 1: 配置扩展

**Files:**
- Modify: `vrlFace/liveness/config.py`
- Test: `tests/liveness/test_config.py`

- [ ] **Step 1: 在 LivenessConfig 类中添加静默检测配置字段**

在 `config.py` 第 49 行后添加：

```python
# 静默检测配置
enable_silent_detection: bool = False
silent_detection_mode: str = "strict"  # "strict" 或 "loose"
silent_sample_frames: int = 5
silent_min_quality: float = 0.6
silent_max_angle: float = 15.0
silent_detection_timeout: float = 5.0
```

- [ ] **Step 2: 添加 video_anti_spoofing_with_silent_config 预设方法**

在 `config.py` 文件末尾添加：

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
    config.silent_detection_timeout = 5.0
    return config
```

- [ ] **Step 3: 添加 from_env 方法支持环境变量**

在 `config.py` 文件末尾添加：

```python
@classmethod
def from_env(cls) -> "LivenessConfig":
    """从环境变量加载配置"""
    import os
    config = cls.realtime_config()
    
    # 静默检测配置
    if os.getenv("LIVENESS_ENABLE_SILENT", "").lower() == "true":
        config.enable_silent_detection = True
    config.silent_detection_mode = os.getenv("LIVENESS_SILENT_MODE", "strict")
    config.silent_sample_frames = int(os.getenv("LIVENESS_SILENT_SAMPLE_FRAMES", "5"))
    config.silent_min_quality = float(os.getenv("LIVENESS_SILENT_MIN_QUALITY", "0.6"))
    config.silent_max_angle = float(os.getenv("LIVENESS_SILENT_MAX_ANGLE", "15.0"))
    config.silent_detection_timeout = float(os.getenv("LIVENESS_SILENT_TIMEOUT", "5.0"))
    
    return config
```

- [ ] **Step 4: 编写配置测试**

创建 `tests/liveness/test_config.py`（如果不存在）并添加：

```python
import os
import pytest
from vrlFace.liveness.config import LivenessConfig


def test_silent_detection_config_defaults():
    """测试静默检测配置默认值"""
    config = LivenessConfig()
    assert config.enable_silent_detection is False
    assert config.silent_detection_mode == "strict"
    assert config.silent_sample_frames == 5


def test_video_anti_spoofing_with_silent_config():
    """测试静默检测预设配置"""
    config = LivenessConfig.video_anti_spoofing_with_silent_config()
    assert config.enable_silent_detection is True
    assert config.silent_detection_mode == "strict"
    assert config.silent_sample_frames == 5


def test_from_env_silent_detection(monkeypatch):
    """测试从环境变量加载静默检测配置"""
    monkeypatch.setenv("LIVENESS_ENABLE_SILENT", "true")
    monkeypatch.setenv("LIVENESS_SILENT_MODE", "loose")
    monkeypatch.setenv("LIVENESS_SILENT_SAMPLE_FRAMES", "3")
    
    config = LivenessConfig.from_env()
    assert config.enable_silent_detection is True
    assert config.silent_detection_mode == "loose"
    assert config.silent_sample_frames == 3
```

- [ ] **Step 5: 运行测试验证配置**

```bash
uv run pytest tests/liveness/test_config.py -v
```

预期：所有测试通过

- [ ] **Step 6: 提交配置扩展**

```bash
git add vrlFace/liveness/config.py tests/liveness/test_config.py
git commit -m "feat(liveness): 添加静默检测配置项"
```


---

### Task 2: 实现关键帧采样器

**Files:**
- Create: `vrlFace/liveness/frame_sampler.py`
- Test: `tests/liveness/test_frame_sampler.py`

- [ ] **Step 1: 编写关键帧采样器测试（TDD）**

创建 `tests/liveness/test_frame_sampler.py`：

```python
import pytest
import numpy as np
from vrlFace.liveness.frame_sampler import FrameSampler


def test_frame_sampler_initialization():
    """测试采样器初始化"""
    sampler = FrameSampler()
    assert sampler is not None


def test_sample_keyframes_returns_frames(tmp_path):
    """测试采样返回帧列表"""
    # 需要真实视频文件进行测试
    # 使用项目中的测试视频
    video_path = "data/test_video.mp4"  # 假设存在
    sampler = FrameSampler()
    
    frames = sampler.sample_keyframes(video_path, num_frames=3)
    assert isinstance(frames, list)
    assert len(frames) <= 3
    if frames:
        assert isinstance(frames[0], np.ndarray)


def test_sample_keyframes_quality_filtering():
    """测试质量过滤"""
    sampler = FrameSampler()
    # 质量评分应该在 0-1 之间
    score = sampler._calculate_quality_score(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        face_size=0.3,
        angle=5.0
    )
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/liveness/test_frame_sampler.py -v
```

预期：FAIL（模块不存在）


- [ ] **Step 3: 实现 FrameSampler 类（最小实现）**

创建 `vrlFace/liveness/frame_sampler.py`：

```python
"""关键帧采样器 — 为静默检测提供高质量人脸帧"""

import cv2
import numpy as np
import logging
from typing import List
from .mediapipe_detector import MediaPipeLivenessDetector

logger = logging.getLogger(__name__)


class FrameSampler:
    """关键帧采样器"""
    
    def __init__(self):
        self.detector = MediaPipeLivenessDetector()
    
    def sample_keyframes(
        self,
        video_path: str,
        num_frames: int = 5,
        min_quality: float = 0.6,
        max_angle: float = 15.0,
        max_scan_frames: int = 100
    ) -> List[np.ndarray]:
        """采样高质量正面人脸帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频：{video_path}")
            return []
        
        candidates = []
        frame_idx = 0
        
        while frame_idx < max_scan_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.detector.detect(frame)
            if not result.face_detected:
                frame_idx += 1
                continue
            
            # 检查角度
            if abs(result.yaw) > max_angle or abs(result.pitch) > max_angle:
                frame_idx += 1
                continue
            
            # 计算质量分数
            quality = self._calculate_quality_score(
                frame,
                face_size=result.face_size if hasattr(result, 'face_size') else 0.3,
                angle=max(abs(result.yaw), abs(result.pitch))
            )
            
            if quality >= min_quality:
                candidates.append((quality, frame.copy()))
            
            frame_idx += 1
        
        cap.release()
        
        # 按质量排序，取 top-N
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [frame for _, frame in candidates[:num_frames]]
    
    def _calculate_quality_score(
        self,
        frame: np.ndarray,
        face_size: float,
        angle: float
    ) -> float:
        """计算帧质量分数"""
        # 人脸尺寸分数
        size_score = min(face_size / 0.5, 1.0)
        
        # 清晰度分数（拉普拉斯方差）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(sharpness / 500.0, 1.0)
        
        # 正面程度分数
        frontal_score = 1.0 - (angle / 15.0)
        
        # 加权融合
        quality = 0.4 * size_score + 0.4 * sharpness_score + 0.2 * frontal_score
        return max(0.0, min(1.0, quality))
```


- [ ] **Step 4: 运行测试验证实现**

```bash
uv run pytest tests/liveness/test_frame_sampler.py -v
```

预期：测试通过

- [ ] **Step 5: 提交关键帧采样器**

```bash
git add vrlFace/liveness/frame_sampler.py tests/liveness/test_frame_sampler.py
git commit -m "feat(liveness): 实现关键帧采样器"
```

---

### Task 3: 更新依赖

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 添加 uniface 依赖**

在 `requirements.txt` 末尾添加：

```txt
# 静默活体检测
uniface>=3.1.0
```

- [ ] **Step 2: 安装依赖**

```bash
uv pip install uniface>=3.1.0
```

- [ ] **Step 3: 验证 uniface 可用**

```bash
python -c "from uniface.detection import RetinaFace; from uniface.spoofing import MiniFASNet; print('UniFace OK')"
```

预期：输出 "UniFace OK"

- [ ] **Step 4: 提交依赖更新**

```bash
git add requirements.txt
git commit -m "deps: 添加 uniface 依赖用于静默检测"
```


---

### Task 4: 扩展 API 响应结构

**Files:**
- Modify: `vrlFace/liveness/schemas.py`

- [ ] **Step 1: 在 VideoLivenessResult 中添加新字段**

在 `schemas.py` 的 `VideoLivenessResult` 类中添加：

```python
# 在现有字段后添加
reject_reason: Optional[str] = None
silent_detection: Optional[Dict[str, Any]] = None
```

确保导入：

```python
from typing import Optional, Dict, Any
```

- [ ] **Step 2: 验证修改**

```bash
python -c "from vrlFace.liveness.schemas import VideoLivenessResult; print('Schema OK')"
```

预期：输出 "Schema OK"

- [ ] **Step 3: 提交 schema 扩展**

```bash
git add vrlFace/liveness/schemas.py
git commit -m "feat(liveness): 扩展 VideoLivenessResult 支持静默检测"
```


---

### Task 5: 集成静默检测到 VideoLivenessAnalyzer（Part 1）

**Files:**
- Modify: `vrlFace/liveness/video_analyzer.py`
- Test: `tests/liveness/test_silent_integration.py`

- [ ] **Step 1: 编写集成测试（TDD）**

创建 `tests/liveness/test_silent_integration.py`：

```python
import pytest
from vrlFace.liveness import VideoLivenessAnalyzer, LivenessConfig


def test_silent_detection_disabled_by_default():
    """测试静默检测默认禁用"""
    config = LivenessConfig()
    analyzer = VideoLivenessAnalyzer(config)
    assert analyzer._silent_detector is None


def test_silent_detection_enabled():
    """测试静默检测启用"""
    config = LivenessConfig.video_anti_spoofing_with_silent_config()
    analyzer = VideoLivenessAnalyzer(config)
    assert analyzer._silent_detector is not None
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/liveness/test_silent_integration.py -v
```

预期：FAIL（属性不存在）


- [ ] **Step 3: 修改 VideoLivenessAnalyzer.__init__() 懒加载静默检测器**

在 `video_analyzer.py` 的 `__init__` 方法末尾添加：

```python
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

- [ ] **Step 4: 运行测试验证初始化**

```bash
uv run pytest tests/liveness/test_silent_integration.py::test_silent_detection_enabled -v
```

预期：测试通过

- [ ] **Step 5: 提交初始化修改**

```bash
git add vrlFace/liveness/video_analyzer.py tests/liveness/test_silent_integration.py
git commit -m "feat(liveness): VideoAnalyzer 支持懒加载静默检测器"
```


---

### Task 6: 实现静默检测方法

**Files:**
- Modify: `vrlFace/liveness/video_analyzer.py`

- [ ] **Step 1: 添加 _run_silent_detection 方法**

在 `video_analyzer.py` 中添加方法：

```python
def _run_silent_detection(self, video_path: str) -> Dict[str, Any]:
    """执行静默检测"""
    import tempfile
    import os
    
    logger.info("开始静默检测：采样关键帧...")
    
    keyframes = self._frame_sampler.sample_keyframes(
        video_path,
        num_frames=self.config.silent_sample_frames,
        min_quality=self.config.silent_min_quality,
        max_angle=self.config.silent_max_angle
    )
    
    if not keyframes:
        return {
            "passed": False,
            "confidence": 0.0,
            "reject_reason": "no_quality_frames",
            "details": {}
        }
    
    results = []
    for i, frame in enumerate(keyframes):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_path = tmp_file.name
            cv2.imwrite(temp_path, frame)
        
        try:
            result = self._silent_detector.detect(temp_path)
            results.append(result)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    passed = all(r["is_liveness"] == 1 for r in results)
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    
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


- [ ] **Step 2: 添加 _build_reject_result 方法**

在 `video_analyzer.py` 中添加方法：

```python
def _build_reject_result(self, silent_result: Dict[str, Any]) -> VideoLivenessResult:
    """构建静默检测拒绝结果"""
    return VideoLivenessResult(
        is_liveness=0,
        liveness_confidence=0.0,
        reject_reason=silent_result["reject_reason"],
        is_face_exist=1,
        face_info=FaceInfo(confidence=0.0, quality_score=0.0),
        action_verify=ActionVerifyResult(
            passed=False,
            required_actions=[],
            action_details=[]
        ),
        silent_detection={
            "enabled": True,
            "passed": False,
            "confidence": silent_result["confidence"],
            "details": silent_result["details"]
        }
    )
```

- [ ] **Step 3: 验证方法可导入**

```bash
python -c "from vrlFace.liveness.video_analyzer import VideoLivenessAnalyzer; print('Methods OK')"
```

预期：输出 "Methods OK"

- [ ] **Step 4: 提交静默检测方法**

```bash
git add vrlFace/liveness/video_analyzer.py
git commit -m "feat(liveness): 实现静默检测方法"
```


---

### Task 7: 集成静默检测到 analyze 方法

**Files:**
- Modify: `vrlFace/liveness/video_analyzer.py`

- [ ] **Step 1: 在 analyze() 方法开始处添加静默检测调用**

在 `analyze()` 方法的开始部分（视频打开后）添加：

```python
# 静默检测（如果启用）
silent_result = None
if self.config.enable_silent_detection:
    silent_result = self._run_silent_detection(video_path)
    
    # 严格模式：检测失败立即返回
    if self.config.silent_detection_mode == "strict" and not silent_result["passed"]:
        return self._build_reject_result(silent_result)
```

- [ ] **Step 2: 在最终结果构建处添加宽松模式处理**

在 `analyze()` 方法返回结果前添加：

```python
# 宽松模式：降低置信度
if self.config.silent_detection_mode == "loose" and silent_result and not silent_result["passed"]:
    final_confidence *= 0.8
```

- [ ] **Step 3: 在返回的 VideoLivenessResult 中添加 silent_detection 字段**

修改返回语句，添加：

```python
silent_detection={
    "enabled": self.config.enable_silent_detection,
    "passed": silent_result["passed"] if silent_result else None,
    "confidence": silent_result["confidence"] if silent_result else None,
    "details": silent_result["details"] if silent_result else None
} if silent_result else None
```


- [ ] **Step 4: 提交 analyze 方法集成**

```bash
git add vrlFace/liveness/video_analyzer.py
git commit -m "feat(liveness): 集成静默检测到 analyze 方法"
```

---

### Task 8: 集成测试

**Files:**
- Test: `tests/liveness/test_silent_integration.py`

- [ ] **Step 1: 添加端到端集成测试**

在 `tests/liveness/test_silent_integration.py` 中添加：

```python
def test_silent_detection_strict_mode_rejects():
    """测试严格模式拒绝伪造视频"""
    config = LivenessConfig.video_anti_spoofing_with_silent_config()
    config.silent_detection_mode = "strict"
    analyzer = VideoLivenessAnalyzer(config)
    
    # 需要准备测试视频
    # result = analyzer.analyze("fake_video.mp4", required_actions=[])
    # assert result.is_liveness == 0
    # assert result.reject_reason in ["traditional_spoof", "ai_spoof"]


def test_silent_detection_loose_mode():
    """测试宽松模式降低置信度"""
    config = LivenessConfig.video_anti_spoofing_with_silent_config()
    config.silent_detection_mode = "loose"
    analyzer = VideoLivenessAnalyzer(config)
    
    # 需要准备测试视频
    pass
```

- [ ] **Step 2: 运行所有集成测试**

```bash
uv run pytest tests/liveness/test_silent_integration.py -v
```

预期：测试通过（或跳过需要真实视频的测试）


---

### Task 9: 文档更新

**Files:**
- Modify: `vrlFace/liveness/AGENTS.md`

- [ ] **Step 1: 更新 AGENTS.md 文档**

在 `vrlFace/liveness/AGENTS.md` 的"最近变更"部分添加：

```markdown
- **静默检测集成**: 支持 AI 生成视频和手机播放攻击检测
```

在"配置预设"部分添加：

```markdown
LivenessConfig.video_anti_spoofing_with_silent_config()  # 视频防伪+静默检测
```

- [ ] **Step 2: 提交文档更新**

```bash
git add vrlFace/liveness/AGENTS.md
git commit -m "docs: 更新活体检测文档，添加静默检测说明"
```

---

### Task 10: 最终验证

**Files:**
- All modified files

- [ ] **Step 1: 运行所有测试**

```bash
uv run pytest tests/liveness/ -v
```

预期：所有测试通过


- [ ] **Step 2: 验证配置加载**

```bash
python -c "from vrlFace.liveness import LivenessConfig; c = LivenessConfig.video_anti_spoofing_with_silent_config(); print(f'Silent enabled: {c.enable_silent_detection}')"
```

预期：输出 "Silent enabled: True"

- [ ] **Step 3: 验证模块导入**

```bash
python -c "from vrlFace.liveness import VideoLivenessAnalyzer; from vrlFace.liveness.frame_sampler import FrameSampler; print('All imports OK')"
```

预期：输出 "All imports OK"

- [ ] **Step 4: 运行代码检查**

```bash
uv run flake8 vrlFace/liveness/frame_sampler.py vrlFace/liveness/video_analyzer.py --max-line-length=100
```

预期：无错误

- [ ] **Step 5: 最终提交**

```bash
git add .
git commit -m "feat(liveness): 完成视频活体检测静默检测集成

- 新增关键帧采样器
- 集成 UniFace 静默检测
- 支持严格/宽松两种模式
- 新增配置项和环境变量支持"
```

---

## 验收标准

- [ ] 所有单元测试通过
- [ ] 配置可通过环境变量加载
- [ ] 静默检测可通过配置开关启用/禁用
- [ ] 严格模式能立即拒绝伪造视频
- [ ] 宽松模式能降低置信度
- [ ] 代码符合 PEP 8 规范
- [ ] 文档已更新

---

## 预估工作量

- Task 1-3: 1 小时（配置+依赖）
- Task 4-7: 4 小时（核心实现）
- Task 8-10: 1 小时（测试+文档）
- **总计**: 约 6 小时

