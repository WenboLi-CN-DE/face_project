# 活体检测阈值配置指南

**更新时间**: 2026-03-16  
**适用版本**: vrlFace >= 1.0.0

---

## 📊 阈值安全范围

### 强制限制（Pydantic 验证）

| 参数 | 最小值 | 最大值 | 默认值 |
|------|--------|--------|--------|
| `liveness_threshold` | **0.30** | **0.75** | 0.45 |
| `action_threshold` | **0.50** | **0.95** | 0.75 |

**超出范围的值将被 API 直接拒绝**，返回 HTTP 422 验证错误。

---

## 🎯 推荐配置

### 标准模式（推荐）

```json
{
  "threshold_config": {
    "liveness_threshold": 0.50,
    "action_threshold": 0.75
  }
}
```

**适用场景**: 大多数生产环境  
**特点**: 平衡安全性和通过率

---

### 宽松模式

```json
{
  "threshold_config": {
    "liveness_threshold": 0.45,
    "action_threshold": 0.75
  }
}
```

**适用场景**:
- 内部测试环境
- 用户体验优先场景
- 低风险业务

**预期效果**:
- 通过率：>95%
- 误拒率：<5%

---

### 严格模式

```json
{
  "threshold_config": {
    "liveness_threshold": 0.60,
    "action_threshold": 0.85
  }
}
```

**适用场景**:
- 金融支付场景
- 高安全要求业务
- 身份验证关键流程

**预期效果**:
- 通过率：>85%
- 误拒率：<1%
- 安全性：最高

---

## 🚫 禁止配置

### ❌ 极端高阈值（已禁止）

```json
// 错误示例 - 将被拒绝
{
  "threshold_config": {
    "liveness_threshold": 0.95,  // ❌ 超过 0.75 上限
    "action_threshold": 0.85
  }
}
```

**问题**: 导致正常用户被误判为攻击（误拒率 >40%）

---

### ❌ 极端低阈值（已禁止）

```json
// 错误示例 - 将被拒绝
{
  "threshold_config": {
    "liveness_threshold": 0.20,  // ❌ 低于 0.30 下限
    "action_threshold": 0.50
  }
}
```

**问题**: 可能导致照片/视频攻击通过（误接受率升高）

---

## 📝 API 请求示例

### 完整请求

```json
{
  "request_id": "req_20260316_001",
  "task_id": "task_vrl_1234567890",
  "video_path": "/data/storage/videos/user_v_001.mp4",
  "actions": ["blink", "mouth_open", "shake_head"],
  "threshold_config": {
    "liveness_threshold": 0.50,
    "action_threshold": 0.75
  },
  "action_config": {
    "max_video_duration": 6.0,
    "per_action_timeout": 2.0
  },
  "callback_url": "http://api.example.com/callbacks/liveness",
  "callback_secret": "your-secret-key"
}
```

### 最小请求（使用默认阈值）

```json
{
  "request_id": "req_20260316_002",
  "task_id": "task_vrl_0987654321",
  "video_path": "/data/storage/videos/user_v_002.mp4",
  "actions": ["blink"]
  // threshold_config 省略，使用默认值 liveness=0.45, action=0.75
}
```

---

## 🔧 服务端日志

### 阈值警告日志

当阈值超出推荐范围（但仍在安全范围内）时，服务端会记录警告：

```
WARNING: vrlMoveLiveness 阈值超出推荐范围 task_id=xxx liveness_threshold=0.65 (推荐 0.45-0.60)
WARNING: vrlMoveLiveness 动作阈值超出推荐范围 task_id=xxx action_threshold=0.90 (推荐 0.70-0.85)
```

**推荐范围**:
- `liveness_threshold`: 0.45 - 0.60
- `action_threshold`: 0.70 - 0.85

---

## 📈 阈值影响分析

### liveness_threshold（活体判定阈值）

| 阈值 | 通过率 | 误拒率 | 安全性 | 适用场景 |
|------|--------|--------|--------|----------|
| 0.30-0.40 | >98% | <2% | 低 | 测试/演示 |
| **0.45-0.50** | **90-95%** | **3-5%** | **中** | **标准生产** |
| 0.55-0.60 | 85-90% | 1-2% | 高 | 金融/支付 |
| 0.65-0.75 | <80% | <1% | 极高 | 特殊场景 |

### action_threshold（动作通过阈值）

| 阈值 | 动作通过率 | 严格度 | 说明 |
|------|------------|--------|------|
| 0.50-0.65 | >95% | 宽松 | 轻微动作即可通过 |
| **0.70-0.85** | **85-95%** | **标准** | **需要明确动作** |
| 0.90-0.95 | <80% | 严格 | 需要标准大幅动作 |

---

## 🧪 测试建议

### 1. 使用测试脚本验证

```bash
# 运行阈值验证测试
uv run python scripts/test_threshold_validation.py
```

### 2. A/B 测试不同阈值

```python
# 分组测试
groups = {
    "A": {"liveness_threshold": 0.45, "action_threshold": 0.75},
    "B": {"liveness_threshold": 0.50, "action_threshold": 0.75},
    "C": {"liveness_threshold": 0.55, "action_threshold": 0.80},
}
```

### 3. 监控指标

调整后监控以下指标：
- **通过率** = 通过次数 / 总请求数
- **误拒率** = 用户投诉 / 总请求数（需人工标注）
- **平均置信度** = 所有请求的平均分数

---

## ⚠️ 历史问题

### 问题：0.95 阈值导致大规模误拒

**时间**: 2026-03-16  
**现象**: 正常用户视频被判定为攻击  
**原因**: 前端配置 `liveness_threshold=0.95` 过高  
**解决**: 
1. 服务端添加阈值范围限制（0.30-0.75）
2. 添加日志警告机制
3. 前端调整推荐配置到 0.50

**影响**: 
- 失败视频实际分数 0.85，但因阈值 0.95 被误拒
- 调整后预期通过率从 50% 提升到 95%

---

## 📚 相关文档

- **API 文档**: `vrlFace/liveness/schemas.py`
- **配置实现**: `vrlFace/liveness/config.py`
- **测试脚本**: `scripts/test_threshold_validation.py`
- **故障分析**: `output/verification_failure_analysis.md`

---

## 🎯 最佳实践

1. **默认使用推荐值**: `liveness=0.50, action=0.75`
2. **不要随意调整**: 除非有充分的数据支持
3. **监控置信度**: 平均置信度应在 0.60-0.80 之间
4. **定期校准**: 每季度用真实数据验证阈值合理性
5. **分级部署**: 不同业务场景使用不同阈值配置

---

**最后更新**: 2026-03-16  
**维护者**: vrlFace Team
