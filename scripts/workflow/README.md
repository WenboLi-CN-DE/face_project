# 活体检测自动化测试工作流

**位置**: `scripts/workflow/`  
**职责**: 自动化诊断活体检测问题，精准定位根因

---

## 📊 工作流概览

```
┌────────────────────────────────────────────────────────────────┐
│ Step 1: auto_diagnose.py                                       │
│ 批量测试视频 → 识别问题视频 → 保存逐帧 CSV                       │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ Step 2: analyze_csv_detailed.py                                │
│ 分析 CSV 数据 → 生成统计报告 → 可视化图表                         │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ Step 3: 诊断报告 (JSON)                                         │
│ 问题汇总 → 阈值建议 → 改进措施                                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 基本用法

```bash
# 1. 运行自动化诊断工作流
uv run python scripts/workflow/auto_diagnose.py --video-dir 15

# 2. 分析特定 CSV 文件
uv run python scripts/workflow/analyze_csv_detailed.py \
  --csv output/diagnose/csv/test_video.webm.frames.csv

# 3. 生成可视化图表
uv run python scripts/workflow/analyze_csv_detailed.py \
  --csv output/diagnose/csv/test_video.webm.frames.csv \
  --plot
```

---

## 📁 输出目录结构

```
output/diagnose/
├── csv/                      # 逐帧 CSV 数据
│   ├── video1.webm.frames.csv
│   └── video2.webm.frames.csv
│
└── reports/                  # 诊断报告
    ├── diagnose_report_20260316_143022.json
    └── diagnose_report_20260316_151245.json
```

---

## 🔧 工具详解

### 1. auto_diagnose.py - 自动化诊断

**功能**:
- 批量测试视频目录中的所有视频
- 自动识别失败视频（活体未通过/动作未通过）
- 为失败视频自动保存逐帧 CSV
- 生成汇总诊断报告

**参数**:
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video-dir` | 视频目录（必需） | - |
| `--action-file` | 动作配置文件 | `<video-dir>/动作对应.txt` |
| `--actions` | 指定测试动作 | 从 action-file 读取 |
| `--no-csv` | 不保存逐帧 CSV | False |
| `--no-report` | 不生成诊断报告 | False |

**使用示例**:
```bash
# 基本用法
uv run python scripts/workflow/auto_diagnose.py --video-dir 15

# 指定动作配置文件
uv run python scripts/workflow/auto_diagnose.py \
  --video-dir 15 \
  --action-file 15/动作对应.txt

# 不保存 CSV（快速测试）
uv run python scripts/workflow/auto_diagnose.py \
  --video-dir 15 \
  --no-csv

# 仅生成报告（使用已有 CSV）
uv run python scripts/workflow/auto_diagnose.py \
  --video-dir 15 \
  --report-only
```

**输出示例**:
```
================================================================================
活体检测自动化诊断工作流
================================================================================
视频目录：15
动作配置：15/动作对应.txt
待测试视频：20 个

[1/20] 测试：test_01.webm
  期望动作：['blink', 'mouth_open']
  活体：✅ (85.00%)
    ✅ blink        置信度：0.92
    ✅ mouth_open   置信度：0.88

[2/20] 测试：test_02.webm
  期望动作：['blink', 'mouth_open']
  活体：❌ (42.00%)
    ❌ blink        置信度：0.35
    ❌ mouth_open   置信度：0.28
  📊 CSV 已保存：output/diagnose/csv/test_02.webm.frames.csv
  📈 CSV 分析:
     人脸检出率：95.0%
     EAR: 0.180 - 0.350
     MAR: 0.250 - 0.580
     Pitch 范围：12.5°
     Yaw 范围：18.3°

================================================================================
诊断报告摘要
================================================================================
测试时间：2026-03-16T14:30:22
测试视频数：20
通过：15 (75.0%)
失败：5 (25.0%)

改进建议:
⚠️  动作 'mouth_open' 通过率过低 (45.0%)，平均置信度 0.52
   → 建议：降低 mar_threshold (当前 0.28)
⚠️  总体活体通过率过低 (75.0%)，建议检查视频质量或调整 threshold

完整报告：output/diagnose/reports/diagnose_report_20260316_143022.json
================================================================================
```

---

### 2. analyze_csv_detailed.py - CSV 详细分析

**功能**:
- 读取 recorder.py 生成的逐帧 CSV
- 计算 EAR/MAR/Pitch/Yaw 统计
- 检测眨眼/张嘴片段
- 生成诊断建议
- 可视化图表（可选）

**参数**:
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--csv` | CSV 文件路径（必需） | - |
| `--output` | JSON 报告输出路径 | 不保存 |
| `--plot` | 生成可视化图表 | False |

**使用示例**:
```bash
# 分析 CSV 并打印报告
uv run python scripts/workflow/analyze_csv_detailed.py \
  --csv output/diagnose/csv/test_02.webm.frames.csv

# 保存 JSON 报告
uv run python scripts/workflow/analyze_csv_detailed.py \
  --csv output/diagnose/csv/test_02.webm.frames.csv \
  --output output/diagnose/reports/test_02_analysis.json

# 生成可视化图表
uv run python scripts/workflow/analyze_csv_detailed.py \
  --csv output/diagnose/csv/test_02.webm.frames.csv \
  --plot
```

**输出示例**:
```
================================================================================
CSV 数据分析报告
================================================================================
文件：output/diagnose/csv/test_02.webm.frames.csv
分析时间：2026-03-16T14:35:10

【基础统计】
  总帧数：180
  人脸检出率：95.0%
  活体通过率：45.0%
  最佳分数：0.5234
  平均分数：0.3891

【EAR 统计（眨眼）】
  范围：0.120 - 0.350
  平均：0.220 ± 0.045
  EAR < 0.15: 12 帧
  EAR < 0.20: 45 帧
  EAR < 0.25: 78 帧

【MAR 统计（张嘴）】
  范围：0.180 - 0.620
  平均：0.350 ± 0.098
  MAR > 0.40: 38 帧
  MAR > 0.50: 22 帧
  MAR > 0.60: 8 帧

【头部姿态】
  Pitch: -5.2° ~ +6.8° (范围：12.0°)
  Yaw:   -15.3° ~ +18.5° (范围：33.8°)

【动作事件】
  眨眼：45 帧
  张嘴：38 帧
  头部动作：92 帧

【诊断建议】
  ⚠️  眨眼帧占比过低 (25.0%)，EAR 范围 0.120-0.350，建议降低 ear_threshold
  ⚠️  张嘴帧占比过低 (21.1%)，MAR 范围 0.180-0.620，建议降低 mar_threshold

================================================================================
```

---

## 📊 CSV 数据格式

**列定义**:

| 列名 | 类型 | 说明 |
|------|------|------|
| `frame_idx` | int | 帧号 |
| `timestamp_s` | float | 时间戳（秒） |
| `face_detected` | bool | 是否检测到人脸 |
| `is_live` | bool | 活体判定 |
| `score_smoothed` | float | 平滑分数 |
| `motion_score` | float | 原始 motion 分数 |
| `ear` | float | 眼睛纵横比（眨眼指标） |
| `mar` | float | 嘴巴纵横比（张嘴指标） |
| `yaw` | float | 左右转头角度 |
| `pitch` | float | 上下点头角度 |
| `blink_detected` | bool | 眨眼检测 |
| `mouth_open` | bool | 张嘴检测 |
| `head_action` | str | 头部动作类型 |

**示例数据**:
```csv
frame_idx,timestamp_s,face_detected,is_live,score_smoothed,ear,mar,yaw,pitch,blink_detected,mouth_open,head_action
1,0.033,True,True,0.623,0.182,0.321,2.1,-1.5,False,False,none
2,0.067,True,True,0.606,0.145,0.583,8.5,3.2,True,False,none
3,0.100,True,True,0.645,0.218,0.645,-6.2,12.5,False,True,shake_head
```

---

## 🎯 典型工作流

### 场景 1: 批量测试新视频集

```bash
# Step 1: 运行自动化诊断
uv run python scripts/workflow/auto_diagnose.py \
  --video-dir new_videos \
  --action-file new_videos/动作对应.txt

# Step 2: 查看诊断报告摘要
# （自动打印在终端）

# Step 3: 分析特定问题视频的 CSV
uv run python scripts/workflow/analyze_csv_detailed.py \
  --csv output/diagnose/csv/problem_video.webm.frames.csv \
  --plot

# Step 4: 根据建议调整阈值
# 修改 vrlFace/liveness/config.py 或 fast_detector.py
```

---

### 场景 2: 调试特定动作失败

```bash
# Step 1: 只测试特定动作
uv run python scripts/workflow/auto_diagnose.py \
  --video-dir 15 \
  --actions mouth_open

# Step 2: 分析失败视频的 CSV
uv run python scripts/workflow/analyze_csv_detailed.py \
  --csv output/diagnose/csv/test_02.webm.frames.csv \
  --output output/diagnose/reports/test_02_analysis.json

# Step 3: 查看 JSON 报告中的详细统计
cat output/diagnose/reports/test_02_analysis.json | jq '.mar'
```

---

### 场景 3: 阈值调优验证

```bash
# Step 1: 修改阈值后重新测试
# 修改 vrlFace/liveness/fast_detector.py:
#   MAR_THRESHOLD = 0.25  # 从 0.28 降低到 0.25

# Step 2: 重新运行诊断
uv run python scripts/workflow/auto_diagnose.py \
  --video-dir 15 \
  --no-csv  # 不保存 CSV，快速验证

# Step 3: 对比通过率变化
# 之前：mouth_open 通过率 45%
# 现在：mouth_open 通过率 78%  ← 改善
```

---

## 🔍 诊断逻辑

### EAR（眨眼）诊断

| EAR 范围 | 眨眼帧占比 | 诊断 | 建议 |
|---------|-----------|------|------|
| < 0.15 | > 30% | 正常 | 无需调整 |
| 0.15-0.20 | 20-30% | 边缘 | 考虑降低 ear_threshold |
| > 0.20 | < 20% | 异常 | 降低 ear_threshold 或检查视频 |

### MAR（张嘴）诊断

| MAR 范围 | 张嘴帧占比 | 诊断 | 建议 |
|---------|-----------|------|------|
| > 0.50 | > 25% | 正常 | 无需调整 |
| 0.40-0.50 | 15-25% | 边缘 | 考虑降低 mar_threshold |
| < 0.40 | < 15% | 异常 | 降低 mar_threshold 或检查视频 |

### 头部姿态诊断

| 峰峰值范围 | 诊断 | 建议 |
|-----------|------|------|
| > 15° | 正常 | 无需调整 |
| 8-15° | 边缘 | 考虑降低 threshold |
| < 8° | 异常 | 降低 threshold 或动作幅度不足 |

---

## 📝 配置阈值

**当前默认值** (`scripts/workflow/auto_diagnose.py`):

```python
class DiagnoseConfig:
    # 阈值配置
    LIVENESS_THRESHOLD = 0.45      # 活体判定阈值
    ACTION_THRESHOLD = 0.75        # 动作通过阈值
    
    # EAR 阈值（眨眼）
    EAR_THRESHOLD = 0.20           # 眨眼触发阈值
    EAR_LOW_RATE = 0.30            # 眨眼帧占比 < 30% 判定为不足
    
    # MAR 阈值（张嘴）
    MAR_THRESHOLD = 0.28           # 张嘴触发阈值
    MAR_LOW_RATE = 0.25            # 张嘴帧占比 < 25% 判定为不足
    
    # 头部动作阈值
    PITCH_THRESHOLD = 8.0          # 点头角度阈值
    YAW_THRESHOLD = 8.0            # 转头角度阈值
```

**修改建议**:
- 根据诊断报告中的建议调整
- 调整后运行 `auto_diagnose.py --report-only` 重新评估

---

## 🛠️ 故障排查

### 问题 1: CSV 文件为空

**原因**: 视频无法读取或无人脸

**解决**:
```bash
# 检查视频格式
ffprobe video.webm

# 使用 diagnose_webm.py 诊断
uv run python scripts/manual_tests/diagnose_webm.py video.webm
```

### 问题 2: 图表生成失败

**原因**: 缺少 matplotlib

**解决**:
```bash
uv add matplotlib
```

### 问题 3: 诊断报告无建议

**原因**: 所有指标正常

**解决**: 这是正常现象，说明视频质量良好

---

## 📚 相关文档

- `vrlFace/liveness/recorder.py` - CSV 录制器源码
- `scripts/analyze_csv.py` - 基础 CSV 分析工具
- `scripts/manual_tests/analyze_action_failure.py` - 实时逐帧分析

---

## 🎯 最佳实践

1. **每次测试都保存 CSV**: 便于后续深入分析
2. **定期生成诊断报告**: 追踪阈值调整效果
3. **使用可视化图表**: 直观展示问题帧分布
4. **建立基准数据集**: 保存通过/失败视频样本
5. **自动化 CI 集成**: 将工作流集成到测试流程

---

## 📊 输出示例

**JSON 报告结构**:

```json
{
  "timestamp": "2026-03-16T14:30:22",
  "total_videos": 20,
  "passed_videos": 15,
  "failed_videos": 5,
  "pass_rate": 75.0,
  "video_results": [...],
  "recommendations": [
    "⚠️  动作 'mouth_open' 通过率过低 (45.0%)",
    "   → 建议：降低 mar_threshold (当前 0.28)"
  ]
}
```

---

## 🔄 工作流演进

**未来改进**:
- [ ] 自动阈值调优（网格搜索）
- [ ] 视频质量评分
- [ ] 动作标准度评估
- [ ] 历史数据对比
- [ ] Web Dashboard 可视化
