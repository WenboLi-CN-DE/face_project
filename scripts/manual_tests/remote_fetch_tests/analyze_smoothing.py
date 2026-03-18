#!/usr/bin/env python3
"""
分析平滑窗口对分数的影响
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from vrlFace.liveness.config import LivenessConfig

# 模拟数据：静止帧分数低，动作帧分数高
scores = [0.05] * 50 + [0.85] * 10 + [0.05] * 40  # 中间 10 帧是动作

config = LivenessConfig.video_fast_config()
window = config.smooth_window  # 10

print(f"平滑窗口大小：{window}")
print(f"\n原始分数:")
print(f"  静止帧：0.05 (共 90 帧)")
print(f"  动作帧：0.85 (共 10 帧)")
print(f"  原始最高分：{max(scores):.4f}")

# 计算平滑分数
smoothed_scores = []
for i in range(len(scores)):
    window_scores = scores[max(0, i - window + 1) : i + 1]
    smoothed = sum(window_scores) / len(window_scores)
    smoothed_scores.append(smoothed)

print(f"\n平滑后分数:")
print(f"  平滑后最高分：{max(smoothed_scores):.4f}")
print(f"  平滑后平均分：{np.mean(smoothed_scores):.4f}")

# 找到最高平滑分数的位置
max_idx = smoothed_scores.index(max(smoothed_scores))
print(f"  最高分出现在第 {max_idx} 帧")

# 计算阈值
threshold = config.threshold
print(f"\n配置阈值：{threshold}")
print(f"活体判定：{'通过' if max(smoothed_scores) >= threshold else '失败'}")

# 不同窗口大小的影响
print(f"\n{'=' * 50}")
print(f"不同平滑窗口大小的影响:")
print(f"{'=' * 50}")

for w in [5, 10, 20, 30]:
    smoothed = []
    for i in range(len(scores)):
        window_scores = scores[max(0, i - w + 1) : i + 1]
        smoothed.append(sum(window_scores) / len(window_scores))

    max_smoothed = max(smoothed)
    passed = max_smoothed >= threshold
    print(f"窗口={w:2d}: 最高平滑分={max_smoothed:.4f}, 判定={'✓' if passed else '✗'}")
