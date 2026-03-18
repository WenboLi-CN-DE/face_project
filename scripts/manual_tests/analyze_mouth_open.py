#!/usr/bin/env python3
"""
分析 mouth_open 通过率低的原因
从远程日志中提取所有 mouth_open 相关的 MAR 数据
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

def analyze_mouth_open_from_log(log_path: str):
    """从日志中分析 mouth_open 数据"""
    print(f"\n{'='*80}")
    print(f"分析日志：{log_path}")
    print(f"{'='*80}")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有 mouth_open 动作的行
    mouth_pattern = re.compile(
        r"动作 'mouth_open': frames=(\d+), events=(\d+), avg_score=([\d.]+)"
    )
    
    # 提取视频信息
    video_pattern = re.compile(
        r"vrlMoveLiveness.*?task_id=([a-f0-9-]+).*?actions=(\[[^\]]+\])"
    )
    
    # 提取 MAR 相关数据（从 diagnose 输出）
    mar_pattern = re.compile(
        r"MAR.*?最小值：([\d.]+).*?最大值：([\d.]+).*?平均值：([\d.]+)"
    )
    
    # 找到所有 mouth_open 测试
    results = []
    current_task = None
    current_actions = None
    
    for line in content.split('\n'):
        # 检测新的任务
        task_match = video_pattern.search(line)
        if task_match:
            current_task = task_match.group(1)
            current_actions = eval(task_match.group(2))
        
        # 检测 mouth_open 结果
        mouth_match = mouth_pattern.search(line)
        if mouth_match and current_task and 'mouth_open' in (current_actions or []):
            frames = int(mouth_match.group(1))
            events = int(mouth_match.group(2))
            avg_score = float(mouth_match.group(3))
            
            passed = events > 0
            
            results.append({
                'task_id': current_task,
                'frames': frames,
                'events': events,
                'avg_score': avg_score,
                'passed': passed,
            })
    
    if not results:
        print("❌ 未找到 mouth_open 测试结果")
        return
    
    # 统计分析
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    failed = total - passed
    
    print(f"\n【汇总统计】")
    print(f"  总测试数：{total}")
    print(f"  通过数：{passed} ({passed/total*100:.1f}%)")
    print(f"  失败数：{failed} ({failed/total*100:.1f}%)")
    
    # 通过和失败的对比
    passed_scores = [r['avg_score'] for r in results if r['passed']]
    failed_scores = [r['avg_score'] for r in results if not r['passed']]
    
    print(f"\n【分数对比】")
    if passed_scores:
        print(f"  通过视频：avg_score = {sum(passed_scores)/len(passed_scores):.3f} (范围：{min(passed_scores):.3f} - {max(passed_scores):.3f})")
    if failed_scores:
        print(f"  失败视频：avg_score = {sum(failed_scores)/len(failed_scores):.3f} (范围：{min(failed_scores):.3f} - {max(failed_scores):.3f})")
    
    # 详细列表
    print(f"\n【详细数据】")
    print(f"{'Task ID':^40} | {'Frames':^6} | {'Events':^6} | {'Avg Score':^10} | {'结果':^6}")
    print(f"{'-'*80}")
    
    for r in results:
        status = '✅' if r['passed'] else '❌'
        print(f"{r['task_id']:^40} | {r['frames']:^6} | {r['events']:^6} | {r['avg_score']:^10.3f} | {status:^6}")
    
    # 诊断建议
    print(f"\n{'='*80}")
    print(f"【诊断结论】")
    print(f"{'='*80}")
    
    if failed_scores:
        avg_failed = sum(failed_scores) / len(failed_scores)
        print(f"\n失败视频的平均分数：{avg_failed:.3f}")
        
        if avg_failed < 0.35:
            print(f"⚠️  MAR 阈值问题：失败视频平均分数 ({avg_failed:.3f}) 远低于当前阈值 (0.35)")
            print(f"   建议：mar_threshold 降低到 {avg_failed * 1.2:.3f} 左右")
        elif avg_failed < 0.50:
            print(f"⚠️  MAR 阈值边缘：失败视频平均分数 ({avg_failed:.3f}) 接近阈值 (0.35)")
            print(f"   建议：保持 mar_threshold=0.35，但需要检查视频质量")
        
        # 分析事件数为 0 的原因
        zero_events = [r for r in results if r['events'] == 0]
        if zero_events:
            print(f"\n⚠️  事件检出率为 0 的视频：{len(zero_events)}/{total}")
            print(f"   这些视频的 avg_score 范围：{min(r['avg_score'] for r in zero_events):.3f} - {max(r['avg_score'] for r in zero_events):.3f}")
            print(f"   说明：MAR 峰值未达到 mar_threshold，需要降低阈值或用户张嘴幅度太小")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分析 mouth_open 通过率")
    parser.add_argument("log", help="日志文件路径")
    args = parser.parse_args()
    
    analyze_mouth_open_from_log(args.log)
