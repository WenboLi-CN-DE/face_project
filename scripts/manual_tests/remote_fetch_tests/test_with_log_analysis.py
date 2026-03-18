#!/usr/bin/env python3
"""
增强版批量测试脚本 - 整合日志分析结果

功能：
1. 从日志中提取服务器端的检测结果
2. 本地重新测试视频
3. 对比服务器端和本地结果
4. 生成详细的问题定位报告
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from vrlFace.liveness.video_analyzer import VideoLivenessAnalyzer
from vrlFace.liveness.config import LivenessConfig


def load_log_analysis(analysis_file: str) -> dict:
    """加载日志分析结果"""
    with open(analysis_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_action_file(file_path: str) -> list:
    """解析动作对应文件"""
    ACTION_MAP = {
        "nod": "nod",
        "nod_up": "nod_up",
        "nod_down": "nod_down",
        "shake_head": "shake_head",
        "turn_left": "turn_left",
        "turn_right": "turn_right",
        "blink": "blink",
        "mouth_open": "mouth_open",
    }
    
    video_actions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not ".webm" in line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                video_name = parts[0].strip()
                actions = [ACTION_MAP[a] for a in parts[1:] if a in ACTION_MAP]
                if actions:
                    video_actions.append((video_name, actions))
    
    return video_actions


def test_single_video(video_path: str, expected_actions: list):
    """测试单个视频"""
    config = LivenessConfig.video_fast_config()
    analyzer = VideoLivenessAnalyzer(
        liveness_config=config,
        liveness_threshold=0.45,
        action_threshold=0.75,
        enable_benchmark=True,
    )
    
    try:
        result = analyzer.analyze(
            video_path=video_path,
            actions=expected_actions,
        )
        
        action_details = []
        for detail in result.action_verify.action_details:
            action_details.append({
                'action': detail.action,
                'passed': detail.passed,
                'confidence': detail.confidence,
                'msg': detail.msg,
            })
        
        return {
            'is_liveness': result.is_liveness == 1,
            'liveness_confidence': result.liveness_confidence,
            'action_details': action_details,
            'all_actions_passed': result.action_verify.passed,
            'error': None,
        }
    except Exception as e:
        return {'error': str(e)}


def compare_results(video_name: str, log_result: dict, local_result: dict) -> dict:
    """对比服务器端和本地测试结果"""
    comparison = {
        'video': video_name,
        'liveness_match': None,
        'action_comparisons': [],
        'issues': [],
    }
    
    # 对比活体判定
    if log_result and local_result and not local_result.get('error'):
        log_liveness = log_result.get('is_liveness', 0) == 1
        local_liveness = local_result.get('is_liveness', False)
        comparison['liveness_match'] = (log_liveness == local_liveness)
        
        if not comparison['liveness_match']:
            comparison['issues'].append(
                f"活体判定不一致：服务器={'通过' if log_liveness else '失败'}, "
                f"本地={'通过' if local_liveness else '失败'}"
            )
        
        # 对比各动作
        log_actions = {a['action']: a for a in log_result.get('action_details', [])}
        local_actions = {a['action']: a for a in local_result.get('action_details', [])}
        
        for action_name in set(list(log_actions.keys()) + list(local_actions.keys())):
            log_action = log_actions.get(action_name, {})
            local_action = local_actions.get(action_name, {})
            
            log_passed = log_action.get('passed', False)
            local_passed = local_action.get('passed', False)
            
            match = (log_passed == local_passed)
            
            comp = {
                'action': action_name,
                'match': match,
                'log_passed': log_passed,
                'local_passed': local_passed,
                'log_confidence': log_action.get('confidence', 0),
                'local_confidence': local_action.get('confidence', 0),
            }
            
            comparison['action_comparisons'].append(comp)
            
            if not match:
                comparison['issues'].append(
                    f"动作 {action_name} 结果不一致：服务器={'通过' if log_passed else '失败'}, "
                    f"本地={'通过' if local_passed else '失败'}"
                )
    
    return comparison


def main():
    video_dir = Path(__file__).parent / "videos"
    action_file = Path(__file__).parent / "动作对应.txt"
    analysis_file = Path(__file__).parent / "analysis.json"
    
    if not analysis_file.exists():
        print(f"❌ 找不到日志分析文件：{analysis_file}")
        print("请先运行：python scripts/log_video_analyzer.py output/remote_fetch/remote_server.log")
        return
    
    if not action_file.exists():
        print(f"❌ 找不到动作文件：{action_file}")
        return
    
    # 加载日志分析结果
    log_results = load_log_analysis(str(analysis_file))
    log_dict = {r['video_filename']: r for r in log_results}
    
    print("=" * 80)
    print("增强版批量测试 - 整合日志分析")
    print("=" * 80)
    print()
    
    video_actions = parse_action_file(str(action_file))
    print(f"找到 {len(video_actions)} 个视频待测试")
    print()
    
    comparisons = []
    total_videos = len(video_actions)
    
    for idx, (video_name, actions) in enumerate(video_actions, 1):
        video_path = video_dir / video_name
        
        if not video_path.exists():
            print(f"[{idx}/{total_videos}] ⚠️  跳过（文件不存在）: {video_name}")
            continue
        
        print(f"[{idx}/{total_videos}] 测试：{video_name}")
        print(f"  期望动作：{actions}")
        
        # 获取日志中的结果
        log_result = log_dict.get(video_name)
        
        # 本地测试
        local_result = test_single_video(str(video_path), actions)
        
        # 对比结果
        comparison = compare_results(video_name, log_result, local_result)
        comparisons.append(comparison)
        
        # 打印日志中的结果
        if log_result:
            print(f"  服务器端:")
            print(
                f"    活体：{'✅' if log_result.get('is_liveness') else '❌'} "
                f"(confidence={log_result.get('liveness_confidence', 0):.2%})"
            )
            for action in log_result.get('action_details', []):
                status = "✅" if action.get('passed') else "❌"
                print(
                    f"    {status} {action['action']:12s} "
                    f"frames={action['frames']}, events={action['events']}, "
                    f"confidence={action.get('confidence', 0):.2%}"
                )
        
        # 打印本地结果
        if local_result.get('error'):
            print(f"  本地测试：❌ 错误：{local_result['error']}")
        else:
            print(f"  本地测试:")
            print(
                f"    活体：{'✅' if local_result.get('is_liveness') else '❌'} "
                f"(confidence={local_result.get('liveness_confidence', 0):.2%})"
            )
            for action in local_result.get('action_details', []):
                status = "✅" if action.get('passed') else "❌"
                print(
                    f"    {status} {action['action']:12s} "
                    f"confidence={action.get('confidence', 0):.2%}  {action.get('msg', '')}"
                )
        
        # 打印对比结果
        if comparison['issues']:
            print(f"  ⚠️  发现问题:")
            for issue in comparison['issues']:
                print(f"    - {issue}")
        
        print()
    
    # 汇总统计
    print("=" * 80)
    print("汇总统计")
    print("=" * 80)
    
    valid_comparisons = [c for c in comparisons if not any('error' in str(i) for i in c.get('issues', []))]
    
    if not valid_comparisons:
        print("没有有效的对比结果")
        return
    
    # 活体判定一致性
    liveness_matches = sum(1 for c in valid_comparisons if c.get('liveness_match'))
    print(f"活体判定一致性：{liveness_matches}/{len(valid_comparisons)} ({liveness_matches/len(valid_comparisons)*100:.1f}%)")
    
    # 动作结果一致性
    total_actions = sum(len(c['action_comparisons']) for c in valid_comparisons)
    action_matches = sum(
        sum(1 for a in c['action_comparisons'] if a.get('match'))
        for c in valid_comparisons
    )
    print(f"动作结果一致性：{action_matches}/{total_actions} ({action_matches/total_actions*100:.1f}%)")
    
    # 问题统计
    videos_with_issues = sum(1 for c in comparisons if c.get('issues'))
    print(f"\n存在问题视频数：{videos_with_issues}/{total_videos}")
    
    if videos_with_issues > 0:
        print("\n问题列表:")
        for c in comparisons:
            if c.get('issues'):
                print(f"  - {c['video']}")
                for issue in c['issues']:
                    print(f"    {issue}")
    
    print()
    print("=" * 80)
    
    # 保存详细报告
    report_file = Path(__file__).parent / "test_comparison_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_videos': total_videos,
                'liveness_match_rate': liveness_matches / len(valid_comparisons) if valid_comparisons else 0,
                'action_match_rate': action_matches / total_actions if total_actions > 0 else 0,
                'videos_with_issues': videos_with_issues,
            },
            'comparisons': comparisons,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 详细报告已保存：{report_file}")


if __name__ == "__main__":
    main()
