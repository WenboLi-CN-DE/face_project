#!/usr/bin/env python3
"""
分析动作检测失败原因 - 使用 HeadActionDetector 的峰峰值检测逻辑
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from collections import deque

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from vrlFace.liveness.config import LivenessConfig
from vrlFace.liveness.fusion_engine import LivenessFusionEngine
from vrlFace.liveness.head_action import HeadActionDetector, HeadActionConfig

def analyze_action_video(video_path: str, actions: list):
    """使用峰峰值检测逻辑分析视频"""
    print(f"\n{'='*80}")
    print(f"分析视频：{video_path}")
    print(f"期望动作：{actions}")
    print(f"{'='*80}")
    
    config = LivenessConfig.video_fast_config()
    
    # 打印当前配置
    print(f"\n【当前配置】")
    print(f"  ear_threshold:   {config.ear_threshold} (眨眼，< 触发)")
    print(f"  mar_threshold:   {config.mar_threshold} (张嘴，> 触发)")
    print(f"  yaw_threshold:   {config.yaw_threshold}° (转头 - 峰峰值)")
    print(f"  pitch_threshold: {config.pitch_threshold}° (点头 - 峰峰值)")
    print(f"  action_threshold: 0.75 (动作通过阈值)")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0:
        temp_cap = cv2.VideoCapture(video_path)
        actual_frames = 0
        while temp_cap.read()[0]:
            actual_frames += 1
        temp_cap.release()
        total_frames = actual_frames
    
    print(f"\n【视频信息】帧数={total_frames}, FPS={fps:.2f}")
    
    engine = LivenessFusionEngine(config)
    head_detector = HeadActionDetector(HeadActionConfig(
        yaw_threshold=config.yaw_threshold,
        pitch_threshold=config.pitch_threshold,
        window_size=config.window_size,
    ))
    
    # 统计数据
    ear_values = []
    mar_values = []
    pitch_values = []
    yaw_values = []
    
    action_events = {
        'blink': 0, 'mouth_open': 0, 
        'nod': 0, 'nod_up': 0, 'nod_down': 0,
        'shake_head': 0, 'turn_left': 0, 'turn_right': 0
    }
    
    frame_count = 0
    max_frames = min(total_frames, 200)
    
    print(f"\n【逐帧分析】（每 5 帧采样，显示 Pitch/Yaw 值和检测到的动作）")
    print(f"{'帧号':^6} | {'EAR':^7} | {'MAR':^7} | {'Pitch':^7} | {'Yaw':^7} | {'检测动作':^25}")
    print(f"{'-'*80}")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 缩小帧
        max_w = config.max_width
        if max_w > 0 and frame.shape[1] > max_w:
            scale = max_w / frame.shape[1]
            frame = cv2.resize(frame, (max_w, int(frame.shape[0] * scale)))
        
        # 提取 landmarks
        lm_data = engine.mp_detector.extract_landmarks(frame)
        if lm_data is None:
            head_detector.reset()  # 人脸丢失，重置检测器
            continue
        
        landmarks = lm_data['landmarks']
        transform_matrix = lm_data.get('transform_matrix')
        aspect_ratio = lm_data.get('aspect_ratio', 1.0)
        
        # 计算 EAR, MAR
        ear = engine.mp_detector.calculate_ear(landmarks, aspect_ratio)
        mar = engine.mp_detector.calculate_mar(landmarks, aspect_ratio)
        
        # 计算头部姿态
        pitch, yaw, roll = engine.mp_detector.calculate_head_pose(
            landmarks, frame.shape, transform_matrix
        )
        
        # 检测头部动作
        head_action = head_detector.detect(pitch, yaw)
        
        # 检测眨眼/张嘴
        blink = ear < config.ear_threshold
        mouth = mar > config.mar_threshold
        
        ear_values.append(ear)
        mar_values.append(mar)
        pitch_values.append(pitch)
        yaw_values.append(yaw)
        
        # 记录事件
        if blink:
            action_events['blink'] += 1
        if mouth:
            action_events['mouth_open'] += 1
        if head_action != 'none':
            action_events[head_action] = action_events.get(head_action, 0) + 1
        
        # 采样显示
        if frame_count % 5 == 0:
            events = []
            if blink: events.append('眨眼')
            if mouth: events.append('张嘴')
            if head_action != 'none': events.append(head_action)
            event_str = ','.join(events) if events else '-'
            print(f"{frame_count:^6} | {ear:^7.3f} | {mar:^7.3f} | {pitch:^7.1f} | {yaw:^7.1f} | {event_str:^25}")
    
    cap.release()
    
    # 统计分析
    print(f"\n{'='*80}")
    print(f"【统计汇总】")
    print(f"{'='*80}")
    
    total = len(ear_values) if ear_values else 1
    
    print(f"\nEAR (眨眼，<{config.ear_threshold}触发):")
    print(f"  最小/最大/平均：{min(ear_values):.3f} / {max(ear_values):.3f} / {np.mean(ear_values):.3f}")
    print(f"  眨眼帧数：{action_events['blink']}/{total} ({action_events['blink']/total*100:.1f}%)")
    
    print(f"\nMAR (张嘴，>{config.mar_threshold}触发):")
    print(f"  最小/最大/平均：{min(mar_values):.3f} / {max(mar_values):.3f} / {np.mean(mar_values):.3f}")
    print(f"  张嘴帧数：{action_events['mouth_open']}/{total} ({action_events['mouth_open']/total*100:.1f}%)")
    
    print(f"\nPitch (点头，峰峰值>={config.pitch_threshold}°触发):")
    print(f"  最小/最大/平均：{min(pitch_values):.1f}° / {max(pitch_values):.1f}° / {np.mean(pitch_values):.1f}°")
    print(f"  峰峰值范围：{max(pitch_values) - min(pitch_values):.1f}°")
    
    print(f"\nYaw (转头，峰峰值>={config.yaw_threshold}°触发):")
    print(f"  最小/最大/平均：{min(yaw_values):.1f}° / {max(yaw_values):.1f}° / {np.mean(yaw_values):.1f}°")
    print(f"  峰峰值范围：{max(yaw_values) - min(yaw_values):.1f}°")
    
    print(f"\n头部动作事件统计:")
    for action in ['nod', 'nod_up', 'nod_down', 'shake_head', 'turn_left', 'turn_right']:
        count = action_events.get(action, 0)
        if count > 0:
            print(f"  {action}: {count} 帧")
    
    # 诊断建议
    print(f"\n{'='*80}")
    print(f"【诊断建议】")
    print(f"{'='*80}")
    
    pitch_range = max(pitch_values) - min(pitch_values)
    yaw_range = max(yaw_values) - min(yaw_values)
    
    if pitch_range < config.pitch_threshold:
        print(f"\n⚠️  点头问题：Pitch 峰峰值 ({pitch_range:.1f}°) < 阈值 ({config.pitch_threshold}°)")
        print(f"   视频中最大低头：{min(pitch_values):.1f}°，最大抬头：{max(pitch_values):.1f}°")
        print(f"   建议：降低 pitch_threshold 到 {pitch_range * 0.8:.1f}°")
    
    if yaw_range < config.yaw_threshold:
        print(f"\n⚠️  转头问题：Yaw 峰峰值 ({yaw_range:.1f}°) < 阈值 ({config.yaw_threshold}°)")
        print(f"   视频中最大左转：{min(yaw_values):.1f}°，最大右转：{max(yaw_values):.1f}°")
        print(f"   建议：降低 yaw_threshold 到 {yaw_range * 0.8:.1f}°")
    
    if action_events['blink'] / total < 0.5:
        print(f"\n⚠️  眨眼不足：眨眼帧数 ({action_events['blink']/total*100:.1f}%) < 50%")
        print(f"   建议：降低 ear_threshold 到 {min(ear_values) * 1.1:.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分析动作检测失败原因")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("--actions", nargs="+", default=[], help="期望的动作列表")
    args = parser.parse_args()
    
    analyze_action_video(args.video, args.actions)
