"""
活体检测命令行工具

支持摄像头实时检测和视频文件检测。

使用示例:
    python -m vrlFace.liveness.cli --camera 0
    python -m vrlFace.liveness.cli --video path/to/video.mp4
    python -m vrlFace.liveness.cli --video path/to/video.mp4 --config video-anti
"""

import cv2
import argparse
import time
import threading
import queue
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .config import LivenessConfig
from .fusion_engine import LivenessFusionEngine, LivenessResult
from .fast_detector import FastLivenessDetector
from .utils import build_fast_detector_config, resolve_current_action


# ---------------------------------------------------------------------------
# 摄像头实时检测
# ---------------------------------------------------------------------------


def run_camera_detection(
    camera_id: int = 0,
    config: Optional[LivenessConfig] = None,
    show_ui: bool = True,
    use_fast_detector: bool = True,
):
    """摄像头实时活体检测"""
    print("=" * 60)
    print("活体检测 - 摄像头模式")
    print("=" * 60)

    if config is None:
        config = LivenessConfig.video_anti_spoofing_config()

    print(f"\n配置:")
    print(f"  活体阈值：{config.threshold}")
    print(f"  跳帧设置：每 {config.skip_frames + 1} 帧检测 1 次")
    print(f"  时间窗口：{config.window_size} 帧")
    print()

    engine = LivenessFusionEngine(config)

    fast_detector = None
    if use_fast_detector:
        fast_detector = FastLivenessDetector(**build_fast_detector_config(config))
        print("✅ 使用快速检测器参与推理")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_id}")
        return

    print("✅ 摄像头已打开")
    print("提示:")
    print("  - 请面对摄像头，自然眨眼、说话、轻微转头")
    print("  - 按 'ESC' 退出  'R' 重置  'L' 开关关键点")
    print()

    last_result = None
    show_landmarks = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面")
            break

        if use_fast_detector and fast_detector is not None:
            small_frame = engine._resize_for_inference(frame)
            lm_data = engine.mp_detector.extract_landmarks(small_frame)

            if lm_data is None:
                result = _make_no_face_result()
            else:
                result = _run_fast_inference(
                    engine, fast_detector, lm_data, small_frame
                )
        else:
            result = engine.process_frame(frame)

        last_result = result

        if show_ui:
            frame = draw_result(
                frame,
                result,
                engine.mp_detector.fps,
                show_landmarks=show_landmarks,
                fast_detector=fast_detector,
            )
            cv2.imshow("Liveness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\n用户退出")
            break
        elif key == ord("r") or key == ord("R"):
            engine.reset()
            if fast_detector:
                fast_detector.reset()
            print("\n重置检测")
        elif key == ord("l") or key == ord("L"):
            show_landmarks = not show_landmarks
            print(f"\n关键点显示：{'ON' if show_landmarks else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    engine.close()

    if last_result:
        _print_final_result(last_result)


# ---------------------------------------------------------------------------
# 视频文件检测
# ---------------------------------------------------------------------------


def run_video_detection(
    video_path: str,
    config: Optional[LivenessConfig] = None,
    show_ui: bool = True,
    fast_mode: bool = True,
    use_fast_detector: bool = True,
):
    """视频文件活体检测"""
    print("=" * 60)
    print("活体检测 - 视频文件模式")
    print("=" * 60)

    if config is None:
        config = (
            LivenessConfig.video_fast_config()
            if fast_mode
            else LivenessConfig.video_anti_spoofing_config()
        )

    engine = LivenessFusionEngine(config)

    fast_detector = None
    if use_fast_detector:
        fast_detector = FastLivenessDetector(**build_fast_detector_config(config))
        print("\n✅ 使用快速检测器")

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{video_path}")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # WebM 文件可能返回无效的帧数，通过实际读取计算
    if total_frames <= 0:
        print(f"⚠️  视频帧数无效，正在计算真实帧数...")
        frame_count = 0
        while True:
            ret = cap.grab()
            if not ret:
                break
            frame_count += 1
        total_frames = frame_count
        print(f"✓ 实际帧数: {total_frames}")

        # 重新打开视频
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"❌ 重新打开视频失败")
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n视频信息: {vid_w}x{vid_h} @ {source_fps:.1f}fps, 共 {total_frames} 帧")
    print("按 'ESC' 退出  'SPACE' 暂停  'L' 开关关键点\n")

    # 解码线程
    _max_w = config.max_width if config.max_width > 0 else 0
    frame_queue: queue.Queue = queue.Queue(maxsize=2)
    decode_stop = threading.Event()

    def _decode_worker():
        while not decode_stop.is_set():
            ret, frm = cap.read()
            if not ret:
                frame_queue.put(None)
                break
            if _max_w > 0:
                h0, w0 = frm.shape[:2]
                if w0 > _max_w:
                    frm = cv2.resize(frm, (_max_w, int(h0 * _max_w / w0)))
            try:
                frame_queue.put(frm, timeout=0.5)
            except queue.Full:
                pass

    decode_thread = threading.Thread(target=_decode_worker, daemon=True)
    decode_thread.start()

    MAX_DISP_W, MAX_DISP_H = 1280, 720
    scale = min(MAX_DISP_W / max(vid_w, 1), MAX_DISP_H / max(vid_h, 1), 1.0)
    disp_w, disp_h = max(1, int(vid_w * scale)), max(1, int(vid_h * scale))

    if show_ui:
        cv2.namedWindow("Liveness Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Liveness Detection", disp_w, disp_h)

    frame_idx = 0
    paused = False
    last_result = None
    frame = None
    disp_frame = None
    show_landmarks = True
    start_time = time.time()
    video_frame_ms = max(1, int(1000.0 / source_fps))

    while True:
        if not paused:
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                frame = None

            if frame is None:
                print("\n视频播放完成")
                break

            frame_idx += 1

            if use_fast_detector and fast_detector is not None:
                lm_data = engine.mp_detector.extract_landmarks(frame)
                result = (
                    _run_fast_inference(engine, fast_detector, lm_data, frame)
                    if lm_data is not None
                    else _make_no_face_result()
                )
            else:
                result = engine.process_frame(frame)

            last_result = result

            if show_ui:
                h_f, w_f = frame.shape[:2]
                base = (
                    cv2.resize(frame, (disp_w, disp_h))
                    if (w_f != disp_w or h_f != disp_h)
                    else frame.copy()
                )
                disp_frame = draw_result(
                    base,
                    result,
                    0,
                    show_landmarks=show_landmarks,
                    fast_detector=fast_detector,
                )
                cv2.putText(
                    disp_frame,
                    f"Frame: {frame_idx}/{total_frames}",
                    (10, disp_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        if show_ui and disp_frame is not None:
            cv2.imshow("Liveness Detection", disp_frame)

        elapsed_ms = 0
        wait_ms = max(1, video_frame_ms - elapsed_ms)
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == 27:
            print("\n用户退出")
            break
        elif key == ord(" "):
            paused = not paused
            print(f"\n{'暂停' if paused else '继续'}")
        elif key == ord("r") or key == ord("R"):
            engine.reset()
        elif key == ord("l") or key == ord("L"):
            show_landmarks = not show_landmarks

        elapsed = time.time() - start_time
        if frame_idx % max(1, int(source_fps)) == 0:
            fps_proc = frame_idx / max(elapsed, 0.001)
            print(
                f"\r帧: {frame_idx}/{total_frames}  推理FPS: {fps_proc:.1f}",
                end="",
                flush=True,
            )

    decode_stop.set()
    decode_thread.join(timeout=2.0)
    cap.release()
    cv2.destroyAllWindows()
    engine.close()

    if last_result:
        _print_final_result(last_result)


# ---------------------------------------------------------------------------
# 共享推理逻辑
# ---------------------------------------------------------------------------


def _make_no_face_result() -> LivenessResult:
    return LivenessResult(
        is_live=False,
        score=0.0,
        confidence=0.0,
        quality_score=0.0,
        motion_score=0.0,
        temporal_score=0.0,
        details={
            "motion": {"face_detected": False},
            "current_action": "none",
            "is_blinking": False,
            "is_mouth_open": False,
            "yaw": 0.0,
        },
        reason="NO_FACE_DETECTED",
    )


def _run_fast_inference(engine, fast_detector, lm_data, frame) -> LivenessResult:
    """执行单状态机推理（mp_detector 仅提取 landmark，fast_detector 负责所有状态机）"""
    landmarks = lm_data["landmarks"]
    quality_score = lm_data["quality_score"]
    fd_result = fast_detector.detect_liveness(
        landmarks, lm_data.get("frame_shape", frame.shape)
    )
    current_action = resolve_current_action(fd_result)

    motion_score = fd_result["score"]
    engine.score_history.append(motion_score)
    smoothed = float(
        sum(list(engine.score_history)[-engine.config.smooth_window :])
        / min(len(engine.score_history), engine.config.smooth_window)
    )
    is_live = smoothed > engine.config.threshold

    motion_dict = {
        "face_detected": True,
        "score": motion_score,
        "ear": fd_result["ear"],
        "mar": fd_result["mar"],
        "yaw": fd_result["yaw"],
        "pitch": fd_result["pitch"],
        "blink_detected": fd_result["blink_detected"],
        "blink_active": fd_result.get("blink_active", False),
        "mouth_moved": fd_result["mouth_open"],
        "mouth_active": fd_result.get("mouth_active", False),
        "head_moved": fd_result["head_action"] != "none",
        "current_action": current_action,
        "is_blinking": fd_result["is_blinking"],
        "is_mouth_open": fd_result["is_mouth_open"],
        "landmarks": landmarks,
        "quality_score": quality_score,
    }
    details = {
        "motion": motion_dict,
        "current_action": current_action,
        "is_blinking": fd_result["is_blinking"],
        "is_mouth_open": fd_result["is_mouth_open"],
        "yaw": fd_result["yaw"],
        "pitch": fd_result["pitch"],
    }
    return LivenessResult(
        is_live=is_live,
        score=smoothed,
        confidence=engine._calculate_confidence(smoothed),
        quality_score=quality_score,
        motion_score=motion_score,
        temporal_score=0.0,
        details=details,
        reason=engine._determine_reason(is_live, details, True),
    )


def _print_final_result(result: LivenessResult):
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"活体判定：{'✅ 是' if result.is_live else '❌ 否'}")
    print(f"置信度：{result.confidence:.2%}")
    print(f"原因：{result.reason}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# UI 绘制
# ---------------------------------------------------------------------------

_ACTION_MAP = {
    "none": "无动作",
    "eye_open": "睁眼",
    "blinking": "眨眼",
    "mouth_open": "张嘴",
    "head_turn_left": "左摇",
    "head_turn_right": "右摇",
    "head_nod_down": "低头",
    "head_nod_up": "抬头",
}
_ACTION_ICON_MAP = {
    "none": "",
    "eye_open": "👀",
    "blinking": "👁",
    "mouth_open": "👄",
    "head_turn_left": "⬅",
    "head_turn_right": "➡",
    "head_nod_down": "⬇",
    "head_nod_up": "⬆",
}
_PIL_CACHE: dict = {"key": None, "patch": None, "ph": 0, "pw": 0}


def _render_action_patch(icon: str, text: str) -> np.ndarray:
    content = f"{icon} {text}" if icon else text
    for fp in [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ]:
        try:
            font = ImageFont.truetype(fp, 20)
            break
        except IOError:
            font = None
    if font is None:
        font = ImageFont.load_default()
    dummy = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), content, font=font)
    tw, th = bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4
    img = Image.new("RGB", (int(tw), int(th)), (0, 0, 0))
    ImageDraw.Draw(img).text((2, 2), content, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_result(
    frame: np.ndarray,
    result: LivenessResult,
    mp_fps: float = 0.0,
    show_landmarks: bool = False,
    fast_detector=None,
) -> np.ndarray:
    """在帧上叠加活体检测结果 UI"""
    h, w = frame.shape[:2]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    PAD, LINE_H = 8, 24
    PANEL_X = 10

    details = result.details
    motion_info = details.get("motion", {})
    current_action = details.get("current_action", "none")
    yaw = details.get("yaw", 0.0)
    pitch = details.get("pitch", 0.0)
    ear_val = motion_info.get("ear", 0.0)
    mar_val = motion_info.get("mar", 0.0)

    # 关键点
    if show_landmarks:
        landmarks = motion_info.get("landmarks")
        if landmarks is not None:
            for lm in landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (80, 80, 80), -1)
            for indices, color in [
                ([133, 160, 158, 33, 153, 144], (0, 255, 255)),
                ([362, 385, 387, 263, 373, 380], (0, 255, 255)),
                ([61, 291, 39, 269, 0, 17], (0, 165, 255)),
            ]:
                pts = [
                    (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices
                ]
                for k in range(len(pts)):
                    cv2.line(frame, pts[k], pts[(k + 1) % len(pts)], color, 1)
                for pt in pts:
                    cv2.circle(frame, pt, 2, color, -1)

    # 主状态文字
    status_color = (0, 255, 0) if result.is_live else (0, 0, 255)
    cv2.putText(
        frame,
        f"{'LIVE' if result.is_live else 'SPOOF'}  score={result.score:.2f}",
        (PANEL_X, 30),
        FONT,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # 信息面板
    ear_thr = fast_detector.EAR_THRESHOLD if fast_detector else 0.20
    mar_thr = fast_detector.MAR_THRESHOLD if fast_detector else 0.55
    yaw_thr = fast_detector.YAW_THRESHOLD if fast_detector else 25.0
    pitch_thr = fast_detector.PITCH_THRESHOLD if fast_detector else 15.0

    lines = [
        (f"Confidence : {result.confidence:.2%}", (255, 255, 255)),
        (f"Quality    : {result.quality_score:.2f}", (255, 255, 255)),
        (f"Motion     : {result.motion_score:.2f}", (255, 255, 255)),
        (
            f"EAR        : {ear_val:.3f}",
            (80, 80, 255) if ear_val < ear_thr else (200, 255, 200),
        ),
        (
            f"MAR        : {mar_val:.3f}",
            (50, 180, 255) if mar_val > mar_thr else (200, 255, 200),
        ),
        (
            f"Yaw        : {yaw:+.1f} deg",
            (0, 220, 255) if abs(yaw) > yaw_thr else (255, 255, 255),
        ),
        (
            f"Pitch      : {pitch:+.1f} deg",
            (0, 220, 255) if abs(pitch) > pitch_thr else (255, 255, 255),
        ),
        (f"FPS        : {mp_fps:.1f}", (255, 255, 255)),
        ("[L]landmark [R]reset [ESC]quit", (0, 220, 255)),
    ]

    panel_y = 40
    panel_w = min(
        max(cv2.getTextSize(ln, FONT, 0.52, 1)[0][0] for ln, _ in lines) + PAD * 2,
        w - PANEL_X - 2,
    )
    panel_h = PAD + (len(lines) + 1) * LINE_H + PAD

    cv2.rectangle(
        frame, (PANEL_X, panel_y), (PANEL_X + panel_w, panel_y + panel_h), (0, 0, 0), -1
    )
    cv2.rectangle(
        frame,
        (PANEL_X, panel_y),
        (PANEL_X + panel_w, panel_y + panel_h),
        (200, 200, 200),
        1,
    )

    y = panel_y + PAD + LINE_H - 4
    for text, color in lines:
        cv2.putText(frame, text, (PANEL_X + PAD, y), FONT, 0.52, color, 1, cv2.LINE_AA)
        y += LINE_H

    # Action 行（PIL 渲染中文）
    action_text = _ACTION_MAP.get(current_action, current_action)
    action_icon = _ACTION_ICON_MAP.get(current_action, "")
    prefix = "Action     : "
    cv2.putText(
        frame,
        prefix,
        (PANEL_X + PAD, y - LINE_H + 4),
        FONT,
        0.52,
        (0, 255, 128),
        1,
        cv2.LINE_AA,
    )

    cache_key = (action_icon, action_text)
    if _PIL_CACHE["key"] != cache_key:
        patch = _render_action_patch(action_icon, action_text)
        _PIL_CACHE.update(
            {
                "key": cache_key,
                "patch": patch,
                "ph": patch.shape[0],
                "pw": patch.shape[1],
            }
        )

    patch = _PIL_CACHE["patch"]
    ph, pw = _PIL_CACHE["ph"], _PIL_CACHE["pw"]
    prefix_w = cv2.getTextSize(prefix, FONT, 0.52, 1)[0][0]
    px0 = PANEL_X + PAD + prefix_w
    py0 = y - LINE_H + 4 - ph + 4
    px1 = min(w, min(PANEL_X + panel_w, px0 + pw))
    py1 = min(h, py0 + ph)
    if px0 >= 0 and py0 >= 0 and px1 > px0 and py1 > py0:
        roi = frame[py0:py1, px0:px1]
        patch_crop = patch[: py1 - py0, : px1 - px0]
        mask = np.any(patch_crop > 10, axis=2, keepdims=True)
        np.copyto(roi, patch_crop, where=mask)

    return frame


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="活体检测命令行工具")
    parser.add_argument(
        "--camera", "-c", type=int, default=0, help="摄像头 ID（默认：0）"
    )
    parser.add_argument("--video", "-v", type=str, default=None, help="视频文件路径")
    parser.add_argument(
        "--config",
        choices=["fast", "accurate", "video-anti", "realtime"],
        default="realtime",
        help="配置模式（默认：realtime）",
    )
    parser.add_argument("--no-ui", action="store_true", help="不显示 UI 界面")
    parser.add_argument("--threshold", type=float, default=None, help="活体阈值")
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="禁用基准帧校准（防替换攻击）",
    )
    args = parser.parse_args()

    config_map = {
        "fast": LivenessConfig.cpu_fast_config,
        "accurate": LivenessConfig.cpu_accurate_config,
        "video-anti": LivenessConfig.video_anti_spoofing_config,
        "realtime": LivenessConfig.realtime_config,
    }
    cfg = config_map[args.config]()
    if args.threshold is not None:
        cfg.threshold = args.threshold
    if args.no_benchmark:
        cfg.enable_benchmark = False

    if args.video:
        run_video_detection(args.video, cfg, show_ui=not args.no_ui)
    else:
        run_camera_detection(args.camera, cfg, show_ui=not args.no_ui)


if __name__ == "__main__":
    main()
