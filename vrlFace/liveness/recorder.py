"""
活体检测 CSV 录制器

对视频文件逐帧运行活体检测，并将结果保存到 CSV 文件。

使用示例:
    python -m vrlFace.liveness.recorder --video path/to/video.mp4
    python -m vrlFace.liveness.recorder --video path/to/video.mp4 --output result.csv --no-ui
"""

import csv
import time
import threading
import queue
import argparse
from pathlib import Path
from typing import Optional

import cv2

from .config import LivenessConfig
from .fusion_engine import LivenessFusionEngine, LivenessResult
from .fast_detector import FastLivenessDetector
from .utils import build_fast_detector_config, resolve_current_action

# ---------------------------------------------------------------------------
# CSV 列定义
# ---------------------------------------------------------------------------

BASE_COLUMNS = [
    "frame_idx",
    "timestamp_s",
    "face_detected",
    "is_live",
    "score_smoothed",
    "motion_score",
    "confidence",
    "quality_score",
    "reason",
    # action state
    "current_action",
    "blink_detected",
    "blink_active",
    "is_blinking",
    "mouth_open",
    "mouth_active",
    "is_mouth_open",
    "head_action",
    "head_moved",
    # key metrics
    "ear",
    "mar",
    "yaw",
    "pitch",
]

# 关键关键点索引 (MediaPipe 478-point model)
KEY_LANDMARK_INDICES = {
    # Left eye (EAR)
    "leye_0": 133,
    "leye_1": 160,
    "leye_2": 158,
    "leye_3": 33,
    "leye_4": 153,
    "leye_5": 144,
    # Right eye (EAR)
    "reye_0": 362,
    "reye_1": 385,
    "reye_2": 387,
    "reye_3": 263,
    "reye_4": 373,
    "reye_5": 380,
    # Lips (MAR)
    "lip_0": 61,
    "lip_1": 291,
    "lip_2": 39,
    "lip_3": 269,
    "lip_4": 0,
    "lip_5": 17,
    # Head pose
    "nose_tip": 1,
    "left_eye_c": 33,
    "right_eye_c": 263,
    "left_mouth": 61,
    "right_mouth": 291,
    "chin": 152,
}

LANDMARK_COLUMNS: list = []
for _name in KEY_LANDMARK_INDICES:
    LANDMARK_COLUMNS.append(f"lm_{_name}_x")
    LANDMARK_COLUMNS.append(f"lm_{_name}_y")
    LANDMARK_COLUMNS.append(f"lm_{_name}_z")

ALL_COLUMNS = BASE_COLUMNS + LANDMARK_COLUMNS


def _landmark_row(landmarks) -> dict:
    row: dict = {}
    for name, idx in KEY_LANDMARK_INDICES.items():
        try:
            lm = landmarks[idx]
            row[f"lm_{name}_x"] = round(lm.x, 6)
            row[f"lm_{name}_y"] = round(lm.y, 6)
            row[f"lm_{name}_z"] = round(lm.z, 6)
        except (IndexError, AttributeError):
            row[f"lm_{name}_x"] = ""
            row[f"lm_{name}_y"] = ""
            row[f"lm_{name}_z"] = ""
    return row


def _empty_landmark_row() -> dict:
    return {
        f"lm_{name}_{axis}": ""
        for name in KEY_LANDMARK_INDICES
        for axis in ("x", "y", "z")
    }


# ---------------------------------------------------------------------------
# 主检测 + CSV 写入函数
# ---------------------------------------------------------------------------


def run_video_detection_with_csv(
    video_path: str,
    output_csv: str,
    config: Optional[LivenessConfig] = None,
    show_ui: bool = True,
    fast_mode: bool = True,
) -> None:
    """对视频文件逐帧运行活体检测，并将结果写入 CSV。"""
    print("=" * 60)
    print("Liveness Detection + CSV recorder")
    print("=" * 60)
    print(f"  Video : {video_path}")
    print(f"  Output: {output_csv}")
    print()

    # --- 配置 ---
    if config is None:
        config = (
            LivenessConfig.video_fast_config()
            if fast_mode
            else LivenessConfig.video_anti_spoofing_config()
        )

    engine = LivenessFusionEngine(config)
    fast_detector = FastLivenessDetector(**build_fast_detector_config(config))

    print(f"  EAR threshold  : {fast_detector.EAR_THRESHOLD:.3f}")
    print(f"  MAR threshold  : {fast_detector.MAR_THRESHOLD:.3f}")
    print(f"  Yaw threshold  : {fast_detector.YAW_THRESHOLD:.1f} deg")
    print(f"  Pitch threshold: {fast_detector.PITCH_THRESHOLD:.1f} deg")
    print()

    # --- 打开视频 ---
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {video_path}")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # WebM 文件可能返回无效的帧数，通过实际读取计算
    if total_frames <= 0:
        print(
            f"  WARNING: Invalid frame count ({total_frames}), calculating actual frames..."
        )
        frame_count = 0
        while True:
            ret = cap.grab()
            if not ret:
                break
            frame_count += 1
        total_frames = frame_count
        print(f"  Actual frames: {total_frames}")

        # 重新打开视频
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"ERROR: cannot reopen video: {video_path}")
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / source_fps if source_fps > 0 else 0

    print(f"  Resolution: {vid_w}x{vid_h}")
    print(f"  FPS       : {source_fps:.1f}")
    print(f"  Frames    : {total_frames}")
    print(f"  Duration  : {duration:.1f} s")
    print()

    # --- 解码线程 ---
    _max_w: int = config.max_width if config.max_width > 0 else 0
    frame_queue: queue.Queue = queue.Queue(maxsize=2)
    decode_stop = threading.Event()

    def _decode_worker() -> None:
        while not decode_stop.is_set():
            ret, frm = cap.read()
            if not ret:
                frame_queue.put(None)
                break
            if _max_w > 0:
                h0, w0 = frm.shape[:2]
                if w0 > _max_w:
                    frm = cv2.resize(
                        frm,
                        (_max_w, int(h0 * _max_w / w0)),
                        interpolation=cv2.INTER_LINEAR,
                    )
            try:
                frame_queue.put(frm, timeout=0.5)
            except queue.Full:
                pass

    decode_thread = threading.Thread(target=_decode_worker, daemon=True)
    decode_thread.start()

    # --- 显示窗口 ---
    MAX_DISP_W, MAX_DISP_H = 1280, 720
    scale = min(MAX_DISP_W / max(vid_w, 1), MAX_DISP_H / max(vid_h, 1), 1.0)
    disp_w = max(1, int(vid_w * scale))
    disp_h = max(1, int(vid_h * scale))
    if show_ui:
        cv2.namedWindow("Liveness CSV Mode", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Liveness CSV Mode", disp_w, disp_h)
        print("  Press ESC to stop early")
        print()

    # --- 打开 CSV ---
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = out_path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(csv_file, fieldnames=ALL_COLUMNS, extrasaction="ignore")
    writer.writeheader()

    # --- 主循环 ---
    frame_idx = 0
    start_time = time.time()
    last_result: Optional[LivenessResult] = None
    rows_written = 0
    best_score = 0.0
    best_reason = "动作不足"

    print("Processing...\n")

    while True:
        try:
            frame = frame_queue.get(timeout=2.0)
        except queue.Empty:
            frame = None

        if frame is None:
            print("\nVideo finished.")
            break

        frame_idx += 1
        timestamp_s = round((frame_idx - 1) / source_fps, 4)

        # --- 推理 ---
        lm_data = engine.mp_detector.extract_landmarks(frame)

        if lm_data is None:
            row: dict = {col: "" for col in ALL_COLUMNS}
            row.update(
                {
                    "frame_idx": frame_idx,
                    "timestamp_s": timestamp_s,
                    "face_detected": False,
                    "is_live": False,
                    "score_smoothed": 0.0,
                    "motion_score": 0.0,
                    "confidence": 0.0,
                    "quality_score": 0.0,
                    "reason": "NO_FACE_DETECTED",
                    "current_action": "none",
                    "blink_detected": False,
                    "blink_active": False,
                    "is_blinking": False,
                    "mouth_open": False,
                    "mouth_active": False,
                    "is_mouth_open": False,
                    "head_action": "none",
                    "head_moved": False,
                    "ear": 0.0,
                    "mar": 0.0,
                    "yaw": 0.0,
                    "pitch": 0.0,
                }
            )
            row.update(_empty_landmark_row())
        else:
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
            result = LivenessResult(
                is_live=is_live,
                score=smoothed,
                confidence=engine._calculate_confidence(smoothed),
                quality_score=quality_score,
                motion_score=motion_score,
                temporal_score=0.0,
                details=details,
                reason=engine._determine_reason(is_live, details, True),
            )
            last_result = result

            if smoothed > best_score:
                best_score = smoothed
                best_reason = result.reason

            row = {
                "frame_idx": frame_idx,
                "timestamp_s": timestamp_s,
                "face_detected": True,
                "is_live": is_live,
                "score_smoothed": round(smoothed, 6),
                "motion_score": round(motion_score, 6),
                "confidence": round(result.confidence, 6),
                "quality_score": round(quality_score, 6),
                "reason": result.reason,
                "current_action": current_action,
                "blink_detected": fd_result["blink_detected"],
                "blink_active": fd_result.get("blink_active", False),
                "is_blinking": fd_result["is_blinking"],
                "mouth_open": fd_result["mouth_open"],
                "mouth_active": fd_result.get("mouth_active", False),
                "is_mouth_open": fd_result["is_mouth_open"],
                "head_action": fd_result["head_action"],
                "head_moved": fd_result["head_action"] != "none",
                "ear": round(fd_result["ear"], 6),
                "mar": round(fd_result["mar"], 6),
                "yaw": round(fd_result["yaw"], 4),
                "pitch": round(fd_result["pitch"], 4),
            }
            row.update(_landmark_row(landmarks))

        writer.writerow(row)
        rows_written += 1

        # --- UI ---
        if show_ui:
            h_f, w_f = frame.shape[:2]
            disp = (
                cv2.resize(frame, (disp_w, disp_h))
                if (w_f != disp_w or h_f != disp_h)
                else frame.copy()
            )
            face_ok = bool(row.get("face_detected", False))
            if face_ok:
                is_live_disp = row.get("is_live", False)
                score_disp = float(row.get("score_smoothed", 0.0))
                ear_disp = float(row.get("ear", 0.0))
                mar_disp = float(row.get("mar", 0.0))
                yaw_disp = float(row.get("yaw", 0.0))
                pitch_disp = float(row.get("pitch", 0.0))
                action_disp = str(row.get("current_action", "none"))

                color = (0, 255, 0) if is_live_disp else (0, 0, 255)
                label = "LIVE" if is_live_disp else "SPOOF"
                cv2.putText(
                    disp,
                    f"{label}  score={score_disp:.3f}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                for i, ln in enumerate(
                    [
                        f"EAR:{ear_disp:.3f}  MAR:{mar_disp:.3f}",
                        f"Yaw:{yaw_disp:+.1f}  Pitch:{pitch_disp:+.1f}",
                        f"Action: {action_disp}",
                    ]
                ):
                    cv2.putText(
                        disp,
                        ln,
                        (10, 56 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            else:
                cv2.putText(
                    disp,
                    "NO FACE",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (128, 128, 128),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                disp,
                f"Frame {frame_idx}/{total_frames}",
                (10, disp_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Liveness CSV Mode", disp)
            if (cv2.waitKey(1) & 0xFF) == 27:
                print("\nUser stopped early.")
                break

        # 进度
        elapsed = time.time() - start_time
        if frame_idx % max(1, int(source_fps)) == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            fps_proc = frame_idx / max(elapsed, 0.001)
            print(
                f"\r  Progress: {pct:.1f}%  Frame: {frame_idx}/{total_frames}"
                f"  FPS: {fps_proc:.1f}",
                end="",
                flush=True,
            )

    # --- 清理 ---
    decode_stop.set()
    decode_thread.join(timeout=2.0)
    cap.release()
    if show_ui:
        cv2.destroyAllWindows()
    engine.close()
    csv_file.close()

    total_time = time.time() - start_time
    print(f"\n\n{'=' * 60}")
    print("Done")
    print(f"  Total time  : {total_time:.1f} s")
    print(f"  Frames      : {frame_idx}")
    print(f"  CSV rows    : {rows_written}")
    print(f"  Output file : {out_path.resolve()}")
    print(f"{'=' * 60}")

    if last_result:
        final_live = best_score > engine.config.threshold
        verdict = "LIVE ✅" if final_live else "SPOOF ❌"
        print(f"\nFinal verdict : {verdict}")
        print(
            f"Best score    : {best_score:.4f}  (threshold={engine.config.threshold:.2f})"
        )
        print(f"Reason        : {best_reason}")

    _print_summary(out_path)


# ---------------------------------------------------------------------------
# 摘要统计
# ---------------------------------------------------------------------------


def _print_summary(csv_path: Path) -> None:
    """打印 CSV 关键字段统计，用于快速诊断。"""
    try:
        rows: list = []
        with csv_path.open("r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append(r)

        if not rows:
            return

        total = len(rows)
        face_rows = [
            r for r in rows if str(r.get("face_detected", "")).lower() == "true"
        ]
        live_rows = [
            r for r in face_rows if str(r.get("is_live", "")).lower() == "true"
        ]

        print(f"\n{'=' * 60}")
        print("Per-frame statistics summary")
        print(f"{'=' * 60}")
        print(f"  Total frames       : {total}")
        print(
            f"  Frames with face   : {len(face_rows)} ({len(face_rows) / max(total, 1):.1%})"
        )
        print(
            f"  Frames LIVE        : {len(live_rows)} ({len(live_rows) / max(total, 1):.1%})"
        )

        def _floats(rows_in: list, col: str) -> list:
            vals = []
            for r in rows_in:
                try:
                    vals.append(float(r[col]))
                except (ValueError, KeyError):
                    pass
            return vals

        for col, label in [
            ("ear", "EAR          "),
            ("mar", "MAR          "),
            ("yaw", "Yaw (deg)    "),
            ("pitch", "Pitch (deg)  "),
            ("score_smoothed", "Score smooth "),
            ("quality_score", "Quality score"),
        ]:
            vals = _floats(face_rows, col)
            if vals:
                print(
                    f"  {label}: min={min(vals):.4f}  max={max(vals):.4f}"
                    f"  mean={sum(vals) / len(vals):.4f}"
                )

        action_counts: dict = {}
        for r in face_rows:
            a = str(r.get("current_action", "none"))
            action_counts[a] = action_counts.get(a, 0) + 1
        print("\n  Action distribution:")
        for action, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
            pct = cnt / max(len(face_rows), 1)
            print(f"    {action:<22}: {cnt:5d} frames ({pct:.1%})")

        def _rising_edges(rows_in: list, col: str) -> int:
            count, prev = 0, False
            for r in rows_in:
                cur = str(r.get(col, "")).lower() == "true"
                if cur and not prev:
                    count += 1
                prev = cur
            return count

        print(f"\n  Event trigger counts (rising edge):")
        print(f"    blink_detected : {_rising_edges(face_rows, 'blink_detected')}")
        print(f"    mouth_active   : {_rising_edges(face_rows, 'mouth_active')}")
        print(f"    head_moved     : {_rising_edges(face_rows, 'head_moved')}")
        print(f"{'=' * 60}\n")

    except Exception as exc:
        print(f"[summary] Failed to read CSV: {exc}")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="活体检测 CSV 录制器",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--video", "-v", required=True, help="视频文件路径")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="输出 CSV 路径（默认：output/<视频名>_liveness.csv）",
    )
    parser.add_argument(
        "--config",
        choices=["fast", "accurate", "video-anti", "realtime"],
        default="realtime",
        help="配置预设（默认：realtime）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="活体分数阈值（覆盖配置默认值）",
    )
    parser.add_argument(
        "--no-ui", action="store_true", help="无 UI 模式（无 OpenCV 窗口）"
    )
    args = parser.parse_args()

    if args.output:
        output_csv = args.output
    else:
        video_stem = Path(args.video).stem
        output_csv = str(Path("output") / f"{video_stem}_liveness.csv")

    config_map = {
        "fast": LivenessConfig.cpu_fast_config,
        "accurate": LivenessConfig.cpu_accurate_config,
        "video-anti": LivenessConfig.video_anti_spoofing_config,
        "realtime": LivenessConfig.realtime_config,
    }
    cfg = config_map[args.config]()
    if args.threshold is not None:
        cfg.threshold = args.threshold

    run_video_detection_with_csv(
        video_path=args.video,
        output_csv=output_csv,
        config=cfg,
        show_ui=not args.no_ui,
    )


if __name__ == "__main__":
    main()
