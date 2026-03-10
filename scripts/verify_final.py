"""Final verification: simulate FULLY FIXED pipeline with threshold=0.35."""
import pandas as pd, numpy as np
from collections import deque

df = pd.read_csv("output/IMG_3714_liveness.csv")

EYE_CLOSE_THRESHOLD = 0.20
EYE_OPEN_THRESHOLD  = 0.20
MAX_CLOSED_FRAMES   = 60
MAR_THRESHOLD       = 0.55   # realtime_config / video_fast_config
MAX_OPEN_FRAMES     = 150
LIVENESS_THRESHOLD  = 0.35   # lowered from 0.45
SMOOTH_WINDOW       = 5
SCORE_WINDOW        = 20
BOOST_FRAMES        = 20
BLINK_BOOST         = 0.85
MOUTH_BOOST         = 0.85

eye_state = "open"; eye_closed_frames = 0
mouth_state = "closed"; mouth_open_frames = 0
ear_h = deque(maxlen=SCORE_WINDOW); mar_h = deque(maxlen=SCORE_WINDOW)
yaw_h = deque(maxlen=SCORE_WINDOW); pitch_h = deque(maxlen=SCORE_WINDOW)
event_h = deque(maxlen=BOOST_FRAMES)
score_h = deque(maxlen=SMOOTH_WINDOW)

blink_count = mouth_count = live_frames = 0
best_score = 0.0; best_reason = ""
timeline = []

for _, row in df.iterrows():
    ear = float(row["ear"]); mar = float(row["mar"])
    yaw = float(row["yaw"]); pitch = float(row["pitch"])

    # Blink SM
    blink_det = False
    if eye_state == "open":
        if ear < EYE_CLOSE_THRESHOLD:
            eye_state = "closed"; eye_closed_frames = 1
    else:
        eye_closed_frames += 1
        if ear > EYE_OPEN_THRESHOLD:
            if 1 <= eye_closed_frames <= MAX_CLOSED_FRAMES:
                blink_det = True; blink_count += 1
            eye_state = "open"; eye_closed_frames = 0

    # Mouth SM
    mouth_det = False
    if mouth_state == "closed":
        if mar > MAR_THRESHOLD:
            mouth_state = "open"; mouth_open_frames = 1
    else:
        mouth_open_frames += 1
        if mar <= MAR_THRESHOLD:
            if 1 <= mouth_open_frames <= MAX_OPEN_FRAMES:
                mouth_det = True; mouth_count += 1
            mouth_state = "closed"; mouth_open_frames = 0

    ear_h.append(ear); mar_h.append(mar)
    yaw_h.append(yaw); pitch_h.append(pitch)

    # Event boost
    if blink_det:
        for _ in range(BOOST_FRAMES): event_h.append(BLINK_BOOST)
    elif mouth_det:
        for _ in range(BOOST_FRAMES): event_h.append(MOUTH_BOOST)
    else:
        event_h.append(0.0)

    # std score
    parts = []
    parts.append(min(float(np.std(list(ear_h)))*10,1.0) if len(ear_h)>=5 else 0.0)
    parts.append(min(float(np.std(list(mar_h)))*8, 1.0) if len(mar_h)>=5 else 0.0)
    if len(yaw_h)>=5:
        yr=float(np.array(list(yaw_h)).max()-np.array(list(yaw_h)).min())
        pr=float(np.array(list(pitch_h)).max()-np.array(list(pitch_h)).min())
        parts.append(min((yr+pr)/30,1.0))
    else:
        parts.append(0.0)
    std_score = float(np.mean(parts))
    event_score = float(np.mean(list(event_h))) if event_h else 0.0
    motion_score = float(np.clip(max(std_score, event_score), 0, 1))

    score_h.append(motion_score)
    smoothed = float(np.mean(list(score_h)))
    is_live = smoothed > LIVENESS_THRESHOLD
    if is_live: live_frames += 1
    if smoothed > best_score:
        best_score = smoothed
        tags = []
        if blink_det: tags.append("眨眼")
        if mouth_det: tags.append("张嘴")
        if not tags: tags.append("运动")
        best_reason = "活体：" + "、".join(tags)

    timeline.append((int(row["frame_idx"]), float(row["timestamp_s"]),
                     round(smoothed,4), is_live, blink_det, mouth_det))

print("="*60)
print("COMPLETE FIX SIMULATION")
print(f"  eye_close_threshold : {EYE_CLOSE_THRESHOLD}  (was 0.10)")
print(f"  max_closed_frames   : {MAX_CLOSED_FRAMES}   (was 15)")
print(f"  max_open_frames     : {MAX_OPEN_FRAMES}  (was 30)")
print(f"  liveness_threshold  : {LIVENESS_THRESHOLD}  (was 0.45)")
print(f"  event_boost         : {BLINK_BOOST} for {BOOST_FRAMES} frames")
print("="*60)
print(f"Blink events detected : {blink_count}  (was 3)")
print(f"Mouth events detected : {mouth_count}  (was 1)")
print(f"is_live=True frames   : {live_frames} / {len(df)} ({live_frames/len(df)*100:.1f}%)")
print(f"Best smoothed score   : {best_score:.4f}")
print()
final_live = best_score > LIVENESS_THRESHOLD
print(f"FINAL VERDICT: {'✅ LIVE' if final_live else '❌ SPOOF'}")
print(f"  best_score={best_score:.4f} > threshold={LIVENESS_THRESHOLD}  => {final_live}")
print(f"  reason: {best_reason}")
print()
print("Timeline (every 60 frames):")
print(f"{'frame':>6} {'t(s)':>6} {'smooth':>7} {'live':>6} {'blink':>6} {'mouth':>6}")
for fr, t, sm, lv, bl, mo in timeline[::60]:
    print(f"{fr:6d} {t:6.2f} {sm:7.4f} {str(lv):>6} {str(bl):>6} {str(mo):>6}")

