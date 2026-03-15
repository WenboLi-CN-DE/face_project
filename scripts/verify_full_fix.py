"""
Simulate the FULLY FIXED pipeline on the CSV:
  1. eye_close_threshold = 0.20 (was 0.10)
  2. eye max closed frames = 60  (was 15)
  3. mouth max open frames = 150 (was 30)
  4. event score boost when blink/mouth fires

Shows per-frame score and live decision.
"""
import pandas as pd
import numpy as np
from collections import deque

df = pd.read_csv("output/IMG_3714_liveness.csv")

# ── Config ─────────────────────────────────────────────────────────────
EYE_CLOSE_THRESHOLD = 0.20   # fixed (was 0.10)
EYE_OPEN_THRESHOLD  = 0.20
MAX_CLOSED_FRAMES   = 60     # fixed (was 15)
MAR_THRESHOLD       = 0.60
MAX_OPEN_FRAMES     = 150    # fixed (was 30)
LIVENESS_THRESHOLD  = 0.45
SMOOTH_WINDOW       = 5      # video_fast_config smooth_window
SCORE_WINDOW        = 20     # rolling window for std-score
BOOST_FRAMES        = 20
BLINK_BOOST         = 0.85
MOUTH_BOOST         = 0.85

# ── State machines ─────────────────────────────────────────────────────
eye_state = "open"
eye_closed_frames = 0
mouth_state = "closed"
mouth_open_frames = 0

ear_history = deque(maxlen=SCORE_WINDOW)
mar_history = deque(maxlen=SCORE_WINDOW)
yaw_history = deque(maxlen=SCORE_WINDOW)
pitch_history = deque(maxlen=SCORE_WINDOW)
event_history = deque(maxlen=BOOST_FRAMES)
score_history = deque(maxlen=SMOOTH_WINDOW)

blink_count = 0
mouth_count = 0
live_frames = 0
results = []

for _, row in df.iterrows():
    ear   = float(row["ear"])
    mar   = float(row["mar"])
    yaw   = float(row["yaw"])
    pitch = float(row["pitch"])

    # Blink state machine
    blink_detected = False
    if eye_state == "open":
        if ear < EYE_CLOSE_THRESHOLD:
            eye_state = "closed"
            eye_closed_frames = 1
    else:
        eye_closed_frames += 1
        if ear > EYE_OPEN_THRESHOLD:
            if 1 <= eye_closed_frames <= MAX_CLOSED_FRAMES:
                blink_detected = True
                blink_count += 1
            eye_state = "open"
            eye_closed_frames = 0

    # Mouth state machine
    mouth_detected = False
    if mouth_state == "closed":
        if mar > MAR_THRESHOLD:
            mouth_state = "open"
            mouth_open_frames = 1
    else:
        mouth_open_frames += 1
        if mar <= MAR_THRESHOLD:
            if 1 <= mouth_open_frames <= MAX_OPEN_FRAMES:
                mouth_detected = True
                mouth_count += 1
            mouth_state = "closed"
            mouth_open_frames = 0

    # Histories
    ear_history.append(ear)
    mar_history.append(mar)
    yaw_history.append(yaw)
    pitch_history.append(pitch)

    # Event boost injection
    if blink_detected:
        for _ in range(BOOST_FRAMES):
            event_history.append(BLINK_BOOST)
    elif mouth_detected:
        for _ in range(BOOST_FRAMES):
            event_history.append(MOUTH_BOOST)
    else:
        event_history.append(0.0)

    # std-based score
    parts = []
    if len(ear_history) >= 5:
        parts.append(min(float(np.std(list(ear_history))) * 10.0, 1.0))
    else:
        parts.append(0.0)
    if len(mar_history) >= 5:
        parts.append(min(float(np.std(list(mar_history))) * 8.0, 1.0))
    else:
        parts.append(0.0)
    if len(yaw_history) >= 5:
        yr = float(np.array(list(yaw_history)).max() - np.array(list(yaw_history)).min())
        pr = float(np.array(list(pitch_history)).max() - np.array(list(pitch_history)).min())
        parts.append(min((yr + pr) / 30.0, 1.0))
    else:
        parts.append(0.0)
    std_score = float(np.mean(parts))

    # event score
    event_score = float(np.mean(list(event_history))) if event_history else 0.0

    # motion score = max(std, event)
    motion_score = float(np.clip(max(std_score, event_score), 0.0, 1.0))

    # smooth
    score_history.append(motion_score)
    smoothed = float(np.mean(list(score_history)))

    is_live = smoothed > LIVENESS_THRESHOLD
    if is_live:
        live_frames += 1

    results.append({
        "frame": int(row["frame_idx"]),
        "t": float(row["timestamp_s"]),
        "ear": ear,
        "mar": mar,
        "blink": blink_detected,
        "mouth": mouth_detected,
        "std_score": round(std_score, 4),
        "event_score": round(event_score, 4),
        "motion_score": round(motion_score, 4),
        "smoothed": round(smoothed, 4),
        "is_live": is_live,
    })

res_df = pd.DataFrame(results)
print("=" * 60)
print("FULLY FIXED SIMULATION RESULTS")
print("=" * 60)
print(f"Blink events: {blink_count}")
print(f"Mouth events: {mouth_count}")
print(f"is_live=True frames: {live_frames} / {len(df)} ({live_frames/len(df)*100:.1f}%)")
print()
print("Score stats (smoothed):")
print(f"  mean={res_df['smoothed'].mean():.4f}, max={res_df['smoothed'].max():.4f}, min={res_df['smoothed'].min():.4f}")
print()

print("Timeline (every 60 frames):")
print(f"{'frame':>6} {'t':>6} {'EAR':>6} {'MAR':>6} {'blink':>6} {'mouth':>6} {'std':>6} {'event':>6} {'smooth':>7} {'live':>6}")
for i in range(0, len(res_df), 60):
    r = res_df.iloc[i]
    print(f"{r['frame']:6.0f} {r['t']:6.2f} {r['ear']:6.3f} {r['mar']:6.3f} {str(r['blink']):>6} {str(r['mouth']):>6} {r['std_score']:6.3f} {r['event_score']:6.3f} {r['smoothed']:7.4f} {str(r['is_live']):>6}")

# Show frames around blink/mouth events
print()
print("Frames near blink events (+/- 5 frames):")
blink_frames = res_df[res_df['blink'] == True]['frame'].tolist()
for bf in blink_frames:
    nearby = res_df[abs(res_df['frame'] - bf) <= 5]
    print(f"  Blink @ frame {bf}:")
    for _, r in nearby.iterrows():
        marker = " <-- BLINK" if r['blink'] else ""
        print(f"    f{r['frame']:.0f} t={r['t']:.2f} smooth={r['smoothed']:.4f} live={r['is_live']}{marker}")

print()
print("Frames near mouth events (+/- 5 frames):")
mouth_frames = res_df[res_df['mouth'] == True]['frame'].tolist()
for mf in mouth_frames:
    nearby = res_df[abs(res_df['frame'] - mf) <= 5]
    print(f"  Mouth @ frame {mf}:")
    for _, r in nearby.iterrows():
        marker = " <-- MOUTH" if r['mouth'] else ""
        print(f"    f{r['frame']:.0f} t={r['t']:.2f} smooth={r['smoothed']:.4f} live={r['is_live']}{marker}")

print()
print("Comparison: ORIGINAL vs FIXED")
orig_live = (df['is_live'] == True).sum()
print(f"  Original: {orig_live} live frames ({orig_live/len(df)*100:.1f}%)")
print(f"  Fixed:    {live_frames} live frames ({live_frames/len(df)*100:.1f}%)")

