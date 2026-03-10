"""
Simulate the FIXED blink/mouth state machines on the CSV data
to verify the fixes work correctly.
"""
import pandas as pd

df = pd.read_csv("output/IMG_3714_liveness.csv")

# ─── Fixed blink state machine ────────────────────────────────────────
# eye_close_threshold = 0.20 (was 0.10)
# eye_open_threshold  = 0.20 (was 0.20)
# max closed frames   = 60   (was 15)

EYE_CLOSE_THRESHOLD = 0.20
EYE_OPEN_THRESHOLD  = 0.20
MAX_CLOSED_FRAMES   = 60

eye_state = "open"
eye_closed_frames = 0
blink_events = []

for _, row in df.iterrows():
    ear = row["ear"]
    if eye_state == "open":
        if ear < EYE_CLOSE_THRESHOLD:
            eye_state = "closed"
            eye_closed_frames = 1
    else:  # closed
        eye_closed_frames += 1
        if ear > EYE_OPEN_THRESHOLD:
            if 1 <= eye_closed_frames <= MAX_CLOSED_FRAMES:
                blink_events.append((int(row["frame_idx"]), round(ear, 4), eye_closed_frames))
            eye_state = "open"
            eye_closed_frames = 0

print("=" * 60)
print(f"FIXED BLINK DETECTION  (threshold={EYE_CLOSE_THRESHOLD}, max_frames={MAX_CLOSED_FRAMES})")
print("=" * 60)
print(f"Total blink events detected: {len(blink_events)}")
for frame, ear_val, dur in blink_events:
    print(f"  frame {frame:4d}  EAR={ear_val:.4f}  closed_for={dur} frames")

print()

# ─── Fixed mouth state machine ─────────────────────────────────────────
# MAR_THRESHOLD      = 0.55  (video_fast_config) or 0.60 (default)
# max open frames    = 150   (was 30)

MAR_THRESHOLD     = 0.60
MAX_OPEN_FRAMES   = 150

mouth_state = "closed"
mouth_open_frames = 0
mouth_events = []

for _, row in df.iterrows():
    mar = row["mar"]
    if mouth_state == "closed":
        if mar > MAR_THRESHOLD:
            mouth_state = "open"
            mouth_open_frames = 1
    else:  # open
        mouth_open_frames += 1
        if mar <= MAR_THRESHOLD:
            if 1 <= mouth_open_frames <= MAX_OPEN_FRAMES:
                mouth_events.append((int(row["frame_idx"]), round(mar, 4), mouth_open_frames))
            mouth_state = "closed"
            mouth_open_frames = 0

print("=" * 60)
print(f"FIXED MOUTH DETECTION  (threshold={MAR_THRESHOLD}, max_frames={MAX_OPEN_FRAMES})")
print("=" * 60)
print(f"Total mouth-open events detected: {len(mouth_events)}")
for frame, mar_val, dur in mouth_events:
    print(f"  frame {frame:4d}  MAR={mar_val:.4f}  open_for={dur} frames")

print()
print("=" * 60)
print("ORIGINAL (broken) vs FIXED comparison")
print("=" * 60)
print(f"  Blink events: {(df['blink_detected'] == True).sum()} → {len(blink_events)}")
print(f"  Mouth events: {(df['mouth_open'] == True).sum()} → {len(mouth_events)}")

