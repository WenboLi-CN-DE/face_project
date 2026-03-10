"""Analyze IMG_3714_liveness.csv for blink/mouth detection issues."""
import pandas as pd
import numpy as np

df = pd.read_csv('output/IMG_3714_liveness.csv')
ear = df['ear']
mar = df['mar']

print("=" * 60)
print("EAR (Eye Aspect Ratio) Stats")
print("=" * 60)
print(f"count={len(ear)}, mean={ear.mean():.4f}, std={ear.std():.4f}")
print(f"min={ear.min():.4f}, max={ear.max():.4f}")
print()

print("EAR < thresholds:")
for t in [0.10, 0.15, 0.20, 0.25]:
    print(f"  < {t}: {(ear < t).sum()} frames")
print()

# EAR segments below 0.20
below = (ear < 0.20).values
segments = []
in_seg = False
start = 0
for i, v in enumerate(below):
    if v and not in_seg:
        in_seg = True
        start = i
    elif not v and in_seg:
        in_seg = False
        segments.append((start, i - 1))
if in_seg:
    segments.append((start, len(below) - 1))

print(f"EAR < 0.20 consecutive segments: {len(segments)}")
for s, e in segments:
    bd = bool(df.iloc[s : e + 1]["blink_detected"].any())
    ib = bool(df.iloc[s : e + 1]["is_blinking"].any())
    mn = round(float(ear.iloc[s : e + 1].min()), 4)
    f1 = int(df.iloc[s]["frame_idx"])
    f2 = int(df.iloc[e]["frame_idx"])
    print(f"  frames {f1}-{f2} ({e-s+1} frames), EAR_min={mn}, blink_detected={bd}, is_blinking={ib}")

print()
print("=" * 60)
print("MAR (Mouth Aspect Ratio) Stats")
print("=" * 60)
print(f"count={len(mar)}, mean={mar.mean():.4f}, std={mar.std():.4f}")
print(f"min={mar.min():.4f}, max={mar.max():.4f}")
print()

print("MAR > thresholds:")
for t in [0.40, 0.50, 0.60, 0.70, 0.80]:
    print(f"  > {t}: {(mar > t).sum()} frames")
print()

# MAR segments above 0.60
above = (mar > 0.60).values
segs_m = []
in_seg = False
for i, v in enumerate(above):
    if v and not in_seg:
        in_seg = True
        start = i
    elif not v and in_seg:
        in_seg = False
        segs_m.append((start, i - 1))
if in_seg:
    segs_m.append((start, len(above) - 1))

print(f"MAR > 0.60 consecutive segments: {len(segs_m)}")
for s, e in segs_m:
    mo = bool(df.iloc[s : e + 1]["mouth_open"].any())
    im = bool(df.iloc[s : e + 1]["is_mouth_open"].any())
    mx = round(float(mar.iloc[s : e + 1].max()), 4)
    f1 = int(df.iloc[s]["frame_idx"])
    f2 = int(df.iloc[e]["frame_idx"])
    print(f"  frames {f1}-{f2} ({e-s+1} frames), MAR_max={mx}, mouth_open={mo}, is_mouth_open={im}")

print()
print("=" * 60)
print("Detection Summary")
print("=" * 60)
print(f"blink_detected=True  : {(df['blink_detected'] == True).sum()} frames")
print(f"blink_active=True    : {(df['blink_active'] == True).sum()} frames")
print(f"is_blinking=True     : {(df['is_blinking'] == True).sum()} frames")
print(f"mouth_open=True      : {(df['mouth_open'] == True).sum()} frames")
print(f"mouth_active=True    : {(df['mouth_active'] == True).sum()} frames")
print(f"is_mouth_open=True   : {(df['is_mouth_open'] == True).sum()} frames")

print()
print("blink_detected=True frames:")
bdf = df[df["blink_detected"] == True][
    ["frame_idx", "timestamp_s", "ear", "is_blinking", "blink_active"]
]
print(bdf.to_string())

print()
print("mouth_open=True frames:")
mdf = df[df["mouth_open"] == True][
    ["frame_idx", "timestamp_s", "mar", "is_mouth_open", "mouth_active"]
]
print(mdf.to_string())

print()
print("=" * 60)
print("Root Cause Analysis")
print("=" * 60)

print("""
BLINK DETECTION ISSUE:
- EAR threshold = 0.20 (config default)
- fast_detector uses: EYE_CLOSE_THRESHOLD=0.10, EYE_OPEN_THRESHOLD=0.20
  (because save_csv.py passes eye_close_threshold=0.10)
- EAR range in video: {:.4f} ~ {:.4f}
- EAR < 0.10 frames: {} (very few true closures detected at strict threshold)
- EAR < 0.20 frames: {} (visible dips but EYE_CLOSE_THRESHOLD=0.10 misses them)

The blink state machine in fast_detector uses:
  ear < EYE_CLOSE_THRESHOLD(0.10) -> transition to "closed"  <-- too strict!
  ear > EYE_OPEN_THRESHOLD(0.20)  -> transition to "open"

With EAR never going below 0.10 in most frames, blink "closed" state is rarely entered.

MOUTH DETECTION ISSUE:
- MAR threshold = 0.60 (config/fast_detector default)
- is_mouth_open=True frames: {} (MAR > 0.60 is reached)
- mouth_open=True frames: {} (but edge transition is missed)
- Max MAR in video: {:.4f}

The mouth state machine requires: closed -> open (MAR > 0.60) -> closed = event
The segments where MAR > 0.60 are {} frames long (see above).
If the video ENDS while mouth is still open, the "closed" transition never fires,
so mouth_open edge event is never recorded.
""".format(
    ear.min(), ear.max(),
    (ear < 0.10).sum(),
    (ear < 0.20).sum(),
    (df['is_mouth_open'] == True).sum(),
    (df['mouth_open'] == True).sum(),
    mar.max(),
    len(segs_m),
))

