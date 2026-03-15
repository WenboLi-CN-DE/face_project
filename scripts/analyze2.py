import pandas as pd, numpy as np
df = pd.read_csv('output/IMG_3714_liveness.csv')

true_min = df[df['is_live']==True]['score_smoothed'].min()
false_max = df[df['is_live']==False]['score_smoothed'].max()
print(f'Threshold: is_live=True when score_smoothed > {true_min:.4f}')
print(f'Max score where is_live=False: {false_max:.4f}')
print()

print('Score timeline (every 60 frames):')
for i in range(0, len(df), 60):
    row = df.iloc[i]
    print(f"  t={row['timestamp_s']:.1f}s  score={row['score_smoothed']:.3f}  live={row['is_live']}  reason={row['reason']}")

print()
print('Live/NotLive breakdown:')
print(f"  is_live=True  frames: {(df['is_live']==True).sum()} ({(df['is_live']==True).sum()/len(df)*100:.1f}%)")
print(f"  is_live=False frames: {(df['is_live']==False).sum()} ({(df['is_live']==False).sum()/len(df)*100:.1f}%)")
print()

# Understand what drives score up
print('Correlation with motion_score:')
for col in ['ear','mar','yaw','pitch']:
    c = df[col].corr(df['motion_score'])
    print(f"  {col}: {c:.3f}")
print()

# motion_score components: std(EAR)*10, std(MAR)*8, (yaw_range+pitch_range)/30 all averaged
# The problem: when person is still or MAR/EAR is stable, score stays low
# Check rolling std
ear_std = df['ear'].rolling(20).std().fillna(0)
mar_std = df['mar'].rolling(20).std().fillna(0)
print('EAR rolling-20 std:')
print(f"  mean={ear_std.mean():.4f}, max={ear_std.max():.4f}")
print(f"  => EAR contribution mean={(ear_std*10).clip(0,1).mean():.4f}")
print()
print('MAR rolling-20 std:')
print(f"  mean={mar_std.mean():.4f}, max={mar_std.max():.4f}")
print(f"  => MAR contribution mean={(mar_std*8).clip(0,1).mean():.4f}")

