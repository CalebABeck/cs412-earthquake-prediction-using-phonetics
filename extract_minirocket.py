"""
MiniRocket feature extraction for LANL Earthquake Prediction
- Segments train.csv into 150k windows
- Fits MiniRocket on training data and transforms train + test
- Saves features to ./features/
"""

import os
import time
import numpy as np
import pandas as pd
import gc
from sktime.transformations.panel.rocket import MiniRocket

DATA_DIR    = "./kaggle_data"
FEATURE_DIR = "./features"
SEG_LEN     = 150_000
CHUNK_SIZE  = 10_000_000

os.makedirs(FEATURE_DIR, exist_ok=True)
feat_path = os.path.join(FEATURE_DIR, "minirocket_features.npz")

# ── Load train ────────────────────────────────────────────────────────────────
print("Loading train.csv in chunks...")
t0 = time.time()
chunks = []
for i, chunk in enumerate(pd.read_csv(
        os.path.join(DATA_DIR, "train.csv"),
        dtype={"acoustic_data": np.int16, "time_to_failure": np.float32},
        chunksize=CHUNK_SIZE)):
    chunks.append(chunk)
    print(f"  Loaded {(i+1)*CHUNK_SIZE:>12,} rows ... ({time.time()-t0:.0f}s)", flush=True)
train = pd.concat(chunks, ignore_index=True)
del chunks; gc.collect()
print(f"  Total: {len(train):,} rows in {time.time()-t0:.0f}s")

n_segments = len(train) // SEG_LEN
print(f"  Cutting into {n_segments} segments of {SEG_LEN:,} points")

X_train_raw = []
y_train = []
for i in range(n_segments):
    start = i * SEG_LEN
    end   = start + SEG_LEN
    X_train_raw.append(train["acoustic_data"].values[start:end].astype(np.float32))
    y_train.append(train["time_to_failure"].values[end - 1])

X_train_3d = np.array(X_train_raw)[:, np.newaxis, :]   # (N, 1, 150000)
y_train    = np.array(y_train)
print(f"  X_train shape: {X_train_3d.shape}, y_train shape: {y_train.shape}")
del train, X_train_raw; gc.collect()

# ── Load test ─────────────────────────────────────────────────────────────────
print("\nLoading test data...")
t0 = time.time()
sub      = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
seg_ids  = sub["seg_id"].values
test_dir = os.path.join(DATA_DIR, "test")

X_test_raw = np.zeros((len(seg_ids), SEG_LEN), dtype=np.float32)
for i, sid in enumerate(seg_ids):
    seg = pd.read_csv(os.path.join(test_dir, f"{sid}.csv"),
                      dtype={"acoustic_data": np.int16})
    X_test_raw[i] = seg["acoustic_data"].values.astype(np.float32)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(seg_ids)} ... ({time.time()-t0:.0f}s)", flush=True)

X_test_3d = X_test_raw[:, np.newaxis, :]   # (N, 1, 150000)
del X_test_raw; gc.collect()
print(f"  Test shape: {X_test_3d.shape} in {time.time()-t0:.0f}s")

# ── Fit MiniRocket ────────────────────────────────────────────────────────────
print("\nFitting MiniRocket...")
t0 = time.time()
rocket = MiniRocket(random_state=42)
rocket.fit(X_train_3d)
print(f"  Fit done in {time.time()-t0:.0f}s")

# ── Transform ─────────────────────────────────────────────────────────────────
print("Transforming train...")
t0 = time.time()
X_train_feat = np.array(rocket.transform(X_train_3d))
print(f"  Train features shape: {X_train_feat.shape} in {time.time()-t0:.0f}s")
del X_train_3d; gc.collect()

print("Transforming test...")
t0 = time.time()
X_test_feat = np.array(rocket.transform(X_test_3d))
print(f"  Test features shape: {X_test_feat.shape} in {time.time()-t0:.0f}s")
del X_test_3d; gc.collect()

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving features to {feat_path}...")
np.savez_compressed(feat_path,
                    X_train=X_train_feat,
                    y_train=y_train,
                    X_test=X_test_feat,
                    seg_ids=seg_ids)
print(f"Done! Train: {X_train_feat.shape}, Test: {X_test_feat.shape}")
