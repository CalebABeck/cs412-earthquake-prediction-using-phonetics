"""
Catch22 + SVR Baseline for LANL Earthquake Prediction
- Segments train.csv into 150k windows
- Extracts 22 canonical time series features via pycatch22
- Trains SVR with RBF kernel
"""

import os
import time
import numpy as np
import pandas as pd
import pycatch22
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

DATA_DIR = "/work/hdd/bfgc/yzhang62/kaggle"
SEG_LEN = 150_000

DOWNSAMPLE = 10  # 150k -> 15k points

def extract_catch22(segment):
    """Extract 22 features from a single segment (downsampled)."""
    ds = segment.reshape(-1, DOWNSAMPLE).mean(axis=1)
    result = pycatch22.catch22_all(ds.tolist())
    return result["values"]


feat_path = os.path.join(DATA_DIR, "catch22_features.npz")

if os.path.exists(feat_path):
    print(f"Loading cached features from {feat_path} ...")
    cached = np.load(feat_path)
    X_train_feat = cached["X_train_feat"]
    X_test_feat = cached["X_test_feat"]
    y_train = cached["y_train"]
    print(f"  Train: {X_train_feat.shape}, Test: {X_test_feat.shape}")
else:
    print("Loading train.csv in chunks ...")
    t0 = time.time()
    chunks = []
    CHUNK_SIZE = 10_000_000
    for i, chunk in enumerate(pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                                           dtype={"acoustic_data": np.int16, "time_to_failure": np.float32},
                                           chunksize=CHUNK_SIZE)):
        chunks.append(chunk)
        print(f"  Loaded {(i+1)*CHUNK_SIZE:>12,} rows ... ({time.time()-t0:.0f}s)", flush=True)
    train = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"  Total: {len(train):,} rows in {time.time()-t0:.0f}s")

    n_segments = len(train) // SEG_LEN
    print(f"  Cutting into {n_segments} segments of {SEG_LEN:,} points")

    acoustic = train["acoustic_data"].values.astype(np.float64)
    ttf = train["time_to_failure"].values

    y_train = np.array([ttf[((i+1)*SEG_LEN) - 1] for i in range(n_segments)])

    # Extract catch22 features for train
    print(f"\nExtracting Catch22 features for {n_segments} train segments ...")
    t0 = time.time()
    X_train_feat = np.zeros((n_segments, 22))
    for i in range(n_segments):
        seg = acoustic[i*SEG_LEN : (i+1)*SEG_LEN]
        X_train_feat[i] = extract_catch22(seg)
        if (i + 1) % 100 == 0:
            print(f"  Train: {i+1}/{n_segments} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  Train features done in {time.time()-t0:.0f}s")

    del train, acoustic, ttf
    import gc; gc.collect()

    # Extract catch22 features for test
    print("\nLoading test data ...")
    t0 = time.time()
    npz_path = os.path.join(DATA_DIR, "test_data.npz")
    data = np.load(npz_path)
    X_test_raw = data["X_test"].astype(np.float64)
    print(f"  Loaded test in {time.time()-t0:.0f}s")

    n_test = X_test_raw.shape[0]
    print(f"Extracting Catch22 features for {n_test} test segments ...")
    t0 = time.time()
    X_test_feat = np.zeros((n_test, 22))
    for i in range(n_test):
        X_test_feat[i] = extract_catch22(X_test_raw[i])
        if (i + 1) % 500 == 0:
            print(f"  Test: {i+1}/{n_test} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  Test features done in {time.time()-t0:.0f}s")

    del X_test_raw; gc.collect()

    print(f"Saving features to {feat_path} ...")
    np.savez(feat_path, X_train_feat=X_train_feat, X_test_feat=X_test_feat, y_train=y_train)
    print("  Saved!")


print("\nCross-validation (5-fold) ...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_train))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_scaled)):
    t0 = time.time()
    model = SVR(kernel="rbf", C=10.0, epsilon=0.1)
    model.fit(X_train_scaled[tr_idx], y_train[tr_idx])
    oof_preds[va_idx] = model.predict(X_train_scaled[va_idx])
    score = mean_absolute_error(y_train[va_idx], oof_preds[va_idx])
    fold_scores.append(score)
    print(f"  Fold {fold}: MAE = {score:.4f} ({time.time()-t0:.0f}s)")

print(f"  Overall CV MAE: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")


print("\nTraining final model on all data ...")
final_model = SVR(kernel="rbf", C=10.0, epsilon=0.1)
final_model.fit(X_train_scaled, y_train)

preds = final_model.predict(X_test_scaled)
preds = np.clip(preds, 0, None)

sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
sub["time_to_failure"] = preds
out_path = os.path.join(DATA_DIR, "submission_catch22_svr.csv")
sub.to_csv(out_path, index=False)
print(f"\nDone! Submission saved to {out_path}")
print(f"Prediction stats: mean={preds.mean():.3f}, std={preds.std():.3f}, "
      f"min={preds.min():.3f}, max={preds.max():.3f}")
