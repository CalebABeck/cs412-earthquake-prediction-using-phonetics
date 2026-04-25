"""
MiniRocket Baseline for LANL Earthquake Prediction
- Segments train.csv into 150k windows
- Extracts features using MiniRocket (random convolutional kernels)
- Trains Ridge regression
- Predicts time_to_failure for test segments
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sktime.transformations.panel.rocket import MiniRocket

DATA_DIR = "./kaggle_data"
SEG_LEN = 150_000


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

X_train_raw = []
y_train = []
for i in range(n_segments):
    start = i * SEG_LEN
    end = start + SEG_LEN
    X_train_raw.append(train["acoustic_data"].values[start:end].astype(np.float32))
    y_train.append(train["time_to_failure"].values[end - 1])

X_train_3d = np.array(X_train_raw)[:, np.newaxis, :]
y_train = np.array(y_train)
print(f"  X_train shape: {X_train_3d.shape}, y_train shape: {y_train.shape}")

del train, X_train_raw
import gc; gc.collect()


print("\nLoading test data ...")
t0 = time.time()
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
npz_path = os.path.join(DATA_DIR, "test_data.npz")
data = np.load(npz_path)
X_test_3d = data["X_test"].astype(np.float32)[:, np.newaxis, :]
print(f"  Loaded test from npz in {time.time()-t0:.0f}s, shape: {X_test_3d.shape}")


print("\nFitting MiniRocket ...")
t0 = time.time()
rocket = MiniRocket(random_state=42)
rocket.fit(X_train_3d)
print(f"  Fit done in {time.time()-t0:.0f}s")

feat_path = os.path.join(DATA_DIR, "minirocket_features.npz")
if os.path.exists(feat_path):
    print(f"  Loading cached features from {feat_path} ...")
    cached = np.load(feat_path)
    X_train_feat = cached["X_train_feat"]
    X_test_feat = cached["X_test_feat"]
    print(f"  Train: {X_train_feat.shape}, Test: {X_test_feat.shape}")
    del X_train_3d, X_test_3d; gc.collect()
else:
    print("Transforming train ...")
    t0 = time.time()
    X_train_feat = np.array(rocket.transform(X_train_3d))
    print(f"  Train transform: {X_train_feat.shape} in {time.time()-t0:.0f}s")

    print("Transforming test ...")
    t0 = time.time()
    X_test_feat = np.array(rocket.transform(X_test_3d))
    print(f"  Test transform: {X_test_feat.shape} in {time.time()-t0:.0f}s")

    del X_train_3d, X_test_3d; gc.collect()

    print(f"  Saving features to {feat_path} ...")
    np.savez(feat_path, X_train_feat=X_train_feat, X_test_feat=X_test_feat)
    print("  Saved!")


print("\nLoading Parselmouth features ...")
gemaps_path = os.path.join(DATA_DIR, "gemaps_parselmouth_features.npz")
using_gemaps = False
if os.path.exists(gemaps_path):
    gemaps_data = np.load(gemaps_path)
    gemaps_dict = {key: gemaps_data[key] for key in gemaps_data.files}

    feature_names = sorted([k for k in gemaps_dict.keys() 
                           if k not in ['segment_id', 'time_to_failure']])
    
    X_train_gemaps = np.column_stack([gemaps_dict[name] for name in feature_names])
    print(f"  Feature names: {feature_names}")

    n_minirocket_segs = X_train_feat.shape[0]
    n_gemaps_segs = X_train_gemaps.shape[0]
    if n_minirocket_segs != n_gemaps_segs:
        print(f"  Warning: Segment count mismatch (MiniRocket: {n_minirocket_segs}, GeMaPS: {n_gemaps_segs})")
        min_segs = min(n_minirocket_segs, n_gemaps_segs)
        X_train_feat = X_train_feat[:min_segs]
        X_train_gemaps = X_train_gemaps[:min_segs]
        y_train = y_train[:min_segs]
        print(f"  Using first {min_segs} segments for both.")
        
    X_train_feat = np.hstack([X_train_feat, X_train_gemaps])
    print(f"  Combined train features shape: {X_train_feat.shape}")
    using_gemaps = True


print("\nCross-validation (5-fold) ...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_train))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_feat)):
    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    model.fit(X_train_feat[tr_idx], y_train[tr_idx])
    oof_preds[va_idx] = model.predict(X_train_feat[va_idx])
    score = mean_absolute_error(y_train[va_idx], oof_preds[va_idx])
    fold_scores.append(score)
    print(f"  Fold {fold}: MAE = {score:.4f}, alpha = {model.alpha_:.2f}")

print(f"  Overall CV MAE: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")


print("\nTraining final model on all data ...")
final_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
final_model.fit(X_train_feat, y_train)

gemaps_test_cache = os.path.join(DATA_DIR, "gemaps_parselmouth_features_test.npz")
if using_gemaps:
    if os.path.exists(gemaps_test_cache):
        print(f"  Loading cached test GeMaPS features from {gemaps_test_cache} ...")
        t0 = time.time()
        test_gemaps_data = np.load(gemaps_test_cache)
        gemaps_test_dict = {key: test_gemaps_data[key] for key in test_gemaps_data.files}
        feature_names = sorted([k for k in gemaps_test_dict.keys() 
                               if k not in ['segment_id']])
        X_test_gemaps = np.column_stack([gemaps_test_dict[name] for name in feature_names])
        print(f"  Loaded test GeMaPS features: {X_test_gemaps.shape} in {time.time()-t0:.0f}s")

        if X_test_gemaps.shape[0] != X_test_feat.shape[0]:
            min_segments = min(X_test_gemaps.shape[0], X_test_feat.shape[0])
            X_test_feat = X_test_feat[:min_segments]
            X_test_gemaps = X_test_gemaps[:min_segments]
            print(f"  Using first {min_segments} test segments.")
        
        X_test_feat = np.hstack([X_test_feat, X_test_gemaps])
        print(f"  Combined test features shape: {X_test_feat.shape}")

preds = final_model.predict(X_test_feat)
preds = np.clip(preds, 0, None)  

sub["time_to_failure"] = preds
out_path = os.path.join(DATA_DIR, "submission_minirocket.csv")
sub.to_csv(out_path, index=False)
print(f"\nDone! Submission saved to {out_path}")
print(f"Prediction stats: mean={preds.mean():.3f}, std={preds.std():.3f}, "
      f"min={preds.min():.3f}, max={preds.max():.3f}")
