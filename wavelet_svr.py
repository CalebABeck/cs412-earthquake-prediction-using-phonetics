import os
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

FEATURE_DIR = "./features"

print("Loading wavelet GeMAPS features...")
train_df = pd.read_csv(os.path.join(FEATURE_DIR, "wavelet_gemaps_features.csv")).iloc[:4194]
test_df  = pd.read_csv(os.path.join(FEATURE_DIR, "wavelet_gemaps_features_test.csv"))

feature_cols = [c for c in train_df.columns if c not in ("segment_id", "time_to_failure")]

X_train = train_df[feature_cols].values
y_train = train_df["time_to_failure"].values
X_test  = test_df[feature_cols].values

print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n5-fold cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds  = np.zeros(len(y_train))
test_preds = np.zeros(len(X_test))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_scaled)):
    t0 = time.time()
    model = SVR(kernel="rbf", C=10.0, epsilon=0.1)
    model.fit(X_train_scaled[tr_idx], y_train[tr_idx])
    oof_preds[va_idx] = model.predict(X_train_scaled[va_idx])
    test_preds        += model.predict(X_test_scaled) / 5
    score = mean_absolute_error(y_train[va_idx], oof_preds[va_idx])
    fold_scores.append(score)
    print(f"  Fold {fold}: MAE = {score:.4f}  ({time.time()-t0:.0f}s)")

print(f"\n  CV MAE: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

print("\nComputing permutation importance (last fold)...")
perm = permutation_importance(
    model, X_train_scaled[va_idx], y_train[va_idx],
    n_repeats=10, scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
)
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": perm.importances_mean,
}).sort_values("importance", ascending=False)
print("Top 10 features by permutation importance:")
print(importance.head(10).to_string(index=False))

preds = np.clip(test_preds, 0, None)
submission = pd.DataFrame({
    "seg_id": "seg_" + test_df["segment_id"].astype(str),
    "time_to_failure": preds,
})
out_path = os.path.join(FEATURE_DIR, "submission_wavelet_svr.csv")
submission.to_csv(out_path, index=False)
print(f"\nSaved {len(submission)} predictions to {out_path}")
print(f"Prediction stats: mean={preds.mean():.3f}, std={preds.std():.3f}, "
      f"min={preds.min():.3f}, max={preds.max():.3f}")
