import os
import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

FEATURE_DIR = "./features"

print("Loading GeMAPS features...")
train_df = pd.read_csv(os.path.join(FEATURE_DIR, "gemaps_parselmouth_features.csv")).iloc[:4194]
test_df  = pd.read_csv(os.path.join(FEATURE_DIR, "gemaps_parselmouth_features_test.csv"))

feature_cols = [c for c in train_df.columns if c not in ("segment_id", "time_to_failure")]

X_train = train_df[feature_cols].values
y_train = train_df["time_to_failure"].values
X_test  = test_df[feature_cols].values

print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

print("\n5-fold cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds  = np.zeros(len(y_train))
test_preds = np.zeros(len(X_test))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    t0 = time.time()
    model = XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        device="cuda",
        verbosity=0,
    )
    model.fit(X_train[tr_idx], y_train[tr_idx])
    oof_preds[va_idx] = model.predict(X_train[va_idx])
    test_preds        += model.predict(X_test) / 5
    score = mean_absolute_error(y_train[va_idx], oof_preds[va_idx])
    fold_scores.append(score)
    print(f"  Fold {fold}: MAE = {score:.4f}  ({time.time()-t0:.0f}s)")

print(f"\n  CV MAE: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nTop 10 features by importance:")
print(importance.head(10).to_string(index=False))

preds = np.clip(test_preds, 0, None)
submission = pd.DataFrame({
    "seg_id": "seg_" + test_df["segment_id"].astype(str),
    "time_to_failure": preds,
})
out_path = os.path.join(FEATURE_DIR, "submission_gemaps_xgboost.csv")
submission.to_csv(out_path, index=False)
print(f"\nSaved {len(submission)} predictions to {out_path}")
print(f"Prediction stats: mean={preds.mean():.3f}, std={preds.std():.3f}, "
      f"min={preds.min():.3f}, max={preds.max():.3f}")
