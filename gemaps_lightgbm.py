import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

FEATURE_DIR = "./features"
SUBMISSION_PATH = "./features/submission_gemaps_lightgbm.csv"

print("Loading features...")
train_df = pd.read_csv(os.path.join(FEATURE_DIR, "gemaps_parselmouth_features.csv")).iloc[:4194]
test_df  = pd.read_csv(os.path.join(FEATURE_DIR, "gemaps_parselmouth_features_test.csv"))

feature_cols = [c for c in train_df.columns if c not in ("segment_id", "time_to_failure")]

X_train = train_df[feature_cols].values
y_train = train_df["time_to_failure"].values
X_test  = test_df[feature_cols].values

print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

params = {
    "objective": "fair",
    "fair_c": 1.0,
    "metric": "mae",
    "learning_rate": 0.02,
    "num_leaves": 16,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "device": "gpu",
    "verbose": -1,
    "n_jobs": -1,
}
NUM_BOOST_ROUND = 1_000_000
EARLY_STOPPING  = 1000

print("\n5-fold cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds  = np.zeros(len(y_train))
test_preds = np.zeros(len(X_test))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    t0 = time.time()
    dtrain = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx])
    dval   = lgb.Dataset(X_train[va_idx], label=y_train[va_idx], reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    oof_preds[va_idx] = model.predict(X_train[va_idx])
    test_preds        += model.predict(X_test) / 5
    score = mean_absolute_error(y_train[va_idx], oof_preds[va_idx])
    fold_scores.append(score)
    print(f"  Fold {fold}: MAE = {score:.4f}  (best iter = {model.best_iteration}, {time.time()-t0:.0f}s)")

print(f"\n  CV MAE: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)
print("\nTop 10 features by gain:")
print(importance.head(10).to_string(index=False))

preds = np.clip(test_preds, 0, None)
submission = pd.DataFrame({
    "seg_id": "seg_" + test_df["segment_id"].astype(str),
    "time_to_failure": preds,
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSaved {len(submission)} predictions to {SUBMISSION_PATH}")
print(f"Prediction stats: mean={preds.mean():.3f}, std={preds.std():.3f}, "
      f"min={preds.min():.3f}, max={preds.max():.3f}")
