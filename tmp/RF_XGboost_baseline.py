import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import os
import warnings
warnings.filterwarnings('ignore')

SEGMENT_SIZE = 150_000
TEST_DIR = '/data/server2/jl126/CS412/test'
TRAIN_PATH = '/data/server2/jl126/CS412/train.csv'
SUBMISSION_PATH = '/data/server2/jl126/CS412/submission.csv'
XGBOOST_SUBMISSION_PATH = '/data/server2/jl126/CS412/XGBoost_submission.csv'
TOP_K_FREQ = 20


def extract_features(signal: np.ndarray) -> dict:
    features = {}
    x = signal.astype(np.float64)

    # Time-domain stats
    features['mean'] = x.mean()
    features['std'] = x.std()
    features['min'] = x.min()
    features['max'] = x.max()
    features['abs_mean'] = np.abs(x).mean()
    features['mad'] = np.abs(x - x.mean()).mean()
    features['p05'] = np.percentile(x, 5)
    features['p25'] = np.percentile(x, 25)
    features['p75'] = np.percentile(x, 75)
    features['p95'] = np.percentile(x, 95)
    features['range'] = features['max'] - features['min']
    features['iqr'] = features['p75'] - features['p25']
    features['zero_crossings'] = ((x[:-1] * x[1:]) < 0).sum()

    # FFT features
    fft_vals = np.abs(rfft(x))
    freqs = rfftfreq(len(x))

    total_energy = (fft_vals ** 2).sum()
    features['fft_energy'] = total_energy

    # Spectral centroid
    features['spectral_centroid'] = (freqs * fft_vals).sum() / (fft_vals.sum() + 1e-10)

    # Spectral rolloff (95% energy)
    cumulative = np.cumsum(fft_vals ** 2)
    rolloff_idx = np.searchsorted(cumulative, 0.95 * total_energy)
    features['spectral_rolloff'] = freqs[rolloff_idx] if rolloff_idx < len(freqs) else freqs[-1]

    # Energy in 3 frequency bands 
    band_size = len(fft_vals) // 3
    features['energy_low'] = (fft_vals[:band_size] ** 2).sum()
    features['energy_mid'] = (fft_vals[band_size:2*band_size] ** 2).sum()
    features['energy_high'] = (fft_vals[2*band_size:] ** 2).sum()

    # Top-20 dominant frequency peaks
    top_idx = np.argsort(fft_vals)[-TOP_K_FREQ:][::-1]
    for rank, idx in enumerate(top_idx):
        features[f'top_freq_{rank}'] = freqs[idx]
        features[f'top_amp_{rank}'] = fft_vals[idx]

    return features


# Training features
print("Extracting training features...")
train_rows = []
chunk_iter = pd.read_csv(TRAIN_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64},
                         chunksize=SEGMENT_SIZE)

for i, chunk in enumerate(chunk_iter):
    if len(chunk) < SEGMENT_SIZE:
        break  # drop last incomplete segment
    signal = chunk['acoustic_data'].values
    label = chunk['time_to_failure'].iloc[-1]
    feats = extract_features(signal)
    feats['time_to_failure'] = label
    train_rows.append(feats)
    if (i + 1) % 100 == 0:
        print(f"  processed {i+1} training segments...")

train_df = pd.DataFrame(train_rows)
print(f"Training segments: {len(train_df)}")

feature_cols = [c for c in train_df.columns if c != 'time_to_failure']
X_train = train_df[feature_cols].values
y_train = train_df['time_to_failure'].values


# Train Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_features='sqrt',
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
rf.fit(X_train, y_train)

train_preds = rf.predict(X_train)
print(f"RF train MAE: {mean_absolute_error(y_train, train_preds):.4f}s")


#Train XGBoost
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=1,
)
xgb.fit(X_train, y_train)

xgb_train_preds = xgb.predict(X_train)
print(f"XGBoost Train MAE: {mean_absolute_error(y_train, xgb_train_preds):.4f}s")


# Extract test features & predict
print("\nExtracting test features and predicting...")
test_files = sorted(os.listdir(TEST_DIR))
results = []

xgb_results = []

for fname in test_files:
    seg_id = fname.replace('.csv', '')
    signal = pd.read_csv(os.path.join(TEST_DIR, fname),
                         dtype={'acoustic_data': np.int16})['acoustic_data'].values
    feats = extract_features(signal)
    feat_vec = np.array([feats[c] for c in feature_cols]).reshape(1, -1)
    rf_pred = rf.predict(feat_vec)[0]
    xgb_pred = xgb.predict(feat_vec)[0]
    results.append({'seg_id': seg_id, 'time_to_failure': rf_pred})
    xgb_results.append({'seg_id': seg_id, 'time_to_failure': xgb_pred})

submission = pd.DataFrame(results)
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved {len(submission)} predictions to {SUBMISSION_PATH}")
print(submission.describe())

xgb_submission = pd.DataFrame(xgb_results)
xgb_submission.to_csv(XGBOOST_SUBMISSION_PATH, index=False)
print(f"\nSaved {len(xgb_submission)} XGBoost predictions to {XGBOOST_SUBMISSION_PATH}")
print(xgb_submission.describe())
