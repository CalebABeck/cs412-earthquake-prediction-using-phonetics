# cs412-earthquake-prediction-using-phonetics

## Setup
Pack test segments: `python pack_test.py`

## GeMAPS Pipeline
Features: 36 GeMAPS speech features.
Extract: `python extract_gemaps_parselmouth.py`
Models: LightGBM (`gemaps_lightgbm.py`), Random Forest (`gemaps_rf.py`), XGBoost (`gemaps_xgboost.py`), SVR (`gemaps_svr.py`)

## Wavelet + GeMAPS Pipeline
Features: Same 36 features, but extracted from wavelet-denoised signals.
Extract: `python extract_wavelet_gemaps.py 8`
Models: LightGBM (`wavelet_lightgbm.py`), Random Forest (`wavelet_rf.py`), XGBoost (`wavelet_xgboost.py`), SVR (`wavelet_svr.py`)

## MiniRocket Pipeline
Features: 9996 random convolutional kernel features extracted via MiniRocket.
Extract: `python extract_minirocket.py`
Models: LightGBM (`minirocket_lightgbm.py`)
