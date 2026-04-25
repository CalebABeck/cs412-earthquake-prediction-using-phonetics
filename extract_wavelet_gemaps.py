import os
import sys
import numpy as np
import pandas as pd
import pywt
from multiprocessing import Pool
from tqdm import tqdm
from extract_gemaps_parselmouth import extract_gemaps_features, SAMPLE_RATE, SEG_LEN

DATA_DIR   = "./kaggle_data"
FEATURE_DIR = "./features"
WAVELET    = "db4"
THRESHOLD  = 10.0


def wavelet_denoise(signal):
    coeffs = pywt.wavedec(signal, WAVELET)
    coeffs = [pywt.threshold(c, THRESHOLD, mode="hard") for c in coeffs]
    denoised = pywt.waverec(coeffs, WAVELET)
    return denoised[:len(signal)]


def _process_train_segment(args):
    segment_index, signal, label = args
    signal = wavelet_denoise(signal)
    features = extract_gemaps_features(signal, sample_rate=SAMPLE_RATE)
    return segment_index, label, features


def _process_test_segment(args):
    segment_index, filename, test_dir = args
    df = pd.read_csv(os.path.join(test_dir, filename), dtype={"acoustic_data": np.int16})
    signal = wavelet_denoise(df["acoustic_data"].values.astype(np.float64))
    features = extract_gemaps_features(signal, sample_rate=SAMPLE_RATE)
    segment_id = filename.replace("seg_", "").replace(".csv", "")
    return segment_id, features


def _train_chunk_generator(csv_path):
    for i, chunk in enumerate(pd.read_csv(
            csv_path,
            dtype={"acoustic_data": np.int16, "time_to_failure": np.float32},
            chunksize=SEG_LEN)):
        signal = chunk["acoustic_data"].values.astype(np.float64)
        label  = float(chunk["time_to_failure"].iloc[-1])
        yield (i, signal, label)


def main(num_workers=4):
    train_csv = os.path.join(DATA_DIR, "train.csv")
    total_rows = 629_145_480
    num_segments = (total_rows + SEG_LEN - 1) // SEG_LEN
    print(f"Extracting wavelet-denoised GeMAPS features ({num_segments} train segments, {num_workers} workers)...")

    segment_features = []
    segment_labels   = []
    segment_ids      = []

    with Pool(processes=num_workers) as pool:
        for segment_index, label, features in tqdm(
                pool.imap(_process_train_segment, _train_chunk_generator(train_csv), chunksize=1),
                total=num_segments, desc="Train"):
            segment_features.append(features)
            segment_labels.append(label)
            segment_ids.append(segment_index)

    df = pd.DataFrame(segment_features)
    df.insert(0, "segment_id", segment_ids)
    df.insert(1, "time_to_failure", segment_labels)

    out_csv = os.path.join(FEATURE_DIR, "wavelet_gemaps_features.csv")
    out_npz = os.path.join(FEATURE_DIR, "wavelet_gemaps_features.npz")
    df.to_csv(out_csv, index=False)
    np.savez_compressed(out_npz, **{col: df[col].values for col in df.columns})
    print(f"Saved {len(df)} train segments to {out_csv}")

    print("\nExtracting test features...")
    test_dir   = os.path.join(DATA_DIR, "test")
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".csv")])
    print(f"Found {len(test_files)} test segments")
    work_args = [(idx, fname, test_dir) for idx, fname in enumerate(test_files)]

    seg_ids      = []
    seg_features = []
    with Pool(processes=num_workers) as pool:
        for seg_id, features in tqdm(
                pool.imap_unordered(_process_test_segment, work_args),
                total=len(test_files), desc="Test"):
            seg_ids.append(seg_id)
            seg_features.append(features)

    sorted_pairs = sorted(zip(seg_ids, seg_features))
    seg_ids, seg_features = zip(*sorted_pairs)

    df_test = pd.DataFrame(seg_features)
    df_test.insert(0, "segment_id", list(seg_ids))

    out_csv_test = os.path.join(FEATURE_DIR, "wavelet_gemaps_features_test.csv")
    out_npz_test = os.path.join(FEATURE_DIR, "wavelet_gemaps_features_test.npz")
    df_test.to_csv(out_csv_test, index=False)
    np.savez_compressed(out_npz_test, **{col: df_test[col].values for col in df_test.columns})
    print(f"Saved {len(df_test)} test segments to {out_csv_test}")


if __name__ == "__main__":
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(num_workers)
