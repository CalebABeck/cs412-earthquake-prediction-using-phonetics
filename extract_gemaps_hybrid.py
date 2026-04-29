import os
import sys
import numpy as np
import pandas as pd
import opensmile
import tempfile
from multiprocessing import Pool

# Import parselmouth extraction function
from extract_gemaps_parselmouth import extract_gemaps_features

DATA_DIR = "./kaggle_data"
SEG_LEN = 150_000
SAMPLE_RATE = 4000
OUTPUT_NPZ = os.path.join(DATA_DIR, "gemaps_hybrid_features.npz")
OUTPUT_CSV = os.path.join(DATA_DIR, "gemaps_hybrid_features.csv")

# Initialize OpenSmile extractor for eGeMAPS
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPS,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# List of OpenSmile features to extract (18 specific features)
OPENSMILE_FEATURES = [
    'loudness_sma3_percentile20.0',
    'loudness_sma3_percentile50.0',
    'loudness_sma3_percentile80.0',
    'loudness_sma3_pctlrange0-2',
    'loudness_sma3_meanRisingSlope',
    'loudness_sma3_stddevRisingSlope',
    'loudness_sma3_meanFallingSlope',
    'loudness_sma3_stddevFallingSlope',
    'spectralFlux_sma3_amean',
    'spectralFlux_sma3_stddevNorm',
    'mfcc1_sma3_amean',
    'mfcc1_sma3_stddevNorm',
    'mfcc2_sma3_amean',
    'mfcc2_sma3_stddevNorm',
    'mfcc3_sma3_amean',
    'mfcc3_sma3_stddevNorm',
    'mfcc4_sma3_amean',
    'mfcc4_sma3_stddevNorm',
]


def extract_opensmile_features(signal, sample_rate=SAMPLE_RATE):
    """
    Extract specific OpenSmile features.
    
    Args:
        signal: numpy array of audio samples
        sample_rate: sampling rate in Hz
    
    Returns:
        dict: Only the 18 requested OpenSmile features
    """
    signal = np.asarray(signal, dtype=np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        import soundfile as sf
        sf.write(tmp_path, signal, sample_rate)
        
        # Extract all eGeMAPS features
        features_df = smile.process_file(tmp_path)
        features_dict = features_df.iloc[0].to_dict()
        
        # Filter to only requested features
        filtered_features = {}
        for feature_name in OPENSMILE_FEATURES:
            if feature_name in features_dict:
                filtered_features[feature_name] = features_dict[feature_name]
            else:
                filtered_features[feature_name] = np.nan
        
        return filtered_features
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def extract_hybrid_features(signal, sample_rate=SAMPLE_RATE):
    """
    Extract both parselmouth and OpenSmile features.
    
    Args:
        signal: numpy array of audio samples
        sample_rate: sampling rate in Hz
    
    Returns:
        dict: Combined parselmouth (36) + OpenSmile (18) = 54 features
    """
    parselmouth_feats = extract_gemaps_features(signal, sample_rate)
    opensmile_feats = extract_opensmile_features(signal, sample_rate)
    
    # Combine both feature sets
    combined = {**parselmouth_feats, **opensmile_feats}
    return combined


def _process_test_segment(args):
    """Worker function for parallel processing of test segments."""
    segment_index, filename, test_dir = args
    csv_path = os.path.join(test_dir, filename)
    df_segment = pd.read_csv(csv_path, dtype={"acoustic_data": np.int16})
    signal = df_segment['acoustic_data'].values.astype(np.float32)
    features = extract_hybrid_features(signal, sample_rate=SAMPLE_RATE)
    segment_id = filename.replace('seg_', '').replace('.csv', '')
    return segment_id, features


def extract_test_features(num_workers=4):
    """Extract test features using parallel processing."""
    test_dir = os.path.join(DATA_DIR, 'test')
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.csv')])
    print(f"Found {len(test_files)} test segment files")
    work_args = [(idx, fname, test_dir) for idx, fname in enumerate(test_files)]
    
    segment_features = []
    segment_ids = []
    
    with Pool(processes=num_workers) as pool:
        for segment_index, (segment_id, features) in enumerate(pool.imap_unordered(_process_test_segment, work_args)):
            segment_ids.append(segment_id)
            segment_features.append(features)
            if (segment_index + 1) % 50 == 0:
                print(f'  Processed {segment_index + 1}/{len(test_files)} test segments...')
    
    # Sort by segment_id to maintain order
    sorted_data = sorted(zip(segment_ids, segment_features))
    segment_ids, segment_features = zip(*sorted_data)
    
    df = pd.DataFrame(segment_features)
    df.insert(0, 'segment_id', segment_ids)

    output_csv_test = os.path.join(DATA_DIR, "gemaps_hybrid_features_test.csv")
    output_npz_test = os.path.join(DATA_DIR, "gemaps_hybrid_features_test.npz")
    
    print(f'Saving test features to {output_csv_test} and {output_npz_test}...')
    df.to_csv(output_csv_test, index=False)
    output_dict = {col: df[col].values for col in df.columns}
    np.savez_compressed(output_npz_test, **output_dict)
    print(f'Done. Extracted {len(df)} test segments.')


def main(num_workers=4):
    train_csv = os.path.join(DATA_DIR, 'train.csv')

    # Count total segments
    total_rows = 629145480  # From train.csv (excluding header)
    num_segments = (total_rows + SEG_LEN - 1) // SEG_LEN
    print(f"Processing {num_segments} training segments...")
    
    segment_features = []
    segment_labels = []
    segment_ids = []
    
    for segment_index, chunk in enumerate(pd.read_csv(
            train_csv,
            dtype={"acoustic_data": np.int16, "time_to_failure": np.float32},
            chunksize=SEG_LEN)):
        if (segment_index + 1) % 100 == 0:
            print(f'  Segment {segment_index + 1}/{num_segments}...')
        
        signal = chunk['acoustic_data'].values.astype(np.float32)
        features = extract_hybrid_features(signal, sample_rate=SAMPLE_RATE)
        segment_features.append(features)
        segment_labels.append(float(chunk['time_to_failure'].iloc[-1]))
        segment_ids.append(segment_index)

    df = pd.DataFrame(segment_features)
    df.insert(0, 'segment_id', segment_ids)
    df.insert(1, 'time_to_failure', segment_labels)

    print(f'Saving features to {OUTPUT_CSV} and {OUTPUT_NPZ}...')
    df.to_csv(OUTPUT_CSV, index=False)
    output_dict = {col: df[col].values for col in df.columns}
    np.savez_compressed(OUTPUT_NPZ, **output_dict)
    print(f'Done. Extracted {len(df)} training segments.')
    
    print("\nExtracting test features...")
    extract_test_features(num_workers=num_workers)


if __name__ == '__main__':
    import sys
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(num_workers=num_workers)
