import os
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from multiprocessing import Pool, Manager
from parselmouth import Sound
from parselmouth.praat import call

DATA_DIR = "./kaggle_data"
SEG_LEN = 150_000
SAMPLE_RATE = 4000
PITCH_FLOOR = 75
PITCH_CEILING = 500
MAX_FORMANT = 2000
NUM_FORMANT_SAMPLES = 30
OUTPUT_NPZ = os.path.join(DATA_DIR, "gemaps_parselmouth_features.npz")
OUTPUT_CSV = os.path.join(DATA_DIR, "gemaps_parselmouth_features.csv")


def get_mean_and_cv(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(values))
    if mean == 0.0:
        cv = 0.0
    else:
        cv = float(np.std(values) / np.abs(mean))
    return mean, cv

def smooth_lld(lld_values, voiced_mask=None):
    lld_values = np.asarray(lld_values, dtype=np.float64)
    
    if voiced_mask is not None:
        voiced_mask = np.asarray(voiced_mask, dtype=bool)
        # Only smooth within voiced regions
        smoothed = np.copy(lld_values)
        for i in range(1, len(lld_values) - 1):
            if voiced_mask[i - 1] and voiced_mask[i] and voiced_mask[i + 1]:
                smoothed[i] = (lld_values[i - 1] + lld_values[i] + lld_values[i + 1]) / 3.0
        return smoothed
    else:
        # Standard moving average (3-frame symmetric)
        if len(lld_values) < 3:
            return lld_values
        smoothed = np.copy(lld_values)
        for i in range(1, len(lld_values) - 1):
            smoothed[i] = (lld_values[i - 1] + lld_values[i] + lld_values[i + 1]) / 3.0
        return smoothed

def get_pitch_lld(sound, pitch=None, frame_interval=0.01):
    if pitch is None:
        pitch = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    duration = sound.duration
    times = np.arange(0.0, duration, frame_interval)
    f0_values = []
    voiced = []
    
    for t in times:
        try:
            f0 = call(pitch, "Get value at time", float(t), "Hertz", "Linear")
            f0_values.append(f0 if f0 > 0 else 0.0)
            voiced.append(f0 > 0)
        except Exception:
            f0_values.append(0.0)
            voiced.append(False)
    
    f0_values = np.asarray(f0_values, dtype=np.float64)
    # Convert to semitones (0 for unvoiced, semitone value for voiced)
    semitone_values = np.where(
        f0_values > 0,
        12.0 * np.log2(np.maximum(f0_values, 1e-8) / 27.5),
        0.0
    )
    
    return semitone_values, np.asarray(voiced, dtype=bool)

def get_jitter_shimmer_lld(sound, frame_interval=0.01):
    point_process = call(sound, 'To PointProcess (periodic, cc)', PITCH_FLOOR, PITCH_CEILING)
    point_times = get_point_times(point_process)
    
    if len(point_times) < 3:
        return np.array([0.0]), np.array([0.0]), np.array([False])
    
    # Extract jitter for each period
    jitter_values = []
    signal = sound.values[0]
    sample_rate = sound.sampling_frequency
    
    for i in range(len(point_times) - 1):
        start_t = point_times[i]
        end_t = point_times[i + 1]
        period_length = end_t - start_t
        if i > 0:
            prev_period_length = point_times[i] - point_times[i - 1]
            jitter = abs(period_length - prev_period_length) / period_length if period_length > 0 else 0.0
            jitter_values.append(jitter)
    
    # Extract shimmer for each period
    shimmer_values = []
    amplitudes = []
    for start_time, end_time in zip(point_times[:-1], point_times[1:]):
        start_sample = max(0, int(np.floor(start_time * sample_rate)))
        end_sample = min(signal.shape[0], int(np.ceil(end_time * sample_rate)))
        frame = np.abs(signal[start_sample:end_sample])
        if frame.size:
            amplitudes.append(np.max(frame))
    
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    for i in range(len(amplitudes) - 1):
        avg_amp = (amplitudes[i] + amplitudes[i + 1]) / 2.0
        shimmer = abs(amplitudes[i + 1] - amplitudes[i]) / avg_amp if avg_amp > 0 else 0.0
        shimmer_values.append(shimmer * 100.0)
    
    jitter_array = np.asarray(jitter_values, dtype=np.float64)
    shimmer_array = np.asarray(shimmer_values, dtype=np.float64)
    voiced_array = np.ones(len(jitter_array), dtype=bool)  # These are only for voiced regions
    
    return jitter_array, shimmer_array, voiced_array

def get_formant_lld(sound, formant_number, formant=None, frame_interval=0.01):
    if formant is None:
        formant = sound.to_formant_burg(0.01, 5, MAX_FORMANT, 0.025, 50)
    duration = sound.duration
    times = np.arange(0.0, duration, frame_interval)
    values = []
    
    for t in times:
        try:
            val = call(formant, "Get value at time", formant_number, float(t), "Hertz", "Linear")
            values.append(val if val > 0 else 0.0)
        except Exception:
            values.append(0.0)
    
    return np.asarray(values, dtype=np.float64)

def get_formant_bandwidth_lld(sound, formant_number, formant=None, frame_interval=0.01):
    if formant is None:
        formant = sound.to_formant_burg(0.01, 5, MAX_FORMANT, 0.025, 50)
    duration = sound.duration
    times = np.arange(0.0, duration, frame_interval)
    values = []
    
    for t in times:
        try:
            val = call(formant, "Get bandwidth at time", formant_number, float(t), "Hertz", "Linear")
            values.append(val if val > 0 else 0.0)
        except Exception:
            values.append(0.0)
    
    return np.asarray(values, dtype=np.float64)

def get_point_times(point_process):
    n_points = int(call(point_process, "Get number of points"))
    if n_points <= 0:
        return []
    return [float(call(point_process, "Get time from index", i)) for i in range(1, n_points + 1)]

def get_loudness_lld(sound, frame_interval=0.01):
    intensity = sound.to_intensity()
    duration = sound.duration
    
    try:
        intensity_matrix = call(intensity, "Down to Matrix")
        
        xmin = call(intensity_matrix, "Get start time")
        xmax = call(intensity_matrix, "Get end time")
        n_frames = call(intensity_matrix, "Get number of columns")
        
        intensity_values = []
        for col in range(1, int(n_frames) + 1):
            val = call(intensity_matrix, "Get value", 1, col)
            intensity_values.append(val)
        
        intensity_values = np.asarray(intensity_values, dtype=np.float64)
        
        dx = (xmax - xmin) / (n_frames - 1) if n_frames > 1 else 0.0
        intensity_times = xmin + np.arange(n_frames) * dx
        
        target_times = np.arange(0.0, duration, frame_interval)
        
        values = np.interp(target_times, intensity_times, intensity_values)
        return values.astype(np.float64)
    except Exception as e:
            intensity_values = intensity.values[0]
            n_frames = len(intensity_values)
            
            dx = duration / (n_frames - 1) if n_frames > 1 else duration
            intensity_times = np.arange(n_frames) * dx
            
            target_times = np.arange(0.0, duration, frame_interval)
            
            values = np.interp(target_times, intensity_times, intensity_values)
            return values.astype(np.float64)

def get_hnr_lld(sound, frame_interval=0.01):
    harmonicity = sound.to_harmonicity_cc()
    duration = sound.duration
    
    try:
        # Get values directly from harmonicity object
        # -200 dB indicates unvoiced frames
        harmonicity_values = harmonicity.values[0]
        harmonicity_values = np.asarray(harmonicity_values, dtype=np.float64)
        
        harmonicity_values = np.where(harmonicity_values <= -200, 0.0, harmonicity_values)
        
        n_frames = len(harmonicity_values)
        
        dx = duration / (n_frames - 1) if n_frames > 1 else duration
        harmonicity_times = np.arange(n_frames) * dx
        
        target_times = np.arange(0.0, duration, frame_interval)
        
        values = np.interp(target_times, harmonicity_times, harmonicity_values)
        return values.astype(np.float64)
    except Exception as e:
        print(f"Warning: Failed to extract HNR LLD: {e}")
        n_frames = int(duration / frame_interval) + 1
        return np.zeros(n_frames, dtype=np.float64)

def get_spectral_features_lld(sound, spectrum=None, pitch=None, formant=None, frame_interval=0.01):
    """Extract spectral features with cached objects and optimized time-loop."""
    sample_rate = sound.sampling_frequency
    duration = sound.duration
    if formant is None:
        formant = sound.to_formant_burg(0.01, 5, MAX_FORMANT, 0.025, 50)
    if pitch is None:
        pitch = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    
    times = np.arange(0.0, duration, frame_interval)
    
    # Compute spectrum once (for whole signal)
    if spectrum is None:
        freqs, power = spectral_energy_power(sound)
    else:
        freqs, power = spectrum
    
    # Pre-compute time-independent spectral features (constant for all frames)
    alpha = band_energy(freqs, power, 50.0, 1000.0) / np.maximum(
        band_energy(freqs, power, 1000.0, min(5000.0, sample_rate / 2.0)), 1e-12)
    alpha_ratio_lld = np.full(len(times), float(alpha), dtype=np.float64)
    
    hammarberg = band_energy(freqs, power, 0.0, 1000.0) / np.maximum(
        band_energy(freqs, power, 1000.0, min(5000.0, sample_rate / 2.0)), 1e-12)
    hammarberg_index_lld = np.full(len(times), float(hammarberg), dtype=np.float64)
    
    slope_0_500 = spectral_slope(freqs, power, 0.0, min(500.0, sample_rate / 2.0))
    spectral_slope_0_500_lld = np.full(len(times), slope_0_500, dtype=np.float64)
    
    slope_500_1500 = spectral_slope(freqs, power, 500.0, min(1500.0, sample_rate / 2.0))
    spectral_slope_500_1500_lld = np.full(len(times), slope_500_1500, dtype=np.float64)
    
    # Time-dependent features (require F0/formants per frame)
    f1_rel_energy_lld = []
    f2_rel_energy_lld = []
    f3_rel_energy_lld = []
    h1_h2_ratio_lld = []
    h1_a3_ratio_lld = []
    
    for t in times:
        # Get F0 and formant frequencies
        try:
            f0 = call(pitch, "Get value at time", float(t), "Hertz", "Linear")
        except Exception:
            f0 = 0.0
        
        try:
            f1 = call(formant, "Get value at time", 1, float(t), "Hertz", "Linear")
        except Exception:
            f1 = 0.0
        
        try:
            f2 = call(formant, "Get value at time", 2, float(t), "Hertz", "Linear")
        except Exception:
            f2 = 0.0
        
        try:
            f3 = call(formant, "Get value at time", 3, float(t), "Hertz", "Linear")
        except Exception:
            f3 = 0.0
        
        f0_energy = peak_energy(freqs, power, f0) if f0 > 0 else 0.0
        f1_energy = peak_energy(freqs, power, f1) if f1 > 0 else 0.0
        f2_energy = peak_energy(freqs, power, f2) if f2 > 0 else 0.0
        f3_energy = peak_energy(freqs, power, f3) if f3 > 0 else 0.0
        
        # Formant relative energies
        f1_rel = f1_energy / np.maximum(f0_energy, 1e-12)
        f2_rel = f2_energy / np.maximum(f0_energy, 1e-12)
        f3_rel = f3_energy / np.maximum(f0_energy, 1e-12)
        f1_rel_energy_lld.append(float(f1_rel))
        f2_rel_energy_lld.append(float(f2_rel))
        f3_rel_energy_lld.append(float(f3_rel))
        
        # H1-H2 ratio
        h2_energy = peak_energy(freqs, power, 2.0 * f0) if f0 > 0 else 0.0
        h1_h2 = f0_energy / np.maximum(h2_energy, 1e-12)
        h1_h2_ratio_lld.append(float(h1_h2))
        
        # H1-A3 ratio
        a3_low = max(0.0, f3 - 200.0) if f3 > 0 else 0.0
        a3_high = min(sample_rate / 2.0, f3 + 200.0) if f3 > 0 else 0.0
        a3_idx = np.where((freqs >= a3_low) & (freqs <= a3_high))[0]
        a3_energy = float(np.max(power[a3_idx])) if a3_idx.size else 0.0
        h1_a3 = f0_energy / np.maximum(a3_energy, 1e-12)
        h1_a3_ratio_lld.append(float(h1_a3))
    
    return (alpha_ratio_lld,
            hammarberg_index_lld,
            spectral_slope_0_500_lld,
            spectral_slope_500_1500_lld,
            np.asarray(f1_rel_energy_lld, dtype=np.float64),
            np.asarray(f2_rel_energy_lld, dtype=np.float64),
            np.asarray(f3_rel_energy_lld, dtype=np.float64),
            np.asarray(h1_h2_ratio_lld, dtype=np.float64),
            np.asarray(h1_a3_ratio_lld, dtype=np.float64))

def spectral_energy_power(sound):
    spectrum = sound.to_spectrum()
    dx = spectrum.dx
    freqs = np.arange(spectrum.nx, dtype=np.float64) * dx
    values = spectrum.values
    power = np.square(values[0]) + np.square(values[1])
    return freqs, power

def band_energy(freqs, power, low, high):
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(power[mask])) if np.any(mask) else 0.0

def peak_energy(freqs, power, center_hz, half_band_hz=20.0):
    if center_hz <= 0:
        return 0.0
    idx = np.where((freqs >= max(0.0, center_hz - half_band_hz)) & (freqs <= center_hz + half_band_hz))[0]
    return float(np.max(power[idx])) if idx.size else 0.0

def spectral_slope(freqs, power, low, high):
    mask = (freqs >= low) & (freqs <= high)
    if np.count_nonzero(mask) < 3:
        return 0.0
    x = freqs[mask]
    y = np.log10(power[mask] + 1e-12)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)

def extract_gemaps_features(signal, sample_rate=SAMPLE_RATE):
    """Extract 18 GeMAPS LLDs, smooth them, and compute mean and CV for 36 final parameters."""
    signal = np.asarray(signal, dtype=np.float64)
    sound = Sound(signal, sampling_frequency=sample_rate)
    
    frame_interval = 0.01  #10ms frames
    
    pitch = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    formant = sound.to_formant_burg(0.01, 5, MAX_FORMANT, 0.025, 50)
    spectrum = spectral_energy_power(sound)
    
    # 1. Pitch (semitone)
    f0_lld, f0_voiced = get_pitch_lld(sound, pitch=pitch, frame_interval=frame_interval)
    f0_smoothed = smooth_lld(f0_lld, voiced_mask=f0_voiced)
    f0_mean, f0_cv = get_mean_and_cv(f0_smoothed)
    
    # 2. Jitter
    jitter_lld, shimmer_lld_raw, jitter_voiced = get_jitter_shimmer_lld(sound, frame_interval)
    jitter_smoothed = smooth_lld(jitter_lld, voiced_mask=jitter_voiced)
    jitter_mean, jitter_cv = get_mean_and_cv(jitter_smoothed)
    
    # 3-5. Formants (F1, F2, F3) frequency
    f1_hz_lld = get_formant_lld(sound, 1, formant=formant, frame_interval=frame_interval)
    f1_hz_smoothed = smooth_lld(f1_hz_lld)
    f1_hz_mean, f1_hz_cv = get_mean_and_cv(f1_hz_smoothed)
    
    f2_hz_lld = get_formant_lld(sound, 2, formant=formant, frame_interval=frame_interval)
    f2_hz_smoothed = smooth_lld(f2_hz_lld)
    f2_hz_mean, f2_hz_cv = get_mean_and_cv(f2_hz_smoothed)
    
    f3_hz_lld = get_formant_lld(sound, 3, formant=formant, frame_interval=frame_interval)
    f3_hz_smoothed = smooth_lld(f3_hz_lld)
    f3_hz_mean, f3_hz_cv = get_mean_and_cv(f3_hz_smoothed)
    
    # 6. Formant 1 bandwidth
    f1_bandwidth_lld = get_formant_bandwidth_lld(sound, 1, formant=formant, frame_interval=frame_interval)
    f1_bandwidth_smoothed = smooth_lld(f1_bandwidth_lld)
    f1_bandwidth_mean, f1_bandwidth_cv = get_mean_and_cv(f1_bandwidth_smoothed)
    
    # 7. Shimmer
    shimmer_smoothed = smooth_lld(shimmer_lld_raw, voiced_mask=jitter_voiced)
    shimmer_mean, shimmer_cv = get_mean_and_cv(shimmer_smoothed)
    
    # 8. Loudness
    loudness_lld = get_loudness_lld(sound, frame_interval)
    loudness_smoothed = smooth_lld(loudness_lld)
    loudness_mean, loudness_cv = get_mean_and_cv(loudness_smoothed)
    
    # 9. HNR
    hnr_lld = get_hnr_lld(sound, frame_interval)
    hnr_smoothed = smooth_lld(hnr_lld)
    hnr_mean, hnr_cv = get_mean_and_cv(hnr_smoothed)
    
    # 10-18. Spectral features
    (alpha_ratio_lld, hammarberg_index_lld, spectral_slope_0_500_lld, 
     spectral_slope_500_1500_lld, f1_rel_energy_lld, f2_rel_energy_lld, 
     f3_rel_energy_lld, h1_h2_ratio_lld, h1_a3_ratio_lld) = get_spectral_features_lld(sound, spectrum=spectrum, pitch=pitch, formant=formant, frame_interval=frame_interval)
    
    alpha_ratio_smoothed = smooth_lld(alpha_ratio_lld)
    alpha_ratio_mean, alpha_ratio_cv = get_mean_and_cv(alpha_ratio_smoothed)
    
    hammarberg_index_smoothed = smooth_lld(hammarberg_index_lld)
    hammarberg_index_mean, hammarberg_index_cv = get_mean_and_cv(hammarberg_index_smoothed)
    
    spectral_slope_0_500_smoothed = smooth_lld(spectral_slope_0_500_lld)
    spectral_slope_0_500_mean, spectral_slope_0_500_cv = get_mean_and_cv(spectral_slope_0_500_smoothed)
    
    spectral_slope_500_1500_smoothed = smooth_lld(spectral_slope_500_1500_lld)
    spectral_slope_500_1500_mean, spectral_slope_500_1500_cv = get_mean_and_cv(spectral_slope_500_1500_smoothed)
    
    f1_rel_energy_smoothed = smooth_lld(f1_rel_energy_lld)
    f1_rel_energy_mean, f1_rel_energy_cv = get_mean_and_cv(f1_rel_energy_smoothed)
    
    f2_rel_energy_smoothed = smooth_lld(f2_rel_energy_lld)
    f2_rel_energy_mean, f2_rel_energy_cv = get_mean_and_cv(f2_rel_energy_smoothed)
    
    f3_rel_energy_smoothed = smooth_lld(f3_rel_energy_lld)
    f3_rel_energy_mean, f3_rel_energy_cv = get_mean_and_cv(f3_rel_energy_smoothed)
    
    h1_h2_ratio_smoothed = smooth_lld(h1_h2_ratio_lld)
    h1_h2_ratio_mean, h1_h2_ratio_cv = get_mean_and_cv(h1_h2_ratio_smoothed)
    
    h1_a3_ratio_smoothed = smooth_lld(h1_a3_ratio_lld)
    h1_a3_ratio_mean, h1_a3_ratio_cv = get_mean_and_cv(h1_a3_ratio_smoothed)
    
    return {
        'f0_semitone_mean': f0_mean,
        'f0_semitone_cv': f0_cv,
        'jitter_local_mean': jitter_mean,
        'jitter_local_cv': jitter_cv,
        'formant1_hz_mean': f1_hz_mean,
        'formant1_hz_cv': f1_hz_cv,
        'formant2_hz_mean': f2_hz_mean,
        'formant2_hz_cv': f2_hz_cv,
        'formant3_hz_mean': f3_hz_mean,
        'formant3_hz_cv': f3_hz_cv,
        'formant1_bandwidth_hz_mean': f1_bandwidth_mean,
        'formant1_bandwidth_hz_cv': f1_bandwidth_cv,
        'shimmer_local_mean': shimmer_mean,
        'shimmer_local_cv': shimmer_cv,
        'loudness_db_mean': loudness_mean,
        'loudness_db_cv': loudness_cv,
        'hnr_db_mean': hnr_mean,
        'hnr_db_cv': hnr_cv,
        'alpha_ratio_mean': alpha_ratio_mean,
        'alpha_ratio_cv': alpha_ratio_cv,
        'hammarberg_index_mean': hammarberg_index_mean,
        'hammarberg_index_cv': hammarberg_index_cv,
        'spectral_slope_0_500_mean': spectral_slope_0_500_mean,
        'spectral_slope_0_500_cv': spectral_slope_0_500_cv,
        'spectral_slope_500_1500_mean': spectral_slope_500_1500_mean,
        'spectral_slope_500_1500_cv': spectral_slope_500_1500_cv,
        'formant1_relative_energy_mean': f1_rel_energy_mean,
        'formant1_relative_energy_cv': f1_rel_energy_cv,
        'formant2_relative_energy_mean': f2_rel_energy_mean,
        'formant2_relative_energy_cv': f2_rel_energy_cv,
        'formant3_relative_energy_mean': f3_rel_energy_mean,
        'formant3_relative_energy_cv': f3_rel_energy_cv,
        'h1_h2_ratio_mean': h1_h2_ratio_mean,
        'h1_h2_ratio_cv': h1_h2_ratio_cv,
        'h1_a3_ratio_mean': h1_a3_ratio_mean,
        'h1_a3_ratio_cv': h1_a3_ratio_cv,
    }

def _process_test_segment(args):
    """Worker function for parallel processing of test segments."""
    segment_index, filename, test_dir = args
    csv_path = os.path.join(test_dir, filename)
    df_segment = pd.read_csv(csv_path, dtype={"acoustic_data": np.int16})
    signal = df_segment['acoustic_data'].values.astype(np.float64)
    features = extract_gemaps_features(signal, sample_rate=SAMPLE_RATE)
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

    output_csv_test = os.path.join(DATA_DIR, "gemaps_parselmouth_features_test.csv")
    output_npz_test = os.path.join(DATA_DIR, "gemaps_parselmouth_features_test.npz")
    
    print(f'Saving test features to {output_csv_test} and {output_npz_test}...')
    df.to_csv(output_csv_test, index=False)
    # Convert to dict-like format for savez_compressed
    output_dict = {col: df[col].values for col in df.columns}
    np.savez_compressed(output_npz_test, **output_dict)
    print(f'Done. Extracted {len(df)} test segments.')

def _process_train_segment(args):
    """Worker function for parallel processing of training segments."""
    segment_index, csv_path, seg_len = args
    chunks = pd.read_csv(csv_path, skiprows=range(1, segment_index * seg_len + 1), nrows=seg_len,
                         dtype={"acoustic_data": np.int16, "time_to_failure": np.float32})
    signal = chunks['acoustic_data'].values.astype(np.float64)
    features = extract_gemaps_features(signal, sample_rate=SAMPLE_RATE)
    label = float(chunks['time_to_failure'].iloc[-1])
    return segment_index, label, features

def main(num_workers=4):
    train_csv = os.path.join(DATA_DIR, 'train.csv')

    # Count total segments
    total_rows = 629145480  # From train.csv (excluding header)
    num_segments = (total_rows + SEG_LEN - 1) // SEG_LEN
    print(f"Processing {num_segments} training segments with {num_workers} workers...")
    
    segment_features = []
    segment_labels = []
    segment_ids = []
    for segment_index, chunk in enumerate(pd.read_csv(
            train_csv,
            dtype={"acoustic_data": np.int16, "time_to_failure": np.float32},
            chunksize=SEG_LEN)):
        if (segment_index + 1) % 100 == 0:
            print(f'  Segment {segment_index + 1}/{num_segments}...')
        signal = chunk['acoustic_data'].values.astype(np.float64)
        features = extract_gemaps_features(signal, sample_rate=SAMPLE_RATE)
        segment_features.append(features)
        segment_labels.append(float(chunk['time_to_failure'].iloc[-1]))
        segment_ids.append(segment_index)

    df = pd.DataFrame(segment_features)
    df.insert(0, 'segment_id', segment_ids)
    df.insert(1, 'time_to_failure', segment_labels)

    print(f'Saving features to {OUTPUT_CSV} and {OUTPUT_NPZ}...')
    df.to_csv(OUTPUT_CSV, index=False)    # Convert to dict-like format for savez_compressed
    output_dict = {col: df[col].values for col in df.columns}
    np.savez_compressed(OUTPUT_NPZ, **output_dict)
    print(f'Done. Extracted {len(df)} training segments.')
    
    print("\nExtracting test features...")
    extract_test_features(num_workers=num_workers)

if __name__ == '__main__':
    import sys
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(num_workers=num_workers)
