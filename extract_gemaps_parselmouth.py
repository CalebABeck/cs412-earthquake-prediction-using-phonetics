import os
import time
import numpy as np
import pandas as pd
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


def get_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0

def get_f0(sound):
    pitch = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0_values = pitch.selected_array['frequency'].astype(np.float64)
    return f0_values[f0_values > 0.0]

def get_formant_statistics(formant, formant_number, duration, num_samples=NUM_FORMANT_SAMPLES):
    if duration <= 0:
        return 0.0
    times = np.linspace(0.05, max(0.05, duration - 0.05), num=num_samples)
    values = []
    for t in times:
        try:
            values.append(call(formant, "Get value at time", formant_number, float(t), "Hertz", "Linear"))
        except Exception:
            values.append(np.nan)
    return get_mean(values)

def get_formant_bandwidth(formant, formant_number, duration, num_samples=NUM_FORMANT_SAMPLES):
    if duration <= 0:
        return 0.0
    times = np.linspace(0.05, max(0.05, duration - 0.05), num=num_samples)
    values = []
    for t in times:
        try:
            values.append(call(formant, "Get bandwidth at time", formant_number, float(t), "Hertz", "Linear"))
        except Exception:
            values.append(np.nan)
    return get_mean(values)

def get_point_times(point_process):
    n_points = int(call(point_process, "Get number of points"))
    if n_points <= 0:
        return []
    return [float(call(point_process, "Get time from index", i)) for i in range(1, n_points + 1)]

def estimate_shimmer(signal, sample_rate, point_times):
    if len(point_times) < 3:
        return 0.0
    amplitudes = []
    signal = np.asarray(signal, dtype=np.float64)
    for start_time, end_time in zip(point_times[:-1], point_times[1:]):
        start_sample = max(0, int(np.floor(start_time * sample_rate)))
        end_sample = min(signal.shape[0], int(np.ceil(end_time * sample_rate)))
        frame = np.abs(signal[start_sample:end_sample])
        if frame.size:
            amplitudes.append(np.max(frame))
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if amplitudes.size < 2:
        return 0.0
    avg_amp = (amplitudes[:-1] + amplitudes[1:]) / 2.0
    ratio = np.abs(amplitudes[1:] - amplitudes[:-1]) / np.maximum(avg_amp, 1e-12)
    return float(np.nanmean(ratio) * 100.0)

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
    signal = np.asarray(signal, dtype=np.float64)
    duration = float(signal.shape[0]) / sample_rate
    sound = Sound(signal, sampling_frequency=sample_rate)

    voiced_f0 = get_f0(sound)
    semitone_f0 = float(np.mean(12.0 * np.log2(np.maximum(voiced_f0, 1e-8) / 27.5))) if voiced_f0.size else 0.0

    point_process = call(sound, 'To PointProcess (periodic, cc)', PITCH_FLOOR, PITCH_CEILING)
    jitter_local = float(call(point_process, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3))
    point_times = get_point_times(point_process)
    shimmer_local = estimate_shimmer(signal, sample_rate, point_times)

    formant = sound.to_formant_burg(0.01, 5, MAX_FORMANT, 0.025, 50)
    f1_hz = get_formant_statistics(formant, 1, duration)
    f2_hz = get_formant_statistics(formant, 2, duration)
    f3_hz = get_formant_statistics(formant, 3, duration)
    f1_bandwidth_hz = get_formant_bandwidth(formant, 1, duration)

    intensity = sound.to_intensity()
    loudness_db = float(call(intensity, 'Get mean', 0, 0))

    harmonicity = sound.to_harmonicity_cc()
    hnr_db = float(call(harmonicity, 'Get mean', 0, 0))

    freqs, power = spectral_energy_power(sound)
    alpha_ratio = float(band_energy(freqs, power, 50.0, 1000.0) / np.maximum(band_energy(freqs, power, 1000.0, min(5000.0, sample_rate / 2.0)), 1e-12))
    hammarberg_index = float(band_energy(freqs, power, 0.0, 1000.0) / np.maximum(band_energy(freqs, power, 1000.0, min(5000.0, sample_rate / 2.0)), 1e-12))
    spectral_slope_0_500 = spectral_slope(freqs, power, 0.0, min(500.0, sample_rate / 2.0))
    spectral_slope_500_1500 = spectral_slope(freqs, power, 500.0, min(1500.0, sample_rate / 2.0))

    f0_center = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
    f0_energy = peak_energy(freqs, power, f0_center)
    formant1_energy = peak_energy(freqs, power, f1_hz)
    formant2_energy = peak_energy(freqs, power, f2_hz)
    formant3_energy = peak_energy(freqs, power, f3_hz)
    f1_rel_energy = float(formant1_energy / np.maximum(f0_energy, 1e-12))
    f2_rel_energy = float(formant2_energy / np.maximum(f0_energy, 1e-12))
    f3_rel_energy = float(formant3_energy / np.maximum(f0_energy, 1e-12))

    h2_energy = peak_energy(freqs, power, 2.0 * f0_center)
    h1_h2_ratio = float(f0_energy / np.maximum(h2_energy, 1e-12))
    a3_low = max(0.0, f3_hz - 200.0)
    a3_high = min(sample_rate / 2.0, f3_hz + 200.0)
    a3_idx = np.where((freqs >= a3_low) & (freqs <= a3_high))[0]
    a3_energy = float(np.max(power[a3_idx])) if a3_idx.size else 0.0
    h1_a3_ratio = float(f0_energy / np.maximum(a3_energy, 1e-12))

    return {
        'f0_semitone': semitone_f0,
        'jitter_local': jitter_local,
        'formant1_hz': f1_hz,
        'formant2_hz': f2_hz,
        'formant3_hz': f3_hz,
        'formant1_bandwidth_hz': f1_bandwidth_hz,
        'shimmer_local': shimmer_local,
        'loudness_db': loudness_db,
        'hnr_db': hnr_db,
        'alpha_ratio': alpha_ratio,
        'hammarberg_index': hammarberg_index,
        'spectral_slope_0_500': spectral_slope_0_500,
        'spectral_slope_500_1500': spectral_slope_500_1500,
        'formant1_relative_energy': f1_rel_energy,
        'formant2_relative_energy': f2_rel_energy,
        'formant3_relative_energy': f3_rel_energy,
        'h1_h2_ratio': h1_h2_ratio,
        'h1_a3_ratio': h1_a3_ratio,
    }

def extract_test_features():
    test_dir = os.path.join(DATA_DIR, 'test')
    segment_features = []
    segment_ids = []

    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.csv')])
    print(f"Found {len(test_files)} test segment files")
    
    for segment_index, filename in enumerate(test_files):
        if (segment_index + 1) % 50 == 0:
            print(f'  Processed {segment_index + 1}/{len(test_files)} test segments...')
        
        csv_path = os.path.join(test_dir, filename)
        df_segment = pd.read_csv(csv_path, dtype={"acoustic_data": np.int16})
        signal = df_segment['acoustic_data'].values.astype(np.float64)
        
        features = extract_gemaps_features(signal, sample_rate=SAMPLE_RATE)
        segment_features.append(features)
        # Extract segment ID from filename (e.g., 'seg_00030f.csv' -> '00030f')
        segment_ids.append(filename.replace('seg_', '').replace('.csv', ''))

    df = pd.DataFrame(segment_features)
    df.insert(0, 'segment_id', segment_ids)

    output_csv_test = os.path.join(DATA_DIR, "gemaps_parselmouth_features_test.csv")
    output_npz_test = os.path.join(DATA_DIR, "gemaps_parselmouth_features_test.npz")
    
    print(f'Saving test features to {output_csv_test} and {output_npz_test}...')
    df.to_csv(output_csv_test, index=False)
    np.savez_compressed(output_npz_test, **df.to_dict(orient='list'))
    print(f'Done. Extracted {len(df)} test segments.')

def main():
    train_csv = os.path.join(DATA_DIR, 'train.csv')

    segment_features = []
    segment_labels = []
    segment_ids = []

    for segment_index, chunk in enumerate(pd.read_csv(
            train_csv,
            dtype={"acoustic_data": np.int16, "time_to_failure": np.float32},
            chunksize=SEG_LEN)):
        print(f'  Segment {segment_index + 1}...')
        signal = chunk['acoustic_data'].values.astype(np.float64)
        features = extract_gemaps_features(signal, sample_rate=SAMPLE_RATE)
        segment_features.append(features)
        segment_labels.append(float(chunk['time_to_failure'].iat[-1]))
        segment_ids.append(segment_index)

    df = pd.DataFrame(segment_features)
    df.insert(0, 'segment_id', segment_ids)
    df.insert(1, 'time_to_failure', segment_labels)

    print(f'Saving features to {OUTPUT_CSV} and {OUTPUT_NPZ}...')
    df.to_csv(OUTPUT_CSV, index=False)
    np.savez_compressed(OUTPUT_NPZ, **df.to_dict(orient='list'))
    print(f'Done. Extracted {len(df)} training segments.')
    
    print("\nExtracting test features...")
    extract_test_features()

if __name__ == '__main__':
    main()
