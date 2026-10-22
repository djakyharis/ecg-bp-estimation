"""
Unit tests untuk modul pra-pemrosesan sinyal ECG.
Verifikasi Notch filter 50 Hz, Butterworth Bandpass filter,
Percentile Clip & Scale [0,1] (normalisasi kanonik pipeline), dan Pipeline end-to-end.

Catatan normalisasi:
  - percentile_clip_scale : normalisasi KANONIK yang digunakan di ECGPreprocessingPipeline.
                            Output bounded [0, 1], robust terhadap outlier via Q1-Q99 clipping.
  - robust_scale          : utilitas alternatif (Z-score berbasis IQR), output unbounded.
                            Tidak digunakan di pipeline produksi.
"""

import pytest
import numpy as np

from src.preprocessing.filters import iir_notch_filter, butterworth_bandpass_filter
from src.preprocessing.normalization import minmax_scale, percentile_clip_scale
from src.preprocessing.pipeline import ECGPreprocessingPipeline


@pytest.fixture
def synthetic_ecg_signal():
    """Membuat sinyal sintetis 256 Hz dengan komponen ECG (1 Hz), noise PLN (50 Hz), dan drift (0.1 Hz)."""
    fs = 256.0
    t = np.linspace(0, 10, int(fs * 10), endpoint=False)
    
    # Komponen sinyal
    clean_ecg = np.sin(2 * np.pi * 1.0 * t)   # Sinyal dasar 1 Hz
    pln_noise = 0.5 * np.sin(2 * np.pi * 50.0 * t) # Noise PLN 50 Hz
    drift = 2.0 * np.sin(2 * np.pi * 0.1 * t)       # Baseline wander 0.1 Hz
    
    raw_signal = clean_ecg + pln_noise + drift
    return raw_signal, fs


def test_iir_notch_filter(synthetic_ecg_signal):
    raw_signal, fs = synthetic_ecg_signal
    filtered = iir_notch_filter(raw_signal, fs=fs, f0=50.0, q=30.0)
    
    assert filtered.shape == raw_signal.shape
    assert not np.isnan(filtered).any()


def test_butterworth_bandpass_filter(synthetic_ecg_signal):
    raw_signal, fs = synthetic_ecg_signal
    filtered = butterworth_bandpass_filter(raw_signal, fs=fs, lowcut=0.5, highcut=40.0, order=4)
    
    assert filtered.shape == raw_signal.shape
    assert not np.isnan(filtered).any()
    # Memastikan baseline wander (0.1 Hz) tereduksi secara signifikan
    assert np.std(filtered) < np.std(raw_signal)


def test_minmax_scale(synthetic_ecg_signal):
    """Verifikasi fungsi normalisasi Min-Max kanonik (Fan et al., 2021 & Li et al., 2020)."""
    raw_signal, _ = synthetic_ecg_signal
    scaled = minmax_scale(raw_signal)

    assert scaled.shape == raw_signal.shape
    assert not np.isnan(scaled).any()
    # Output harus bounded tepat dalam [0, 1]
    assert np.isclose(scaled.min(), 0.0, atol=1e-6), f"Nilai min: {scaled.min()}"
    assert np.isclose(scaled.max(), 1.0, atol=1e-6), f"Nilai max: {scaled.max()}"


def test_percentile_clip_scale(synthetic_ecg_signal):
    """Verifikasi utilitas normalisasi persentil clipping [Q1, Q99]."""
    raw_signal, _ = synthetic_ecg_signal
    scaled = percentile_clip_scale(raw_signal, q_low_percentile=1.0, q_high_percentile=99.0)

    assert scaled.shape == raw_signal.shape
    assert not np.isnan(scaled).any()
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0


def test_ecg_preprocessing_pipeline(synthetic_ecg_signal):
    raw_signal, fs = synthetic_ecg_signal
    pipeline = ECGPreprocessingPipeline()
    clean_signal = pipeline.process(raw_signal, input_fs=fs)
    
    # Hasil resampling dari 256 Hz ke 125 Hz untuk 10 detik data
    expected_length = int(10 * 125.0)
    assert abs(len(clean_signal) - expected_length) <= 5
    assert not np.isnan(clean_signal).any()


def test_abp_skewness_filter():
    """Verifikasi ABP quality filter berbasis kemiringan positif sesuai Fan et al. (2021)."""
    from src.preprocessing.kachuee_preprocessor import check_abp_skewness, compute_bp_metrics

    fs = 125.0
    t = np.linspace(0, 10, int(fs * 10), endpoint=False)
    # Sinyal ABP realistis (upstroke tajam, penurunan bertahap) -> skewness positif
    abp_valid = 80.0 + 40.0 * np.exp(-((t % 0.8) / 0.15))
    assert check_abp_skewness(abp_valid, num_windows=10) is True

    # Sinyal terbalik (artefak terdistorsi) -> skewness negatif
    abp_invalid = 120.0 - 40.0 * np.exp(-((t % 0.8) / 0.15))
    assert check_abp_skewness(abp_invalid, num_windows=10) is False

    # Uji integrasi dalam compute_bp_metrics
    bp_valid = compute_bp_metrics(abp_valid, fs=fs, check_skewness=True)
    assert bp_valid["valid"] is True
    assert 100.0 <= bp_valid["sbp"] <= 130.0
    assert 70.0 <= bp_valid["dbp"] <= 90.0

    bp_invalid = compute_bp_metrics(abp_invalid, fs=fs, check_skewness=True)
    assert bp_invalid["valid"] is False

