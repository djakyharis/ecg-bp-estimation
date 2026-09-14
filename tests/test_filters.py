"""
Unit tests untuk modul pra-pemrosesan sinyal ECG.
Verifikasi Notch filter 50 Hz, Butterworth Bandpass filter, Robust Scaling, dan Pipeline.
"""

import pytest
import numpy as np

from src.preprocessing.filters import iir_notch_filter, butterworth_bandpass_filter
from src.preprocessing.normalization import robust_scale
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


def test_robust_scale(synthetic_ecg_signal):
    raw_signal, _ = synthetic_ecg_signal
    scaled = robust_scale(raw_signal)
    
    assert scaled.shape == raw_signal.shape
    assert not np.isnan(scaled).any()
    # Median sinyal hasil scaling harus mendekati 0
    assert np.isclose(np.median(scaled), 0.0, atol=1e-2)


def test_ecg_preprocessing_pipeline(synthetic_ecg_signal):
    raw_signal, fs = synthetic_ecg_signal
    pipeline = ECGPreprocessingPipeline()
    clean_signal = pipeline.process(raw_signal, input_fs=fs)
    
    # Hasil resampling dari 256 Hz ke 125 Hz untuk 10 detik data
    expected_length = int(10 * 125.0)
    assert abs(len(clean_signal) - expected_length) <= 5
    assert not np.isnan(clean_signal).any()
