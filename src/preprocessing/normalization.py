"""
Modul Normalisasi Amplitudo Sinyal ECG.

Fungsi yang tersedia:
  - minmax_scale          : normalisasi KANONIK untuk ECGPreprocessingPipeline.
                            Memetakan amplitudo sinyal ke rentang [0, 1] menggunakan Min-Max Normalization
                            sesuai paper acuan Fan et al. (2021) & Li et al. (2020).
  - percentile_clip_scale : utilitas normalisasi persentil clipping [Q1, Q99] -> [0, 1].
  - robust_scale          : utilitas alternatif berbasis Median & IQR (output unbounded).

Referensi Paper Acuan Utama:
  1. Fan et al. (2021) — "An Adaptive Weight Learning-Based Multitask Deep Network
     for Continuous Blood Pressure Estimation Using Electrocardiogram Signals" (Sensors, 2021):
     "To mitigate the different amplitudes of ECG signals, the min-max normalization
     technology is utilized for ECG signals to map the amplitudes ranging from 0 to 1."
     Formula: x_norm = (x - x_min) / (x_max - x_min)
  2. Li et al. (2020) — "Real-Time Cuffless Continuous Blood Pressure Estimation
     Using Deep Learning Model" (Sensors, 2020).
"""

import numpy as np


def minmax_scale(
    signal: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Menormalisasi amplitudo sinyal ECG ke rentang [0, 1] menggunakan Min-Max Normalization
    sesuai metode Fan et al. (2021) & Li et al. (2020):
        x_norm = (signal - x_min) / (x_max - x_min + eps)

    Parameters
    ----------
    signal : np.ndarray
        Sinyal ECG (1D array atau 2D array [samples, channels]).
    eps : float, optional
        Konstanta pencegah pembagian dengan nol (default 1e-8).

    Returns
    -------
    np.ndarray
        Sinyal ECG ternormalisasi dalam rentang tertutup [0, 1].
    """
    if signal.size == 0:
        return signal

    if signal.ndim == 1:
        x_min = np.min(signal)
        x_max = np.max(signal)
    else:
        x_min = np.min(signal, axis=0, keepdims=True)
        x_max = np.max(signal, axis=0, keepdims=True)

    range_val = x_max - x_min
    range_val = np.where(range_val < eps, 1.0, range_val)

    x_norm = (signal - x_min) / range_val
    return x_norm


def percentile_clip_scale(
    signal: np.ndarray,
    q_low_percentile: float = 1.0,
    q_high_percentile: float = 99.0,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalisasi berbasis persentil clipping [0, 1]:
        x_clipped = clip(signal, Q_low, Q_high)
        x_norm = (x_clipped - Q_low) / (Q_high - Q_low + eps)
    """
    if signal.size == 0:
        return signal

    q_low = np.percentile(signal, q_low_percentile)
    q_high = np.percentile(signal, q_high_percentile)

    x_clipped = np.clip(signal, q_low, q_high)

    range_val = q_high - q_low
    if range_val < eps:
        range_val = 1.0

    x_norm = (x_clipped - q_low) / range_val
    return x_norm


def robust_scale(
    signal: np.ndarray,
    quantile_low: float = 25.0,
    quantile_high: float = 75.0,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Menormalisasi amplitudo sinyal ECG menggunakan Robust Scaling:
        scaled = (signal - median) / (IQR + eps)
    """
    if signal.size == 0:
        return signal

    if signal.ndim == 1:
        median = np.median(signal)
        q25 = np.percentile(signal, quantile_low)
        q75 = np.percentile(signal, quantile_high)
        iqr = q75 - q25
    else:
        median = np.median(signal, axis=0, keepdims=True)
        q25 = np.percentile(signal, quantile_low, axis=0, keepdims=True)
        q75 = np.percentile(signal, quantile_high, axis=0, keepdims=True)
        iqr = q75 - q25

    iqr = np.where(iqr < eps, 1.0, iqr)
    scaled_signal = (signal - median) / iqr
    return scaled_signal
