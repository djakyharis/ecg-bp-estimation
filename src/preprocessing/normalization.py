"""
Modul Normalisasi Amplitudo Sinyal ECG.
Mengimplementasikan Robust Scaling berbasis Median & Interquartile Range (IQR)
serta Persentil Clipping & Normalisasi [0, 1] (Proposal Subbab 2.2.4 & Butarbutar & Widasari, 2026).
"""

import numpy as np


def robust_scale(
    signal: np.ndarray,
    quantile_low: float = 25.0,
    quantile_high: float = 75.0,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Menormalisasi amplitudo sinyal ECG menggunakan Robust Scaling:
        scaled = (signal - median) / (IQR + eps)

    Parameters
    ----------
    signal : np.ndarray
        Sinyal ECG (1D array atau 2D array [samples, channels]).
    quantile_low : float, optional
        Persentil batas bawah IQR (default 25.0).
    quantile_high : float, optional
        Persentil batas atas IQR (default 75.0).
    eps : float, optional
        Konstanta kecil untuk mencegah pembagian dengan nol (default 1e-8).

    Returns
    -------
    np.ndarray
        Sinyal ECG yang telah dinormalisasi.
    """
    if signal.size == 0:
        return signal

    # Hitung median dan IQR sepanjang aksis sinyal
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

    # Hindari pembagian dengan nol apabila sinyal datar (flat line)
    iqr = np.where(iqr < eps, 1.0, iqr)
    
    scaled_signal = (signal - median) / iqr
    return scaled_signal


def percentile_clip_scale(
    signal: np.ndarray,
    q_low_percentile: float = 1.0,
    q_high_percentile: float = 99.0,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalisasi Robust Scaling berbasis persentil clipping [0, 1]
    sesuai formulasi Persamaan Subbab 2.2.4 Proposal Skripsi & Butarbutar & Widasari (2026):
        x_clipped = clip(signal, Q_low, Q_high)
        x_norm = (x_clipped - Q_low) / (Q_high - Q_low + eps)
    """
    if signal.size == 0:
        return signal

    q_low = np.percentile(signal, q_low_percentile)
    q_high = np.percentile(signal, q_high_percentile)

    # Potong sampel di luar persentil batas (outlier clipping)
    x_clipped = np.clip(signal, q_low, q_high)

    range_val = q_high - q_low
    if range_val < eps:
        range_val = 1.0

    x_norm = (x_clipped - q_low) / range_val
    return x_norm
