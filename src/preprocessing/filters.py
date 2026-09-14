"""
Modul Filter Digital Sinyal ECG.
Mengimplementasikan IIR Notch Filter (50 Hz) dan Butterworth Bandpass Filter (0.5 - 40 Hz)
menggunakan pendekatan Zero-Phase Filtering (filtfilt) sesuai metode Alhamid (2026) & Hakim (2025).
"""

import numpy as np
from scipy.signal import iirnotch, butter, filtfilt


def iir_notch_filter(
    signal: np.ndarray,
    fs: float = 125.0,
    f0: float = 50.0,
    q: float = 30.0
) -> np.ndarray:
    """
    Mengeliminasi interferensi frekuensi listrik PLN (50 Hz) menggunakan IIR Notch Filter.
    (Metode Alhamid, 2026 & Hakim, 2025).

    Parameters
    ----------
    signal : np.ndarray
        Sinyal ECG mentah (1D array atau 2D array [samples, channels]).
    fs : float, optional
        Laju sampling sinyal (Hz), default 125.0.
    f0 : float, optional
        Frekuensi notch yang dieliminasi (Hz), default 50.0.
    q : float, optional
        Quality factor filter (Q = f0 / bandwidth), default 30.0.

    Returns
    -------
    np.ndarray
        Sinyal ECG hasil penyaringan takik zero-phase.
    """
    if signal.size == 0:
        return signal

    # Desain IIR Notch Filter (b, a)
    b, a = iirnotch(w0=f0, Q=q, fs=fs)
    
    # Filter zero-phase dua arah (filtfilt)
    filtered_signal = filtfilt(b, a, signal, axis=0)
    return filtered_signal


def butterworth_bandpass_filter(
    signal: np.ndarray,
    fs: float = 125.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 2
) -> np.ndarray:
    """
    Membatasi bandwidth sinyal ECG (0.5 - 40 Hz) menggunakan Butterworth Bandpass Filter.
    (Metode Alhamid, 2026 & Hakim, 2025).

    Parameters
    ----------
    signal : np.ndarray
        Sinyal ECG (1D array atau 2D array [samples, channels]).
    fs : float, optional
        Laju sampling sinyal (Hz), default 125.0.
    lowcut : float, optional
        Frekuensi batas bawah (Hz), default 0.5.
    highcut : float, optional
        Frekuensi batas atas (Hz), default 40.0.
    order : int, optional
        Orde filter Butterworth (default 2, setara Orde 4 total saat filtfilt).

    Returns
    -------
    np.ndarray
        Sinyal ECG hasil penyaringan pita frekuensi zero-phase.
    """
    if signal.size == 0:
        return signal

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Validasi batas Nyquist
    if high >= 1.0:
        high = 0.99  # Batasi di bawah frekuensi Nyquist
        
    # Desain Butterworth Bandpass Filter (b, a)
    b, a = butter(N=order, Wn=[low, high], btype='bandpass')
    
    # Filter zero-phase dua arah (filtfilt)
    filtered_signal = filtfilt(b, a, signal, axis=0)
    return filtered_signal
