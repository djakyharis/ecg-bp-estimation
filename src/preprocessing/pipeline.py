"""
Pipeline Pemrosesan Sinyal ECG End-to-End.
Menggabungkan resampling, Notch Filter 50 Hz, Bandpass Butterworth 0.5-40 Hz,
dan Robust Scaling dalam satu alur eksekusi sekuensial.
"""

from typing import Optional
import numpy as np
from scipy.signal import resample_poly

from ..config import PreprocessingConfig, config
from .filters import iir_notch_filter, butterworth_bandpass_filter
from .normalization import minmax_scale, percentile_clip_scale, robust_scale


class ECGPreprocessingPipeline:
    """
    Pipeline pra-pemrosesan sekuensial sinyal ECG dari mentah hingga siap diumpankan ke model.
    """

    def __init__(self, cfg: Optional[PreprocessingConfig] = None):
        """
        Inisialisasi pipeline dengan konfigurasi.

        Parameters
        ----------
        cfg : PreprocessingConfig, optional
            Konfigurasi pemrosesan sinyal. Jika None, menggunakan config default.
        """
        self.cfg = cfg if cfg is not None else config.preprocessing

    def resample_signal(
        self,
        signal: np.ndarray,
        source_fs: float,
        target_fs: float
    ) -> np.ndarray:
        """
        Merubah laju sampling sinyal (misal dari 256 Hz Shimmer3 ke 125 Hz target model).
        """
        if int(source_fs) == int(target_fs) or signal.size == 0:
            return signal

        # Resampling menggunakan rasio integer rasional paling sederhana
        up = int(target_fs)
        down = int(source_fs)
        
        # Penyederhanaan FPB untuk resample_poly
        gcd = np.gcd(up, down)
        up //= gcd
        down //= gcd

        resampled = resample_poly(signal, up, down, axis=0)
        return resampled

    def process(
        self,
        raw_signal: np.ndarray,
        input_fs: Optional[float] = None
    ) -> np.ndarray:
        """
        Mengeksekusi rantai pra-pemrosesan sinyal ECG:
            raw_signal -> resample -> notch (50Hz) -> bandpass (0.5-40Hz) -> robust_scale -> clean_signal

        Parameters
        ----------
        raw_signal : np.ndarray
            Sinyal ECG mentah (1D array atau 2D array [samples, channels]).
        input_fs : float, optional
            Laju sampling input sinyal mentah. Jika None, diasumsikan shimmer_fs (256 Hz).

        Returns
        -------
        np.ndarray
            Sinyal ECG bersih yang telah difilter dan dinormalisasi.
        """
        if raw_signal.size == 0:
            return raw_signal

        fs = input_fs if input_fs is not None else self.cfg.shimmer_fs

        # 1. Resampling jika frekuensi input berbeda dengan frekuensi target
        signal = self.resample_signal(raw_signal, source_fs=fs, target_fs=self.cfg.target_fs)
        working_fs = self.cfg.target_fs

        # 2. IIR Notch Filter (Penghilang noise 50 Hz PLN)
        signal = iir_notch_filter(
            signal,
            fs=working_fs,
            f0=self.cfg.notch_freq,
            q=self.cfg.notch_q
        )

        # 3. Butterworth Bandpass Filter (0.5 - 40 Hz Orde 4)
        signal = butterworth_bandpass_filter(
            signal,
            fs=working_fs,
            lowcut=self.cfg.lowcut,
            highcut=self.cfg.highcut,
            order=self.cfg.filter_order
        )

        # 4. Normalisasi Min-Max [0, 1] sesuai Fan et al. (2021) & Li et al. (2020)
        # Formula: x_norm = (signal - x_min) / (x_max - x_min + eps)
        signal = minmax_scale(
            signal,
            eps=self.cfg.epsilon
        )

        return signal
