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

    def filter_only(
        self,
        raw_signal: np.ndarray,
        input_fs: Optional[float] = None
    ) -> np.ndarray:
        """
        Mengeksekusi rantai filter digital zero-phase tanpa normalisasi skala amplitudo
        (mempertahankan satuan milivolt untuk visualisasi osiloskop GUI & deteksi puncak R):
            raw_signal -> resample (125Hz) -> IIR Notch (50Hz, Q=30) -> Butterworth Bandpass (0.5-40Hz, Orde 4)
        """
        if raw_signal.size == 0:
            return raw_signal

        fs = input_fs if input_fs is not None else self.cfg.shimmer_fs
        signal = self.resample_signal(raw_signal, source_fs=fs, target_fs=self.cfg.target_fs)
        working_fs = self.cfg.target_fs

        # Terapkan filter hanya jika panjang sinyal memadai untuk padlen filtfilt (> 18 sampel)
        if signal.shape[0] > 18:
            signal = iir_notch_filter(
                signal,
                fs=working_fs,
                f0=self.cfg.notch_freq,
                q=self.cfg.notch_q
            )
            signal = butterworth_bandpass_filter(
                signal,
                fs=working_fs,
                lowcut=self.cfg.lowcut,
                highcut=self.cfg.highcut,
                order=self.cfg.filter_order
            )
        return signal

    def process(
        self,
        raw_signal: np.ndarray,
        input_fs: Optional[float] = None,
        norm_mode: Optional[str] = None
    ) -> np.ndarray:
        """
        Mengeksekusi rantai pra-pemrosesan sinyal ECG sesuai Proposal Skripsi (Subbab 2.2.2 - 2.2.4 & Bab 3):
            raw_signal -> resample (125Hz) -> notch (50Hz) -> bandpass (0.5-40Hz) -> normalisasi [0, 1] -> clean_signal

        Parameters
        ----------
        raw_signal : np.ndarray
            Sinyal ECG mentah (1D array atau 2D array [samples, channels]).
        input_fs : float, optional
            Laju sampling input sinyal mentah. Jika None, diasumsikan shimmer_fs (256 Hz).
        norm_mode : str, optional
            Metode normalisasi: "percentile_clip" (Proposal Subbab 2.2.4: Q1-Q99 -> [0,1]),
            "minmax" (Fan et al. 2021: [0,1]), atau "robust" (Median/IQR).

        Returns
        -------
        np.ndarray
            Sinyal ECG bersih yang telah difilter dan dinormalisasi.
        """
        if raw_signal.size == 0:
            return raw_signal

        # 1-3. Resampling + IIR Notch 50 Hz + Butterworth Bandpass 0.5-40 Hz Orde 4
        signal = self.filter_only(raw_signal, input_fs=input_fs)

        # 4. Normalisasi Amplitudo sesuai konfigurasi / Proposal Subbab 2.2.4
        mode = (norm_mode or getattr(self.cfg, "norm_mode", "percentile_clip")).lower()
        if mode == "percentile_clip":
            signal = percentile_clip_scale(
                signal,
                q_low_percentile=self.cfg.clip_q_low,
                q_high_percentile=self.cfg.clip_q_high,
                eps=self.cfg.epsilon
            )
        elif mode == "robust":
            signal = robust_scale(signal, eps=self.cfg.epsilon)
        else:
            signal = minmax_scale(
                signal,
                eps=self.cfg.epsilon
            )

        return signal
