"""
Modul Ekstraksi Fitur Statistik ECG untuk Estimasi Tekanan Darah (SBP, DBP, MAP).

Diadaptasi dari:
  1. Nour et al. (2022) — "Ranking Fitur Korelasi Spearman" (Tabel 3.1)
  2. Muhammad Ilham Hakim (2025) — "Sistem Monitoring Tekanan Darah Menggunakan
     Random Forest Regression Berbasis Sensor Shimmer ECG" (Tabel 5.3)

Partisi Fitur Fisiologis:
  - SBP Head (6 Fitur Daya/Puncak):
      1. Shape Factor (Rank 1, r=0.93)
      2. Mean (Rank 2, r=0.41)
      3. Average Energy (Rank 3, r=0.39)
      4. Root Mean Square (RMS) (Rank 3, r=0.39)
      5. SVD Dominant Singular Value (Rank 3, r=0.39)
      6. Central Moment ke-10 / CM10 (Rank 4, r=0.26)
  - DBP Head (6 Fitur Dinamika/Tahanan Perifer):
      1. Hjorth Mobility (Rank 5, r=0.24)
      2. Hjorth Complexity (Rank 5, r=0.24)
      3. Hjorth Activity (Rank 5, r=0.24)
      4. Coefficient of Variation / CV (Rank 5, r=0.24)
      5. Skewness (Rank 5, r=0.24)
      6. Standard Deviation (Rank 5, r=0.24)
  - MAP Head (12 Fitur Gabungan):
      Seluruh gabungan 12 fitur SBP dan DBP untuk merefleksikan relasi fisik
      MAP ≈ DBP + 1/3 (SBP - DBP).
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch


SBP_FEATURE_NAMES = [
    "shape_factor",
    "mean",
    "average_energy",
    "rms",
    "svd_1",
    "cm10",
]

DBP_FEATURE_NAMES = [
    "mobility_hjorth",
    "complexity_hjorth",
    "activity_hjorth",
    "coef_var",
    "skewness",
    "std",
]

ALL_FEATURE_NAMES = SBP_FEATURE_NAMES + DBP_FEATURE_NAMES


def extract_statistical_features(
    ecg_signals: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    Mengekstrak 12 fitur statistik terpandu secara efisien (vectorized)
    dari array segmen ECG berdimensi [N, 1250] atau [1250].

    Parameters
    ----------
    ecg_signals : np.ndarray atau torch.Tensor
        Array sinyal ECG shape [N, seq_len] atau [seq_len] atau [N, seq_len, 1].
    eps : float, optional
        Konstanta stabilitas numerik untuk mencegah pembagian dengan nol.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary berisi:
          - 'sbp': np.ndarray shape [N, 6]
          - 'dbp': np.ndarray shape [N, 6]
          - 'map': np.ndarray shape [N, 12] (gabungan SBP + DBP)
          - 'all': np.ndarray shape [N, 12]
    """
    if isinstance(ecg_signals, torch.Tensor):
        x = ecg_signals.detach().cpu().numpy()
    else:
        x = np.array(ecg_signals, copy=False)

    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    n_samples, seq_len = x.shape

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Fitur Statistik SBP (Daya & Karakteristik Puncak Kontraksi)
    # ──────────────────────────────────────────────────────────────────────────
    # Mean (Rank 2, r=0.41)
    mean_val = np.mean(x, axis=-1, keepdims=True)

    # Average Energy (Rank 3, r=0.39)
    energy_val = np.mean(x ** 2, axis=-1, keepdims=True)

    # RMS (Rank 3, r=0.39)
    rms_val = np.sqrt(np.maximum(energy_val, 0.0))

    # Mean Square Amplitude (MSA) untuk Shape Factor
    msa_val = np.mean(np.sqrt(np.maximum(np.abs(x), eps)), axis=-1, keepdims=True)

    # Shape Factor (Rank 1, r=0.93) - Ilham Hakim Tabel 5.3
    shape_factor = np.where(msa_val > eps, rms_val / msa_val, 0.0)

    # Central Moment ke-10 (Rank 4, r=0.26) - Ilham Hakim Tabel 5.3
    # Mengukur keberadaan puncak ekstrem QRS sistolik
    diff_mean = x - mean_val
    # Untuk stabilitas numerik agar tidak meledak jika sinyal belum dinormalisasi
    scale = np.max(np.abs(diff_mean), axis=-1, keepdims=True) + eps
    cm10_val = np.mean((diff_mean / scale) ** 10, axis=-1, keepdims=True) * (scale ** 2)

    # SVD Komponen Pertama (Rank 3, r=0.39)
    # Reshape sekuens (misal 1250 -> 50 x 25) untuk mengisolasi energi komponen dominan
    if seq_len % 50 == 0:
        rows = 50
        cols = seq_len // 50
        x_matrix = x.reshape(n_samples, rows, cols)
        svd_val = np.linalg.svd(x_matrix, compute_uv=False)[:, 0:1]
    else:
        # Fallback jika panjang sekuens tidak kelipatan 50: gunakan Frobenius norm
        svd_val = np.linalg.norm(x, axis=-1, keepdims=True) / np.sqrt(seq_len)

    sbp_feats = np.concatenate([
        shape_factor,  # (N, 1)
        mean_val,      # (N, 1)
        energy_val,    # (N, 1)
        rms_val,       # (N, 1)
        svd_val,       # (N, 1)
        cm10_val,      # (N, 1)
    ], axis=-1).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Fitur Statistik DBP (Dinamika & Tahanan Perifer / Relaksasi Vaskular)
    # ──────────────────────────────────────────────────────────────────────────
    var_s = np.var(x, axis=-1, ddof=1, keepdims=True)
    std_val = np.sqrt(np.maximum(var_s, 0.0))

    # Turunan pertama dan kedua sinyal
    d1 = np.diff(x, axis=-1)
    var_d1 = np.var(d1, axis=-1, ddof=1, keepdims=True)

    d2 = np.diff(d1, axis=-1)
    var_d2 = np.var(d2, axis=-1, ddof=1, keepdims=True)

    # Hjorth Mobility (Rank 5, r=0.24) - Ilham Hakim Tabel 5.3
    mob_x = np.where(var_s > eps, np.sqrt(np.maximum(var_d1 / (var_s + eps), 0.0)), 0.0)

    # Hjorth Complexity (Rank 5, r=0.24) - Ilham Hakim Tabel 5.3
    mob_d1 = np.where(var_d1 > eps, np.sqrt(np.maximum(var_d2 / (var_d1 + eps), 0.0)), 0.0)
    complexity_val = np.where(mob_x > eps, mob_d1 / (mob_x + eps), 0.0)

    # Hjorth Activity (Rank 5, r=0.24) = Variansi sinyal dasar
    activity_val = var_s

    # Coefficient of Variation / CV (Rank 5, r=0.24) - Ilham Hakim Tabel 5.3
    coef_var_val = np.where(np.abs(mean_val) > eps, (std_val / (np.abs(mean_val) + eps)) * 100.0, 0.0)

    # Skewness (Rank 5, r=0.24) - Ilham Hakim Tabel 5.3
    m3 = np.mean(diff_mean ** 3, axis=-1, keepdims=True)
    skewness_val = np.where(std_val > eps, m3 / (std_val ** 3 + eps), 0.0)

    dbp_feats = np.concatenate([
        mob_x,           # (N, 1)
        complexity_val,  # (N, 1)
        activity_val,    # (N, 1)
        coef_var_val,    # (N, 1)
        skewness_val,    # (N, 1)
        std_val,         # (N, 1)
    ], axis=-1).astype(np.float32)

    # MAP menerima seluruh 12 fitur gabungan
    all_feats = np.concatenate([sbp_feats, dbp_feats], axis=-1).astype(np.float32)

    return {
        "sbp": sbp_feats,
        "dbp": dbp_feats,
        "map": all_feats,
        "all": all_feats,
    }


class FeatureStandardScaler:
    """
    Scaler Standarisasi Fitur Statistik (Z-score: (x - mean) / std)
    untuk memastikan nilai fitur berada dalam rentang teratur dan stabil
    saat digabungkan dengan representasi laten BiLSTM (512 dimensi).
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_dict: Dict[str, np.ndarray] = {}
        self.scale_dict: Dict[str, np.ndarray] = {}
        self.is_fitted = False

    def fit(self, features_dict: Dict[str, np.ndarray]) -> "FeatureStandardScaler":
        """Menghitung nilai mean dan std dari data pelatihan."""
        for key in ["sbp", "dbp", "map"]:
            if key in features_dict:
                feat = features_dict[key]
                self.mean_dict[key] = np.mean(feat, axis=0, keepdims=True)
                scale = np.std(feat, axis=0, keepdims=True)
                self.scale_dict[key] = np.where(scale < self.eps, 1.0, scale)
        self.is_fitted = True
        return self

    def transform(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Melakukan standarisasi (z-score) pada fitur."""
        if not self.is_fitted:
            raise RuntimeError("FeatureStandardScaler belum di-fit dengan data latih!")

        out = {}
        for key, feat in features_dict.items():
            if key in self.mean_dict:
                scaled = (feat - self.mean_dict[key]) / self.scale_dict[key]
                # Menghindari outlier ekstrem tak terbatas
                out[key] = np.clip(scaled, -10.0, 10.0).astype(np.float32)
            else:
                out[key] = feat
        return out

    def fit_transform(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.fit(features_dict).transform(features_dict)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean_dict": self.mean_dict,
            "scale_dict": self.scale_dict,
            "is_fitted": self.is_fitted,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.mean_dict = state["mean_dict"]
        self.scale_dict = state["scale_dict"]
        self.is_fitted = state["is_fitted"]
