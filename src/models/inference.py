"""
Pipeline Inferensi Estimasi Tekanan Darah End-to-End & Interpretasi Klinis AHA/ACC.
Menggabungkan ECGPreprocessingPipeline dan ECGMultiTaskBiLSTM.
"""

from typing import Dict, Any, Union, Optional
import numpy as np
import torch

from ..preprocessing.pipeline import ECGPreprocessingPipeline
from ..features.statistical import extract_statistical_features, FeatureStandardScaler
from .bilstm import ECGMultiTaskBiLSTM


def classify_blood_pressure_aha(sbp: float, dbp: float) -> Dict[str, str]:
    """
    Mengklasifikasikan kategori tekanan darah berdasarkan standar AHA/ACC (American Heart Association).

    Parameters
    ----------
    sbp : float
        Systolic Blood Pressure (mmHg).
    dbp : float
        Diastolic Blood Pressure (mmHg).

    Returns
    -------
    Dict[str, str]
        Kategori, deskripsi, warna status, dan rekomendasi klinis.
    """
    if sbp > 180 or dbp > 120:
        return {
            "category": "Krisis Hipertensi (Hypertensive Crisis)",
            "severity": "critical",
            "color": "#e63946",  # Merah tua
            "description": "Kondisi darurat medis: Tekanan darah sangat tinggi yang memerlukan evaluasi klinis segera.",
        }
    elif sbp >= 140 or dbp >= 90:
        return {
            "category": "Hipertensi Stage 2",
            "severity": "high",
            "color": "#f77f00",  # Oranye tua
            "description": "Tekanan darah tinggi tingkat 2: Memerlukan penanganan gaya hidup dan evaluasi terapi medis.",
        }
    elif (130 <= sbp <= 139) or (80 <= dbp <= 89):
        return {
            "category": "Hipertensi Stage 1",
            "severity": "moderate",
            "color": "#fcbf49",  # Kuning / amber
            "description": "Tekanan darah tinggi tingkat 1: Disarankan perubahan pola hidup dan pemantauan berkala.",
        }
    elif (120 <= sbp <= 129) and dbp < 80:
        return {
            "category": "Pre-Hipertensi (Elevated)",
            "severity": "mild",
            "color": "#90be6d",  # Hijau muda
            "description": "Tekanan darah di atas normal: Tingkatkan aktivitas fisik dan kurangi konsumsi natrium.",
        }
    else:
        return {
            "category": "Normal",
            "severity": "normal",
            "color": "#2a9d8f",  # Hijau toska
            "description": "Tekanan darah dalam batas ideal optimal.",
        }


class ECGBPPredictor:
    """
    Kelas inferensi terpadu: Menerima sinyal ECG mentah -> Pra-pemrosesan -> Ekstraksi Fitur (jika Hybrid) -> Prediksi BiLSTM -> Hasil Klinis.
    """

    def __init__(
        self,
        model: Optional[ECGMultiTaskBiLSTM] = None,
        weights_path: Optional[str] = None,
        feature_scaler: Optional[FeatureStandardScaler] = None,
        device: Optional[str] = None,
        target_fs: float = 125.0,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.target_fs = target_fs
        self.pipeline = ECGPreprocessingPipeline()
        self.feature_scaler = feature_scaler

        if weights_path is not None:
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Checkpoint format lengkap dari train.py
                model_kwargs = checkpoint.get("model_kwargs", {})
                if model is None:
                    self.model = ECGMultiTaskBiLSTM(**model_kwargs).to(self.device)
                else:
                    self.model = model.to(self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])

                if self.feature_scaler is None and checkpoint.get("feature_scaler_state") is not None:
                    self.feature_scaler = FeatureStandardScaler()
                    self.feature_scaler.load_state_dict(checkpoint["feature_scaler_state"])
            elif isinstance(checkpoint, dict):
                # Format state_dict langsung
                if model is None:
                    sbp_w = checkpoint.get("sbp_head.fc.0.weight")
                    if sbp_w is not None and sbp_w.shape[1] > 512:
                        sbp_feat = sbp_w.shape[1] - 512
                        dbp_w = checkpoint.get("dbp_head.fc.0.weight")
                        dbp_feat = dbp_w.shape[1] - 512 if dbp_w is not None else sbp_feat
                        map_w = checkpoint.get("map_head.fc.0.weight")
                        map_feat = map_w.shape[1] - 512 if map_w is not None else sbp_feat * 2
                        self.model = ECGMultiTaskBiLSTM(
                            input_size=1, hidden_size=256, num_layers=2,
                            sbp_feat_dim=sbp_feat, dbp_feat_dim=dbp_feat, map_feat_dim=map_feat
                        ).to(self.device)
                    else:
                        self.model = ECGMultiTaskBiLSTM(input_size=1, hidden_size=256, num_layers=2).to(self.device)
                else:
                    self.model = model.to(self.device)
                self.model.load_state_dict(checkpoint)
            else:
                raise ValueError(f"Format checkpoint tidak dikenali dari {weights_path}")
        else:
            if model is not None:
                self.model = model.to(self.device)
            else:
                self.model = ECGMultiTaskBiLSTM(input_size=1, hidden_size=256, num_layers=2).to(self.device)

        self.model.eval()

    def predict(
        self,
        raw_ecg_signal: np.ndarray,
        input_fs: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Menjalankan estimasi tekanan darah lengkap dari sinyal ECG mentah.

        Parameters
        ----------
        raw_ecg_signal : np.ndarray
            Sinyal ECG mentah 1D array (minimal 1250 sampel untuk 10 detik @ 125 Hz).
        input_fs : float, optional
            Laju sampling input (default 125.0 Hz).

        Returns
        -------
        Dict[str, Any]
            Hasil estimasi [SBP, DBP, MAP], sinyal bersih pra-pemrosesan, dan analisis AHA/ACC.
        """
        if raw_ecg_signal.ndim > 1:
            raw_ecg_signal = raw_ecg_signal.flatten()

        # 1. Jalankan pra-pemrosesan (Notch -> Bandpass -> Percentile Scale)
        clean_ecg = self.pipeline.process(raw_ecg_signal, input_fs=input_fs)

        # Pastikan ukuran pas 1250 sampel (10 detik @ 125 Hz)
        epoch_len = int(self.target_fs * 10.0)
        if len(clean_ecg) >= epoch_len:
            clean_epoch = clean_ecg[:epoch_len]
        else:
            # Padding jika kurang
            pad_len = epoch_len - len(clean_ecg)
            clean_epoch = np.pad(clean_ecg, (0, pad_len), mode="edge")

        # 2. Forward pass model
        tensor_in = torch.from_numpy(clean_epoch).float().unsqueeze(0).unsqueeze(-1).to(self.device)  # [1, 1250, 1]

        # Cek apakah model membutuhkan injeksi fitur statistik (Hybrid)
        use_features = (
            getattr(self.model, "sbp_feat_dim", 0) > 0
            or getattr(self.model, "dbp_feat_dim", 0) > 0
            or getattr(self.model, "map_feat_dim", 0) > 0
        )

        with torch.no_grad():
            if use_features:
                feats = extract_statistical_features(clean_epoch)
                if self.feature_scaler is not None and getattr(self.feature_scaler, "is_fitted", False):
                    feats = self.feature_scaler.transform(feats)
                features_tensors = {
                    k: torch.from_numpy(v).float().to(self.device)
                    for k, v in feats.items()
                }
                sbp_t, dbp_t, map_t = self.model(tensor_in, features=features_tensors)
            else:
                sbp_t, dbp_t, map_t = self.model(tensor_in)

            sbp = float(sbp_t.squeeze().cpu().numpy())
            dbp = float(dbp_t.squeeze().cpu().numpy())
            map_val = float(map_t.squeeze().cpu().numpy())

        # 3. Interpretasi Klinis
        clinical_info = classify_blood_pressure_aha(sbp, dbp)

        return {
            "sbp": round(sbp, 2),
            "dbp": round(dbp, 2),
            "map": round(map_val, 2),
            "clean_ecg": clean_epoch.tolist(),
            "raw_ecg": raw_ecg_signal[:epoch_len].tolist(),
            "clinical": clinical_info,
        }
