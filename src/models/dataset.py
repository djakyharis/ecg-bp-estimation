"""
Modul PyTorch Dataset & DataLoader untuk Sinyal ECG dan Target Tekanan Darah.
Mendukung pemuatan berkas .npz hasil pra-pemrosesan (dari data/processed/)
serta ekstraksi dan penyuntikan fitur statistik terpandu (Hybrid BiLSTM).
"""

import os
import glob
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.features.statistical import extract_statistical_features, FeatureStandardScaler


class ECGBPDataset(Dataset):
    """
    Dataset PyTorch untuk pasangan sinyal ECG (1250 sampel), target BP (SBP, DBP, MAP),
    dan fitur statistik terpandu (SBP: 6 fitur, DBP: 6 fitur, MAP: 12 fitur).
    """

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        features: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
        compute_features: bool = False,
        feature_scaler: Optional[FeatureStandardScaler] = None,
        transform=None
    ):
        """
        Parameters
        ----------
        X : np.ndarray atau torch.Tensor
            Array sinyal ECG shape [N, 1250] atau [N, 1250, 1].
        y : np.ndarray atau torch.Tensor
            Array target tekanan darah shape [N, 3] = [SBP, DBP, MAP].
        features : dict, optional
            Dictionary fitur statistik yang sudah diekstrak: {'sbp': [N, 6], 'dbp': [N, 6], 'map': [N, 12]}.
        compute_features : bool, default False
            Jika True dan features belum ada, otomatis mengekstrak 12 fitur statistik dari X.
        feature_scaler : FeatureStandardScaler, optional
            Scaler untuk menormalkan fitur statistik (z-score).
        transform : callable, optional
            Fungsi transformasi augmentasi tambahan.
        """
        if isinstance(X, np.ndarray):
            X_arr = X
            X = torch.from_numpy(X).float()
        else:
            X_arr = X.detach().cpu().numpy()

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        if X.dim() == 2:
            X = X.unsqueeze(-1)  # [N, 1250, 1]

        self.X = X
        self.y = y
        self.transform = transform
        self.feature_scaler = feature_scaler

        # Ekstraksi atau inisialisasi fitur statistik
        if features is not None:
            self.features = self._format_features(features)
        elif compute_features:
            raw_feats = extract_statistical_features(X_arr)
            if feature_scaler is not None:
                if not feature_scaler.is_fitted:
                    raw_feats = feature_scaler.fit_transform(raw_feats)
                else:
                    raw_feats = feature_scaler.transform(raw_feats)
            self.features = self._format_features(raw_feats)
        else:
            self.features = None

    def _format_features(self, feats: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        formatted = {}
        for k in ["sbp", "dbp", "map"]:
            if k in feats:
                v = feats[k]
                if isinstance(v, np.ndarray):
                    formatted[k] = torch.from_numpy(v).float()
                else:
                    formatted[k] = v.float()
        return formatted

    def set_features(self, features: Dict[str, Union[np.ndarray, torch.Tensor]]):
        """Menyetel atau memperbarui fitur statistik yang sudah diskalakan."""
        self.features = self._format_features(features)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        if self.features is not None:
            f = {
                "sbp": self.features["sbp"][idx],
                "dbp": self.features["dbp"][idx],
                "map": self.features["map"][idx],
            }
            return x, y, f

        return x, y

    @classmethod
    def from_npz_files(
        cls,
        file_paths: Union[str, List[str]],
        max_samples: Optional[int] = None,
        compute_features: bool = False,
        feature_scaler: Optional[FeatureStandardScaler] = None,
    ) -> "ECGBPDataset":
        """
        Membuat instance ECGBPDataset dari satu atau beberapa file .npz.
        """
        if isinstance(file_paths, str):
            if os.path.isdir(file_paths):
                file_paths = sorted(glob.glob(os.path.join(file_paths, "*.npz")))
            else:
                file_paths = [file_paths]

        all_X, all_y = [], []
        for path in file_paths:
            if not os.path.exists(path):
                continue
            data = np.load(path)
            all_X.append(data["X"])
            all_y.append(data["y"])

        if not all_X:
            raise ValueError(f"Tidak ada data valid yang dapat dimuat dari: {file_paths}")

        X_cat = np.concatenate(all_X, axis=0)
        y_cat = np.concatenate(all_y, axis=0)

        if max_samples is not None and max_samples < len(X_cat):
            indices = np.random.choice(len(X_cat), max_samples, replace=False)
            X_cat = X_cat[indices]
            y_cat = y_cat[indices]

        return cls(
            X=X_cat,
            y=y_cat,
            compute_features=compute_features,
            feature_scaler=feature_scaler,
        )


def create_dataloaders(
    dataset: ECGBPDataset,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Membagi dataset menjadi Train, Validation, dan Test DataLoaders.
    """
    torch.manual_seed(seed)
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len]
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
