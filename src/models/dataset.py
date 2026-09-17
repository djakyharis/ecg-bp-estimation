"""
Modul PyTorch Dataset & DataLoader untuk Sinyal ECG dan Target Tekanan Darah.
Mendukung pemuatan berkas .npz hasil pra-pemrosesan (dari data/processed/).
"""

import os
import glob
from typing import List, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ECGBPDataset(Dataset):
    """
    Dataset PyTorch untuk pasangan sinyal ECG (1250 sampel) dan target BP (SBP, DBP, MAP).
    """

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        transform=None
    ):
        """
        Parameters
        ----------
        X : np.ndarray atau torch.Tensor
            Array sinyal ECG shape [N, 1250] atau [N, 1250, 1].
        y : np.ndarray atau torch.Tensor
            Array target tekanan darah shape [N, 3] = [SBP, DBP, MAP].
        transform : callable, optional
            Fungsi transformasi augmentasi tambahan.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        if X.dim() == 2:
            X = X.unsqueeze(-1)  # [N, 1250, 1]

        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    @classmethod
    def from_npz_files(
        cls,
        file_paths: Union[str, List[str]],
        max_samples: Optional[int] = None
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

        return cls(X=X_cat, y=y_cat)


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
