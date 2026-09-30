"""
Arsitektur Deep Learning Multi-Task BiLSTM untuk Estimasi Tekanan Darah (SBP, DBP, MAP).
Sesuai rancangan skripsi dan ARCHITECTURE.md:
  - Input: Sinyal ECG ternormalisasi shape [batch, 1250, 1] atau [batch, 1250] @ 125 Hz (10 detik)
  - Shared Encoder: 2-Layer Bidirectional LSTM (hidden=256, output concat=512)
  - 3 Dedicated Heads:
      * SBP Head: Linear(512 -> 512) -> ReLU -> Dropout(0.2) -> Linear(512 -> 256) -> ReLU -> Dropout(0.2) -> Linear(256 -> 1)
      * DBP Head: Linear(512 -> 512) -> ReLU -> Dropout(0.2) -> Linear(512 -> 256) -> ReLU -> Dropout(0.2) -> Linear(256 -> 1)
      * MAP Head: Linear(512 -> 512) -> ReLU -> Dropout(0.2) -> Linear(512 -> 256) -> ReLU -> Dropout(0.2) -> Linear(256 -> 1)
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


class TaskSpecificHead(nn.Module):
    """
    Fully Connected Regression Head untuk masing-masing target tekanan darah.
    Fan et al. (2021) Section 2.3 & Table 1:
      - Hidden Layer 1: 512 units + ReLU
      - Hidden Layer 2: 256 units + ReLU
      - Output Layer: 1 unit (Linear murni / tanpa aktivasi)
    """

    def __init__(
        self,
        in_features: int = 512,
        hidden_dim1: int = 512,
        hidden_dim2: int = 256,
        dropout: float = 0.0
    ):
        super().__init__()
        layers = [
            nn.Linear(in_features, hidden_dim1),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
        ])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim2, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Feature representation dari shared encoder, shape [batch, in_features].

        Returns
        -------
        torch.Tensor
            Prediksi skalar per sampel, shape [batch, 1].
        """
        return self.net(x)


class ECGMultiTaskBiLSTM(nn.Module):
    """
    Model Multi-Task BiLSTM untuk estimasi simultan SBP, DBP, dan MAP dari sinyal ECG 10 detik.
    Sesuai Tabel 1 & Section 2.3 Fan et al. (2021, Sensors).
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Shared BiLSTM Encoder (Tabel 1 Fan et al., 2021)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dimensi output BiLSTM (2 * hidden_size karena bidirectional: 256 * 2 = 512)
        encoder_out_dim = hidden_size * 2

        # 3 Dedicated Task Heads (2 hidden layers: 512 -> 256 -> 1 sesuai Fan et al. 2021)
        fc1_dim = encoder_out_dim if encoder_out_dim != 512 else 512
        fc2_dim = encoder_out_dim // 2 if encoder_out_dim != 512 else 256
        self.sbp_head = TaskSpecificHead(
            in_features=encoder_out_dim, hidden_dim1=fc1_dim, hidden_dim2=fc2_dim, dropout=dropout
        )
        self.dbp_head = TaskSpecificHead(
            in_features=encoder_out_dim, hidden_dim1=fc1_dim, hidden_dim2=fc2_dim, dropout=dropout
        )
        self.map_head = TaskSpecificHead(
            in_features=encoder_out_dim, hidden_dim1=fc1_dim, hidden_dim2=fc2_dim, dropout=dropout
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mengekstrak representasi fitur terminal dari Shared BiLSTM Encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch, seq_len] atau [batch, seq_len, 1].
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, seq_len, 1]

        # lstm_out shape: [batch, seq_len, 2 * hidden_size]
        lstm_out, _ = self.lstm(x)

        # Ekstraksi terminal state: forward akhir dan backward awal (Fan et al., 2021)
        fwd_last = lstm_out[:, -1, : self.hidden_size]
        bwd_first = lstm_out[:, 0, self.hidden_size :]
        features = torch.cat([fwd_last, bwd_first], dim=-1)  # [batch, 512]

        return features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass model untuk memprediksi SBP, DBP, dan MAP.

        Parameters
        ----------
        x : torch.Tensor
            Input batch sinyal ECG, shape [batch, 1250, 1] atau [batch, 1250].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (sbp_pred, dbp_pred, map_pred) masing-masing shape [batch, 1].
        """
        features = self.extract_features(x)

        sbp = self.sbp_head(features)
        dbp = self.dbp_head(features)
        map_val = self.map_head(features)

        return sbp, dbp, map_val

    def predict_all(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper yang mengembalikan tensor gabungan [batch, 3] = [SBP, DBP, MAP].
        """
        sbp, dbp, map_val = self.forward(x)
        return torch.cat([sbp, dbp, map_val], dim=-1)
