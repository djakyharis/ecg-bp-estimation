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

from typing import Dict, Tuple, Optional, Union
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
        lstm_dropout: Optional[float] = None,
        fc_dropout: Optional[float] = None,
        sbp_feat_dim: int = 0,
        dbp_feat_dim: int = 0,
        map_feat_dim: int = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        eff_lstm_dropout = lstm_dropout if lstm_dropout is not None else dropout
        eff_fc_dropout = fc_dropout if fc_dropout is not None else dropout
        self.lstm_dropout = eff_lstm_dropout
        self.fc_dropout = eff_fc_dropout

        # Shared BiLSTM Encoder (Tabel 1 Fan et al., 2021)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=eff_lstm_dropout if num_layers > 1 else 0.0,
        )

        # Dimensi output BiLSTM (2 * hidden_size karena bidirectional: 256 * 2 = 512)
        encoder_out_dim = hidden_size * 2

        self.sbp_feat_dim = sbp_feat_dim
        self.dbp_feat_dim = dbp_feat_dim
        self.map_feat_dim = map_feat_dim

        # 3 Dedicated Task Heads (2 hidden layers: 512 -> 256 -> 1 sesuai Fan et al. 2021)
        # Mendukung Hybrid Architecture: in_features = 512 (BiLSTM) + fitur statistik spesifik
        fc1_dim = encoder_out_dim if encoder_out_dim != 512 else 512
        fc2_dim = encoder_out_dim // 2 if encoder_out_dim != 512 else 256

        self.sbp_head = TaskSpecificHead(
            in_features=encoder_out_dim + sbp_feat_dim,
            hidden_dim1=fc1_dim,
            hidden_dim2=fc2_dim,
            dropout=eff_fc_dropout,
        )
        self.dbp_head = TaskSpecificHead(
            in_features=encoder_out_dim + dbp_feat_dim,
            hidden_dim1=fc1_dim,
            hidden_dim2=fc2_dim,
            dropout=eff_fc_dropout,
        )
        self.map_head = TaskSpecificHead(
            in_features=encoder_out_dim + map_feat_dim,
            hidden_dim1=fc1_dim,
            hidden_dim2=fc2_dim,
            dropout=eff_fc_dropout,
        )

        self._init_weights()

    def _init_weights(self):
        """Inisialisasi bobot dan prior bias fisiologis (SBP~120, DBP~80, MAP~93.3 mmHg)."""
        # Set prior bias fisiologis pada layer regresi output akhir
        if hasattr(self.sbp_head.net[-1], "bias") and self.sbp_head.net[-1].bias is not None:
            nn.init.constant_(self.sbp_head.net[-1].bias, 120.0)
        if hasattr(self.dbp_head.net[-1], "bias") and self.dbp_head.net[-1].bias is not None:
            nn.init.constant_(self.dbp_head.net[-1].bias, 80.0)
        if hasattr(self.map_head.net[-1], "bias") and self.map_head.net[-1].bias is not None:
            nn.init.constant_(self.map_head.net[-1].bias, 93.3)

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

    def forward(
        self,
        x: torch.Tensor,
        features: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass model untuk memprediksi SBP, DBP, dan MAP.

        Parameters
        ----------
        x : torch.Tensor
            Input batch sinyal ECG, shape [batch, 1250, 1] atau [batch, 1250].
        features : dict atau torch.Tensor, optional
            Fitur statistik tambahan per tugas {'sbp': [B, 6], 'dbp': [B, 6], 'map': [B, 12]}
            atau gabungan [B, 12].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (sbp_pred, dbp_pred, map_pred) masing-masing shape [batch, 1].
        """
        enc_feat = self.extract_features(x)  # [batch, 512]

        if features is not None and isinstance(features, dict):
            sbp_in = torch.cat([enc_feat, features["sbp"]], dim=-1) if self.sbp_feat_dim > 0 and "sbp" in features else enc_feat
            dbp_in = torch.cat([enc_feat, features["dbp"]], dim=-1) if self.dbp_feat_dim > 0 and "dbp" in features else enc_feat
            map_in = torch.cat([enc_feat, features["map"]], dim=-1) if self.map_feat_dim > 0 and "map" in features else enc_feat
        elif features is not None and isinstance(features, torch.Tensor):
            # Jika berupa tensor utuh [B, 12]
            f_sbp = features[:, : self.sbp_feat_dim] if self.sbp_feat_dim > 0 else None
            f_dbp = features[:, self.sbp_feat_dim : self.sbp_feat_dim + self.dbp_feat_dim] if self.dbp_feat_dim > 0 else None
            f_map = features[:, : self.map_feat_dim] if self.map_feat_dim > 0 else None

            sbp_in = torch.cat([enc_feat, f_sbp], dim=-1) if f_sbp is not None else enc_feat
            dbp_in = torch.cat([enc_feat, f_dbp], dim=-1) if f_dbp is not None else enc_feat
            map_in = torch.cat([enc_feat, f_map], dim=-1) if f_map is not None else enc_feat
        else:
            sbp_in = enc_feat
            dbp_in = enc_feat
            map_in = enc_feat

        sbp = self.sbp_head(sbp_in)
        dbp = self.dbp_head(dbp_in)
        map_val = self.map_head(map_in)

        return sbp, dbp, map_val

    def predict_all(
        self,
        x: torch.Tensor,
        features: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Helper yang mengembalikan tensor gabungan [batch, 3] = [SBP, DBP, MAP].
        """
        sbp, dbp, map_val = self.forward(x, features=features)
        return torch.cat([sbp, dbp, map_val], dim=-1)
