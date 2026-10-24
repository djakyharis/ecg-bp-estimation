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

from typing import Dict, Tuple, Optional, Union, Any
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


class FeatureProjector(nn.Module):
    """
    Subnet MLP untuk memproyeksikan fitur statistik sebelum digabungkan dengan representasi BiLSTM.
    Mencegah fenomena Feature Drowning (Blueprint Fan et al. Pilar 2):
      Linear(in_features -> 128) -> LayerNorm(128) -> ReLU -> Dropout(dropout) -> Linear(128 -> out_dim) -> ReLU
    """

    def __init__(self, in_features: int, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFeatureFusion(nn.Module):
    """
    Subnet Adaptive Gated Fusion:
    Menggabungkan representasi BiLSTM laten (misal 512-dim) dengan proyeksi fitur statistik (misal 64-dim).
    Menghitung bobot gating adaptif alpha via Sigmoid agar model mampu menentukan seberapa
    besar mengandalkan informasi statistik morfologi untuk setiap denyut/pasien.
    Output: Concat [enc_feat, alpha * f_proj] berdimensi (enc_dim + proj_dim = 576).
    """

    def __init__(self, enc_dim: int, proj_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(enc_dim + proj_dim, proj_dim),
            nn.Sigmoid(),
        )

    def forward(self, enc_feat: torch.Tensor, proj_feat: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(torch.cat([enc_feat, proj_feat], dim=-1))
        gated_proj = proj_feat * alpha
        return torch.cat([enc_feat, gated_proj], dim=-1)


class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention Pooling over timesteps:
    Menghitung bobot atensi untuk setiap timestep sekuens dan menghasilkan
    vektor konteks berbobot shape [batch, hidden_dim].
    Menggantikan terminal states agar seluruh siklus jantung terbaca optimal.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, hidden_dim]
        scores = self.proj(x)  # [batch, seq_len, 1]
        weights = torch.softmax(scores, dim=1)  # [batch, seq_len, 1]
        context = torch.sum(x * weights, dim=1)  # [batch, hidden_dim]
        return context


class ECGMultiTaskBiLSTM(nn.Module):
    """
    Model Multi-Task BiLSTM untuk estimasi simultan SBP, DBP, dan MAP dari sinyal ECG 10 detik.
    Sesuai Tabel 1 & Section 2.3 Fan et al. (2021, Sensors) dengan ekstensi optimasi
    Spatial-Temporal Patching (Pilar 1), Feature Projection MLP (Pilar 2), Gated Fusion, dan Temporal Attention (Pilar 3).
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
        patch_size: int = 1,
        use_projection: bool = False,
        proj_dim: int = 64,
        proj_dropout: float = 0.1,
        use_gating: bool = True,
        use_attention: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_projection = use_projection
        self.proj_dim = proj_dim
        self.proj_dropout = proj_dropout
        self.use_gating = use_gating
        self.use_attention = use_attention

        eff_lstm_dropout = lstm_dropout if lstm_dropout is not None else dropout
        eff_fc_dropout = fc_dropout if fc_dropout is not None else dropout
        self.lstm_dropout = eff_lstm_dropout
        self.fc_dropout = eff_fc_dropout

        # Jika patch_size > 1, input per timestep adalah patch_size (misal 25 sampel = 0.2 detik)
        lstm_in_dim = patch_size if patch_size > 1 else input_size

        # Shared BiLSTM Encoder (Tabel 1 Fan et al., 2021)
        self.lstm = nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=eff_lstm_dropout if num_layers > 1 else 0.0,
        )

        # Dimensi output BiLSTM (2 * hidden_size karena bidirectional: 256 * 2 = 512)
        encoder_out_dim = hidden_size * 2

        # Temporal Attention Pooling (opsional)
        if use_attention:
            self.attention = TemporalAttention(encoder_out_dim)
        else:
            self.attention = None

        self.sbp_feat_dim = sbp_feat_dim
        self.dbp_feat_dim = dbp_feat_dim
        self.map_feat_dim = map_feat_dim

        # Inisialisasi Feature Projector jika use_projection aktif (Pilar 2)
        if use_projection and sbp_feat_dim > 0:
            self.sbp_proj = FeatureProjector(sbp_feat_dim, proj_dim, dropout=proj_dropout)
            sbp_in_dim = encoder_out_dim + proj_dim
        else:
            self.sbp_proj = None
            sbp_in_dim = encoder_out_dim + sbp_feat_dim

        if use_projection and dbp_feat_dim > 0:
            self.dbp_proj = FeatureProjector(dbp_feat_dim, proj_dim, dropout=proj_dropout)
            dbp_in_dim = encoder_out_dim + proj_dim
        else:
            self.dbp_proj = None
            dbp_in_dim = encoder_out_dim + dbp_feat_dim

        if use_projection and map_feat_dim > 0:
            self.map_proj = FeatureProjector(map_feat_dim, proj_dim, dropout=proj_dropout)
            map_in_dim = encoder_out_dim + proj_dim
        else:
            self.map_proj = None
            map_in_dim = encoder_out_dim + map_feat_dim

        # Gated Feature Fusion (Adaptive Gating)
        if use_projection and use_gating and sbp_feat_dim > 0:
            self.sbp_fusion = GatedFeatureFusion(encoder_out_dim, proj_dim)
        else:
            self.sbp_fusion = None

        if use_projection and use_gating and dbp_feat_dim > 0:
            self.dbp_fusion = GatedFeatureFusion(encoder_out_dim, proj_dim)
        else:
            self.dbp_fusion = None

        if use_projection and use_gating and map_feat_dim > 0:
            self.map_fusion = GatedFeatureFusion(encoder_out_dim, proj_dim)
        else:
            self.map_fusion = None

        # 3 Dedicated Task Heads (2 hidden layers: 512 -> 256 -> 1 sesuai Fan et al. 2021)
        fc1_dim = encoder_out_dim if encoder_out_dim != 512 else 512
        fc2_dim = encoder_out_dim // 2 if encoder_out_dim != 512 else 256

        self.sbp_head = TaskSpecificHead(
            in_features=sbp_in_dim,
            hidden_dim1=fc1_dim,
            hidden_dim2=fc2_dim,
            dropout=eff_fc_dropout,
        )
        self.dbp_head = TaskSpecificHead(
            in_features=dbp_in_dim,
            hidden_dim1=fc1_dim,
            hidden_dim2=fc2_dim,
            dropout=eff_fc_dropout,
        )
        self.map_head = TaskSpecificHead(
            in_features=map_in_dim,
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
            Input tensor:
            - Jika patch_size == 1: [batch, seq_len] atau [batch, seq_len, 1]
            - Jika patch_size > 1: [batch, 1250], [batch, 1250, 1], atau [batch, 50, patch_size]
        """
        if self.patch_size > 1:
            if x.dim() == 3 and x.shape[-1] == 1:
                x = x.squeeze(-1)  # [batch, total_len]
            if x.dim() == 2:
                batch_size, total_len = x.shape
                if total_len % self.patch_size != 0:
                    raise ValueError(
                        f"Panjang sinyal {total_len} tidak habis dibagi patch_size {self.patch_size}"
                    )
                seq_len = total_len // self.patch_size
                x = x.view(batch_size, seq_len, self.patch_size)  # [batch, seq_len, patch_size]
            elif x.dim() == 3 and x.shape[-1] != self.patch_size:
                raise ValueError(
                    f"Dimensi fitur input {x.shape[-1]} tidak cocok dengan patch_size {self.patch_size}"
                )
        else:
            if x.dim() == 2:
                x = x.unsqueeze(-1)  # [batch, seq_len, 1]

        # lstm_out shape: [batch, seq_len, 2 * hidden_size]
        lstm_out, _ = self.lstm(x)

        if self.use_attention and self.attention is not None:
            features = self.attention(lstm_out)  # [batch, 512]
        else:
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
            Input batch sinyal ECG, shape [batch, 1250, 1], [batch, 1250], atau [batch, 50, patch_size].
        features : dict atau torch.Tensor, optional
            Fitur statistik tambahan per tugas {'sbp': [B, d_sbp], 'dbp': [B, d_dbp], 'map': [B, d_map]}
            atau gabungan [B, total_feat_dim].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (sbp_pred, dbp_pred, map_pred) masing-masing shape [batch, 1].
        """
        enc_feat = self.extract_features(x)  # [batch, 512]

        def _fuse(enc: torch.Tensor, f_proj: torch.Tensor, fusion_mod: Optional[GatedFeatureFusion]) -> torch.Tensor:
            if fusion_mod is not None:
                return fusion_mod(enc, f_proj)
            return torch.cat([enc, f_proj], dim=-1)

        if features is not None and isinstance(features, dict):
            # SBP
            if self.sbp_feat_dim > 0 and "sbp" in features:
                f_sbp = self.sbp_proj(features["sbp"]) if self.sbp_proj is not None else features["sbp"]
                sbp_in = _fuse(enc_feat, f_sbp, self.sbp_fusion)
            else:
                sbp_in = enc_feat

            # DBP
            if self.dbp_feat_dim > 0 and "dbp" in features:
                f_dbp = self.dbp_proj(features["dbp"]) if self.dbp_proj is not None else features["dbp"]
                dbp_in = _fuse(enc_feat, f_dbp, self.dbp_fusion)
            else:
                dbp_in = enc_feat

            # MAP
            if self.map_feat_dim > 0 and "map" in features:
                f_map = self.map_proj(features["map"]) if self.map_proj is not None else features["map"]
                map_in = _fuse(enc_feat, f_map, self.map_fusion)
            else:
                map_in = enc_feat

        elif features is not None and isinstance(features, torch.Tensor):
            # Jika berupa tensor utuh [B, total_feat_dim]
            raw_sbp = features[:, : self.sbp_feat_dim] if self.sbp_feat_dim > 0 else None
            raw_dbp = features[:, self.sbp_feat_dim : self.sbp_feat_dim + self.dbp_feat_dim] if self.dbp_feat_dim > 0 else None
            raw_map = features[:, : self.map_feat_dim] if self.map_feat_dim > 0 else None

            if raw_sbp is not None:
                f_sbp = self.sbp_proj(raw_sbp) if self.sbp_proj is not None else raw_sbp
                sbp_in = _fuse(enc_feat, f_sbp, self.sbp_fusion)
            else:
                sbp_in = enc_feat

            if raw_dbp is not None:
                f_dbp = self.dbp_proj(raw_dbp) if self.dbp_proj is not None else raw_dbp
                dbp_in = _fuse(enc_feat, f_dbp, self.dbp_fusion)
            else:
                dbp_in = enc_feat

            if raw_map is not None:
                f_map = self.map_proj(raw_map) if self.map_proj is not None else raw_map
                map_in = _fuse(enc_feat, f_map, self.map_fusion)
            else:
                map_in = enc_feat
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

    def get_config(self) -> Dict[str, Any]:
        """Mengembalikan kamus konfigurasi arsitektur model untuk serialisasi & deployment."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "lstm_dropout": self.lstm_dropout,
            "fc_dropout": self.fc_dropout,
            "sbp_feat_dim": self.sbp_feat_dim,
            "dbp_feat_dim": self.dbp_feat_dim,
            "map_feat_dim": self.map_feat_dim,
            "patch_size": self.patch_size,
            "use_projection": self.use_projection,
            "proj_dim": self.proj_dim,
            "proj_dropout": self.proj_dropout,
            "use_gating": self.use_gating,
            "use_attention": self.use_attention,
        }


