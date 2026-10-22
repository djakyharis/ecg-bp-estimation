"""
Fungsi Loss Multi-Task dengan Pembobotan Adaptif (Adaptive Weight Learning).
Diadaptasi 100% dari Paper Acuan Utama:
  Fan et al. (2021) — "An Adaptive Weight Learning-Based Multitask Deep Network
  for Continuous Blood Pressure Estimation Using Electrocardiogram Signals" (Sensors, 2021).

Formulasi (Fan et al., 2021 - Section 2.3 & Algorithm 1):
  1. Loss Gabungan (Eq. 4):
     L_total = theta_s * L_sbp + theta_d * L_dbp + theta_m * L_map
     di mana theta_s + theta_d + theta_m = 1.0 (Inisialisasi awal: 1/3, 1/3, 1/3).

  2. Pembaruan Bobot Berdasarkan Tren Validation Loss (Eq. 8, 9, 10):
     - T_mean^i = |L_mean^i(k) - L_mean^i(k-1)| / L_mean^i(k)
     - T_std^i  = |L_std^i(k) - L_std^i(k-1)| / L_std^i(k)
     - theta_tilde^i = T_mean^i * T_std^i * (L_mean^i + L_std^i)
     - theta^i = theta_tilde^i / sum_j(theta_tilde^j)
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMultiTaskLoss(nn.Module):
    """
    Fungsi Loss Multi-Task dengan Skema Pembobotan Adaptif berbasis Tren Validation Loss
    sesuai Paper Acuan Fan et al. (2021, Sensors) Algorithm 1.
    """

    def __init__(
        self,
        loss_fn_type: str = "mae",
        min_weight: float = 0.15,
        momentum: float = 0.70,
        target_scales: Optional[List[float]] = None,
        task_weights: Optional[List[float]] = None,
    ):
        """
        Parameters
        ----------
        loss_fn_type : str, optional
            Tipe loss regresi: 'mae' (L1Loss), 'mse' (MSELoss), 'smooth_l1' (Huber), default 'mae'.
        min_weight : float, optional
            Batas bawah bobot minimum tugas (default: 0.15) untuk mencegah weight collapse ke 0.
        momentum : float, optional
            Faktor smoothing eksponensial antarepoch (default: 0.70) untuk meredam osilasi liar.
        target_scales : list of float, optional
            Skala normalisasi fisiologis [scale_sbp, scale_dbp, scale_map] (misal: [120.0, 80.0, 93.3]).
            Mencegah bias gradien dominan pada target berangka absolut besar (SBP) dan menyeimbangkan DBP.
        task_weights : list of float, optional
            Bobot tetap untuk tugas [w_sbp, w_dbp, w_map] jika ingin menonaktifkan adaptasi dinamis (misal: [0.5, 0.3, 0.2]).
        """
        super().__init__()
        self.loss_fn_type = loss_fn_type
        self.min_weight = min_weight
        self.momentum = momentum
        self.target_scales = list(target_scales) if target_scales is not None else None

        if task_weights is not None:
            norm_tw = np.array(task_weights, dtype=float)
            norm_tw = norm_tw / np.sum(norm_tw)
            self.is_adaptive = False
            self.register_buffer("weights", torch.tensor(norm_tw, dtype=torch.float32))
        else:
            self.is_adaptive = True
            # Inisialisasi bobot tugas: theta_i = 1/3 (Fan et al., 2021 Algorithm 1 Step 1)
            self.register_buffer("weights", torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=torch.float32))

        # Riwayat nilai rata-rata dan standar deviasi validation loss untuk perhitungan tren (Eq. 8 & 9)
        self.prev_val_mean: Optional[np.ndarray] = None
        self.prev_val_std: Optional[np.ndarray] = None

    def _compute_single_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_fn_type == "mse":
            return F.mse_loss(pred, target)
        elif self.loss_fn_type == "rmse":
            return torch.sqrt(F.mse_loss(pred, target) + 1e-8)
        elif self.loss_fn_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target, beta=1.0)
        else:
            return F.l1_loss(pred, target)

    def forward(
        self,
        sbp_pred: torch.Tensor,
        dbp_pred: torch.Tensor,
        map_pred: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Menghitung total loss adaptif multi-task sesuai Equation (4) Fan et al. (2021):
            L_total = theta_s * L_sbp + theta_d * L_dbp + theta_m * L_map

        Parameters
        ----------
        sbp_pred : torch.Tensor
            Prediksi SBP shape [batch, 1] atau [batch].
        dbp_pred : torch.Tensor
            Prediksi DBP shape [batch, 1] atau [batch].
        map_pred : torch.Tensor
            Prediksi MAP shape [batch, 1] atau [batch].
        targets : torch.Tensor
            Ground truth BP shape [batch, 3] = [SBP, DBP, MAP].

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            (total_loss_tensor, metrics_dictionary)
        """
        if isinstance(targets, torch.Tensor):
            if targets.dim() == 1:
                targets = targets.unsqueeze(0)
            sbp_true = targets[:, 0:1]
            dbp_true = targets[:, 1:2]
            map_true = targets[:, 2:3]
        else:
            sbp_true, dbp_true, map_true = targets

        sbp_pred = sbp_pred.view_as(sbp_true)
        dbp_pred = dbp_pred.view_as(dbp_true)
        map_pred = map_pred.view_as(map_true)

        # Hitung loss per tugas (MAE/RMSE/MSE/Smooth L1 sesuai loss_fn_type)
        loss_sbp = self._compute_single_loss(sbp_pred, sbp_true)
        loss_dbp = self._compute_single_loss(dbp_pred, dbp_true)
        loss_map = self._compute_single_loss(map_pred, map_true)

        if self.target_scales is not None:
            scale_s, scale_d, scale_m = self.target_scales
            eff_loss_sbp = loss_sbp / scale_s
            eff_loss_dbp = loss_dbp / scale_d
            eff_loss_map = loss_map / scale_m
        else:
            eff_loss_sbp = loss_sbp
            eff_loss_dbp = loss_dbp
            eff_loss_map = loss_map

        # Bobot aktif: theta_s, theta_d, theta_m
        theta_s, theta_d, theta_m = self.weights[0], self.weights[1], self.weights[2]

        # Total Loss (Eq. 4 Fan et al., 2021)
        total_loss = theta_s * eff_loss_sbp + theta_d * eff_loss_dbp + theta_m * eff_loss_map

        metrics = {
            "loss_total": float(total_loss.detach().cpu().item()),
            "mae_sbp": float(F.l1_loss(sbp_pred, sbp_true).detach().cpu().item()),
            "mae_dbp": float(F.l1_loss(dbp_pred, dbp_true).detach().cpu().item()),
            "mae_map": float(F.l1_loss(map_pred, map_true).detach().cpu().item()),
            "weight_sbp": float(theta_s.detach().cpu().item()),
            "weight_dbp": float(theta_d.detach().cpu().item()),
            "weight_map": float(theta_m.detach().cpu().item()),
        }

        return total_loss, metrics

    def update_weights_from_validation(
        self,
        val_sbp_losses: List[float],
        val_dbp_losses: List[float],
        val_map_losses: List[float]
    ) -> Dict[str, float]:
        """
        Memperbarui bobot tugas theta_i di akhir fase validasi sesuai Algorithm 1 (Lines 11-15)
        dan Equations (8, 9, 10) Fan et al. (2021).
        Jika target_scales aktif, tren dinormalisasi terhadap skala fisiologis target.

        Parameters
        ----------
        val_sbp_losses : List[float]
            Kumpulan error loss SBP dari seluruh batch validasi di epoch saat ini.
        val_dbp_losses : List[float]
            Kumpulan error loss DBP dari seluruh batch validasi di epoch saat ini.
        val_map_losses : List[float]
            Kumpulan error loss MAP dari seluruh batch validasi di epoch saat ini.

        Returns
        -------
        Dict[str, float]
            Dictionary bobot baru {'weight_sbp': ..., 'weight_dbp': ..., 'weight_map': ...}
        """
        if not getattr(self, "is_adaptive", True):
            return {
                "weight_sbp": float(self.weights[0].item()),
                "weight_dbp": float(self.weights[1].item()),
                "weight_map": float(self.weights[2].item()),
            }

        # Step 11: Compute mean value Li_mean(k) and standard deviation Li_std(k)
        raw_mean = np.array([
            float(np.mean(val_sbp_losses)),
            float(np.mean(val_dbp_losses)),
            float(np.mean(val_map_losses))
        ])
        raw_std = np.array([
            float(np.std(val_sbp_losses)),
            float(np.std(val_dbp_losses)),
            float(np.std(val_map_losses))
        ])

        if self.target_scales is not None:
            scales = np.array(self.target_scales, dtype=float)
            cur_mean = raw_mean / scales
            cur_std = raw_std / scales
        else:
            cur_mean = raw_mean
            cur_std = raw_std

        # Pastikan tidak ada pembagian dengan nol
        eps = 1e-8

        if self.prev_val_mean is not None and self.prev_val_std is not None:
            # Step 13: Compute trend of mean value Ti_mean based on Eq. (8)
            # Ti_mean = |Li_mean(k) - Li_mean(k-1)| / (Li_mean(k) + eps)
            t_mean = np.abs(cur_mean - self.prev_val_mean) / (cur_mean + eps)

            # Step 14: Compute trend of standard deviation Ti_std based on Eq. (9)
            # Ti_std = |Li_std(k) - Li_std(k-1)| / (Li_std(k) + eps)
            t_std = np.abs(cur_std - self.prev_val_std) / (cur_std + eps)

            # Step 15: Compute task-specific weight theta_i based on Eq. (10)
            # theta_tilde = Ti_mean * Ti_std * (Li_mean + Li_std)
            raw_weights = t_mean * t_std * (cur_mean + cur_std)

            sum_raw = np.sum(raw_weights)
            if sum_raw > eps:
                norm_weights = raw_weights / sum_raw
            else:
                norm_weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

            # Stabilisasi 1: Batas Bawah Bobot (Floor) agar tidak ada tugas yang diabaikan (weight=0)
            if self.min_weight > 0.0:
                norm_weights = np.maximum(norm_weights, self.min_weight)
                norm_weights = norm_weights / np.sum(norm_weights)

            # Stabilisasi 2: Exponential Moving Average (Momentum) untuk meredam osilasi antarepoch
            if self.momentum > 0.0:
                old_w = self.weights.detach().cpu().numpy()
                norm_weights = self.momentum * old_w + (1.0 - self.momentum) * norm_weights
                norm_weights = norm_weights / np.sum(norm_weights)

            # Update buffer weights di PyTorch
            self.weights.copy_(torch.tensor(norm_weights, dtype=torch.float32, device=self.weights.device))

        # Simpan nilai epoch saat ini untuk perhitungan tren epoch berikutnya
        self.prev_val_mean = cur_mean
        self.prev_val_std = cur_std

        return {
            "weight_sbp": float(self.weights[0].item()),
            "weight_dbp": float(self.weights[1].item()),
            "weight_map": float(self.weights[2].item()),
        }

    def get_state(self) -> Dict[str, any]:
        """Menyimpan status bobot dan tren validasi untuk checkpointing."""
        return {
            "weights": self.weights.detach().cpu().clone(),
            "prev_val_mean": self.prev_val_mean.copy() if self.prev_val_mean is not None else None,
            "prev_val_std": self.prev_val_std.copy() if self.prev_val_std is not None else None,
            "target_scales": self.target_scales,
        }

    def load_state(self, state: Dict[str, any]) -> None:
        """Memulihkan status bobot dan tren validasi saat melanjutkan training (resume)."""
        if "weights" in state and state["weights"] is not None:
            w = state["weights"]
            if not isinstance(w, torch.Tensor):
                w = torch.tensor(w, dtype=torch.float32)
            self.weights.copy_(w.to(device=self.weights.device))
        if "prev_val_mean" in state and state["prev_val_mean"] is not None:
            self.prev_val_mean = np.array(state["prev_val_mean"], dtype=float)
        if "prev_val_std" in state and state["prev_val_std"] is not None:
            self.prev_val_std = np.array(state["prev_val_std"], dtype=float)
        if "target_scales" in state:
            self.target_scales = state["target_scales"]

