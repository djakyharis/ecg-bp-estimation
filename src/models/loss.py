"""
Fungsi Loss Multi-Task dengan Pembobotan Adaptif (Adaptive Loss).
Mengimplementasikan Homoscedastic Uncertainty Weighting (Kendall et al., 2018)
untuk menyeimbangkan gradien pembelajaran antara tugas SBP, DBP, dan MAP.

Formulasi:
    L_total = sum_{t in {sbp, dbp, map}} ( 0.5 * exp(-log_var_t) * L_t + 0.5 * log_var_t )
    di mana log_var_t adalah parameter terpelajari (learnable log variance).
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMultiTaskLoss(nn.Module):
    """
    Multi-Task Loss dengan pembobotan dinamis adaptif berbasis ketidakpastian homoskedastik.
    """

    def __init__(self, loss_fn_type: str = "mae", init_log_var: float = 0.0):
        """
        Parameters
        ----------
        loss_fn_type : str, optional
            Tipe loss regresi: 'mae' (L1Loss) atau 'mse' (MSELoss), default 'mae'.
        init_log_var : float, optional
            Nilai inisialisasi log variance (log(sigma^2)), default 0.0 (artinya weight awal ~ 1.0).
        """
        super().__init__()
        self.loss_fn_type = loss_fn_type

        # Parameter terpelajari log(sigma^2) untuk masing-masing task
        # Dioptimasi bersamaan dengan parameter bobot jaringan
        self.log_var_sbp = nn.Parameter(torch.tensor(init_log_var, dtype=torch.float32))
        self.log_var_dbp = nn.Parameter(torch.tensor(init_log_var, dtype=torch.float32))
        self.log_var_map = nn.Parameter(torch.tensor(init_log_var, dtype=torch.float32))

    def _compute_single_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_fn_type == "mse":
            return F.mse_loss(pred, target)
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
        Menghitung total loss adaptif multi-task.

        Parameters
        ----------
        sbp_pred : torch.Tensor
            Prediksi SBP shape [batch, 1] atau [batch].
        dbp_pred : torch.Tensor
            Prediksi DBP shape [batch, 1] atau [batch].
        map_pred : torch.Tensor
            Prediksi MAP shape [batch, 1] atau [batch].
        targets : torch.Tensor
            Ground truth BP shape [batch, 3] = [SBP, DBP, MAP] atau tuple 3 tensor.

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

        # Reshape prediksi jika diperlukan
        sbp_pred = sbp_pred.view_as(sbp_true)
        dbp_pred = dbp_pred.view_as(dbp_true)
        map_pred = map_pred.view_as(map_true)

        # Hitung base loss (MAE atau MSE) per task
        loss_sbp = self._compute_single_loss(sbp_pred, sbp_true)
        loss_dbp = self._compute_single_loss(dbp_pred, dbp_true)
        loss_map = self._compute_single_loss(map_pred, map_true)

        # Presisi (precision = exp(-log_var))
        prec_sbp = torch.exp(-self.log_var_sbp)
        prec_dbp = torch.exp(-self.log_var_dbp)
        prec_map = torch.exp(-self.log_var_map)

        # Weighted uncertainty loss + regularization term
        weighted_loss_sbp = 0.5 * prec_sbp * loss_sbp + 0.5 * self.log_var_sbp
        weighted_loss_dbp = 0.5 * prec_dbp * loss_dbp + 0.5 * self.log_var_dbp
        weighted_loss_map = 0.5 * prec_map * loss_map + 0.5 * self.log_var_map

        total_loss = weighted_loss_sbp + weighted_loss_dbp + weighted_loss_map

        metrics = {
            "loss_total": float(total_loss.detach().cpu().item()),
            "mae_sbp": float(F.l1_loss(sbp_pred, sbp_true).detach().cpu().item()),
            "mae_dbp": float(F.l1_loss(dbp_pred, dbp_true).detach().cpu().item()),
            "mae_map": float(F.l1_loss(map_pred, map_true).detach().cpu().item()),
            "weight_sbp": float(prec_sbp.detach().cpu().item()),
            "weight_dbp": float(prec_dbp.detach().cpu().item()),
            "weight_map": float(prec_map.detach().cpu().item()),
        }

        return total_loss, metrics
