"""
Unit Tests untuk Modul Arsitektur Deep Learning (BiLSTM, Loss, Dataset, Inference).
"""

import pytest
import numpy as np
import torch

from src.models.bilstm import ECGMultiTaskBiLSTM, TaskSpecificHead, FeatureProjector
from src.models.loss import AdaptiveMultiTaskLoss
from src.models.dataset import ECGBPDataset, create_dataloaders
from src.models.inference import ECGBPPredictor, classify_blood_pressure_aha


def test_bilstm_forward_pass_shape():
    """Memverifikasi bahwa forward pass menghasilkan tensor berdimensi tepat."""
    model = ECGMultiTaskBiLSTM(input_size=1, hidden_size=64, num_layers=2)
    model.eval()

    batch_size = 4
    seq_len = 1250

    # Input 3D: [batch, seq_len, 1]
    x_3d = torch.randn(batch_size, seq_len, 1)
    sbp, dbp, map_val = model(x_3d)

    assert sbp.shape == (batch_size, 1)
    assert dbp.shape == (batch_size, 1)
    assert map_val.shape == (batch_size, 1)

    # Input 2D: [batch, seq_len]
    x_2d = torch.randn(batch_size, seq_len)
    pred_all = model.predict_all(x_2d)
    assert pred_all.shape == (batch_size, 3)


def test_feature_projector():
    """Memverifikasi subnet FeatureProjector (Pilar 2) memproyeksikan fitur dengan benar."""
    proj = FeatureProjector(in_features=25, out_dim=64, dropout=0.1)
    x = torch.randn(8, 25)
    out = proj(x)
    assert out.shape == (8, 64)
    assert (out >= 0.0).all()  # Terminal ReLU


def test_bilstm_patching_forward_pass():
    """Memverifikasi spatial-temporal patching 50x25 (Pilar 1) bekerja dengan berbagai format input."""
    model = ECGMultiTaskBiLSTM(
        hidden_size=64,
        num_layers=1,
        patch_size=25,
    )
    model.eval()

    batch_size = 4

    # Format 1: 1D flat [B, 1250]
    x_flat = torch.randn(batch_size, 1250)
    sbp, dbp, mp = model(x_flat)
    assert sbp.shape == (batch_size, 1)
    assert dbp.shape == (batch_size, 1)
    assert mp.shape == (batch_size, 1)

    # Format 2: Shimmer sensor format [B, 1250, 1]
    x_3d = torch.randn(batch_size, 1250, 1)
    sbp, dbp, mp = model(x_3d)
    assert sbp.shape == (batch_size, 1)

    # Format 3: Pre-patched [B, 50, 25]
    x_patched = torch.randn(batch_size, 50, 25)
    sbp, dbp, mp = model(x_patched)
    assert sbp.shape == (batch_size, 1)

    # Verifikasi konfigurasi
    cfg = model.get_config()
    assert cfg["patch_size"] == 25
    assert cfg["hidden_size"] == 64


def test_bilstm_with_patching_and_feature_projection():
    """Memverifikasi integrasi Pilar 1 (Patch 50x25) dan Pilar 2 (Feature Projection MLP)."""
    model = ECGMultiTaskBiLSTM(
        hidden_size=64,
        num_layers=1,
        patch_size=25,
        use_projection=True,
        proj_dim=64,
        sbp_feat_dim=13,
        dbp_feat_dim=12,
        map_feat_dim=25,
    )

    batch_size = 4
    x = torch.randn(batch_size, 1250)
    feats = {
        "sbp": torch.randn(batch_size, 13),
        "dbp": torch.randn(batch_size, 12),
        "map": torch.randn(batch_size, 25),
    }

    sbp, dbp, mp = model(x, features=feats)
    assert sbp.shape == (batch_size, 1)
    assert dbp.shape == (batch_size, 1)
    assert mp.shape == (batch_size, 1)

    # Backward pass
    loss = sbp.sum() + dbp.sum() + mp.sum()
    loss.backward()
    assert model.lstm.weight_ih_l0.grad is not None
    assert model.sbp_proj.net[0].weight.grad is not None


def test_adaptive_multi_task_loss_backward():
    """Memverifikasi perhitungan loss adaptif dan kalkulasi gradien backward pass."""
    model = ECGMultiTaskBiLSTM(input_size=1, hidden_size=32, num_layers=1)
    criterion = AdaptiveMultiTaskLoss(loss_fn_type="mae")

    x = torch.randn(8, 1250, 1)
    targets = torch.tensor([[120.0, 80.0, 93.3]] * 8, dtype=torch.float32)

    sbp, dbp, map_val = model(x)
    total_loss, metrics = criterion(sbp, dbp, map_val, targets)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() > 0.0
    assert "mae_sbp" in metrics
    assert "mae_dbp" in metrics
    assert "mae_map" in metrics

    # Test backward pass
    total_loss.backward()
    assert model.lstm.weight_ih_l0.grad is not None


def test_dataset_and_dataloaders():
    """Memverifikasi fungsionalitas ECGBPDataset dan pembagian DataLoader."""
    n_samples = 50
    X_dummy = np.random.randn(n_samples, 1250).astype(np.float32)
    y_dummy = np.random.randn(n_samples, 3).astype(np.float32)

    dataset = ECGBPDataset(X=X_dummy, y=y_dummy)
    assert len(dataset) == n_samples

    x_0, y_0 = dataset[0]
    assert x_0.shape == (1250, 1)
    assert y_0.shape == (3,)

    train_l, val_l, test_l = create_dataloaders(dataset, batch_size=8, train_ratio=0.8, val_ratio=0.1)
    assert len(train_l) > 0
    assert len(val_l) > 0
    assert len(test_l) > 0


def test_aha_blood_pressure_classification():
    """Memverifikasi keakuratan klasifikasi klinis AHA/ACC."""
    normal = classify_blood_pressure_aha(115.0, 75.0)
    assert normal["severity"] == "normal"

    elevated = classify_blood_pressure_aha(125.0, 78.0)
    assert elevated["severity"] == "mild"

    stage1 = classify_blood_pressure_aha(135.0, 85.0)
    assert stage1["severity"] == "moderate"

    stage2 = classify_blood_pressure_aha(145.0, 95.0)
    assert stage2["severity"] == "high"

    crisis = classify_blood_pressure_aha(190.0, 125.0)
    assert crisis["severity"] == "critical"


def test_end_to_end_inference_predictor():
    """Memverifikasi alur inferensi terpadu dari sinyal ECG mentah (baseline)."""
    raw_ecg = np.random.randn(1500)  # Sinyal mentah 1500 sampel
    predictor = ECGBPPredictor()

    result = predictor.predict(raw_ecg, input_fs=125.0)

    assert "sbp" in result
    assert "dbp" in result
    assert "map" in result
    assert "clean_ecg" in result
    assert "clinical" in result
    assert len(result["clean_ecg"]) == 1250


def test_end_to_end_inference_predictor_hybrid():
    """Memverifikasi alur inferensi terpadu model hybrid (dengan ekstraksi 12 fitur)."""
    hybrid_model = ECGMultiTaskBiLSTM(
        input_size=1, hidden_size=64, num_layers=1,
        sbp_feat_dim=6, dbp_feat_dim=6, map_feat_dim=12
    )
    predictor = ECGBPPredictor(model=hybrid_model)
    raw_ecg = np.random.randn(1500)

    result = predictor.predict(raw_ecg, input_fs=125.0)

    assert "sbp" in result
    assert "dbp" in result
    assert "map" in result
    assert isinstance(result["sbp"], float)
    assert isinstance(result["dbp"], float)
    assert isinstance(result["map"], float)
    assert "clinical" in result


def test_end_to_end_inference_predictor_patched_projection():
    """Memverifikasi inferensi terpadu dengan model hasil optimasi Fan et al. (Patch 50x25 + Proj)."""
    opt_model = ECGMultiTaskBiLSTM(
        hidden_size=64,
        num_layers=1,
        patch_size=25,
        use_projection=True,
        proj_dim=64,
        sbp_feat_dim=6,
        dbp_feat_dim=6,
        map_feat_dim=12,
    )
    predictor = ECGBPPredictor(model=opt_model)
    raw_ecg = np.random.randn(1500)

    result = predictor.predict(raw_ecg, input_fs=125.0)

    assert "sbp" in result
    assert "dbp" in result
    assert "map" in result
    assert isinstance(result["sbp"], float)
    assert isinstance(result["dbp"], float)
    assert "clinical" in result


def test_adaptive_loss_with_target_scales():
    """Memverifikasi normalisasi loss multi-task terhadap skala target fisiologis."""
    criterion = AdaptiveMultiTaskLoss(
        loss_fn_type="rmse",
        target_scales=[120.0, 80.0, 93.3]
    )

    sbp = torch.tensor([[130.0], [140.0]])
    dbp = torch.tensor([[85.0], [90.0]])
    mp = torch.tensor([[100.0], [105.0]])
    targets = torch.tensor([[120.0, 80.0, 93.3], [120.0, 80.0, 93.3]])

    total_loss, metrics = criterion(sbp, dbp, mp, targets)
    assert total_loss.item() > 0.0

    # Simulasikan update bobot
    weights_info = criterion.update_weights_from_validation([15.0, 16.0], [8.0, 9.0], [10.0, 11.0])
    assert "weight_sbp" in weights_info
    assert "weight_dbp" in weights_info
    assert "weight_map" in weights_info
    assert criterion.target_scales == [120.0, 80.0, 93.3]


def test_bilstm_temporal_attention():
    """Memverifikasi mekanisme Temporal Attention Pooling (Pilar 3) pada BiLSTM."""
    model = ECGMultiTaskBiLSTM(
        hidden_size=64,
        num_layers=1,
        patch_size=25,
        use_attention=True,
    )
    model.eval()

    x = torch.randn(4, 1250)
    sbp, dbp, mp = model(x)
    assert sbp.shape == (4, 1)
    assert dbp.shape == (4, 1)
    assert mp.shape == (4, 1)

    cfg = model.get_config()
    assert cfg["use_attention"] is True


def test_adaptive_multi_task_loss_with_phys_regularization():
    """Memverifikasi perhitungan penalti konsistensi fisiologis MAP: MAP ≈ DBP + 1/3(SBP - DBP)."""
    criterion = AdaptiveMultiTaskLoss(loss_fn_type="mae", lambda_phys=0.20)

    # Kasus 1: Prediksi konsisten (SBP=120, DBP=80 -> MAP=80 + 40/3 = 93.333)
    sbp_cons = torch.tensor([[120.0]])
    dbp_cons = torch.tensor([[80.0]])
    map_cons = torch.tensor([[93.3333]])
    targets = torch.tensor([[120.0, 80.0, 93.3333]])

    loss_cons, met_cons = criterion(sbp_cons, dbp_cons, map_cons, targets)
    assert met_cons["loss_phys"] < 1e-3

    # Kasus 2: Prediksi inkonsisten (SBP=140, DBP=90, tapi MAP=80 -> MAP harusnya ~106.67)
    sbp_incons = torch.tensor([[140.0]])
    dbp_incons = torch.tensor([[90.0]])
    map_incons = torch.tensor([[80.0]])
    loss_incons, met_incons = criterion(sbp_incons, dbp_incons, map_incons, targets)
    assert met_incons["loss_phys"] > 20.0
    assert loss_incons.item() > loss_cons.item()


def test_bilstm_with_gated_feature_fusion():
    """Memverifikasi fungsionalitas Adaptive Gated Feature Fusion pada BiLSTM + Projector."""
    model = ECGMultiTaskBiLSTM(
        input_size=25,
        hidden_size=64,
        num_layers=1,
        sbp_feat_dim=6,
        dbp_feat_dim=6,
        map_feat_dim=12,
        patch_size=25,
        use_projection=True,
        proj_dim=32,
        use_gating=True,
    )
    model.eval()

    x = torch.randn(4, 1250)
    feats = {
        "sbp": torch.randn(4, 6),
        "dbp": torch.randn(4, 6),
        "map": torch.randn(4, 12),
    }

    sbp, dbp, mp = model(x, features=feats)
    assert sbp.shape == (4, 1)
    assert dbp.shape == (4, 1)
    assert mp.shape == (4, 1)

    cfg = model.get_config()
    assert cfg["use_gating"] is True
    assert cfg["use_projection"] is True
    assert model.sbp_fusion is not None




