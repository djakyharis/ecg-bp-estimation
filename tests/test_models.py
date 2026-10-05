"""
Unit Tests untuk Modul Arsitektur Deep Learning (BiLSTM, Loss, Dataset, Inference).
"""

import pytest
import numpy as np
import torch

from src.models.bilstm import ECGMultiTaskBiLSTM, TaskSpecificHead
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
    """Memverifikasi alur inferensi terpadu dari sinyal ECG mentah."""
    raw_ecg = np.random.randn(1500)  # Sinyal mentah 1500 sampel
    predictor = ECGBPPredictor()

    result = predictor.predict(raw_ecg, input_fs=125.0)

    assert "sbp" in result
    assert "dbp" in result
    assert "map" in result
    assert "clean_ecg" in result
    assert "clinical" in result
    assert len(result["clean_ecg"]) == 1250
