"""
Script Pelatihan Model Deep Learning Multi-Task BiLSTM untuk Estimasi Tekanan Darah.
Mendukung Apple Silicon (MPS), CUDA, dan CPU.

Penggunaan:
    python src/train.py --parts 1 --epochs 10 --batch-size 64
    python src/train.py --parts all --epochs 25 --batch-size 128
"""

import os
import sys
import glob
import json
import time
import argparse
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Setup PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.bilstm import ECGMultiTaskBiLSTM
from src.models.loss import AdaptiveMultiTaskLoss
from src.models.dataset import ECGBPDataset


def get_device(requested: str = "auto") -> torch.device:
    """Mendeteksi akselerator hardware terbaik (MPS / CUDA / CPU)."""
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dataset_parts(
    data_dir: str,
    parts_spec: str = "1",
    max_samples: int = None
) -> ECGBPDataset:
    """Memuat berkas .npz berdasarkan spesifikasi part (misal: '1', '1,2', atau 'all')."""
    if parts_spec.strip().lower() == "all":
        file_paths = sorted(glob.glob(os.path.join(data_dir, "*_preprocessed.npz")))
    else:
        part_nums = [p.strip() for p in parts_spec.split(",") if p.strip()]
        file_paths = []
        for p in part_nums:
            target = os.path.join(data_dir, f"part_{p}_preprocessed.npz")
            if os.path.exists(target):
                file_paths.append(target)
            else:
                print(f"[Peringatan] Berkas part tidak ditemukan: {target}")

    if not file_paths:
        raise FileNotFoundError(f"Tidak ada berkas .npz yang cocok di {data_dir} dengan spesifikasi '{parts_spec}'")

    print(f"\n[Data] Memuat {len(file_paths)} berkas dataset preprocessed:")
    for fp in file_paths:
        print(f"  - {os.path.basename(fp)} ({os.path.getsize(fp) / (1024 * 1024):.1f} MB)")

    dataset = ECGBPDataset.from_npz_files(file_paths, max_samples=max_samples)
    print(f"[Data] Total sampel terkumpul: {len(dataset):,} epoch (10 detik @ 125 Hz)")
    return dataset


def train_one_epoch(
    model: nn.Module,
    criterion: AdaptiveMultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Satu epoch training."""
    model.train()
    total_loss = 0.0
    mae_sbp_sum, mae_dbp_sum, mae_map_sum = 0.0, 0.0, 0.0
    n_batches = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        sbp_pred, dbp_pred, map_pred = model(x_batch)
        loss, metrics = criterion(sbp_pred, dbp_pred, map_pred, y_batch)

        loss.backward()
        # Gradient clipping untuk stabilitas LSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += metrics["loss_total"]
        mae_sbp_sum += metrics["mae_sbp"]
        mae_dbp_sum += metrics["mae_dbp"]
        mae_map_sum += metrics["mae_map"]
        n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "mae_sbp": mae_sbp_sum / max(1, n_batches),
        "mae_dbp": mae_dbp_sum / max(1, n_batches),
        "mae_map": mae_map_sum / max(1, n_batches),
    }


def evaluate(
    model: nn.Module,
    criterion: AdaptiveMultiTaskLoss,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluasi pada validation / test set."""
    model.eval()
    total_loss = 0.0
    mae_sbp_sum, mae_dbp_sum, mae_map_sum = 0.0, 0.0, 0.0
    n_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            sbp_pred, dbp_pred, map_pred = model(x_batch)
            loss, metrics = criterion(sbp_pred, dbp_pred, map_pred, y_batch)

            total_loss += metrics["loss_total"]
            mae_sbp_sum += metrics["mae_sbp"]
            mae_dbp_sum += metrics["mae_dbp"]
            mae_map_sum += metrics["mae_map"]
            n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "mae_sbp": mae_sbp_sum / max(1, n_batches),
        "mae_dbp": mae_dbp_sum / max(1, n_batches),
        "mae_map": mae_map_sum / max(1, n_batches),
    }


def main():
    parser = argparse.ArgumentParser(description="Training Deep Learning Multi-Task BiLSTM ECG-to-BP")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "processed"), help="Direktori berkas .npz")
    parser.add_argument("--parts", type=str, default="1", help="Pilihan part: '1', '1,2', atau 'all'")
    parser.add_argument("--max-samples", type=int, default=None, help="Batas maksimal sampel (opsional, untuk smoke-test)")
    parser.add_argument("--batch-size", type=int, default=128, help="Ukuran batch (Fan et al. 2021: 128)")
    parser.add_argument("--epochs", type=int, default=15, help="Jumlah epoch")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate (Fan et al. 2021: 0.0008)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"], help="Pilihan optimizer: 'adamw' atau 'sgd' (Fan et al. 2021)")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 regularization")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'mps', 'cuda', atau 'cpu'")
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(PROJECT_ROOT, "checkpoints"), help="Folder checkpoint")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = get_device(args.device)

    print("=" * 70)
    print("  PELATIHAN DEEP LEARNING: MULTI-TASK BiLSTM (ECG -> SBP/DBP/MAP)")
    print("  Parameter Sesuai Paper Acuan: Fan et al. (2021, Sensors)")
    print("=" * 70)
    print(f"  Device Akselerator : {device}")
    print(f"  Batch Size         : {args.batch_size} (Fan et al., 2021)")
    print(f"  Learning Rate      : {args.lr} (Fan et al., 2021)")
    print(f"  Optimizer          : {args.optimizer.upper()}")
    print(f"  Jumlah Epoch       : {args.epochs}")
    print(f"  Folder Checkpoint  : {args.checkpoint_dir}")
    print("=" * 70)

    # 1. Muat Dataset
    dataset = load_dataset_parts(args.data_dir, parts_spec=args.parts, max_samples=args.max_samples)

    # 2. Split Train / Val / Test (80% / 10% / 10%)
    total_len = len(dataset)
    train_len = int(0.80 * total_len)
    val_len = int(0.10 * total_len)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print(f"[Split] Train: {len(train_set):,} | Val: {len(val_set):,} | Test: {len(test_set):,}")

    # 3. Inisialisasi Model, Loss, dan Optimizer
    model = ECGMultiTaskBiLSTM(input_size=1, hidden_size=256, num_layers=2, lstm_dropout=0.3, head_dropout=0.2)
    model.to(device)

    criterion = AdaptiveMultiTaskLoss(loss_fn_type="mae", init_log_var=0.0)
    criterion.to(device)

    # Parameter yang dioptimasi mencakup bobot model + learnable log variance pada adaptive loss
    params_to_optimize = list(model.parameters()) + list(criterion.parameters())
    if args.optimizer == "sgd":
        # Sesuai Fan et al. (2021): Momentum-SGD dengan momentum=0.9
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    # LR Scheduler sesuai Fan et al. (2021): factor=0.9, patience=4
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=4)

    best_val_mae = float("inf")
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    history = {"train_loss": [], "val_loss": [], "val_mae_sbp": [], "val_mae_dbp": [], "val_mae_map": []}

    print("\nMemulai proses pelatihan...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(model, criterion, optimizer, train_loader, device)
        val_res = evaluate(model, criterion, val_loader, device)
        scheduler.step(val_res["loss"])
        elapsed = time.time() - t0

        avg_val_mae = (val_res["mae_sbp"] + val_res["mae_dbp"] + val_res["mae_map"]) / 3.0

        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_mae_sbp"].append(val_res["mae_sbp"])
        history["val_mae_dbp"].append(val_res["mae_dbp"])
        history["val_mae_map"].append(val_res["mae_map"])

        is_best = avg_val_mae < best_val_mae
        if is_best:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), best_model_path)
            mark = "★ BEST"
        else:
            mark = ""

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_res['loss']:.3f} | "
            f"Val Loss: {val_res['loss']:.3f} | "
            f"Val MAE -> SBP: {val_res['mae_sbp']:.2f}, DBP: {val_res['mae_dbp']:.2f}, MAP: {val_res['mae_map']:.2f} mmHg {mark}"
        )

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Pelatihan Selesai dalam {total_time / 60:.1f} menit!")
    print(f"Model terbaik disimpan di: {best_model_path} (Best Avg Val MAE: {best_val_mae:.2f} mmHg)")

    # 4. Evaluasi Akhir pada Test Set menggunakan Model Terbaik
    print("\nMelakukan evaluasi akhir pada Test Set (unseen data)...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_res = evaluate(model, criterion, test_loader, device)

    print("=" * 70)
    print("  HASIL EVALUASI TEST SET (STANDAR BHS / AAMI):")
    print(f"  - SBP MAE : {test_res['mae_sbp']:.2f} mmHg")
    print(f"  - DBP MAE : {test_res['mae_dbp']:.2f} mmHg")
    print(f"  - MAP MAE : {test_res['mae_map']:.2f} mmHg")
    print(f"  - Rata-rata Error Keseluruhan : {(test_res['mae_sbp'] + test_res['mae_dbp'] + test_res['mae_map'])/3.0:.2f} mmHg")
    print("=" * 70)

    # Simpan riwayat training
    history_path = os.path.join(args.checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Riwayat training disimpan di: {history_path}")


if __name__ == "__main__":
    main()
