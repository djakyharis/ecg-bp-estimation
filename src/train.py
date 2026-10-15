"""
Script Pelatihan Model Deep Learning Multi-Task BiLSTM untuk Estimasi Tekanan Darah.
Dioptimalkan untuk Lingkungan GPU Server via SSH & Sinkron dengan Notebook 06 (Kachuee Dataset / Fan et al. 2021).

Fitur Utama:
1. Resilient SSH Training:
   - Graceful Signal Handling (SIGHUP, SIGINT, SIGTERM) -> simpan checkpoint darurat saat koneksi putus
   - Checkpoint Ganda: 'last_checkpoint.pt' (tiap epoch) & 'best_model.pt' (best validation)
   - Auto-Resume / Resume fleksibel (--resume <path> atau --auto-resume)
   - Dual Logging: stdout + logfile dengan buffer autoflush (aman dipantau via tail -f)
2. Akselerasi GPU Modern:
   - Automatic Mixed Precision (AMP FP16) dengan GradScaler
   - DataLoader teroptimasi (num_workers, pin_memory, persistent_workers)
   - cuDNN benchmark auto-tuner
3. Evaluasi & Visualisasi Lengkap (Kaggle Notebook 06):
   - Standar Medis Internasional: BHS Protocol (Grade A/B/C/D) & AAMI SP10
   - Grafik Evaluasi Otomatis: training_curves.png, scatter_pred_vs_true.png, bland_altman.png
   - Artefak deployment: final_model_full.pt, evaluation_results.json, training_history.json
"""

import os
import sys
import glob
import json
import time
import signal
import logging
import argparse
from typing import List, Tuple, Dict, Optional, Any

# Headless safe agar matplotlib tidak crash di server remote tanpa X11/GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from src.preprocessing.kachuee_preprocessor import preprocess_kachuee_part
from src.features.statistical import (
    extract_statistical_features,
    FeatureStandardScaler,
    SBP_FEATURE_NAMES,
    DBP_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
)


# ── Global State untuk Penanganan Sinyal SSH (SIGHUP/SIGINT) ──────────────────
TRAINING_STATE = {
    "model": None,
    "optimizer": None,
    "scheduler": None,
    "scaler": None,
    "feature_scaler": None,
    "criterion": None,
    "epoch": 0,
    "best_val_mae": float("inf"),
    "history": None,
    "checkpoint_dir": None,
    "args": None,
}


def register_signal_handlers(logger: logging.Logger):
    """Mencegah proses langsung mati seketika saat SSH disconnect (SIGHUP) atau di-terminate."""
    def handle_signal(sig, frame):
        sig_name = "SIGHUP (SSH Disconnect)" if sig == signal.SIGHUP else f"Signal {sig}"
        logger.warning(f"\n⚠️  [{sig_name}] Terdeteksi! Menjalankan penyimpanan status darurat...")
        save_checkpoint(
            checkpoint_dir=TRAINING_STATE["checkpoint_dir"],
            filename="emergency_checkpoint.pt",
            epoch=TRAINING_STATE["epoch"],
            model=TRAINING_STATE["model"],
            optimizer=TRAINING_STATE["optimizer"],
            scheduler=TRAINING_STATE["scheduler"],
            scaler=TRAINING_STATE["scaler"],
            criterion=TRAINING_STATE["criterion"],
            best_val_mae=TRAINING_STATE["best_val_mae"],
            history=TRAINING_STATE["history"],
            args=TRAINING_STATE["args"],
            feature_scaler=TRAINING_STATE["feature_scaler"],
            logger=logger,
        )
        logger.warning("✅ Checkpoint darurat berhasil disimpan. Anda dapat melanjutkan training via --resume.")
        sys.exit(0)

    try:
        signal.signal(signal.SIGHUP, handle_signal)
    except AttributeError:
        pass  # Windows tidak memiliki SIGHUP
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger yang menulis ke terminal dan berkas log secara simultan dengan autoflush."""
    logger = logging.getLogger("ECG_BP_Trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler jika ditentukan
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_device(requested: str = "auto", logger: Optional[logging.Logger] = None) -> torch.device:
    """Mendeteksi akselerator hardware terbaik (CUDA / MPS / CPU)."""
    if requested != "auto":
        dev = torch.device(requested)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    if logger:
        if dev.type == "cuda":
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"✅ Device Akselerator : CUDA ({device_name}, VRAM: {vram_gb:.1f} GB)")
        elif dev.type == "mps":
            logger.info("✅ Device Akselerator : Apple Silicon MPS")
        else:
            logger.warning("⚠️  Device Akselerator : CPU (Performa training akan lambat)")
    return dev


def load_dataset_parts(
    data_dir: str,
    parts_spec: str = "1",
    max_samples: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> ECGBPDataset:
    """
    Memuat dataset sinyal ECG & BP ground truth.
    Mendukung alur End-to-End: Jika berkas .npz preprocessed belum tersedia di folder
    tujuan (data/processed), fungsi ini otomatis memproses berkas mentah .mat dari
    folder data/kachuee_mat/ secara transparan.
    """
    log_fn = logger.info if logger else print
    warn_fn = logger.warning if logger else print

    processed_dir = data_dir
    # Deteksi lokasi file mentah kachuee_mat
    kachuee_mat_dir = os.path.join(PROJECT_ROOT, "data", "kachuee_mat")
    if os.path.basename(os.path.normpath(data_dir)) == "kachuee_mat":
        processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
        kachuee_mat_dir = data_dir

    os.makedirs(processed_dir, exist_ok=True)

    # Identifikasi part yang diminta
    if parts_spec.strip().lower() == "all":
        # Cari semua part dari .npz atau dari .mat jika .npz belum lengkap
        mat_candidates = sorted(glob.glob(os.path.join(kachuee_mat_dir, "part_*.mat")))
        if mat_candidates:
            part_nums = [
                os.path.splitext(os.path.basename(p))[0].replace("part_", "")
                for p in mat_candidates
            ]
        else:
            npz_candidates = sorted(glob.glob(os.path.join(processed_dir, "*_preprocessed.npz")))
            part_nums = [
                os.path.splitext(os.path.basename(p))[0].replace("part_", "").replace("_preprocessed", "")
                for p in npz_candidates
            ]
    else:
        part_nums = [p.strip() for p in parts_spec.split(",") if p.strip()]

    file_paths = []
    for p in part_nums:
        npz_target = os.path.join(processed_dir, f"part_{p}_preprocessed.npz")
        mat_target = os.path.join(kachuee_mat_dir, f"part_{p}.mat")

        if os.path.exists(npz_target):
            file_paths.append(npz_target)
        elif os.path.exists(mat_target):
            log_fn(f"🔄 Berkas preprocessed belum ada: {npz_target}")
            log_fn(f"⚡ Menjalankan alur End-to-End Pra-Pemrosesan untuk Part {p} dari: {mat_target}...")
            
            # Jika hanya smoke-test (max_samples <= 500) dan .npz belum ada, proses secara cepat
            is_smoke_test = (max_samples is not None and max_samples <= 500)
            max_epochs_to_proc = max(max_samples * 2, 200) if is_smoke_test else None
            
            out_file = npz_target if not is_smoke_test else os.path.join(processed_dir, f"part_{p}_smoke_preprocessed.npz")
            if is_smoke_test and os.path.exists(out_file):
                file_paths.append(out_file)
                continue

            preprocess_kachuee_part(
                mat_path=mat_target,
                output_npz_path=out_file,
                epoch_len=1250,
                fs=125.0,
                max_epochs=max_epochs_to_proc,
                logger=logger,
            )
            file_paths.append(out_file)
        else:
            warn_fn(f"[Peringatan] Berkas data untuk Part {p} tidak ditemukan di {npz_target} maupun {mat_target}")

    if not file_paths:
        raise FileNotFoundError(
            f"Tidak ada berkas data yang cocok dengan spesifikasi parts '{parts_spec}'. "
            f"Pastikan file .mat tersedia di '{kachuee_mat_dir}' atau .npz di '{processed_dir}'."
        )

    log_fn(f"Memuat {len(file_paths)} berkas dataset preprocessed...")
    for fp in file_paths:
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        log_fn(f"  • {os.path.basename(fp)} ({size_mb:.1f} MB)")

    dataset = ECGBPDataset.from_npz_files(file_paths, max_samples=max_samples)
    log_fn(f"Total sampel terkumpul: {len(dataset):,} epoch (10 detik @ 125 Hz)")
    return dataset


def save_checkpoint(
    checkpoint_dir: str,
    filename: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any],
    criterion: AdaptiveMultiTaskLoss,
    best_val_mae: float,
    history: Dict[str, list],
    args: argparse.Namespace,
    feature_scaler: Optional[Any] = None,
    logger: Optional[logging.Logger] = None
):
    """Menyimpan checkpoint lengkap (model, optimizer, scheduler, scaler, status loss, dan riwayat)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    target_path = os.path.join(checkpoint_dir, filename)
    temp_path = target_path + ".tmp"

    checkpoint_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "feature_scaler_state": feature_scaler.state_dict() if feature_scaler else None,
        "criterion_state": criterion.get_state(),
        "best_val_mae": best_val_mae,
        "history": history,
        "args": vars(args),
    }

    # Atomic write: simpan ke file sementara lalu rename agar tidak korup jika diinterupsi
    torch.save(checkpoint_dict, temp_path)
    if os.path.exists(target_path):
        try:
            os.remove(target_path)
        except OSError:
            pass
    os.rename(temp_path, target_path)

    if logger:
        logger.info(f"💾 Checkpoint tersimpan: {filename} (Epoch {epoch})")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any],
    criterion: AdaptiveMultiTaskLoss,
    device: torch.device,
    feature_scaler: Optional[Any] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[int, float, Dict[str, list]]:
    """Memulihkan proses training dari checkpoint (Resume)."""
    log_fn = logger.info if logger else print
    log_fn(f"🔄 Memulihkan status training dari: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    if feature_scaler and ckpt.get("feature_scaler_state"):
        feature_scaler.load_state_dict(ckpt["feature_scaler_state"])

    if "criterion_state" in ckpt:
        criterion.load_state(ckpt["criterion_state"])

    start_epoch = ckpt.get("epoch", 0)
    best_val_mae = ckpt.get("best_val_mae", float("inf"))
    history = ckpt.get("history", {
        "train_loss": [], "val_loss": [],
        "val_mae_sbp": [], "val_mae_dbp": [], "val_mae_map": [],
        "lr": [], "weights": []
    })

    log_fn(f"✅ Berhasil memulihkan Epoch {start_epoch} | Best Val MAE sebelumnya: {best_val_mae:.2f} mmHg")
    return start_epoch, best_val_mae, history


def train_one_epoch(
    model: nn.Module,
    criterion: AdaptiveMultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_clip: float = 5.0
) -> Dict[str, float]:
    """Satu epoch training dengan dukungan Automatic Mixed Precision (AMP)."""
    model.train()
    total_loss = 0.0
    mae_sbp_sum, mae_dbp_sum, mae_map_sum = 0.0, 0.0, 0.0
    n_batches = 0
    use_cuda = device.type == "cuda"

    for batch in dataloader:
        if len(batch) == 3:
            x_batch, y_batch, feat_batch = batch
            feat_batch = {k: v.to(device, non_blocking=use_cuda) for k, v in feat_batch.items()}
        else:
            x_batch, y_batch = batch
            feat_batch = None

        x_batch = x_batch.to(device, non_blocking=use_cuda)
        y_batch = y_batch.to(device, non_blocking=use_cuda)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                sbp_pred, dbp_pred, map_pred = model(x_batch, features=feat_batch)
                loss, metrics = criterion(sbp_pred, dbp_pred, map_pred, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            sbp_pred, dbp_pred, map_pred = model(x_batch, features=feat_batch)
            loss, metrics = criterion(sbp_pred, dbp_pred, map_pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: AdaptiveMultiTaskLoss,
    dataloader: DataLoader,
    device: torch.device,
    update_adaptive_weights: bool = False
) -> Dict[str, float]:
    """Evaluasi pada validation / test set dengan pembaruan bobot adaptif (Fan et al., 2021)."""
    model.eval()
    total_loss = 0.0
    mae_sbp_list, mae_dbp_list, mae_map_list = [], [], []
    n_batches = 0
    use_cuda = device.type == "cuda"

    for batch in dataloader:
        if len(batch) == 3:
            x_batch, y_batch, feat_batch = batch
            feat_batch = {k: v.to(device, non_blocking=use_cuda) for k, v in feat_batch.items()}
        else:
            x_batch, y_batch = batch
            feat_batch = None

        x_batch = x_batch.to(device, non_blocking=use_cuda)
        y_batch = y_batch.to(device, non_blocking=use_cuda)

        sbp_pred, dbp_pred, map_pred = model(x_batch, features=feat_batch)
        loss, metrics = criterion(sbp_pred, dbp_pred, map_pred, y_batch)

        total_loss += metrics["loss_total"]
        mae_sbp_list.append(metrics["mae_sbp"])
        mae_dbp_list.append(metrics["mae_dbp"])
        mae_map_list.append(metrics["mae_map"])
        n_batches += 1

    if update_adaptive_weights and len(mae_sbp_list) > 0:
        criterion.update_weights_from_validation(mae_sbp_list, mae_dbp_list, mae_map_list)

    return {
        "loss": total_loss / max(1, n_batches),
        "mae_sbp": float(np.mean(mae_sbp_list)) if mae_sbp_list else 0.0,
        "mae_dbp": float(np.mean(mae_dbp_list)) if mae_dbp_list else 0.0,
        "mae_map": float(np.mean(mae_map_list)) if mae_map_list else 0.0,
    }


def evaluate_test_set_full(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Evaluasi komprehensif pada Test Set sesuai Fase III Notebook 06:
    - Metrik MAE, ME (Bias), STD, RMSE
    - Standar British Hypertension Society (BHS Protocol Grade A/B/C/D)
    - Standar AAMI SP10 (MAE <= 5 mmHg dan STD <= 8 mmHg)
    - Visualisasi: training_curves, scatter_pred_vs_true, bland_altman
    """
    model.eval()
    logger.info("\n" + "=" * 75)
    logger.info("  EVALUASI AKHIR TEST SET (UNSEEN DATA) — STANDAR BHS & AAMI SP10")
    logger.info("=" * 75)

    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x_batch, y_batch, feat_batch = batch
                feat_batch = {k: v.to(device) for k, v in feat_batch.items()}
            else:
                x_batch, y_batch = batch
                feat_batch = None
            preds = model.predict_all(x_batch.to(device), features=feat_batch)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    errors = all_preds - all_trues
    abs_errors = np.abs(errors)

    target_names = ["SBP", "DBP", "MAP"]
    results = {}

    for i, name in enumerate(target_names):
        mae = float(np.mean(abs_errors[:, i]))
        std = float(np.std(errors[:, i]))
        me = float(np.mean(errors[:, i]))
        rmse = float(np.sqrt(np.mean(errors[:, i] ** 2)))

        p5 = float(np.mean(abs_errors[:, i] <= 5.0) * 100)
        p10 = float(np.mean(abs_errors[:, i] <= 10.0) * 100)
        p15 = float(np.mean(abs_errors[:, i] <= 15.0) * 100)

        # BHS Protocol grading
        if p5 >= 60.0 and p10 >= 85.0 and p15 >= 95.0:
            bhs = "A"
        elif p5 >= 50.0 and p10 >= 75.0 and p15 >= 90.0:
            bhs = "B"
        elif p5 >= 40.0 and p10 >= 65.0 and p15 >= 85.0:
            bhs = "C"
        else:
            bhs = "D"

        # AAMI SP10 criteria: MAE <= 5.0 and STD <= 8.0
        aami_pass = bool(mae <= 5.0 and std <= 8.0)
        aami_label = "✅ LULUS" if aami_pass else "❌ GAGAL"

        results[name] = {
            "MAE": mae,
            "STD": std,
            "ME": me,
            "RMSE": rmse,
            "P_le_5": p5,
            "P_le_10": p10,
            "P_le_15": p15,
            "BHS_Grade": bhs,
            "AAMI_Pass": aami_pass,
        }

        logger.info(f"\n  🎯 {name}:")
        logger.info(f"     MAE: {mae:.2f} mmHg | ME (Bias): {me:+.2f} ± {std:.2f} mmHg | RMSE: {rmse:.2f} mmHg")
        logger.info(f"     BHS Protocol : ≤5mmHg={p5:.1f}% | ≤10mmHg={p10:.1f}% | ≤15mmHg={p15:.1f}% → Grade {bhs}")
        logger.info(f"     AAMI SP10    : {aami_label} (Kriteria: MAE ≤ 5 & STD ≤ 8)")

    avg_mae = float(np.mean([r["MAE"] for r in results.values()]))
    logger.info("-" * 75)
    logger.info(f"  📊 RATA-RATA MAE KESELURUHAN: {avg_mae:.2f} mmHg")
    logger.info("=" * 75)

    # ── Visualisasi 1: Scatter Plot Prediksi vs Ground Truth ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, (name, c) in enumerate(zip(target_names, colors)):
        t_vals, p_vals = all_trues[:, i], all_preds[:, i]
        axes[i].scatter(t_vals, p_vals, alpha=0.15, s=8, color=c, edgecolors="none")
        lims = [min(t_vals.min(), p_vals.min()) - 5, max(t_vals.max(), p_vals.max()) + 5]
        axes[i].plot(lims, lims, "k--", lw=1.5, alpha=0.7, label="Ideal (y=x)")
        axes[i].fill_between(lims, [l - 5 for l in lims], [l + 5 for l in lims], alpha=0.1, color="green", label="±5 mmHg")
        axes[i].set_title(f"{name} — MAE: {results[name]['MAE']:.2f} mmHg", fontweight="bold", fontsize=12)
        axes[i].set_xlabel(f"{name} Ground Truth (mmHg)")
        axes[i].set_ylabel(f"{name} Prediksi (mmHg)")
        axes[i].set_xlim(lims)
        axes[i].set_ylim(lims)
        axes[i].set_aspect("equal")
        axes[i].legend(loc="upper left", fontsize=9)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f"Prediksi vs Ground Truth — Test Set ({len(all_trues):,} sampel)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "scatter_pred_vs_true.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"📈 Grafik scatter tersimpan: {scatter_path}")

    # ── Visualisasi 2: Bland-Altman Agreement Analysis ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, c) in enumerate(zip(target_names, colors)):
        t_vals, p_vals = all_trues[:, i], all_preds[:, i]
        mean_v = (t_vals + p_vals) / 2.0
        diff_v = p_vals - t_vals
        md, sd = float(np.mean(diff_v)), float(np.std(diff_v))

        axes[i].scatter(mean_v, diff_v, alpha=0.1, s=6, color=c, edgecolors="none")
        axes[i].axhline(md, color="black", ls="-", lw=1.5, label=f"Bias: {md:+.2f}")
        axes[i].axhline(md + 1.96 * sd, color="red", ls="--", lw=1.0, label=f"+1.96SD: {md + 1.96*sd:+.2f}")
        axes[i].axhline(md - 1.96 * sd, color="red", ls="--", lw=1.0, label=f"−1.96SD: {md - 1.96*sd:+.2f}")
        axes[i].axhline(0, color="gray", ls=":", alpha=0.5)
        axes[i].set_title(f"Bland-Altman: {name}", fontweight="bold", fontsize=12)
        axes[i].set_xlabel("Mean (mmHg)")
        axes[i].set_ylabel("Diff (Pred − True) (mmHg)")
        axes[i].legend(fontsize=8, loc="upper right")
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Bland-Altman Agreement Analysis — Test Set", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    ba_path = os.path.join(output_dir, "bland_altman.png")
    plt.savefig(ba_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"📈 Grafik Bland-Altman tersimpan: {ba_path}")

    return {
        "test_samples": len(all_trues),
        "results": results,
        "avg_mae": avg_mae,
    }


def plot_training_curves(history: Dict[str, list], output_dir: str, logger: logging.Logger):
    """Menghasilkan grafik 4 panel kurva pelatihan (Loss, Val MAE, LR, dan Bobot Adaptif theta)."""
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Loss Train vs Val
    axes[0, 0].plot(ep, history["train_loss"], "o-", label="Train", color="#1f77b4", lw=2)
    axes[0, 0].plot(ep, history["val_loss"], "s--", label="Val", color="#ff7f0e", lw=2)
    axes[0, 0].set_title("Loss (Train vs Val)", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Weighted MAE")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Val MAE per Target dengan Garis AAMI <= 5
    axes[0, 1].plot(ep, history["val_mae_sbp"], "o-", label="SBP", color="#e74c3c", lw=2)
    axes[0, 1].plot(ep, history["val_mae_dbp"], "s-", label="DBP", color="#3498db", lw=2)
    axes[0, 1].plot(ep, history["val_mae_map"], "^-", label="MAP", color="#2ecc71", lw=2)
    axes[0, 1].axhline(5.0, color="gray", ls=":", alpha=0.7, label="Target AAMI ≤5 mmHg")
    axes[0, 1].set_title("Val MAE per Target", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MAE (mmHg)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Learning Rate
    if "lr" in history and len(history["lr"]) == len(ep):
        axes[1, 0].plot(ep, history["lr"], "D-", color="#9b59b6", lw=2)
        axes[1, 0].set_title("Learning Rate Schedule", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Adaptive Weights theta
    if "weights" in history and len(history["weights"]) == len(ep):
        wa = np.array(history["weights"])
        axes[1, 1].plot(ep, wa[:, 0], "o-", label="θ_SBP", color="#e74c3c", lw=2)
        axes[1, 1].plot(ep, wa[:, 1], "s-", label="θ_DBP", color="#3498db", lw=2)
        axes[1, 1].plot(ep, wa[:, 2], "^-", label="θ_MAP", color="#2ecc71", lw=2)
        axes[1, 1].axhline(1.0 / 3.0, color="gray", ls=":", alpha=0.5, label="Init (1/3)")
        axes[1, 1].set_title("Adaptive Task Weights (θ)", fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("θ Value")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Monitoring Pelatihan Multi-Task BiLSTM (Fan et al., 2021)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    curve_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"📈 Grafik kurva pelatihan tersimpan: {curve_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pelatihan Model Deep Learning Multi-Task BiLSTM (GPU Server SSH + Fan et al. 2021)"
    )
    # Jalur Data & Checkpoint
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "processed"), help="Direktori berkas .npz")
    parser.add_argument("--parts", type=str, default="1", help="Part dataset: '1', '1,2', atau 'all'")
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(PROJECT_ROOT, "checkpoints"), help="Folder checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Folder output artefak & grafik (default: sama dengan checkpoint-dir)")
    parser.add_argument("--log-file", type=str, default=None, help="Jalur file log (default: <checkpoint-dir>/train.log)")

    # Hyperparameter Sesuai Paper Fan et al. (2021) & Notebook 06
    parser.add_argument("--batch-size", type=int, default=128, help="Ukuran batch (Fan et al. 2021: 128)")
    parser.add_argument("--epochs", type=int, default=500, help="Jumlah total epoch (Fan et al. 2021: 500)")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate (Fan et al. 2021: 0.0008)")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"], help="Optimizer: 'sgd' (Fan et al. 2021 Momentum-SGD) atau 'adamw'")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum untuk SGD (Fan et al. 2021: 0.9)")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="L2 Regularization (Fan et al. 2021: 0.1, atau 1e-5 untuk AdamW)")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clipping norm")
    parser.add_argument("--scheduler-factor", type=float, default=0.9, help="ReduceLROnPlateau decay factor")
    parser.add_argument("--scheduler-patience", type=int, default=4, help="ReduceLROnPlateau patience")
    parser.add_argument("--loss-fn", type=str, default="rmse", choices=["rmse", "mae", "mse", "smooth_l1"], help="Fungsi loss pelatihan (Fan et al. 2021: rmse, default: rmse)")
    parser.add_argument("--loss-min-weight", type=float, default=0.15, help="Batas bawah bobot minimum tugas adaptif (default: 0.15)")
    parser.add_argument("--loss-momentum", type=float, default=0.70, help="Faktor smoothing momentum antarepoch bobot adaptif (default: 0.70)")

    # Arsitektur BiLSTM (Tabel 1 Fan et al., 2021)
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden units per arah BiLSTM (output concat: 512)")
    parser.add_argument("--num-layers", type=int, default=2, help="Jumlah layer BiLSTM")
    parser.add_argument("--lstm-dropout", type=float, default=0.0, help="Dropout antar layer LSTM (Fan et al. 2021: 0.0)")
    parser.add_argument("--fc-dropout", type=float, default=0.0, help="Dropout pada FC heads (Fan et al. 2021: 0.0)")

    # Optimasi GPU & Sistem
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', 'mps', atau 'cpu'")
    parser.add_argument("--amp", action="store_true", default=True, help="Aktifkan Automatic Mixed Precision FP16 (default: True)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Nonaktifkan Mixed Precision")
    parser.add_argument("--num-workers", type=int, default=-1, help="DataLoader workers (-1 = otomatis: 4 di GPU, 0 di CPU)")
    parser.add_argument("--max-samples", type=int, default=None, help="Batas maksimal sampel (opsional, untuk smoke-test)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Fitur Statistik Terpandu (Nour et al. 2022 / Ilham Hakim 2025)
    parser.add_argument("--use-features", action="store_true", default=True, help="Gunakan 12 fitur statistik terpandu (default: True)")
    parser.add_argument("--no-features", dest="use_features", action="store_false", help="Nonaktifkan fitur statistik (mode Pure BiLSTM)")

    # Mitigasi SSH: Resume Capabilities
    parser.add_argument("--resume", type=str, default=None, help="Jalur checkpoint .pt untuk melanjutkan training")
    parser.add_argument("--auto-resume", action="store_true", default=False, help="Otomatis melanjutkan dari 'last_checkpoint.pt' jika tersedia di folder checkpoint")

    args = parser.parse_args()

    # Inisialisasi Jalur
    output_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(args.checkpoint_dir, "train.log")

    logger = setup_logger(log_file)
    register_signal_handlers(logger)

    device = get_device(args.device, logger)
    use_cuda = (device.type == "cuda")

    # cuDNN auto-tuner untuk sequence length statis
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # Workers otomatis
    if args.num_workers < 0:
        args.num_workers = min(os.cpu_count() or 4, 8) if use_cuda else 0

    logger.info("=" * 75)
    logger.info("  PELATIHAN DEEP LEARNING: MULTI-TASK BiLSTM (ECG -> SBP/DBP/MAP)")
    logger.info("  Parameter & Metodologi Sesuai Paper Acuan: Fan et al. (2021, Sensors)")
    logger.info("=" * 75)
    logger.info(f"  Batch Size         : {args.batch_size}")
    logger.info(f"  Learning Rate      : {args.lr}")
    logger.info(f"  Optimizer          : {args.optimizer.upper()} (Momentum: {args.momentum}, Weight Decay: {args.weight_decay})")
    logger.info(f"  BiLSTM Dropout     : LSTM={args.lstm_dropout}, FC Heads={args.fc_dropout}")
    logger.info(f"  Mixed Precision    : {'AKTIF (FP16 GradScaler)' if (args.amp and use_cuda) else 'NON-AKTIF'}")
    logger.info(f"  DataLoader Workers : {args.num_workers} (Pin Memory: {use_cuda})")
    logger.info(f"  Folder Checkpoint  : {args.checkpoint_dir}")
    logger.info(f"  File Log Aktif     : {log_file}")
    logger.info("=" * 75)

    # 1. Muat Dataset
    dataset = load_dataset_parts(
        args.data_dir, parts_spec=args.parts, max_samples=args.max_samples, logger=logger
    )

    # 2. Split Train (80%) / Val (10%) / Test (10%) Sesuai Fan et al. (2021)
    total_len = len(dataset)
    train_len = int(0.80 * total_len)
    val_len = int(0.10 * total_len)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Ekstraksi dan Injeksi Fitur Statistik (Nour et al. 2022 / Ilham Hakim 2025)
    if args.use_features:
        logger.info("🔍 Mengekstrak 12 Fitur Statistik Terpandu (Nour et al. 2022 / Ilham Hakim 2025)...")
        raw_x = dataset.X.squeeze(-1).numpy()
        all_extracted = extract_statistical_features(raw_x)

        # Fit scaler HANYA pada train_set (mencegah data leakage)
        feature_scaler = FeatureStandardScaler()
        train_idx = list(train_set.indices)
        train_feats = {k: v[train_idx] for k, v in all_extracted.items()}
        feature_scaler.fit(train_feats)

        # Terapkan standarisasi ke seluruh dataset
        scaled_features = feature_scaler.transform(all_extracted)
        dataset.set_features(scaled_features)

        sbp_feat_dim = len(SBP_FEATURE_NAMES)
        dbp_feat_dim = len(DBP_FEATURE_NAMES)
        map_feat_dim = len(ALL_FEATURE_NAMES)
        logger.info(
            f"✅ Mode Hybrid Aktif: SBP Head (+{sbp_feat_dim} fitur) | "
            f"DBP Head (+{dbp_feat_dim} fitur) | MAP Head (+{map_feat_dim} fitur)"
        )
    else:
        feature_scaler = None
        sbp_feat_dim = 0
        dbp_feat_dim = 0
        map_feat_dim = 0
        logger.info("ℹ️  Mode Pure BiLSTM Aktif (tanpa injeksi fitur statistik terpandu)")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
    )

    logger.info(f"📊 Pembagian Data: Train={len(train_set):,} | Val={len(val_set):,} | Test={len(test_set):,}")
    logger.info(f"   Jumlah Iterasi per Epoch: {len(train_loader)} batch")

    # 3. Inisialisasi Model, Loss, dan Optimizer
    model = ECGMultiTaskBiLSTM(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        lstm_dropout=args.lstm_dropout,
        fc_dropout=args.fc_dropout,
        sbp_feat_dim=sbp_feat_dim,
        dbp_feat_dim=dbp_feat_dim,
        map_feat_dim=map_feat_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Inisialisasi Model: {total_params:,} parameter terlatih")

    criterion = AdaptiveMultiTaskLoss(
        loss_fn_type=args.loss_fn,
        min_weight=args.loss_min_weight,
        momentum=args.loss_momentum,
    ).to(device)

    # Pisahkan parameter weight decay: jangan kenakan weight decay pada bias (mencegah penarikan paksa ke 0)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
        )
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    # Scaler untuk Automatic Mixed Precision
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and use_cuda))

    # Cek Resume Checkpoint
    start_epoch = 0
    best_val_mae = float("inf")
    history = {
        "train_loss": [], "val_loss": [],
        "val_mae_sbp": [], "val_mae_dbp": [], "val_mae_map": [],
        "lr": [], "weights": []
    }

    resume_target = None
    if args.resume:
        resume_target = args.resume
    elif args.auto_resume:
        last_ckpt_path = os.path.join(args.checkpoint_dir, "last_checkpoint.pt")
        if os.path.exists(last_ckpt_path):
            resume_target = last_ckpt_path

    if resume_target and os.path.exists(resume_target):
        start_epoch, best_val_mae, history = load_checkpoint(
            resume_target, model, optimizer, scheduler, scaler, criterion, device,
            feature_scaler=feature_scaler, logger=logger
        )

    # Daftarkan state ke signal handler
    TRAINING_STATE.update({
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "feature_scaler": feature_scaler,
        "criterion": criterion,
        "epoch": start_epoch,
        "best_val_mae": best_val_mae,
        "history": history,
        "checkpoint_dir": args.checkpoint_dir,
        "args": args,
    })

    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    logger.info(f"\n🚀 Memulai proses pelatihan dari Epoch {start_epoch + 1} hingga {args.epochs}...")
    t_start = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        TRAINING_STATE["epoch"] = epoch
        t0 = time.time()

        train_res = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            scaler=scaler if (args.amp and use_cuda) else None,
            grad_clip=args.grad_clip,
        )

        val_res = evaluate(
            model=model,
            criterion=criterion,
            dataloader=val_loader,
            device=device,
            update_adaptive_weights=True,
        )

        avg_val_mae = (val_res["mae_sbp"] + val_res["mae_dbp"] + val_res["mae_map"]) / 3.0
        scheduler.step(avg_val_mae)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_mae_sbp"].append(val_res["mae_sbp"])
        history["val_mae_dbp"].append(val_res["mae_dbp"])
        history["val_mae_map"].append(val_res["mae_map"])
        history["lr"].append(lr_now)
        history["weights"].append(criterion.weights.tolist())

        is_best = avg_val_mae < best_val_mae
        if is_best:
            best_val_mae = avg_val_mae
            TRAINING_STATE["best_val_mae"] = best_val_mae
            save_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                filename="best_model.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                criterion=criterion,
                best_val_mae=best_val_mae,
                history=history,
                args=args,
                feature_scaler=feature_scaler,
                logger=None,  # Tidak perlu spam log
            )
            mark = "★ BEST"
        else:
            mark = ""

        # Simpan checkpoint epoch terakhir (untuk jaminan auto-resume bila putus koneksi)
        save_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            filename="last_checkpoint.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            criterion=criterion,
            best_val_mae=best_val_mae,
            history=history,
            args=args,
            feature_scaler=feature_scaler,
            logger=None,
        )

        w = criterion.weights.tolist()
        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"TrLoss: {train_res['loss']:.4f} | VlLoss: {val_res['loss']:.4f} | "
            f"Val MAE -> SBP: {val_res['mae_sbp']:.2f}, DBP: {val_res['mae_dbp']:.2f}, MAP: {val_res['mae_map']:.2f} mmHg {mark} | "
            f"θ=[{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}] | LR={lr_now:.1e}"
        )

    total_time = time.time() - t_start
    logger.info("=" * 75)
    logger.info(f"🎉 Pelatihan Selesai dalam {total_time / 60:.1f} menit ({total_time / 3600:.2f} jam)!")
    logger.info(f"Model terbaik tersimpan di: {best_model_path} (Best Avg Val MAE: {best_val_mae:.2f} mmHg)")

    # 4. Visualisasi Kurva Pelatihan
    plot_training_curves(history, output_dir, logger)

    # 5. Evaluasi Akhir Test Set Menggunakan Checkpoint Model Terbaik
    logger.info("\nMemuat model terbaik untuk evaluasi akhir pada Test Set...")
    best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    eval_results = evaluate_test_set_full(model, test_loader, device, output_dir, logger)

    # 6. Simpan Artefak Lengkap (JSON & Deployment Model)
    # A. training_history.json
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"📄 Riwayat training tersimpan: {history_path}")

    # B. evaluation_results.json
    eval_full = {
        "test_samples": eval_results["test_samples"],
        "results": eval_results["results"],
        "avg_mae": eval_results["avg_mae"],
        "best_epoch": best_ckpt["epoch"],
        "best_val_mae": best_val_mae,
        "hyperparameters": vars(args),
    }
    eval_path = os.path.join(output_dir, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_full, f, indent=2)
    logger.info(f"📄 Ringkasan evaluasi tersimpan: {eval_path}")

    # C. final_model_full.pt (Artefak Terintegrasi untuk Deployment/Inference)
    final_bundle_path = os.path.join(output_dir, "final_model_full.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": 1,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "lstm_dropout": args.lstm_dropout,
            "fc_dropout": args.fc_dropout,
        },
        "training_args": vars(args),
        "evaluation": eval_full,
    }, final_bundle_path)
    logger.info(f"📦 Model final deployment tersimpan: {final_bundle_path}")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
