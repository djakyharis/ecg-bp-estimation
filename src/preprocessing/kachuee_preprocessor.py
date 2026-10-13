"""
Modul Pra-Pemrosesan Batch Dataset Kachuee et al.
Mengekstrak rekaman .mat (ECG Lead II + ABP), segmentasi 10 detik (1250 titik @ 125 Hz),
melakukan filtering Notch 50 Hz + Bandpass 0.5-40 Hz + MinMax [0, 1],
mengekstrak ground-truth BP (SBP, DBP, MAP), dan menyimpan ke .npz terkompresi.
"""

import os
import glob
import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import scipy.io
from scipy.signal import find_peaks
from tqdm import tqdm

from src.preprocessing.pipeline import ECGPreprocessingPipeline


def load_mat_file(file_path: str) -> Any:
    """Membaca file dataset .mat (format MATLAB v5/v7 atau HDF5 v7.3)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    try:
        mat_data = scipy.io.loadmat(file_path)
        keys = [k for k in mat_data.keys() if not k.startswith("__")]
        main_key = keys[0] if keys else None
        return mat_data[main_key]
    except Exception:
        import h5py

        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            main_key = keys[0]
            return np.array(f[main_key])


def extract_record_matrix(cell_data: Any, index: int = 0) -> Optional[np.ndarray]:
    """
    Ekstrak matriks sinyal [3, N_samples] dari cell array.
    Baris 0: PPG, Baris 1: ABP, Baris 2: ECG.
    """
    try:
        if isinstance(cell_data, np.ndarray):
            if cell_data.dtype == object:
                raw_matrix = cell_data[0, index] if cell_data.ndim == 2 else cell_data[index]
            else:
                raw_matrix = cell_data[index]
        else:
            raw_matrix = cell_data[index]

        matrix = np.array(raw_matrix, dtype=np.float64)
        if matrix.ndim == 2 and matrix.shape[0] > matrix.shape[1] and matrix.shape[1] == 3:
            matrix = matrix.T
        return matrix
    except Exception:
        return None


def get_num_records(cell_data: Any) -> int:
    """Menghitung jumlah record dalam cell array MATLAB."""
    if isinstance(cell_data, np.ndarray) and cell_data.dtype == object:
        return cell_data.shape[1] if cell_data.ndim == 2 else cell_data.shape[0]
    if hasattr(cell_data, "shape"):
        return cell_data.shape[0]
    return len(cell_data)


def compute_bp_metrics(
    abp_signal: np.ndarray,
    fs: float = 125.0,
    sbp_range: Tuple[float, float] = (60.0, 200.0),
    dbp_range: Tuple[float, float] = (40.0, 130.0),
) -> Dict[str, Any]:
    """
    Menghitung SBP, DBP, dan MAP dari segmen sinyal ABP menggunakan deteksi puncak beat.
    """
    min_peak_distance = int(0.3 * fs)  # minimal 0.3s (maks ~200 BPM)
    min_peak_height = np.percentile(abp_signal, 50)

    systolic_peaks, _ = find_peaks(abp_signal, distance=min_peak_distance, height=min_peak_height)
    diastolic_troughs, _ = find_peaks(-abp_signal, distance=min_peak_distance)

    sbp = float(np.mean(abp_signal[systolic_peaks])) if len(systolic_peaks) > 0 else float(np.max(abp_signal))
    dbp = float(np.mean(abp_signal[diastolic_troughs])) if len(diastolic_troughs) > 0 else float(np.min(abp_signal))
    map_val = float(np.mean(abp_signal))

    is_valid = (
        len(systolic_peaks) > 0
        and sbp_range[0] <= sbp <= sbp_range[1]
        and dbp_range[0] <= dbp <= dbp_range[1]
        and sbp > dbp + 10.0  # Pulse pressure minimal 10 mmHg
    )

    return {
        "sbp": sbp,
        "dbp": dbp,
        "map": map_val,
        "valid": is_valid,
    }


def preprocess_kachuee_part(
    mat_path: str,
    output_npz_path: str,
    epoch_len: int = 1250,
    fs: float = 125.0,
    max_records: Optional[int] = None,
    max_epochs: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Memproses 1 berkas .mat menjadi berkas .npz terkompresi siap latih.
    """
    log_fn = logger.info if logger else print
    os.makedirs(os.path.dirname(os.path.abspath(output_npz_path)), exist_ok=True)

    part_name = os.path.basename(mat_path)
    part_stem = os.path.splitext(part_name)[0]
    log_fn(f"⚡ Melakukan pra-pemrosesan end-to-end: {part_name} -> {os.path.basename(output_npz_path)}")

    cell_array = load_mat_file(mat_path)
    num_records = get_num_records(cell_array)
    if max_records is not None:
        num_records = min(num_records, max_records)

    pipeline = ECGPreprocessingPipeline()
    X_part: List[np.ndarray] = []
    y_part: List[List[float]] = []

    pbar = tqdm(range(num_records), desc=f"  {part_stem}", disable=(logger is not None))
    for r_idx in pbar:
        rec = extract_record_matrix(cell_array, r_idx)
        if rec is None or rec.ndim != 2 or rec.shape[0] < 3:
            continue

        raw_abp = rec[1, :]
        raw_ecg = rec[2, :]

        if len(raw_ecg) < epoch_len:
            continue

        for start in range(0, len(raw_ecg) - epoch_len + 1, epoch_len):
            seg_ecg = raw_ecg[start : start + epoch_len]
            seg_abp = raw_abp[start : start + epoch_len]

            # Validasi sinyal dasar
            if np.std(seg_ecg) < 1e-4 or np.isnan(seg_ecg).any() or np.isnan(seg_abp).any():
                continue

            bp = compute_bp_metrics(seg_abp, fs=fs)
            if not bp["valid"]:
                continue

            clean_ecg = pipeline.process(seg_ecg, input_fs=fs)
            X_part.append(clean_ecg.astype(np.float32))
            y_part.append([bp["sbp"], bp["dbp"], bp["map"]])

            if max_epochs is not None and len(X_part) >= max_epochs:
                break

        if max_epochs is not None and len(X_part) >= max_epochs:
            break

    X_arr = np.array(X_part, dtype=np.float32)
    y_arr = np.array(y_part, dtype=np.float32)

    if len(X_arr) == 0:
        raise ValueError(f"Tidak ada segmen valid yang berhasil diekstrak dari {mat_path}")

    np.savez_compressed(output_npz_path, X=X_arr, y=y_arr, fs=fs, part_name=part_name)
    size_mb = os.path.getsize(output_npz_path) / (1024 * 1024)
    log_fn(f"  ✅ Selesai: {len(X_arr):,} epoch tersimpan -> {output_npz_path} ({size_mb:.1f} MB)")
    return output_npz_path
