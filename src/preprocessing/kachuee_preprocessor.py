"""
Modul Pra-Pemrosesan Batch Dataset Kachuee et al.
Mengekstrak rekaman .mat (ECG Lead II + ABP), segmentasi 10 detik (1250 titik @ 125 Hz),
melakukan filtering Notch 50 Hz + Bandpass 0.5-40 Hz + MinMax [0, 1],
mengekstrak ground-truth BP (SBP, DBP, MAP), dan menyimpan ke .npz terkompresi.
"""

import os
import sys
import glob
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Pastikan root direktori selalu ada di sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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


def check_abp_skewness(abp_signal: np.ndarray, num_windows: int = 10) -> bool:
    """
    Quality assessment gelombang ABP sesuai Fan et al. (2021) Section 2.2:
    Membagi segmen 10 detik (1250 titik @ 125 Hz) menjadi 10 sub-jendela 1 detik (125 titik).
    Jika terdapat minimal satu jendela dengan nilai skewness < 0,
    maka segmen terdistorsi (kateter tertekan / artefak) dan dieliminasi (return False).
    """
    total_len = len(abp_signal)
    if total_len < num_windows:
        return False
    window_len = total_len // num_windows
    trimmed_len = window_len * num_windows
    windows = abp_signal[:trimmed_len].reshape(num_windows, window_len)

    # Menghitung momen ke-2 dan ke-3
    means = np.mean(windows, axis=1, keepdims=True)
    diffs = windows - means
    m2 = np.mean(diffs ** 2, axis=1)
    m3 = np.mean(diffs ** 3, axis=1)

    std_cubed = np.where(m2 < 1e-8, 1e-8, np.power(m2, 1.5))
    skews = m3 / std_cubed

    # Kriteria eliminasi Fan et al. (2021): Semua jendela 1s harus memiliki skewness >= 0
    return bool(np.all(skews >= 0.0))


def compute_bp_metrics(
    abp_signal: np.ndarray,
    fs: float = 125.0,
    sbp_range: Tuple[float, float] = (60.0, 200.0),
    dbp_range: Tuple[float, float] = (40.0, 130.0),
    check_skewness: bool = True,
) -> Dict[str, Any]:
    """
    Menghitung SBP, DBP, dan MAP dari segmen sinyal ABP menggunakan deteksi puncak beat.
    Dilengkapi filter skewness positif sesuai Fan et al. (2021).
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

    if is_valid and check_skewness:
        is_valid = check_abp_skewness(abp_signal, num_windows=10)

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
    check_skewness: bool = True,
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

            bp = compute_bp_metrics(seg_abp, fs=fs, check_skewness=check_skewness)
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


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Pra-pemrosesan batch dataset Kachuee .mat ke .npz dengan ABP Skewness Filter (Fan et al. 2021)"
    )
    parser.add_argument(
        "--parts", type=str, default="1",
        help="Nomor part yang akan diproses (misal: '1', '1,2,3', atau 'all')",
    )
    parser.add_argument(
        "--mat-dir", type=str, default="data/kachuee_mat",
        help="Folder penyimpanan berkas .mat mentah",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed_clean",
        help="Folder penyimpanan berkas .npz bersih",
    )
    parser.add_argument(
        "--check-skewness", action="store_true", default=True,
        help="Aktifkan filter skewness positif ABP (Fan et al. 2021)",
    )
    parser.add_argument(
        "--no-skewness", dest="check_skewness", action="store_false",
        help="Nonaktifkan filter skewness",
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="Maksimal records per part untuk pengujian cepat",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="Maksimal epoch per part untuk pengujian cepat",
    )

    cli_args = parser.parse_args()
    os.makedirs(cli_args.output_dir, exist_ok=True)

    if cli_args.parts.strip().lower() == "all":
        mat_files = sorted(glob.glob(os.path.join(cli_args.mat_dir, "part_*.mat")))
        parts_list = [
            os.path.splitext(os.path.basename(p))[0].replace("part_", "")
            for p in mat_files
        ]
    else:
        parts_list = [p.strip() for p in cli_args.parts.split(",") if p.strip()]

    print(f"🚀 Memproses {len(parts_list)} berkas part dari '{cli_args.mat_dir}'...")
    print(f"   Quality Filter Skewness ABP : {'AKTIF (Fan et al. 2021)' if cli_args.check_skewness else 'NONAKTIF'}")
    print(f"   Folder Output               : '{cli_args.output_dir}'")

    for part_idx in parts_list:
        in_mat = os.path.join(cli_args.mat_dir, f"part_{part_idx}.mat")
        out_npz = os.path.join(cli_args.output_dir, f"part_{part_idx}_clean.npz")
        if not os.path.exists(in_mat):
            print(f"⚠️ Berkas {in_mat} tidak ditemukan, dilewati.")
            continue
        try:
            preprocess_kachuee_part(
                mat_path=in_mat,
                output_npz_path=out_npz,
                check_skewness=cli_args.check_skewness,
                max_records=cli_args.max_records,
                max_epochs=cli_args.max_epochs,
            )
        except Exception as exc:
            print(f"❌ Gagal memproses {in_mat}: {exc}")

