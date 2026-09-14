"""
Modul Pemuat, Inspeksi, dan Visualisasi Dataset Kachuee et al.
(Cuff-Less Blood Pressure Estimation Dataset - Kaggle/PhysioNet).
"""

import os
import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Tambahkan direktori root src ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.pipeline import ECGPreprocessingPipeline


class KachueeDatasetInspector:
    """
    Kelas pemuat dan penginspeksi dataset Cuff-Less Blood Pressure Estimation (Kachuee et al., 2016).
    """

    def __init__(self, target_fs: float = 125.0):
        self.fs = target_fs
        self.pipeline = ECGPreprocessingPipeline()

    def download_kaggle_dataset(self) -> str:
        """
        Mengunduh dataset mkachuee/BloodPressureDataset via kagglehub.
        """
        import kagglehub

        print("Mengunduh/Memverifikasi dataset Kaggle mkachuee/BloodPressureDataset...")
        path = kagglehub.dataset_download("mkachuee/BloodPressureDataset")
        print(f"Dataset berhasil dimuat di: {path}")
        return path

    def find_mat_files(self, dataset_dir: str):
        """
        Mencari file .mat di direktori dataset.
        """
        mat_files = []
        for root, _, files in os.walk(dataset_dir):
            for file in sorted(files):
                if file.endswith(".mat"):
                    mat_files.append(os.path.join(root, file))
        return mat_files

    def load_mat_file(self, file_path: str):
        """
        Membaca file dataset .mat (misal part_1.mat).
        Mendukung format MATLAB v7 dan HDF5 v7.3.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

        print(f"Membaca berkas MATLAB: {os.path.basename(file_path)}...")
        try:
            mat_data = scipy.io.loadmat(file_path)
            keys = [k for k in mat_data.keys() if not k.startswith("__")]
            main_key = keys[0] if keys else None
            data = mat_data[main_key]
            return data
        except Exception:
            import h5py

            with h5py.File(file_path, "r") as f:
                keys = list(f.keys())
                main_key = keys[0]
                data = np.array(f[main_key])
                return data

    def extract_record_matrix(self, cell_data, index: int = 0) -> np.ndarray:
        """
        Mengekstrak matriks sinyal tunggal dari cell array (misal elemen ke-0).
        Di Kachuee dataset:
          - Baris 0: PPG (125 Hz)
          - Baris 1: ABP (125 Hz)
          - Baris 2: ECG Lead II (125 Hz)
        """
        if isinstance(cell_data, np.ndarray):
            if cell_data.dtype == object:
                raw_matrix = cell_data[0, index]
            else:
                raw_matrix = cell_data[index]
        else:
            raw_matrix = cell_data[index]

        matrix = np.array(raw_matrix, dtype=np.float64)

        # Jika bentuknya (N_samples, 3), transpose menjadi (3, N_samples)
        if matrix.shape[0] > matrix.shape[1] and matrix.shape[1] == 3:
            matrix = matrix.T

        return matrix

    def inspect_and_plot(self, matrix: np.ndarray, record_name: str = "Record #1", save_path: str = None, duration_sec: float = 8.0):
        """
        Mengekstrak sinyal, memfilter ECG, menghitung SBP/DBP/MAP, dan memvisualisasikan grafik 3 panel.
        """
        num_samples = int(duration_sec * self.fs)
        num_samples = min(num_samples, matrix.shape[1])

        ppg = matrix[0, :num_samples]
        abp = matrix[1, :num_samples]
        ecg_raw = matrix[2, :num_samples]

        # Hitung metrik Tekanan Darah dari sinyal ABP
        sbp = float(np.max(abp))
        dbp = float(np.min(abp))
        map_val = float(np.mean(abp))

        # Pra-pemrosesan Sinyal ECG Lead II via Pipeline (input 125 Hz)
        ecg_clean = self.pipeline.process(ecg_raw, input_fs=125.0)

        time_axis = np.arange(len(ecg_raw)) / self.fs

        # Plot Visualisasi
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Panel 1: Sinyal ECG Lead II
        axes[0].plot(time_axis, ecg_raw, color="gray", alpha=0.6, label="ECG Mentah (Raw 125 Hz)")
        axes[0].plot(time_axis, ecg_clean, color="#1f77b4", linewidth=1.5, label="ECG Preprocessed (Pipeline)")
        axes[0].set_title(f"Dataset Kachuee et al. ({record_name}) — Sinyal ECG Lead II", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Amplitudo Skala")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Sinyal PPG
        axes[1].plot(time_axis, ppg, color="#ff7f0e", linewidth=1.5, label="PPG Fingertip Signal (125 Hz)")
        axes[1].set_title("Sinyal PPG (Photoplethysmogram Fingertip)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Amplitudo Optik")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        # Panel 3: Sinyal ABP & Target Ground Truth
        axes[2].plot(time_axis, abp, color="#d62728", linewidth=1.5, label="ABP Waveform (Invasive)")
        axes[2].axhline(y=sbp, color="darkred", linestyle=":", linewidth=1.5, label=f"SBP (Systolic) = {sbp:.1f} mmHg")
        axes[2].axhline(y=map_val, color="green", linestyle="--", linewidth=1.5, label=f"MAP (Mean) = {map_val:.1f} mmHg")
        axes[2].axhline(y=dbp, color="blue", linestyle=":", linewidth=1.5, label=f"DBP (Diastolic) = {dbp:.1f} mmHg")
        axes[2].set_title(f"Sinyal ABP & Ground Truth BP (SBP: {sbp:.1f} | DBP: {dbp:.1f} | MAP: {map_val:.1f} mmHg)", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Waktu (detik)")
        axes[2].set_ylabel("Tekanan Darah (mmHg)")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Grafik visualisasi berhasil disimpan di: {save_path}")

        plt.close()

        return {
            "record_name": record_name,
            "sbp": sbp,
            "dbp": dbp,
            "map": map_val,
            "samples": len(ecg_raw),
            "duration_sec": len(ecg_raw) / self.fs,
        }


def run_inspection():
    inspector = KachueeDatasetInspector()

    # 1. Unduh dataset dari Kaggle
    dataset_dir = inspector.download_kaggle_dataset()

    # 2. Cari file .mat (misal part_1.mat)
    mat_files = inspector.find_mat_files(dataset_dir)
    print("Daftar berkas dataset .mat terdeteksi:", mat_files)

    if not mat_files:
        print("Tidak ada berkas .mat ditemukan.")
        return

    first_mat = mat_files[0]
    cell_data = inspector.load_mat_file(first_mat)

    # 3. Ekstrak rekam data pertama
    matrix = inspector.extract_record_matrix(cell_data, index=0)
    print("Bentuk matriks rekaman sinyal:", matrix.shape)

    # 4. Visualisasikan & Simpan Grafik
    output_png = os.path.join(os.path.dirname(__file__), "..", "data", "visualisasi_dataset_kachuee_real.png")
    metrics = inspector.inspect_and_plot(matrix, record_name=os.path.basename(first_mat) + " (Record #1)", save_path=output_png, duration_sec=8.0)

    print("\nMetrik Hasil Inspeksi Sinyal Nyata:")
    print(f"  - Rekam Data: {metrics['record_name']}")
    print(f"  - Durasi: {metrics['duration_sec']} detik ({metrics['samples']} sampel @ 125 Hz)")
    print(f"  - Systolic BP (SBP): {metrics['sbp']:.2f} mmHg")
    print(f"  - Diastolic BP (DBP): {metrics['dbp']:.2f} mmHg")
    print(f"  - Mean Arterial Pressure (MAP): {metrics['map']:.2f} mmHg")


if __name__ == "__main__":
    run_inspection()
