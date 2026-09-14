"""
Modul Pemuat, Inspeksi, dan Visualisasi Dataset Kachuee et al.
(Cuff-Less Blood Pressure Estimation Dataset - Kaggle/PhysioNet).

Struktur dataset (per baris matriks, sesuai deskripsi resmi Kachuee et al.):
  Baris 0: PPG signal  @ 125 Hz  (photoplethysmograph from fingertip)
  Baris 1: ABP signal  @ 125 Hz  (invasive arterial blood pressure, mmHg)
  Baris 2: ECG signal  @ 125 Hz  (electrocardiogram channel II)
"""

import os
import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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

        Struktur dataset Kachuee et al. (per baris matriks):
          - Baris 0: PPG signal  @ 125 Hz  (photoplethysmograph from fingertip)
          - Baris 1: ABP signal  @ 125 Hz  (invasive arterial blood pressure, mmHg)
          - Baris 2: ECG signal  @ 125 Hz  (electrocardiogram channel II)
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

    def compute_bp_metrics(self, abp_signal: np.ndarray) -> dict:
        """
        Menghitung SBP, DBP, dan MAP dari sinyal ABP menggunakan deteksi puncak per beat.

        Metode beat-by-beat sesuai standar klinis:
          - SBP = rata-rata puncak sistolik (systolic peaks) per beat
          - DBP = rata-rata lembah diastolik (diastolic troughs) per beat
          - MAP = rata-rata amplitudo ABP (mean arterial pressure)

        Ini menggantikan pendekatan global max/min yang tidak representatif secara klinis,
        karena global max overestimate SBP dan global min underestimate DBP akibat artefak.

        Parameters
        ----------
        abp_signal : np.ndarray
            Sinyal ABP invasif 1D dalam satuan mmHg @ target_fs Hz.

        Returns
        -------
        dict
            Dictionary berisi 'sbp', 'dbp', 'map', 'systolic_peaks_idx',
            'diastolic_troughs_idx', 'n_beats'.
        """
        # --- Deteksi Puncak Sistolik (SBP per beat) ---
        # Jarak minimum antar puncak berdasarkan asumsi HR 40–200 bpm:
        #   HR max 200 bpm → jarak min = 60/200 * fs = 0.3 * 125 = 37 sampel
        #   HR min  40 bpm → jarak max = 60/40  * fs = 1.5 * 125 = 188 sampel
        min_peak_distance = int(0.3 * self.fs)          # ~37 sampel (HR ≤ 200 bpm)
        min_peak_height = np.percentile(abp_signal, 50) # Di atas median sinyal ABP

        systolic_peaks_idx, _ = find_peaks(
            abp_signal,
            distance=min_peak_distance,
            height=min_peak_height,
        )

        # --- Deteksi Lembah Diastolik (DBP per beat) ---
        # Invert sinyal untuk menemukan lembah sebagai puncak pada sinyal negatif
        diastolic_troughs_idx, _ = find_peaks(
            -abp_signal,
            distance=min_peak_distance,
        )

        # --- Hitung rata-rata per beat ---
        # Fallback ke global max/min hanya jika deteksi puncak gagal total
        if len(systolic_peaks_idx) > 0:
            sbp = float(np.mean(abp_signal[systolic_peaks_idx]))
        else:
            sbp = float(np.max(abp_signal))

        if len(diastolic_troughs_idx) > 0:
            dbp = float(np.mean(abp_signal[diastolic_troughs_idx]))
        else:
            dbp = float(np.min(abp_signal))

        map_val = float(np.mean(abp_signal))

        return {
            "sbp": sbp,
            "dbp": dbp,
            "map": map_val,
            "systolic_peaks_idx": systolic_peaks_idx,
            "diastolic_troughs_idx": diastolic_troughs_idx,
            "n_beats": len(systolic_peaks_idx),
        }

    def inspect_and_plot(
        self,
        matrix: np.ndarray,
        record_name: str = "Record #1",
        save_path: str = None,
        duration_sec: float = 8.0,
    ):
        """
        Mengekstrak sinyal, memfilter ECG, menghitung SBP/DBP/MAP per beat dari ABP,
        dan memvisualisasikan grafik 3 panel.

        Struktur sinyal (Kachuee et al.):
          - matrix[0]: PPG signal  (photoplethysmograph from fingertip)
          - matrix[1]: ABP signal  (invasive arterial blood pressure, mmHg)
          - matrix[2]: ECG signal  (electrocardiogram channel II)
        """
        num_samples = int(duration_sec * self.fs)
        num_samples = min(num_samples, matrix.shape[1])

        # Ekstrak setiap kanal sesuai spesifikasi dataset Kachuee et al.
        ppg     = matrix[0, :num_samples]   # row 0: PPG
        abp     = matrix[1, :num_samples]   # row 1: ABP (mmHg)
        ecg_raw = matrix[2, :num_samples]   # row 2: ECG Lead II

        # Hitung metrik BP dari sinyal ABP menggunakan deteksi puncak per beat
        bp_metrics           = self.compute_bp_metrics(abp)
        sbp                  = bp_metrics["sbp"]
        dbp                  = bp_metrics["dbp"]
        map_val              = bp_metrics["map"]
        systolic_peaks_idx   = bp_metrics["systolic_peaks_idx"]
        diastolic_troughs_idx = bp_metrics["diastolic_troughs_idx"]
        n_beats              = bp_metrics["n_beats"]

        # Pra-pemrosesan Sinyal ECG Lead II via Pipeline (input sudah 125 Hz)
        ecg_clean = self.pipeline.process(ecg_raw, input_fs=125.0)

        time_axis = np.arange(num_samples) / self.fs

        # --- Plot 3-panel ---
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Panel 1: Sinyal ECG Lead II (mentah vs setelah pipeline)
        axes[0].plot(time_axis, ecg_raw, color="gray", alpha=0.6, label="ECG Mentah (Raw 125 Hz)")
        axes[0].plot(time_axis, ecg_clean, color="#1f77b4", linewidth=1.5, label="ECG Preprocessed (Pipeline)")
        axes[0].set_title(
            f"Dataset Kachuee et al. ({record_name}) — Sinyal ECG Lead II",
            fontsize=12, fontweight="bold"
        )
        axes[0].set_ylabel("Amplitudo Skala [0–1]")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Sinyal PPG
        axes[1].plot(time_axis, ppg, color="#ff7f0e", linewidth=1.5, label="PPG Fingertip Signal (125 Hz)")
        axes[1].set_title("Sinyal PPG (Photoplethysmogram Fingertip)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Amplitudo Optik")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        # Panel 3: Sinyal ABP + penanda puncak/lembah per beat + garis rata-rata
        axes[2].plot(time_axis, abp, color="#d62728", linewidth=1.5, label="ABP Waveform (Invasive)")
        if len(systolic_peaks_idx) > 0:
            axes[2].scatter(
                time_axis[systolic_peaks_idx], abp[systolic_peaks_idx],
                color="darkred", s=45, zorder=5, label=f"Puncak Sistolik ({n_beats} beats)"
            )
        if len(diastolic_troughs_idx) > 0:
            axes[2].scatter(
                time_axis[diastolic_troughs_idx], abp[diastolic_troughs_idx],
                color="royalblue", s=45, marker="v", zorder=5, label="Lembah Diastolik"
            )
        axes[2].axhline(
            y=sbp, color="darkred", linestyle=":", linewidth=1.5,
            label=f"SBP (rata-rata per beat) = {sbp:.1f} mmHg"
        )
        axes[2].axhline(
            y=map_val, color="green", linestyle="--", linewidth=1.5,
            label=f"MAP (mean ABP) = {map_val:.1f} mmHg"
        )
        axes[2].axhline(
            y=dbp, color="royalblue", linestyle=":", linewidth=1.5,
            label=f"DBP (rata-rata per beat) = {dbp:.1f} mmHg"
        )
        axes[2].set_title(
            f"Sinyal ABP & Ground Truth BP per Beat  "
            f"(SBP: {sbp:.1f} | DBP: {dbp:.1f} | MAP: {map_val:.1f} mmHg | n={n_beats} beats)",
            fontsize=12, fontweight="bold"
        )
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
            "n_beats": n_beats,
            "samples": num_samples,
            "duration_sec": num_samples / self.fs,
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
    print(f"Kanal: [0]=PPG | [1]=ABP | [2]=ECG  ({matrix.shape[1]} sampel @ 125 Hz)")

    # 4. Visualisasikan & Simpan Grafik
    output_png = os.path.join(
        os.path.dirname(__file__), "..", "data", "visualisasi_dataset_kachuee_real.png"
    )
    metrics = inspector.inspect_and_plot(
        matrix,
        record_name=os.path.basename(first_mat) + " (Record #1)",
        save_path=output_png,
        duration_sec=8.0,
    )

    print("\nMetrik Hasil Inspeksi Sinyal Nyata:")
    print(f"  - Rekam Data      : {metrics['record_name']}")
    print(f"  - Durasi          : {metrics['duration_sec']:.1f} detik ({metrics['samples']} sampel @ 125 Hz)")
    print(f"  - Beat terdeteksi : {metrics['n_beats']} beats")
    print(f"  - SBP             : {metrics['sbp']:.2f} mmHg  [rata-rata {metrics['n_beats']} puncak sistolik]")
    print(f"  - DBP             : {metrics['dbp']:.2f} mmHg  [rata-rata {metrics['n_beats']} lembah diastolik]")
    print(f"  - MAP             : {metrics['map']:.2f} mmHg  [mean ABP]")


if __name__ == "__main__":
    run_inspection()
