"""
Konfigurasi Terpusat Sistem Estimasi Tekanan Darah.
Mengatur parameter sampling, filter digital, arsitektur model, dan dataset.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PreprocessingConfig:
    """Konfigurasi untuk Pemrosesan Sinyal Digital ECG."""
    
    # Sampling Rates (Hz)
    shimmer_fs: float = 256.0      # Physical sampling rate dari sensor Shimmer3 ECG
    mimic_fs: float = 125.0        # Sampling rate dari dataset MIMIC-III
    target_fs: float = 125.0       # Target sampling rate untuk keselarasan model
    
    # Notch Filter (Penghilang interferensi listrik PLN 50 Hz)
    notch_freq: float = 50.0       # Frekuensi notch (Hz)
    notch_q: float = 30.0          # Quality factor
    
    # Bandpass Butterworth Filter
    lowcut: float = 0.5            # Cutoff frekuensi bawah (Hz) - eliminasi baseline wander
    highcut: float = 40.0          # Cutoff frekuensi atas (Hz) - eliminasi noise otot/EMG
    filter_order: int = 2          # Orde filter Butterworth (N=2, setara Orde 4 total saat filtfilt)
    
    # Normalisasi Persentil Clipping & Scaling [0, 1] (Proposal Subbab 2.2.4)
    clip_q_low: float = 1.0        # Persentil batas bawah (Q_low = persentil ke-1)
    clip_q_high: float = 99.0      # Persentil batas atas (Q_high = persentil ke-99)
    epsilon: float = 1e-8          # Mencegah pembagian dengan nol


@dataclass
class ModelConfig:
    """Konfigurasi Arsitektur Deep Learning BiLSTM Multitask."""
    
    input_channels: int = 1
    seq_length: int = 1250          # 10 detik data @ 125 Hz
    
    # Sharing Layers (BiLSTM)
    bilstm_hidden_units: int = 256  # Output bidirectional = 2 * 256 = 512
    bilstm_num_layers: int = 2
    bilstm_dropout: float = 0.3
    
    # Task-Specific FC Heads (SBP, DBP, MAP)
    fc_layers: List[int] = field(default_factory=lambda: [512, 256])
    fc_dropout: float = 0.2
    
    # Adaptive Weight Learning
    init_task_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class PathConfig:
    """Konfigurasi Jalur Direktori dan Dataset."""
    
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    
    # Kaggle — Kachuee Cuff-Less Blood Pressure Dataset
    # https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset
    kaggle_dataset_slug: str = "mkachuee/BloodPressureDataset"
    kachuee_mat_dir: str = "data/raw/kachuee_mat"   # folder .mat files lokal


@dataclass
class GlobalConfig:
    """Konfigurasi Utama Gabungan."""
    
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# Instance konfigurasi global default
config = GlobalConfig()
