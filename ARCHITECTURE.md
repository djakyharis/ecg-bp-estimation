 

# System Architecture

> **Full pipeline** dari sinyal ECG mentah hingga output estimasi tekanan darah (SBP, DBP, MAP).

---

## Overview Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         INPUT                                            │
│  Sinyal ECG mentah — np.ndarray shape: (N_samples,) atau (N_samples, 1) │
│  Sensor Shimmer3 ECG @ 256 Hz  ──OR──  Dataset Kachuee @ 125 Hz         │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      STAGE 1 — PREPROCESSING                             │
│                    ECGPreprocessingPipeline.process()                    │
│                                                                          │
│  1a. Resampling          256 Hz → 125 Hz  (resample_poly, GCD-reduced)  │
│  1b. IIR Notch Filter    eliminasi noise PLN 50 Hz  (Q = 30, filtfilt)  │
│  1c. Butterworth BPF     0.5 – 40 Hz, Orde 4 efektif (filtfilt)         │
│  1d. Normalisasi         Min-Max Normalization [0, 1] (Fan et al., 2021)│
│                                                                          │
│  Output: np.ndarray shape: (1250,) atau (1250, 1)                       │
│          = 10 detik sinyal bersih @ 125 Hz                               │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      STAGE 2 — SHARED ENCODER                            │
│                    Bidirectional LSTM (BiLSTM)                           │
│                                                                          │
│  Input:  (batch, seq_len=1250, input_channels=1)                        │
│  Layer 1: BiLSTM  hidden=256  →  output: (batch, 1250, 512)             │
│  Layer 2: BiLSTM  hidden=256  →  output: (batch, 1250, 512)             │
│           dropout=0.3 antar layer                                        │
│  Pooling: Last hidden state  →  shape: (batch, 512)                     │
│                                                                          │
│  Output: feature vector — (batch, 512)                                  │
└──────────────┬──────────────┬──────────────┬────────────────────────────┘
               │              │              │
          SBP Head        DBP Head       MAP Head
               │              │              │
               ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  STAGE 3 — TASK-SPECIFIC HEADS (×3)                     │
│                  Fully Connected (FC) Layers                             │
│                                                                          │
│  Shared input: (batch, 512)                                              │
│                                                                          │
│  FC Head (tiap task):                                                   │
│    Linear(512 → 256) → ReLU → Dropout(0.2)                             │
│    Linear(256 → 1)                                                      │
│                                                                          │
│  Output masing-masing head: scalar — (batch, 1)                         │
└──────────────┬──────────────┬──────────────┬────────────────────────────┘
               │              │              │
               ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   STAGE 4 — ADAPTIVE WEIGHT LEARNING                    │
│                                                                          │
│  Loss gabungan:                                                          │
│    L_total = λ_sbp · L_sbp + λ_dbp · L_dbp + λ_map · L_map            │
│                                                                          │
│  λ_sbp, λ_dbp, λ_map = learnable parameters (diinisialisasi 1.0)       │
│  Dioptimasi bersama parameter model via backpropagation.                │
│                                                                          │
│  L_task = MAE(y_pred, y_true)  per task                                 │
└──────────────┬──────────────┬──────────────┬────────────────────────────┘
               │              │              │
               ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                          │
│                                                                          │
│  ŷ_SBP — Systolic Blood Pressure   (mmHg)                              │
│  ŷ_DBP — Diastolic Blood Pressure  (mmHg)                              │
│  ŷ_MAP — Mean Arterial Pressure    (mmHg)                              │
│                                                                          │
│  Shape: (batch, 3) atau 3 × (batch, 1)                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1 — Preprocessing Detail

### 1a. Resampling

| Parameter            | Nilai                                             |
| -------------------- | ------------------------------------------------- |
| Source fs (Shimmer3) | 256 Hz                                            |
| Source fs (Kachuee)  | 125 Hz                                            |
| Target fs            | 125 Hz                                            |
| Metode               | `scipy.signal.resample_poly` dengan reduksi GCD |

Jika input sudah 125 Hz (dataset Kachuee), tahap ini dilewati.

### 1b. IIR Notch Filter

Menghilangkan interferensi frekuensi jala-jala listrik PLN (50 Hz).

| Parameter             | Nilai                     |
| --------------------- | ------------------------- |
| Frekuensi notch (f₀) | 50 Hz                     |
| Quality factor (Q)    | 30                        |
| Filtering method      | Zero-phase (`filtfilt`) |

```
Formula: H(z) = B(z) / A(z)  dari iirnotch(w0=50, Q=30, fs=125)
filtfilt → tidak ada phase shift, orde efektif 2×
```

### 1c. Butterworth Bandpass Filter

Membatasi bandwidth sinyal ECG pada rentang fisiologis.

| Parameter        | Nilai                                   |
| ---------------- | --------------------------------------- |
| Lowcut           | 0.5 Hz (eliminasi baseline wander)      |
| Highcut          | 40 Hz (eliminasi noise otot/EMG)        |
| Orde desain (N)  | 2                                       |
| Orde efektif     | 4 (karena`filtfilt` forward-backward) |
| Filtering method | Zero-phase (`filtfilt`)               |

### 1d. Normalisasi Min-Max [0, 1]

Sesuai metode yang digunakan pada paper acuan utama Fan et al. (2021) & Li et al. (2020).

| Parameter    | Nilai                                |
| ------------ | ------------------------------------ |
| x_min        | Nilai minimum pada segmen sinyal     |
| x_max        | Nilai maksimum pada segmen sinyal    |
| Output range | [0, 1]                               |
| ε            | 1 × 10⁻⁸                             |

```
x_norm = (signal - x_min) / (x_max - x_min + ε)
```

---

## Stage 2 — Shared Encoder (BiLSTM)

```
Input: (batch, 1250, 1)
  │
  ├─ BiLSTM Layer 1: hidden=256, bidirectional=True
  │    forward  LSTM: (batch, 1250, 256)
  │    backward LSTM: (batch, 1250, 256)
  │    concat        → (batch, 1250, 512)
  │    dropout(0.3)
  │
  └─ BiLSTM Layer 2: hidden=256, bidirectional=True
       concat        → (batch, 1250, 512)
       last timestep → (batch, 512)          ← feature vector
```

| Parameter             | Nilai                |
| --------------------- | -------------------- |
| Hidden units per arah | 256                  |
| Jumlah layer          | 2                    |
| Output dimensi        | 512 (2 × 256)       |
| Dropout antar layer   | 0.3                  |
| Sequence length       | 1250 (10 s @ 125 Hz) |

---

## Stage 3 — Task-Specific FC Heads

Arsitektur **identik** untuk ketiga head (SBP, DBP, MAP):

```
Input: (batch, 512)
  │
  Linear(512 → 256)
  ReLU
  Dropout(0.2)
  │
  Linear(256 → 1)
  │
Output: (batch, 1)  ← estimasi tekanan darah dalam mmHg
```

---

## Stage 4 — Adaptive Weight Learning

Loss function gabungan multitask dengan bobot yang dapat dipelajari:

```
L_total = λ_sbp · MAE(ŷ_sbp, y_sbp)
        + λ_dbp · MAE(ŷ_dbp, y_dbp)
        + λ_map · MAE(ŷ_map, y_map)
```

| Parameter | Inisialisasi | Sifat                              |
| --------- | ------------ | ---------------------------------- |
| λ_sbp    | 1.0          | Learnable, dioptimasi via backprop |
| λ_dbp    | 1.0          | Learnable, dioptimasi via backprop |
| λ_map    | 1.0          | Learnable, dioptimasi via backprop |

---

## Ringkasan Shape Tensor

| Stage                        | Shape                                       |
| ---------------------------- | ------------------------------------------- |
| Raw ECG input                | `(N,)` atau `(N, 1)`                    |
| Setelah resampling           | `(1250,)` atau `(1250, 1)`              |
| Setelah filter + normalisasi | `(1250,)` atau `(1250, 1)`              |
| Input model (batched)        | `(batch, 1250, 1)`                        |
| Setelah BiLSTM (feature)     | `(batch, 512)`                            |
| Output tiap head             | `(batch, 1)`                              |
| **Output final**       | **`(batch, 3)` — [SBP, DBP, MAP]** |

---

## Konfigurasi Referensi

Semua nilai di atas bersumber dari [`src/config.py`](src/config.py):

```python
PreprocessingConfig(
    shimmer_fs=256.0, target_fs=125.0,
    notch_freq=50.0, notch_q=30.0,
    lowcut=0.5, highcut=40.0, filter_order=2,
    clip_q_low=1.0, clip_q_high=99.0, epsilon=1e-8
)

ModelConfig(
    input_channels=1, seq_length=1250,
    bilstm_hidden_units=256, bilstm_num_layers=2, bilstm_dropout=0.3,
    fc_layers=[512, 256], fc_dropout=0.2,
    init_task_weights=[1.0, 1.0, 1.0]
)
```
