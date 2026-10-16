# CONTEXT & SYSTEM SPECIFICATION: ECG-BP ESTIMATION (FAN ET AL., 2021)

Dokumen ini dirancang sebagai **konteks menyeluruh (*system context / prompt injection*)** untuk agent AI lain yang bekerja pada workspace repositori ini. Dokumen ini menjelaskan perbandingan antara **spesifikasi paper acuan utama (Fan et al., Sensors 2021)** dan **penyesuaian metodologis skripsi** yang diterapkan di repositori ini.

---

## 1. Identitas Proyek & Referensi Utama

- **Topik Skripsi**: Estimasi Tekanan Darah Non-Invasif (SBP, DBP, MAP) Kontinu Berbasis Sinyal Elektrokardiogram (ECG) Menggunakan Deep Learning Multi-Task.
- **Paper Acuan Utama**:
  > Xiaomao Fan, Hailiang Wang, Yang Zhao, Ye Li, and Kwok Leung Tsui,
  > *"An Adaptive Weight Learning-Based Multitask Deep Network for Continuous Blood Pressure Estimation Using Electrocardiogram Signals"*,
  > **Sensors 2021**, 21(5), 1595. DOI: [10.3390/s21051595](https://doi.org/10.3390/s21051595).
  >

---

## 2. Spesifikasi Asli Paper Fan et al. (2021)

| Aspek                                | Spesifikasi Resmi Paper Fan et al. (Sensors 2021)                                                                                                                                                                                                                       |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset**                    | PhysioNet MIMIC-II Version 3.0 Waveform Database (21.422 sinyal sinkron ECG Lead II & ABP).                                                                                                                                                                             |
| **Data Split**                 | Random split independen:**80% Training**, **10% Validation**, **10% Testing**.                                                                                                                                                                        |
| **Filtering ECG**              | **Discrete Wavelet Transform (DWT)** wavelet Daubechies 8 (`db8`) 8 tingkat dekomposisi. Koefisien aproksimasi tingkat ke-8 (0–0,24 Hz) dan detail tingkat ke-1 (31,25–62,5 Hz) dinolkan, lalu direkonstruksi via inverse DWT (pita efektif: 0,24–31,25 Hz). |
| **Resampling**                 | Sinyal ECG direkonstruksi di-downsample dari**125 Hz ke 100 Hz**.                                                                                                                                                                                                 |
| **Panjang Jendela**            | 10 detik (@ 100 Hz = 1.000 titik sampel).                                                                                                                                                                                                                               |
| **Normalisasi ECG**            | Min-Max Normalization:$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1]$.                                                                                                                                                                        |
| **Ekstraksi ABP Ground Truth** | •$\text{SBP} = \max(\text{segmen 10s})$• $\text{DBP} = \min(\text{segmen 10s})$• $\text{MAP} = \frac{\text{SBP} + 2\cdot\text{DBP}}{3}$.                                                                                                                       |
| **Quality Assessment ABP**     | Skewness index dihitung pada sub-segmen ABP 1 detik*non-overlapping*. Jika ada nilai skewness $< 0$, seluruh segmen 10 detik (ECG & ABP) **dibuang**.                                                                                                         |
| **Struktur Input BiLSTM**      | **`Time steps: 50`**, **`Sequence length: 20`** ($50 \times 20 = 1.000$ titik). Sinyal 1.000 sampel di-reshape menjadi matriks $50 \times 20$.                                                                                                      |
| **Sha red Backbone**          | 2-Layer Bidirectional LSTM (BiLSTM),`hidden_size = 256` per arah (output terminal forward akhir + backward awal di-concat = **512 unit**).                                                                                                                      |
| **Task-Specific Heads**        | 3 head terpisah (SBP, DBP, MAP) dengan arsitektur identik:`Linear(512 → 512)` → `ReLU` → `Linear(512 → 256)` → `ReLU` → `Linear(256 → 1)`.                                                                                                             |
| **Dropout**                    | **Tidak ada dropout** (baik pada BiLSTM maupun FC heads).                                                                                                                                                                                                         |
| **Inisialisasi Bobot**         | Seluruh bobot dan bias diinisialisasi dari distribusi seragam$U(-\sqrt{d}, \sqrt{d})$ dengan $d = 256$.                                                                                                                                                             |
| **Optimizer**                  | **Mini-batch SGD with Momentum**: `lr = 0.0008`, `momentum = 0.9`, `batch_size = 128`.                                                                                                                                                                      |
| **Regularisasi L2**            | **Weight decay = 0.1** pada seluruh bobot dan bias.                                                                                                                                                                                                               |
| **Jumlah Epoch**               | **500 Epoch**.                                                                                                                                                                                                                                                    |
| **LR Scheduler**               | `ReduceLROnPlateau(mode='min', factor=0.9, patience=4)`. **Dipicu saat Training Loss tidak lagi menurun**.                                                                                                                                                      |
| **Loss Function Training**     | **RMSE / Root Mean Square Error** (Caption Gambar 1: *"computing the root mean square loss values"*).                                                                                                                                                           |
| **Adaptive Weight Learning**   | Formulasi Eq. (4), (8), (9), (10) & Algorithm 1:• Inisialisasi:$\theta_s = \theta_d = \theta_m = 1/3$• Dihitung di akhir tiap epoch validasi berdasarkan *Mean* ($L_{\text{mean}}$) & *Std* ($L_{\text{std}}$) MAE dari $M$ batch validasi:               |

$$
T_{\text{mean}}^i = \frac{\|L_{\text{mean}}^i(k) - L_{\text{mean}}^i(k-1)\|_1}{L_{\text{mean}}^i(k)}
$$

$$
T_{\text{std}}^i = \frac{\|L_{\text{std}}^i(k) - L_{\text{std}}^i(k-1)\|_1}{L_{\text{std}}^i(k)}
$$

$$
\tilde{\theta}^i = T_{\text{mean}}^i \cdot T_{\text{std}}^i \cdot (L_{\text{mean}}^i + L_{\text{std}}^i)
$$

$$
\theta^i = \frac{\tilde{\theta}^i}{\sum_j \tilde{\theta}^j}
$$

 |
| **Standar Evaluasi**           | •**AAMI SP10**: Mean Error $\le 5\text{ mmHg}$, STD $\le 8\text{ mmHg}$• **BHS Protocol**: Persentase kumulatif eror absolut $\le 5, \le 10, \le 15\text{ mmHg}$ (Grade A/B/C/D)• **Bland-Altman Analysis**                                                                                                                                                                                                                                                                                                                                                                                                                  |

---

## 3. Penyesuaian Metodologi Skripsi di Workspace Ini

Repositori ini mengadopsi paper Fan et al. (2021) dengan sejumlah **penyesuaian terencana** demi kebutuhan konteks skripsi (integrasi hardware sensor Shimmer3 ECG & ketepatan fisiologis klinis):

1. **Filtering Sinyal ECG**:
   - Menggunakan **IIR Notch Filter (50 Hz, Q = 30)** + **Butterworth Bandpass Filter (0.5 – 40 Hz, orde 2 / orde 4 efektif via `filtfilt`)**.
   - *Alasan*: Menyesuaikan dengan frekuensi jala-jala listrik PLN Indonesia (50 Hz) dan spesifikasi sensor Shimmer3 ECG (metode Alhamid 2026 & Hakim 2025).
2. **Laju Sampling & Dimensi Sekuens**:
   - Mempertahankan laju sampling **125 Hz** (panjang jendela 10 detik = **1.250 titik sampel**).
   - Input model saat ini adalah `[batch, 1250, 1]` (unrolled BiLSTM 1.250 timestep).
3. **Pembersihan Ground Truth ABP**:
   - Menggunakan **deteksi puncak per detak (*beat-by-beat peak detection*)** via `scipy.signal.find_peaks` (rata-rata puncak sistolik = SBP, rata-rata lembah diastolik = DBP).
   - Filter fisiologis berbasis ambang batas: $50 \le \text{SBP} \le 220\text{ mmHg}$ dan $30 \le \text{DBP} \le 140\text{ mmHg}$.
   - *Alasan*: Min/max global 10 detik rentan terdistorsi oleh satu lonjakan loncatan artefak, sedangkan deteksi beat-by-beat lebih valid secara klinis.
4. **Regularisasi Tambahan**:
   - Ditambahkan `LSTM_DROPOUT = 0.3` dan `FC_DROPOUT = 0.2` serta `grad_clip = 5.0` untuk mencegah *gradient explosion* dan *overfitting* saat bereksperimen dengan subset data yang lebih kecil.
5. **Loss Training & Scheduler**:
   - Loss regresi training menggunakan **MAE (L1Loss)**, konsisten dengan metrik validasi.
   - `scheduler.step(val_loss)` memantau **Validation Loss** (standar *deep learning* modern untuk menghindari *overfitting*).
6. **Path Dataset di Lingkungan Kaggle**:
   - Dataset input terpasang di:
     `/kaggle/input/datasets/mkachuee/BloodPressureDataset`
     *(tanpa download otomatis via `kagglehub`)*.

---

## 4. Panduan Aturan untuk AI Agent Berikutnya

Saat mengembangkan, memodifikasi kode, atau menjalankan pelatihan di repositori ini:

1. **JANGAN sembarangan mengubah formula Adaptive Weight Learning**:
   Implementasi pada `src/models/loss.py` sudah tepat mengikuti Persamaan (8), (9), dan (10) Fan et al. (2021). Bobot $\theta$ diperbarui di akhir epoch validasi tanpa backpropagation.
2. **JANGAN menghapus filter 50 Hz / Butterworth 0.5–40 Hz**:
   Filter ini adalah keputusan metodologis skripsi untuk kompatibilitas sensor Shimmer3.
3. **Fleksibilitas Jumlah Epoch**:
   Secara literatur paper disetel **500 epoch**, namun pada pengujian cepat di Kaggle atau lokal dapat dijalankan 15–50 epoch menggunakan mekanisme *checkpoint resume* (`--auto-resume` atau `--resume`).
4. **Optimizer**:
   Paper menggunakan **Momentum-SGD (lr=0.0008, momentum=0.9, weight_decay=0.1)**. Repositori mendukung SGD ini sebagai default, serta opsi AdamW untuk stabilitas konvergensi alternatif.
5. **Kesesuaian Metrik**:
   Evaluasi test set WAJIB selalu menyertakan tabel standar **AAMI SP10** dan **BHS Protocol Grade**, scatter plot prediksi vs true, serta plot Bland-Altman agreement.
