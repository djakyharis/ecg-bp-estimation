# Konteks Alur Kerja Pelatihan GPU & Remote SSH (AI & Developer Guide)

Dokumen ini menyediakan konteks teknis terpusat mengenai sistem estimasi tekanan darah berbasis sinyal ECG untuk digunakan oleh **AI Assistant**, **Developer**, atau **Peneliti** yang bekerja di lingkungan server GPU remote via SSH.

---

## 1. Ikhtisar Penelitian & Domain Masalah

* **Topik Skripsi**: Estimasi Tekanan Darah Non-Invasif Kontinu Berbasis Sinyal ECG Menggunakan Deep Learning.
* **Target Estimasi (Multi-Output)**:
  1. **SBP** (*Systolic Blood Pressure*) — Tekanan darah sistolik (mmHg).
  2. **DBP** (*Diastolic Blood Pressure*) — Tekanan darah diastolik (mmHg).
  3. **MAP** (*Mean Arterial Pressure*) — Tekanan arteri rata-rata (mmHg).
* **Dataset Acuan**: *Cuff-less Blood Pressure Estimation Dataset* (Kachuee et al., 2016 / PhysioNet MIMIC-II). Terdiri dari file `.mat` (part 1–12) yang telah diproses menjadi segmen `.npz` 10 detik @ 125 Hz (1250 sampel).
* **Paper Metodologi Utama**: Fan et al. (2021, Sensors, MDPI) — *"An Adaptive Weight Learning-Based Multitask Deep Network for Continuous Blood Pressure Estimation Using Electrocardiogram Signals"*.

---

## 2. Arsitektur Model & Formulasi Loss

```
Sinyal ECG [batch, 1250, 1]
         │
         ▼
┌────────────────────────────────┐
│   Shared BiLSTM Encoder        │  Layer 1: hidden=256, bidirectional → 512 units (dropout=0.3)
│   (Fan et al., 2021 Tabel 1)   │  Layer 2: hidden=256, bidirectional → 512 units (dropout=0.3)
└────────────────────────────────┘
         │ Ekstraksi Terminal State: [fwd_last ‖ bwd_first] → [batch, 512]
    ┌────┴────────────┬──────────────┐
    ▼                 ▼              ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   SBP Head    │ │   DBP Head    │ │   MAP Head    │  Masing-masing Head:
│ FC(512→512)   │ │ FC(512→512)   │ │ FC(512→512)   │  Linear(512→512) → ReLU → Drop(0.2) →
│ FC(512→256)   │ │ FC(512→256)   │ │ FC(512→256)   │  Linear(512→256) → ReLU → Drop(0.2) →
│ FC(256→1)     │ │ FC(256→1)     │ │ FC(256→1)     │  Linear(256→1) [Linear murni tanpa aktivasi]
└───────────────┘ └───────────────┘ └───────────────┘
```

### Formulasi Adaptive Weight Learning (Eq. 4, 8, 9, 10 Fan et al. 2021)
* **Total Loss**:
  $$\mathcal{L}_{\text{total}} = \theta_{\text{sbp}} \mathcal{L}_{\text{sbp}} + \theta_{\text{dbp}} \mathcal{L}_{\text{dbp}} + \theta_{\text{map}} \mathcal{L}_{\text{map}}$$
* **Inisialisasi**: $\theta = [\frac{1}{3}, \frac{1}{3}, \frac{1}{3}]$
* **Update Otomatis**: Setiap akhir epoch validasi, bobot $\theta_i$ dihitung ulang berdasarkan laju perubahan rata-rata error ($T_{\text{mean}}^i$) dan standar deviasi error ($T_{\text{std}}^i$) pada validation set.

---

## 3. Komponen Alur Kerja Pelatihan ([src/train.py](file:///Users/sir/Documents/Skripsi/ecg-bp-estimation/src/train.py))

Skrip utama [src/train.py](file:///Users/sir/Documents/Skripsi/ecg-bp-estimation/src/train.py) telah dirancang khusus untuk lingkungan GPU jarak jauh dengan fitur berikut:

### A. Akselerasi Hardware GPU
1. **Automatic Mixed Precision (AMP FP16)**:
   * Menggunakan `torch.amp.autocast('cuda')` dan `torch.amp.GradScaler('cuda')`.
   * Mempercepat komputasi forward-backward pada Tensor Cores NVIDIA hingga 2x dan memangkas penggunaan VRAM hingga ~50%.
2. **DataLoader Asynchronous (DMA)**:
   * `pin_memory=True`: Mengunci halaman memori CPU host untuk transfer DMA langsung ke VRAM GPU.
   * `num_workers=4`: Multiprocess CPU workers untuk mencegah kelaparan data (*GPU starvation*).
   * `persistent_workers=True`: Mencegah *overhead* penghancuran dan pembuatan ulang proses worker di tiap pergantian epoch.
3. **cuDNN Benchmark Autotuner**:
   * `torch.backends.cudnn.benchmark = True`: Mencari algoritma konvolusi/LSTM tercepat untuk sequence length statis (1250 sampel).
4. **Headless Safe Visualization**:
   * `matplotlib.use('Agg')`: Menghasilkan grafik langsung ke disk tanpa ketergantungan pada server grafis X11/GUI.

---

### B. Ketahanan Sesi SSH & Mitigasi Putus Koneksi (*Fault Tolerance*)
1. **Penanganan Sinyal POSIX (`SIGHUP`, `SIGINT`, `SIGTERM`)**:
   * Saat koneksi SSH terputus, kernel Linux mengirim sinyal `SIGHUP`.
   * Handler di [src/train.py](file:///Users/sir/Documents/Skripsi/ecg-bp-estimation/src/train.py#L58) menangkap sinyal tersebut dan **segera melakukan penyimpanan darurat** (`emergency_checkpoint.pt`) sebelum keluar.
2. **Sistem Checkpoint Ganda (*Dual Checkpointing*)**:
   * `last_checkpoint.pt`: Disimpan setiap akhir epoch dengan mekanisme penulisan atomik (`.tmp` → `os.rename`) agar berkas tidak rusak jika server mati mendadak.
   * `best_model.pt`: Disimpan saat rata-rata validation MAE mencapai nilai terbaik.
3. **Mekanisme Resume & Auto-Resume**:
   * Tambahkan argumen `--auto-resume`: Skrip akan otomatis mendeteksi `last_checkpoint.pt`, memulihkan `epoch`, model, optimizer, scheduler, scaler AMP, riwayat loss, dan bobot adaptif $\theta$, lalu melanjutkan epoch berikutnya tanpa intervensi.
   * Argumen `--resume <path>`: Untuk memulihkan checkpoint spesifik secara manual.
4. **Dual Logging Real-Time**:
   * Menulis ke *stdout* dan file log (`train.log`) secara simultan dengan *autoflush*, sehingga dapat dipantau dari luar SSH via `tail -f`.

---

## 4. Standar Evaluasi Medis & Artefak Output

Setiap sesi training yang selesai akan mengevaluasi unseen Test Set (10% data) dan menghasilkan artefak lengkap di direktori `checkpoints/` (atau `--output-dir`):

| File Artefak | Deskripsi Isi |
| :--- | :--- |
| `best_model.pt` | Model dengan MAE validasi terendah (berisi bobot, optimizer, scheduler, riwayat). |
| `last_checkpoint.pt` | Checkpoint epoch terakhir untuk fitur `--auto-resume`. |
| `final_model_full.pt` | Paket model siap pakai untuk deployment (arsitektur + bobot + metrik evaluasi). |
| `training_history.json` | Log metrik numerik per epoch: train loss, val loss, MAE SBP/DBP/MAP, LR, dan $\theta$. |
| `evaluation_results.json` | Hasil evaluasi test set lengkap: MAE, RMSE, ME, BHS grade, kelulusan AAMI, dan hyperparameter. |
| `training_curves.png` | 4 panel grafik: Loss, MAE vs ambang AAMI $\le 5\text{ mmHg}$, Learning Rate, dan Bobot Adaptif $\theta$. |
| `scatter_pred_vs_true.png` | Scatter plot prediksi vs aktual per target (SBP, DBP, MAP) dengan pita toleransi $\pm 5\text{ mmHg}$. |
| `bland_altman.png` | Analisis kesepakatan klinis Bland-Altman dengan garis *Bias* dan rentang $\pm 1.96\text{SD}$. |

### Kriteria Klinis yang Diuji:
1. **AAMI SP10 Standard**:
   * $\text{Mean Error (ME / Bias)} \le \pm 5.0\text{ mmHg}$
   * $\text{Standard Deviation (STD)} \le 8.0\text{ mmHg}$
2. **BHS Protocol (British Hypertension Society)**:
   * Dihitung berdasarkan persentase kumulatif kesalahan absolut pada batas $\le 5$, $\le 10$, dan $\le 15\text{ mmHg}$ untuk menentukan Grade A, B, C, atau D.

---

## 5. Panduan Menjalankan Perintah (Runbook untuk AI & Engineer)

### A. Menggunakan Skrip Otomasi ([scripts/train_remote_ssh.sh](file:///Users/sir/Documents/Skripsi/ecg-bp-estimation/scripts/train_remote_ssh.sh))
Skrip ini mengelola sesi `tmux`, pengalihan log, dan deteksi GPU secara otomatis:
```bash
# Sintaks: bash scripts/train_remote_ssh.sh [mode] [parts] [epochs] [batch_size]

# 1. Menjalankan di dalam background session tmux (Paling Direkomendasikan di Server SSH):
bash scripts/train_remote_ssh.sh tmux all 50 128

# 2. Menjalankan via background nohup:
bash scripts/train_remote_ssh.sh nohup all 50 128

# 3. Menjalankan langsung di foreground terminal:
bash scripts/train_remote_ssh.sh foreground 1 15 64
```

### B. Perintah Manual Python Langsung
```bash
# Melatih seluruh part dataset Kachuee (1-12) dengan auto-resume aktif di CUDA
python src/train.py \
    --parts all \
    --epochs 50 \
    --batch-size 128 \
    --optimizer sgd \
    --lr 8e-4 \
    --momentum 0.9 \
    --weight-decay 0.1 \
    --device cuda \
    --amp \
    --auto-resume \
    --checkpoint-dir checkpoints

# Melakukan smoke-test cepat (100 sampel, 2 epoch) untuk memverifikasi fungsionalitas pipeline
python src/train.py \
    --parts 1 \
    --max-samples 100 \
    --batch-size 32 \
    --epochs 2 \
    --checkpoint-dir /tmp/test_checkpoints
```

---

## 6. Prosedur Operasional Saat Terjadi SSH Disconnect

Jika koneksi SSH ke server terputus mendadak saat pelatihan sedang berlangsung:

1. **Sambungkan kembali SSH ke server**:
   ```bash
   ssh user@ip-server
   ```
2. **Periksa status GPU & Proses**:
   ```bash
   nvidia-smi                      # Memastikan Python masih memanfaatkan GPU
   ps aux | grep train.py          # Memeriksa PID proses
   ```
3. **Jika menggunakan `tmux`**:
   ```bash
   tmux ls                         # Melihat daftar sesi aktif
   tmux attach -t ecg_bp_train     # Menyambung kembali ke antarmuka terminal pelatihan
   ```
   *Catatan Shortcut tmux*: Tekan `Ctrl + B` lalu lepaskan, kemudian tekan tombol `D` untuk *detach* (keluar tampilan) tanpa mematikan proses training.
4. **Jika menggunakan `nohup` atau ingin memantau log**:
   ```bash
   tail -f checkpoints/train.log
   ```
5. **Jika proses terhenti / server mati**:
   Cukup jalankan kembali perintah awal dengan argumen `--auto-resume`. Skrip akan melanjutkan otomatis dari epoch terakhir yang tersimpan di `checkpoints/last_checkpoint.pt`.
