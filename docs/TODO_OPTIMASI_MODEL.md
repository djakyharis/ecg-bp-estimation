# TODO & Action Plan: Optimasi Model Estimasi Tekanan Darah ECG
*Daftar Rencana Kerja dan Checklist Tahapan Optimasi (Acuan: Fan et al., 2021)*

---

## Acuan Titik Tolak (Baseline Terbaik Saat Ini)

* **Checkpoint Rujukan:** `checkpoints/run_patch_proj_gpu2/best_model.pt`
* **Arsitektur Inti:** Patching 50x25 (50 langkah waktu) + 2-layer BiLSTM (hidden 128) + Feature Projector MLP (64-dim) + Loss Absolut Murni.
* **Hasil Evaluasi Saat Ini (Part 1 Test Set):**
  * SBP MAE: **15,70 mmHg** (STD: 19,81 mmHg)
  * DBP MAE: **9,06 mmHg** (STD: 11,89 mmHg)
  * MAP MAE: **10,72 mmHg** (STD: 13,87 mmHg)
  * SBP Error $\le 5\text{ mmHg}$: **21,13%**

---

## Checklist Tahapan Optimasi

### Tahap 1: Pelatihan Skala Penuh 12 Part Data (Skalabilitas Data)
Target: Meningkatkan kapasitas generalisasi representasi BiLSTM + Projector dengan memanfaatkan seluruh data yang tersedia (261.117 sampel).

- [ ] **1.1 Verifikasi Data:** Cek ketersediaan dan integritas file `part_1_preprocessed.npz` hingga `part_12_preprocessed.npz` di folder `data/processed/`.
- [ ] **1.2 Setup Parameter Training:**
  - Arsitektur: `--patch-size 25 --use-projection --proj-dim 64`
  - Loss function: Loss adaptif absolut murni (`--loss-normalize false` atau tanpa pembagi skala)
  - Optimizer & Scheduler: SGD momentum 0.9, lr 0.0008, `--scheduler-type plateau --scheduler-patience 10 --min-lr 1e-5`
- [ ] **1.3 Eksekusi Training 12 Part:** Jalankan training di GPU yang tersedia dan pantau konvergensi loss serta log evaluasi.
- [ ] **1.4 Evaluasi Test Set 12 Part:** Catat metrik evaluasi test set (MAE SBP, DBP, MAP, STD, dan persentase error $\le 5$ mmHg).

---

### Tahap 2: Pembersihan Noise Target dengan Filter Skewness ABP (Kualitas Data)
Target: Menyingkirkan data ICU yang terdistorsi oleh redaman kateter arteri (*catheter damping*) untuk menembus batas bawah DBP (~9 mmHg) menuju target Fan et al. (~4–5 mmHg).

- [ ] **2.1 Buat Script Filter Skewness:** Kembangkan script pra-pemrosesan (misal `scripts/filter_abp_skewness.py`) yang membagi sinyal ABP ke jendela 1 detik dan menghitung kemiringan (*skewness*).
- [ ] **2.2 Eliminasi Segmen Buruk:** Buang segmen 10 detik apabila terdapat minimal satu jendela 1 detik dengan nilai $\text{Skewness} < 0$.
- [ ] **2.3 Simpan Dataset Bersih:** Simpan partisi data terfilter ke direktori `data/cleaned/` dan catat jumlah data yang tersisa.
- [ ] **2.4 Re-Training Model Terbaik pada Cleaned Data:** Latih ulang arsitektur `run_patch_proj` pada dataset yang sudah disaring skewness-nya.
- [ ] **2.5 Evaluasi Ulang DBP & SBP:** Verifikasi apakah batas bawah DBP berhasil turun signifikan di bawah 9 mmHg.

---

### Tahap 3: Pengujian Modul Temporal Attention dengan Loss Absolut (Penyempurnaan Model)
Target: Menguji efektivitas pembobotan patch dinamis (fokus pada interval QRS dan gelombang T) tanpa terhambat oleh *gradient underflow*.

- [ ] **3.1 Verifikasi Modul Attention:** Pastikan modul `TemporalAttention` pada `src/models/bilstm.py` sudah terhubung dengan tepat setelah layer BiLSTM.
- [ ] **3.2 Jalankan Training Attention Tanpa Skala Loss:**
  - Parameter: `--patch-size 25 --use-projection --proj-dim 64 --use-attention` dengan loss RMSE murni.
- [ ] **3.3 Analisis Bobot Attention:** Tinjau distribusi bobot perhatian model pada 50 patch waktu sinyal ECG.
- [ ] **3.4 Bandingkan Hasil:** Bandingkan performa MAE antara model dengan attention vs model tanpa attention.

---

### Tahap 4: Implementasi Kalibrasi 1-Titik Subjek (Standar Klinis AAMI SP10)
Target: Memenuhi ambang batas medis $\text{MAE} \le 5\text{ mmHg}$ dan $\text{STD} \le 8\text{ mmHg}$ melalui kalibrasi baseline personal saat inferensi.

- [ ] **4.1 Kembangkan Modul Kalibrasi:** Tambahkan logika kalibrasi 1-titik pada script evaluasi/inferensi:
  $$\hat{y}_{\text{terkalibrasi}} = \hat{y} + (y_{\text{baseline}} - \hat{y}_{\text{baseline}})$$
- [ ] **4.2 Simulasi Kalibrasi pada Test Set:** Uji skenario kalibrasi di mana 1 segmen pertama tiap subjek digunakan sebagai baseline acuan untuk segmen-segmen berikutnya.
- [ ] **4.3 Hitung Metrik Pasca-Kalibrasi:** Hitung penurunan MAE, STD, dan persentase kepatuhan terhadap standar AAMI SP10 serta BHS Grade A/B.

---

### Tahap 5: Finalisasi Deployment & Dokumentasi Skripsi
Target: Memastikan model siap digunakan untuk streaming data nyata dan hasil terdokumentasi lengkap.

- [ ] **5.1 Uji Inferensi Sensor Shimmer3:** Pastikan pipeline inferensi pada `src/models/inference.py` menerima sinyal raw 1-lead 1250 titik (10 detik @ 125 Hz) secara mulus.
- [ ] **5.2 Update Tabel Perbandingan Final:** Masukkan seluruh hasil eksperimen ke dalam naskah laporan / skripsi.
- [ ] **5.3 Simpan Checkpoint Model Final:** Simpan file bobot dan konfigurasi model final ke direktori rilis yang aman.
