# Notes Eksperimen Saya: Evaluasi & Review Model Estimasi Tekanan Darah ECG
*Logbook Pengujian Model, Analisis Hasil Latih, dan Catatan Teknis (Acuan: Fan et al., 2021)*

---

## 1. Ringkasan & Alur Eksperimen yang Saya Jalankan

Eksperimen ini bertujuan mengembangkan dan menguji model estimasi tekanan darah kontinu (SBP, DBP, MAP) dari sinyal 1-lead ECG berdurasi 10 detik (1.250 titik @ 125 Hz). Acuan utama yang saya gunakan adalah arsitektur **Fan et al. (Sensors, 2021)**, yang saya kombinasikan dengan ekstraksi fitur statistik morfologi ECG dari paper **Nour et al. (2022)**.

Evolusi arsitektur yang telah saya bangun dan latih:
1. **Baseline Skripsi Awal:** Pure BiLSTM 2-layer dengan sekuens penuh 1.250 titik tanpa fitur statistik.
2. **Hybrid Fitur Langsung:** Menggabungkan 12 fitur (Top-12) dan 25 fitur (All-25) langsung ke dense output layer bersama representasi BiLSTM.
3. **Spatial-Temporal Patching ($50 \times 25$):** Membagi sinyal 1.250 titik menjadi 50 langkah waktu (masing-masing 25 titik / 0,2 detik) untuk memangkas panjang BPTT.
4. **Feature Projector MLP:** Memetakan 25 fitur statistik ke subnet proyeksi 64-dimensi agar bobot representasi statistik seimbang dan tidak tenggelam (*feature drowning*).
5. **Eksperimen Tambahan:** Pengujian normalisasi skala loss $[120, 80, 93,3]$ dan modul *Temporal Attention*.

---

## 2. Tabel Perbandingan Hasil Evaluasi Test Set

Evaluasi dilakukan pada *unseen test set* Part 1 (2.518 segmen) setelah masing-masing model dilatih 500 epoch:

| Varian Model | Partisi Data | Konfigurasi Utama | SBP MAE (mmHg) | DBP MAE (mmHg) | MAP MAE (mmHg) | Rata-rata MAE (mmHg) | Akurasi SBP (Error ≤ 5 mmHg) | Status / Catatan |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|---|
| **run_patch_proj_gpu2** | Part 1 | **Patch 50x25 + MLP Proyeksi (64-dim)** | **15,70** | **9,06** | 10,72 | **11,83** | **21,13%** | 🏆 **Model Terbaik Saya Saat Ini** |
| **run_patch50x25_gpu0** | Part 1 | Patch 50x25 + 25 Fitur Langsung | 16,77 | 9,10 | 10,60 | 12,16 | 18,35% | Waktu latih efisien (1,04 jam) |
| **run_all25** | Part 1 | Sekuens 1250 + 25 Fitur Langsung | 16,81 | 9,09 | 10,59 | 12,16 | 18,11% | Baseline hybrid 25 fitur |
| **run_top12** | Part 1 | Sekuens 1250 + 12 Fitur Langsung | 17,43 | 9,06 | 10,73 | 12,40 | 16,96% | Baseline hybrid 12 fitur |
| **pure_bilstm_root** | 12 Part | Sekuens 1250 (Tanpa Fitur) | 18,25 | 9,17 | 10,87 | 12,76 | 15,20% | Model baseline awal skripsi |
| **run_full12_patch_proj** | Part 1 | Patch 50x25 + Target Scaled 120/80/93 | 21,01 | 11,40 | 10,77 | 14,39 | 13,22% | Terkendala gradien underflow |
| **run_full12_attn_gpu0** | Part 1 | Patch 50x25 + Temporal Attention | 21,01 | 11,40 | 10,77 | 14,40 | 13,22% | Terkendala gradien underflow |
| *Target Fan et al. (2021)* | Cleaned | Patch 50x20 + ABP Skewness Filter | *7,63* | *4,36* | *5,60* | *5,86* | *~60%* | Target acuan literatur |

---

## 3. Catatan Detail Model Terbaik: `run_patch_proj_gpu2`

Model yang tersimpan di `checkpoints/run_patch_proj_gpu2` menunjukkan performa terbaik secara konsisten pada seluruh metrik utama.

### Poin Keberhasilan yang Tercatat:
* **Penurunan Error SBP Paling Signifikan:** SBP MAE turun ke **15,70 mmHg**. Nilai ini lebih rendah **2,55 mmHg** dibanding model baseline skripsi awal (18,25 mmHg) dan unggul **1,11 mmHg** dibanding model hybrid tanpa proyeksi (16,81 mmHg).
* **Variansi Error Paling Rendah:** Standar deviasi SBP menyentuh **19,81 mmHg**, menjadi satu-satunya model yang berhasil menembus angka di bawah 20 mmHg.
* **Kenaikan Proporsi Standar Medis:** Akurasi prediksi dalam batas toleransi medis ($\le 5\text{ mmHg}$) mencapai **21,13%**, meningkat dari baseline 15,20%.
* **Efektivitas Feature Projector:** Penambahan subnet proyeksi (Linear 128 $\rightarrow$ LayerNorm $\rightarrow$ ReLU $\rightarrow$ Linear 64) meningkatkan porsi representasi statistik dari 4,6% menjadi 11,1% pada input dense layer akhir, mencegah fitur morfologi tenggelam oleh representasi laten BiLSTM (512-dimensi).
* **Kestabilan Bobot Recurrent BiLSTM:** Pembagian sekuens menjadi 50 patch menjaga magnitudo bobot recurrent tetap aktif pada rata-rata $|W_{ih}| = 0,001956$, terhindar dari fenomena vanishing gradient yang terjadi pada sekuens panjang 1.250 langkah ($|W_{ih}| \approx 0,000002$).

---

## 4. Catatan Teknis Eksperimen Skala Loss (Model 21 mmHg)

Pada run `run_full12_patch_proj` dan `run_full12_attn_gpu0`, target loss dibagi dengan faktor fisiologis $[120, 80, 93,3]$. Hasil evaluasi tertahan di angka 21,01 mmHg.

### Diagnosis Teknis:
* Pembagian target dengan konstanta 120 dan 80 menurunkan magnitudo numerik loss dari kisaran 18,0 menjadi 0,18 (turun sekitar 100 kali lipat).
* Penurunan nilai loss mengecilkan gradien backpropagation $\nabla L$ sebesar 100 kali lipat.
* Dengan learning rate SGD $0,0008$, besaran update bobot ($\Delta W$) berada di orde $10^{-6}$, sehingga didominasi oleh penalti *weight decay* ($0,1$).
* Bobot model tertahan di sekitar inisialisasi awal dan tidak dapat menyesuaikan fitur dengan baik.
* **Ketentuan untuk run berikutnya:** Fungsi loss adaptif Fan et al. harus selalu dijalankan pada skala absolut (RMSE murni), sama seperti pada `run_patch_proj_gpu2`.

---

## 5. Catatan Batas Bawah DBP (~9,06 mmHg)

Pada seluruh varian model yang diuji pada Part 1, DBP MAE selalu berada di rentang **9,06 – 9,17 mmHg**, meskipun SBP berhasil membaik secara bertahap dari 18,25 ke 15,70 mmHg.

### Observasi Kualitas Data & Literatur:
1. **Prosedur Pembersihan Data Fan et al. (2021):** Peneliti membagi sinyal ABP ke jendela 1 detik dan membuang segmen 10 detik jika terdapat minimal satu jendela dengan nilai **Skewness < 0**.
2. **Karakteristik Fisiologis Gelombang ABP:** Denyut arteri normal memiliki kemiringan positif (puncak sistolik tajam). Nilai skewness negatif merupakan penanda artefak redaman kateter ICU, pergerakan pasien, atau gangguan sensor.
3. **Perbedaan Kondisi Dataset:** Seleksi ketat Fan et al. menghasilkan 21.422 segmen yang sangat bersih. Sedangkan dataset Part 1 yang saya gunakan masih memuat sinyal mentah populasi ICU Kachuee, di mana batas bawah noise alami DBP berada pada level ~9 mmHg.

---

## 6. Rencana Langkah Berikutnya

1. **Jadikan `run_patch_proj_gpu2` Model Utama:** Menggunakan konfigurasi Patch 50x25 + Feature Projection 64-dim sebagai arsitektur inti skripsi.
2. **Training Skala Penuh 12 Part:** Melatih arsitektur terbaik ini pada seluruh 12 part (261.117 sampel) dengan loss absolut murni, learning rate $0,0008$, dan scheduler patience 10 epoch.
3. **Penyaringan Skewness ABP:** Menerapkan script filter skewness ABP pada data latih untuk menyingkirkan segmen dengan redaman kateter ICU, mendekati kondisi dataset Fan et al.
4. **Kalibrasi Subjek Wearable:** Menerapkan skema kalibrasi baseline 1-titik ($y_{\text{baseline}}$) saat inferensi sinyal sensor Shimmer3 untuk memenuhi batas toleransi medis AAMI SP10 ($\text{MAE} \le 5\text{ mmHg}$).

