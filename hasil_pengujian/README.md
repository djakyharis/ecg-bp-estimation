# Data Hasil Pengujian dan Evaluasi (Bab 6)
## Estimasi Tekanan Darah Berbasis Sinyal ECG dengan Kalibrasi 1-Titik

Direktori ini memuat seluruh data pengujian eksperimental sistem estimasi tekanan darah berbasis ECG (*Lead II*) menggunakan perangkat keras **Shimmer3 ECG Unit (ADS1292R)** dan model **Multitask BiLSTM** dengan metode **Kalibrasi 1-Titik Manset**.

Seluruh identitas partisipan telah disamarkan (*anonymized*) sesuai kaidah privasi data penelitian medis menjadi **Subjek 1, Subjek 2, Subjek 3, Subjek 4, dan Subjek 5**.

---

### 1. Struktur Direktori dan Berkas

```text
hasil_pengujian/
├── Subjek_1/ ... Subjek_5/
│   ├── Take_1/
│   │   ├── SubjekX_Take1_ecg.csv       # Sinyal ECG mentah (ecg_raw_mV) & terfilter (ecg_clean_mV) 125 Hz
│   │   ├── SubjekX_Take1_evaluasi.csv  # Hasil prediksi tekanan darah & latensi per jendela 10 detik
│   │   └── SubjekX_Take1_summary.json  # Statistik evaluasi metrik (MAE, RMSE, ME, STD, AAMI, BHS)
│   ├── Take_2/
│   └── Take_3/
├── rekap_evaluasi_tekanan_darah.csv     # Rekap tabular lengkap 15 sesi pengujian (5 subjek × 3 take)
├── tabel_kesesuaian_sinyal_ecg_rm1.csv  # Data kesesuaian amplitudo masukan sinyal ECG (RM 1)
└── README.md                           # Panduan dan ringkasan hasil untuk penulisan laporan
```

Setiap subjek menjalani **3 kali pengambilan data (Take 1, Take 2, Take 3)** secara berturut-turut. Setiap take berlangsung selama **60 detik**, yang dianalisis menjadi **6 jendela bergeser (10 detik per jendela, pergeseran 10 detik)**. 
Total data evaluasi: **5 subjek × 3 take × 6 jendela = 90 jendela inferensi**.

---

### 2. Tabel 6.1 Hasil Pengujian Kesesuaian Sinyal ECG (Menjawab Rumusan Masalah 1)

*Pengujian ini mengevaluasi apakah sinyal biopotensial ECG yang diakuisisi modul Shimmer3 (ADS1292R) terstandarisasi, bebas dari saturasi, dan sesuai dengan rentang fisiologis baku gelombang R Lead II (Noble et al., 0,5 – 2,0 mV).*

| Subjek | Pengukuran | Min Amplitudo Teramati (Lembah S) | Max Amplitudo Teramati (Puncak R) | Referensi Max Amplitudo R-Peak | Kesesuaian Klinis |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Subjek 1** | Take 1 | -1,8 — -1,4 mV | 0,5 — 0,8 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| | Take 2 | -1,7 — -1,4 mV | 0,5 — 0,7 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| | Take 3 | -1,7 — -1,4 mV | 0,5 — 0,7 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| **Subjek 2** | Take 1 | -2,5 — -1,3 mV | 0,5 — 1,6 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| | Take 2 | -1,7 — -1,3 mV | 0,5 — 0,6 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| | Take 3 | -1,7 — -1,3 mV | 0,5 — 0,9 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| **Subjek 3** | Take 1 | -1,6 — -0,5 mV | 0,5 — 1,1 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| | Take 2 | -2,2 — -0,6 mV | 0,5 — 1,2 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| | Take 3 | -1,6 — -0,5 mV | 0,5 — 0,9 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| **Subjek 4** | Take 1 | -1,8 — -1,5 mV | 0,5 — 0,7 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| | Take 2 | -1,8 — -0,7 mV | 0,5 — 0,7 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| | Take 3 | -1,8 — -1,4 mV | 0,5 — 0,7 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| **Subjek 5** | Take 1 | -1,7 — -0,5 mV | 0,5 — 0,8 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| | Take 2 | -2,0 — -0,5 mV | 0,5 — 0,9 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |
| | Take 3 | -1,9 — -0,5 mV | 0,5 — 0,8 mV | 0,5 — 2,0 mV | **Masuk Kriteria** |

---

### 3. Tabel 6.2 Hasil Pengujian Akurasi Estimasi Tekanan Darah (Menjawab Rumusan Masalah 2)

*Standar medis: AAMI mensyaratkan MAE $\le 5\text{ mmHg}$ dan STD $\le 8\text{ mmHg}$. BHS Grade A mensyaratkan kumulatif galat $\le 5\text{ mmHg}$ minimal 60%.*

| Subjek | Pengukuran | Nilai Manset (Ref) | Prediksi SBP / DBP (mmHg) | MAE SBP (mmHg) | MAE DBP (mmHg) | Status AAMI (SBP / DBP) | Grade BHS (SBP / DBP) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Subjek 1** | Take 1 | 112 / 53 | 111,8 / 52,8 | 1,33 | 0,45 | LULUS / LULUS | A / A |
| | Take 2 | 100 / 62 | 99,2 / 61,9 | 0,83 | 0,32 | LULUS / LULUS | A / A |
| | Take 3 | 97 / 52 | 101,8 / 53,4 | 4,82 | 1,40 | LULUS / LULUS | B / A |
| **Subjek 2** | Take 1 | 96 / 70 | 94,9 / 69,5 | 1,73 | 1,00 | LULUS / LULUS | A / A |
| | Take 2 | 95 / 65 | 94,2 / 64,0 | 0,97 | 1,00 | LULUS / LULUS | A / A |
| | Take 3 | 93 / 68 | 95,1 / 70,1 | 2,08 | 2,10 | LULUS / LULUS | A / A |
| **Subjek 3** | Take 1 | 117 / 74 | 117,7 / 74,8 | 3,17 | 1,10 | LULUS / LULUS | B / A |
| | Take 2 | 107 / 83 | 104,2 / 81,1 | 3,42 | 2,23 | LULUS / LULUS | A / A |
| | Take 3 | 98 / 78 | 94,5 / 77,1 | 4,00 | 1,17 | LULUS / LULUS | A / A |
| **Subjek 4** | Take 1 | 99 / 76 | 97,9 / 75,4 | 1,43 | 0,60 | LULUS / LULUS | A / A |
| | Take 2 | 104 / 72 | 104,8 / 72,4 | 1,12 | 0,53 | LULUS / LULUS | A / A |
| | Take 3 | 103 / 75 | 102,4 / 74,9 | 1,52 | 0,65 | LULUS / LULUS | A / A |
| **Subjek 5** | Take 1 | 122 / 81 | 123,2 / 80,5 | 1,83 | 1,08 | LULUS / LULUS | A / A |
| | Take 2 | 118 / 78 | 123,8 / 80,6 | 5,75 | 2,58 | TIDAK / LULUS | B / A |
| | Take 3 | 118 / 78 | 111,9 / 77,4 | 6,12 | 0,85 | TIDAK / LULUS | D / A |
| **Rata-rata** | — | — | — | **2,27 mmHg** | **1,14 mmHg** | **LULUS / LULUS** | **Grade A / A** |

---

### 4. Tabel 6.3 Hasil Pengujian Waktu Komputasi Sistem (Menjawab Rumusan Masalah 3)

*Ambang batas real-time pada proposal: $< 100\text{ ms}$.*

| Subjek | Pengukuran | Latensi Rata-rata | Latensi Maksimum | Laju Tampilan GUI | Kondisi Buffer | Status Target |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Subjek 1** | Take 1–3 | 6,25 ms | 19,34 ms | 30,1 FPS | Zero Congestion | LULUS (< 100 ms) |
| **Subjek 2** | Take 1–3 | 5,27 ms | 18,45 ms | 28,5 FPS | Zero Congestion | LULUS (< 100 ms) |
| **Subjek 3** | Take 1–3 | 5,23 ms | 17,80 ms | 29,2 FPS | Zero Congestion | LULUS (< 100 ms) |
| **Subjek 4** | Take 1–3 | 4,83 ms | 16,50 ms | 30,4 FPS | Zero Congestion | LULUS (< 100 ms) |
| **Subjek 5** | Take 1–3 | 5,00 ms | 17,10 ms | 28,8 FPS | Zero Congestion | LULUS (< 100 ms) |
| **Rata-rata** | — | **5,32 ms** | **19,34 ms** | **29,4 FPS** | **Bebas Antrean** | **LULUS SANGAT CEPAT** |

---

### 5. Tabel 6.4 Efektivitas Kalibrasi 1-Titik vs Tanpa Kalibrasi (Ablation Study)

| Parameter | Tanpa Kalibrasi (Pure AI) | Dengan Kalibrasi 1-Titik (Usulan) | Peningkatan Akurasi |
| :--- | :---: | :---: | :---: |
| **MAE SBP** | 14,35 mmHg | **2,27 mmHg** | **Membaik 84,2%** |
| **MAE DBP** | 8,98 mmHg | **1,14 mmHg** | **Membaik 87,3%** |
| **Tingkat Kelulusan AAMI SBP** | 0% | **86,7% (13 dari 15 sesi)** | Lompatan klinis signifikan |
| **Tingkat Kelulusan AAMI DBP** | 0% | **100,0% (15 dari 15 sesi)** | Sesuai standar medis |
| **Klasifikasi BHS** | Grade D (Gagal) | **Grade A (Klinis Sangat Baik)** | Peningkatan mutu medis |
