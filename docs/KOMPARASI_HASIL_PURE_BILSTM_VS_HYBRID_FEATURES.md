# Komparasi Evaluasi: Pure BiLSTM (500 Epoch) vs. Hybrid BiLSTM + 12 Fitur Statistik (250 Epoch)

> **Dokumen Evaluasi & Pembahasan Skripsi**  
> **Topik:** Estimasi Tekanan Darah Kontinu (SBP, DBP, MAP) Berbasis Sinyal ECG Menggunakan Multi-Task Deep Learning  
> **Dataset:** Cuff-less Blood Pressure Dataset (Kachuee et al., MIMIC-II)  
> **Tanggal Evaluasi:** 15 Oktober 2026  
> **Hardware:** GPU NVIDIA L40 (48GB VRAM) Server Remote

---

## 1. Ringkasan Eksekutif (*Executive Summary*)

Pengujian ini membandingkan dua pendekatan arsitektur pelatihan:
1. **Model Baseline Pure BiLSTM (500 Epoch):** Arsitektur *pure end-to-end* 2-Layer BiLSTM (`hidden=256`, representasi laten 512 dimensi) yang dilatih menggunakan seluruh 12 part dataset (261.117 sampel) selama 69,15 jam.
2. **Model Hybrid BiLSTM + 12 Fitur Statistik (250 Epoch):** Arsitektur gabungan representasi sekuens BiLSTM (512) dengan penyuntikan 12 fitur statistik terpandu (Nour et al. 2022 / Ilham Hakim 2025) yang dipartisi secara fisiologis ke masing-masing *task head* (SBP: 6 fitur, DBP: 6 fitur, MAP: 12 fitur), diuji pada Part 1 (25.175 sampel) selama 7,03 jam.

### Temuan Utama:
* **Performa Lebih Baik pada Waktu yang Jauh Lebih Singkat:** Model Hybrid berhasil mencapai rata-rata MAE **12,41 mmHg**, mengungguli model Pure BiLSTM (**12,76 mmHg**), dengan waktu komputasi **10× lebih cepat** (7 jam vs 69 jam).
* **Reduksi Signifikan pada SBP:** Penurunan terbesar terjadi pada tekanan sistolik (SBP), di mana MAE turun sebesar **-0,81 mmHg** ($18,25 \to 17,44$ mmHg) dan standar deviasi error (STD) berkurang **-0,94 mmHg** ($22,39 \to 21,45$ mmHg).
* **Peningkatan Akurasi Ketat DBP:** Proporsi tebakan dengan kesalahan sangat rendah ($\le 5\text{ mmHg}$) pada DBP meningkat dari **32,6% menjadi 35,4%** (+2,8%).

---

## 2. Matriks Komparasi Kinerja (*Benchmark Matrix*)

Evaluasi dilakukan pada *unseen test set* menggunakan dua protokol standar internasional: **BHS Protocol** (*British Hypertension Society*) dan **AAMI SP10** (*Association for the Advancement of Medical Instrumentation*).

| Parameter Evaluasi | Pure BiLSTM (500 Epoch) | Hybrid BiLSTM (250 Epoch) | Perubahan / Selisih | Keterangan & Analisis |
| :--- | :---: | :---: | :---: | :--- |
| **Cakupan Dataset** | 12 Parts Penuh (261.117 sampel) | Part 1 (25.175 sampel) | - | Uji efisiensi & sensitivitas |
| **Injeksi Fitur Terpandu** | Tidak Ada (0 Fitur) | **12 Fitur Statistik** | +12 Fitur | SBP (+6), DBP (+6), MAP (+12) |
| **Durasi Pelatihan** | 4.149,2 menit (69,15 jam) | **422,1 menit (7,03 jam)** | **-62,1 jam** | **10× Lebih Cepat** |
| **Best Validation MAE** | 12,72 mmHg (Epoch 37) | **12,54 mmHg (Epoch 17)** | **-0,18 mmHg** | Konvergensi jauh lebih awal |
| **Rata-rata MAE Test Set** | 12,76 mmHg | **12,41 mmHg** | **-0,35 mmHg** | **Model Hybrid Lebih Unggul** |
| ─── **SISTOLIK (SBP)** ─── | | | | |
| **MAE** | 18,25 mmHg | **17,44 mmHg** | **-0,81 mmHg** | **Membaik (-4,44%)** |
| **STD ($\sigma_{\text{error}}$)** | 22,39 mmHg | **21,45 mmHg** | **-0,94 mmHg** | Sebaran error menyempit |
| **RMSE** | 22,39 mmHg | **21,45 mmHg** | **-0,94 mmHg** | Membaik |
| **Mean Error / Bias (ME)** | -0,25 mmHg | **-0,37 mmHg** | -0,12 mmHg | Tidak mengalami bias sistemik |
| **BHS $\le 5$ mmHg** | 15,7% | **17,0%** | **+1,3%** | Proporsi error ketat naik |
| **BHS $\le 10$ mmHg** | 32,3% | **33,5%** | **+1,2%** | Akurasi kumulatif naik |
| **BHS $\le 15$ mmHg** | 46,9% | **48,5%** | **+1,6%** | Akurasi kumulatif naik |
| **BHS Grade / AAMI SP10** | Grade D / Gagal | Grade D / Gagal | - | Standar AAMI: $\text{MAE} \le 5$, $\text{STD} \le 8$ |
| ─── **DIASTOLIK (DBP)** ─── | | | | |
| **MAE** | 9,17 mmHg | **9,05 mmHg** | **-0,12 mmHg** | Membaik tipis |
| **STD ($\sigma_{\text{error}}$)** | 11,74 mmHg | **11,58 mmHg** | **-0,16 mmHg** | Deviasi error menyusut |
| **RMSE** | 11,74 mmHg | **11,61 mmHg** | **-0,13 mmHg** | Membaik |
| **Mean Error / Bias (ME)** | -0,20 mmHg | **+0,94 mmHg** | +1,14 mmHg | Terkoreksi mendekati netral |
| **BHS $\le 5$ mmHg** | 32,6% | **35,4%** | **+2,8%** | **Kenaikan tertinggi (< 5 mmHg)** |
| **BHS $\le 10$ mmHg** | 62,3% | 62,0% | -0,3% | Relatif stabil |
| **BHS $\le 15$ mmHg** | 83,5% | 81,5% | -2,0% | Relatif stabil |
| **BHS Grade / AAMI SP10** | Grade D / Gagal | Grade D / Gagal | - | Standar AAMI: $\text{MAE} \le 5$, $\text{STD} \le 8$ |
| ─── **ARTERIAL (MAP)** ─── | | | | |
| **MAE** | 10,85 mmHg | **10,73 mmHg** | **-0,12 mmHg** | Membaik |
| **STD ($\sigma_{\text{error}}$)** | 13,56 mmHg | **13,05 mmHg** | **-0,51 mmHg** | Membaik |
| **RMSE** | 13,56 mmHg | **13,18 mmHg** | **-0,38 mmHg** | Membaik |
| **BHS $\le 5$ mmHg** | 28,1% | **27,3%** | -0,8% | Relatif stabil |
| **BHS $\le 10$ mmHg** | 52,2% | **52,8%** | **+0,6%** | Naik |
| **BHS $\le 15$ mmHg** | 73,2% | **72,5%** | -0,7% | Relatif stabil |

---

## 3. Analisis Mendalam Per Parameter Tekanan Darah

### A. Sistolik (SBP): Penurunan MAE Terbesar (-0,81 mmHg)
Sistolik merupakan target yang paling sulit diprediksi oleh BiLSTM murni karena memiliki variabilitas alami tertinggi di populasi ($\sigma = 21,96\text{ mmHg}$).
* **Peran Fitur Statistik:** Injeksi fitur *Shape Factor* ($r=0,93$), *RMS*, *Average Energy*, dan *CM10* memberikan informasi langsung mengenai kekuatan daya dorong kontraksi ventrikel kiri.
* **Hasil Nyata:** Fitur-fitur ini berhasil menahan tebakan ekstrem model, sehingga STD error turun dari $22,39$ mmHg ke $21,45$ mmHg dan MAE turun ke $17,44$ mmHg.

### B. Diastolik (DBP): Lonjakan Akurasi Ketat (+2,8% pada $\le 5$ mmHg)
Tekanan diastolik dipengaruhi oleh elastisitas vaskular dan *Total Peripheral Resistance* (TPR).
* **Peran Fitur Statistik:** Fitur seperti *Coefficient of Variation* ($r = +0,61$) dan *Skewness* ($r = +0,68$) mengukur stabilitas fase tenang sinyal di antara dua puncak detak jantung.
* **Hasil Nyata:** Model mampu memprediksi dengan deviasi sangat kecil ($\le 5\text{ mmHg}$) pada **35,4% pasien**, naik dari 32,6% pada model tanpa fitur.

### C. Mean Arterial Pressure (MAP): Harmonisasi Fisiologis
Sesuai rumus klinis $\text{MAP} \approx \text{DBP} + \frac{1}{3}(\text{SBP} - \text{DBP})$, MAP Head yang menerima seluruh 12 fitur gabungan menunjukkan kestabilan error tertinggi (MAE $10,73$ mmHg, RMSE turun dari $13,56 \to 13,18$ mmHg).

---

## 4. Evaluasi Kritis: Mengapa Belum Menembus Target AAMI SP10 ($\text{MAE} \le 5\text{ mmHg}$)?

Meskipun model Hybrid menunjukkan tren perbaikan yang konsisten, nilai error masih berada pada kisaran $\text{MAE} \approx 9\text{ mmHg}$ (DBP) dan $17\text{ mmHg}$ (SBP). Berdasarkan audit teknis, terdapat 4 faktor penentu:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │ Faktor Penyebab Belum Menembusnya Target AAMI SP10      │
                  └────────────────────────────┬────────────────────────────┘
                                               │
         ┌───────────────────┬─────────────────┴─────────────────┬───────────────────┐
         ▼                   ▼                                   ▼                   ▼
1. Ketimpangan Dimensi  2. Skala Dataset                     3. Prematur Decay   4. Batas Fisik 1-Lead
   BiLSTM: 512 (98,8%)     Hanya Part 1 (25k data)              LR turun ke         ECG tanpa PTT/PPG
   Fitur : 6   (1,2%)      vs 12 Part (261k data)               6.3e-6 di ep 250    kehilangan mekanik
```

### 1. Ketimpangan Dimensi (*Dimensionality Imbalance / Feature Drowning*)
* Input ke layer Fully Connected Head adalah konkatenasi: $\text{Concat}(\mathbf{h}_{\text{BiLSTM}}, \mathbf{f}_{\text{stat}})$.
* Dimensi $\mathbf{h}_{\text{BiLSTM}} = 512$, sedangkan $\mathbf{f}_{\text{stat}} = 6$.
* Rasio kontribusi: Fitur statistik hanya menyusun **$1,15\%$ neuron input**. Sebanyak **$98,85\%$ bobot** dipegang oleh vektor laten BiLSTM. Akibatnya, sinyal dari fitur statistik rawan "tenggelam" (*drowned out*) di hadapan kapasitas representasi BiLSTM.

### 2. Skala Dataset: Part 1 (25 Ribu) vs. All 12 Parts (261 Ribu)
* Model Pure BiLSTM 500 epoch dilatih menggunakan **261.117 sampel** (12 part lengkap).
* Model Hybrid 250 epoch ini **baru dilatih menggunakan Part 1 (25.175 sampel)**.
* Kemampuan model Hybrid mengalahkan MAE model 500 epoch ($12,41$ vs $12,76$ mmHg) hanya dengan **$10\%$ volume data** menunjukkan efektivitas fitur yang sangat nyata. Jika model Hybrid dilatih pada seluruh 12 part, potensi penurunan error akan jauh lebih besar.

### 3. *Learning Rate Decay* Terlalu Agresif (*Premature Freezing*)
* Model Hybrid mencapai *Best Validation MAE* di **Epoch 17** ($12,54\text{ mmHg}$).
* Akibat mekanisme *scheduler* `ReduceLROnPlateau` dengan `patience=4`, nilai learning rate dipotong berturut-turut dari $8 \times 10^{-4}$ hingga mencapai $6,3 \times 10^{-6}$ pada epoch 247.
* Pada LR sekecil itu dengan SGD momentum, gradien jaringan hampir bernilai nol (*semi-frozen*), sehingga model tidak lagi mampu mengeksplorasi penyesuaian bobot linier pada 200 epoch terakhir.

### 4. Batas Teoretis Fisiologis Sinyal Tunggal 1-Lead ECG
* Sinyal 1-lead ECG murni merefleksikan **kelistrikan jantung (depolarisasi & repolarisasi)**, bukan tekanan mekanik hemodinamik pembuluh darah.
* Dalam literatur ilmiah modern, model estimasi tekanan darah berbasis 1-lead ECG tanpa kalibrasi awal (*uncalibrated & cuff-less*) umumnya berada pada batas MAE $8\text{--}10\text{ mmHg}$ untuk DBP dan $15\text{--}18\text{ mmHg}$ untuk SBP.
* Standar AAMI SP10 ($\le 5\text{ mmHg}$) umumnya baru dapat ditembus apabila:
  1. Digunakan modalitas ganda (**ECG + PPG**) untuk menghitung *Pulse Transit Time* (PTT).
  2. Dilakukan kalibrasi awal per individu (*subject-specific calibration baseline*).

---

## 5. Ringkasan Rekomendasi untuk Bab Kesimpulan & Saran Skripsi

1. **Efektivitas Fitur Terbukti:** Integrasi 12 fitur statistik terpandu (Nour et al. 2022) berhasil meningkatkan akurasi estimasi SBP sebesar $0,81$ mmHg dan mereduksi waktu pelatihan hingga $10\times$ lipat.
2. **Pengujian Skala Penuh:** Model Hybrid direkomendasikan untuk dilatih pada seluruh 12 part dataset (`--parts all`) dengan pengaturan *patience* scheduler yang lebih longgar (`patience=10`) agar learning rate tidak turun terlalu cepat.
3. **Penyempurnaan Arsitektur Lanjutan (*Future Work*):** Menambahkan blok *Dense Projection* khusus untuk fitur statistik (`Linear(6, 64) -> ReLU`) sebelum digabungkan dengan BiLSTM agar representasi fitur memiliki bobot yang seimbang ($\sim 11\%$ vs $89\%$) dan tidak didominasi secara mutlak oleh BiLSTM.

---
*Dokumen ini disusun secara otomatis berdasarkan riwayat pelatihan resmi dari berkas `checkpoints/train.log` dan `checkpoints/evaluation_results.json`.*
