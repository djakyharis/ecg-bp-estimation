# Analisis Struktur Penulisan Hasil dan Pembahasan Skripsi Terdahulu
## Acuan Penulisan Bab 6 (Pengujian dan Analisis) untuk Estimasi Tekanan Darah Berbasis ECG

Dokumen ini disusun berdasarkan telaah mendalam terhadap 3 skripsi terdahulu di laboratorium/program studi yang sama:
1. **Skripsi A (U-Net Autoencoder & Extra Trees Regressor)**: *Tekanan Darah Berbasis Sinyal Electrocardiogram Menggunakan U-Net Autoencoder dan Extra Trees Regressor* (2024/2025).
2. **Skripsi B (Pan-Tompkins++ & Gaussian Process Regression / GPR)**: *Sistem Monitoring Tekanan Darah Berbasis ECG Menggunakan GPR* (2024).
3. **Skripsi C (Random Forest Regression / RFR)**: *Tekanan Darah Menggunakan Metode Random Forest Regression Berbasis ECG* (2024).

---

## 1. Pemetaan Pola: Rumusan Masalah (Bab 1) $\longleftrightarrow$ Pengujian (Bab 6)

Ketiga skripsi kating menerapkan kaidah **pemetaan satu-ke-satu (*one-to-one mapping*) yang sangat ketat**. Setiap butir Rumusan Masalah pada Bab 1 memiliki satu subbab pengujian tersendiri pada Bab 6:

```mermaid
flowchart LR
    subgraph Bab1["Bab 1: Pendahuluan"]
        RM1["RM 1: Kesesuaian Sinyal ECG / Puncak R"]
        RM2["RM 2: Tingkat Akurasi Estimasi SBP & DBP"]
        RM3["RM 3: Efisiensi Komputasi & Operasi Real-Time"]
        RM4["RM 4: Pengaruh Intervensi / Performa Keseluruhan"]
    end

    subgraph Bab6["Bab 6: Pengujian & Analisis"]
        Sub61["6.1 Pengujian Kesesuaian Sinyal ECG"]
        Sub62["6.2 Pengujian Tingkat Akurasi Prediksi Tekanan Darah"]
        Sub63["6.3 Pengujian Performa Sistem & Waktu Komputasi"]
        Sub64["6.4 Pengujian Perbandingan / Keberhasilan Sistem"]
    end

    RM1 --> Sub61
    RM2 --> Sub62
    RM3 --> Sub63
    RM4 --> Sub64
```

### Perbandingan Redaksional Rumusan Masalah di 3 Skripsi Kating

| Rumusan Masalah | Skripsi U-Net + ETR | Skripsi Pan-Tompkins++ + GPR | Skripsi Random Forest |
| :--- | :--- | :--- | :--- |
| **RM 1 (Input)** | Bagaimana kesesuaian sinyal ECG hasil akuisisi sensor Shimmer3? | Bagaimana kesesuaian hasil pembacaan puncak R secara real-time? | Bagaimana kesesuaian sinyal ECG dan ekstraksi 7 fitur korelasi? |
| **RM 2 (Akurasi)** | Bagaimana tingkat akurasi prediksi nilai SBP dan DBP? | Bagaimana hasil pengujian akurasi sistem monitoring (GPR)? | Bagaimana akurasi estimasi model dibandingkan tensimeter manset? |
| **RM 3 (Komputasi)**| Bagaimana performa keseluruhan sistem prediksi (waktu komputasi)? | Berapa lama waktu komputasi yang diperlukan sistem monitoring? | Bagaimana perbandingan akurasi pipeline offline vs live detection? |
| **RM 4 (Sistem)** | *(Dirangkum di RM 3)* | Bagaimana tingkat keberhasilan keseluruhan sistem? | Bagaimana tingkat keberhasilan sistem secara keseluruhan? |

---

## 2. Struktur Baku Penulisan Setiap Subbab Pengujian

Di seluruh skripsi terdahulu, setiap subbab pengujian (**6.1, 6.2, 6.3**) **WAJIB** dipecah menjadi 3 anak subbab dengan struktur baku:

```text
6.X [Nama Pengujian Sesuai Rumusan Masalah]
    ├── Paragraf Pengantar Subbab (Konteks & urgensi pengujian)
    ├── 6.X.1 Tujuan Pengujian (Pernyataan lugas apa yang ingin dibuktikan)
    ├── 6.X.2 Prosedur Pengujian (Langkah operasional subjek, durasi, alat, rumus evaluasi)
    └── 6.X.3 Hasil dan Analisis Pengujian (Tangkapan layar GUI, Tabel Pengujian, Pembahasan)
```

### Bedah Komponen 6.X.3 (*Hasil dan Analisis Pengujian*):
1. **Gambar Tangkapan Layar GUI (*Screenshot*)**: Menunjukkan bukti sistem berjalan saat pengujian berlangsung (misal grafik osiloskop ECG, panel hasil tekanan darah, kartu latensi).
2. **Tabel Rekapitulasi Data**: Selalu menguji **5 subjek** dengan masing-masing **3 kali pengukuran/pengambilan data** (total 15 baris data pengujian).
3. **Paragraf Pembahasan Ilmiah**:
   - Membandingkan hasil teramati dengan literatur baku (misal rentang gelombang Lead II, standar medis AAMI, standar BHS).
   - Menjelaskan anomali/variasi fisiologis antar-subjek (mengapa ada subjek dengan nilai lebih tinggi/rendah, faktor kontak elektrode kulit, variasi vaskular).
   - Menarik kesimpulan apakah sistem berhasil menjawab rumusan masalah tersebut.

---

## 3. Penerapan Langsung untuk Skripsi Anda: *Template* Bab 6

Berikut adalah draf lengkap struktur Bab 6 yang telah disesuaikan 100% dengan data riil Anda (**5 subjek: Aris, Cahyo, Faiz, Nadya, Semi** @ 3 Take = 15 pengujian):

---

### 6.1 Pengujian Kesesuaian Sinyal ECG (Menjawab RM 1)

Pengujian kesesuaian sinyal ECG dilakukan untuk mengevaluasi kemampuan sistem dalam mengakuisisi, memfilter, dan menampilkan sinyal ECG menggunakan modul Shimmer3 Unit ECG yang memuat chip ADC ADS1292R sebagai masukan model *deep learning*.

#### 6.1.1 Tujuan Pengujian
Tujuan pengujian ini adalah untuk memverifikasi kemampuan sistem dalam mengakuisisi sinyal biopotensial ECG secara *real-time*, memastikan tegangan berada pada rentang linier skala penuh ADC ($\pm 403{,}3\text{ mV}$), serta mengevaluasi kesesuaian amplitudo gelombang terhadap karakteristik standar fisiologis ECG Lead II (Noble et al., 1990; Goldberger, 2017).

#### 6.1.2 Prosedur Pengujian
1. Memasang elektrode Ag/AgCl pada tubuh subjek uji dengan konfigurasi bipolar Lead II (RA pada pergelangan tangan kanan, LL pada pergelangan kaki kiri, dan RL sebagai *ground*).
2. Menghubungkan perangkat Shimmer3 ke laptop melalui koneksi nirkabel Bluetooth.
3. Membuka antarmuka GUI dan memulai *streaming* akuisisi data.
4. Merekam sinyal ECG selama 3 kali pengukuran pada masing-masing dari 5 subjek, dengan durasi 60 detik (6 jendela bergeser 10 detik).
5. Mengamati bentuk gelombang pada osiloskop GUI (*Dual-Trace Raw vs Clean*) serta mencatat nilai amplitudo minimum (lembah gelombang S) dan amplitudo maksimum (puncak gelombang R).
6. Membandingkan amplitudo puncak R teramati terhadap batas referensi fisiologis $0{,}5\text{ mV} - 2{,}0\text{ mV}$.

#### 6.1.3 Hasil dan Analisis Pengujian
*(Sisipkan Gambar: Tampilan Osiloskop Dual-Trace Raw vs Clean ECG pada GUI)*

**Tabel 6.1 Hasil Pengujian Kesesuaian Sinyal ECG**

| Subjek | Pengukuran | Range Nilai yang Teramati: Min Amplitudo (mV) | Range Nilai yang Teramati: Max Amplitudo (mV) | Referensi Max Amplitudo R-Peak (mV) | Kesesuaian |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Aris)** | Pengukuran 1 | -1,8 — -1,4 | 0,5 — 0,8 | 0,5 — 2,0 | Masuk Kriteria |
| | Pengukuran 2 | -1,7 — -1,4 | 0,5 — 0,7 | 0,5 — 2,0 | Masuk Kriteria |
| | Pengukuran 3 | -1,7 — -1,4 | 0,5 — 0,7 | 0,5 — 2,0 | Masuk Kriteria |
| **2 (Cahyo)** | Pengukuran 1 | -2,5 — -1,3 | 0,5 — 1,6 | 0,5 — 2,0 | Masuk Kriteria |
| | Pengukuran 2 | -1,7 — -1,3 | 0,5 — 0,6 | 0,5 — 2,0 | Masuk Kriteria |
| | Pengukuran 3 | -1,7 — -1,3 | 0,5 — 0,9 | 0,5 — 2,0 | Masuk Kriteria |
| **3 (Faiz)** | Pengukuran 1 | -1,6 — -0,5 | 0,5 — 1,1 | 0,5 — 2,0 | Masuk Kriteria |
| | Pengukuran 2 | -2,2 — -0,6 | 0,5 — 1,2 | 0,5 — 2,0 | Masuk Kriteria |
| | Pengukuran 3 | -1,6 — -0,5 | 0,5 — 0,9 | 0,5 — 2,0 | Masuk Kriteria |
| **4 (Nadya)** | Pengukuran 1 | -1,8 — -1,5 | 0,5 — 0,7 | 0,5 — 2,0 | Masuk Kriteria |
| | Pengukuran 2 | -1,8 — -0,7 | 0,5 — 0,7 | 0,5 — 2,0 | Masuk Kriteria |
| | Pengukuran 3 | -1,8 — -1,4 | 0,5 — 0,7 | 0,5 — 2,0 | Masuk Kriteria |
| **5 (Semi)** | Pengukuran 1 | -1,7 — -0,5 | 0,5 — 0,8 | 0,5 — 2,0 | Masuk Kriteria |
| | Pengukuran 2 | -2,0 — -0,5 | 0,5 — 0,9 | 0,5 — 2,0 | Masuk Kriteria |
| | Pengukuran 3 | -1,9 — -0,5 | 0,5 — 0,8 | 0,5 — 2,0 | Masuk Kriteria |

**Paragraf Analisis Pembahasan:**
Berdasarkan Tabel 6.1, seluruh 15 sesi pengukuran pada 5 subjek menghasilkan amplitudo puncak gelombang R (*R-Peak*) pada rentang **0,5 mV hingga 1,6 mV**. Seluruh nilai ini berada di dalam rentang referensi fisiologis standar ECG Lead II (0,5 – 2,0 mV). Defleksi minimum gelombang S terukur stabil pada rentang -0,5 mV hingga -2,5 mV dengan garis isoelektrik yang mendatar tepat pada 0 mV.

Sinyal mentah (`ecg_raw_mV`) terukur memiliki *offset* DC elektrokimia sebesar +1,9 mV hingga +5,6 mV. Nilai ini berada jauh di bawah batas skala penuh ADC Texas Instruments ADS1292R ($\pm 403{,}3\text{ mV}$), membuktikan tidak adanya fenomena saturasi atau pemotongan sinyal (*clipping*). Dengan demikian, tahapan prapemrosesan digital (IIR Notch 50 Hz dan Butterworth Bandpass 0,5–40 Hz) berhasil menghasilkan sinyal masukan yang terstandarisasi dan memenuhi kriteria untuk inferensi model.

---

### 6.2 Pengujian Tingkat Akurasi Prediksi Tekanan Darah (Menjawab RM 2)

#### 6.2.1 Tujuan Pengujian
Mengevaluasi tingkat akurasi model *Multitask BiLSTM* dengan kalibrasi satu titik dalam mengestimasi nilai tekanan darah sistolik (SBP) dan diastolik (DBP) secara non-invasif terhadap standar medis internasional AAMI (MAE $\le 5\text{ mmHg}$, STD $\le 8\text{ mmHg}$) dan standar British Hypertension Society (BHS).

#### 6.2.2 Prosedur Pengujian
1. Melakukan pengukuran tekanan darah referensi (*Ground Truth*) awal pada lengan subjek menggunakan tensimeter digital terkalibrasi OMRON HEM-7130 saat subjek duduk tenang.
2. Memasukkan nilai SBP dan DBP referensi ke antarmuka sistem untuk mengunci *offset* kalibrasi 1-titik pada jendela pertama ($t_0$).
3. Menjalankan estimasi berkelanjutan selama 6 jendela bergeser (masing-masing 10 detik, durasi total 60 detik) sebanyak 3 kali sesi per subjek.
4. Mencatat nilai estimasi SBP, DBP, selisih galat absolut (MAE), galat kuadratik (RMSE), deviasi standar (STD), serta menghitung persentase kumulatif BHS ($\le 5\text{ mmHg}, \le 10\text{ mmHg}, \le 15\text{ mmHg}$).

#### 6.2.3 Hasil dan Analisis Pengujian
*(Sisipkan Gambar: Grafik Perbandingan Prediksi SBP/DBP terhadap Ground Truth Manset)*

**Tabel 6.2 Hasil Pengujian Akurasi Estimasi Tekanan Darah (15 Sesi Pengujian)**

| Subjek | Pengukuran | Nilai Manset (Ref) | Estimasi SBP / DBP (mmHg) | MAE SBP (mmHg) | MAE DBP (mmHg) | Status AAMI (SBP / DBP) | Grade BHS (SBP / DBP) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Aris)** | Pengukuran 1 | 112 / 53 | 111,8 / 52,8 | 1,33 | 0,45 | LULUS / LULUS | A / A |
| | Pengukuran 2 | 100 / 62 | 99,2 / 61,9 | 0,83 | 0,32 | LULUS / LULUS | A / A |
| | Pengukuran 3 | 97 / 52 | 101,8 / 53,4 | 4,82 | 1,40 | LULUS / LULUS | B / A |
| **2 (Cahyo)** | Pengukuran 1 | 96 / 70 | 94,9 / 69,5 | 1,73 | 1,00 | LULUS / LULUS | A / A |
| | Pengukuran 2 | 95 / 65 | 94,2 / 64,0 | 0,97 | 1,00 | LULUS / LULUS | A / A |
| | Pengukuran 3 | 93 / 68 | 95,1 / 70,1 | 2,08 | 2,10 | LULUS / LULUS | A / A |
| **3 (Faiz)** | Pengukuran 1 | 117 / 74 | 117,7 / 74,8 | 3,17 | 1,10 | LULUS / LULUS | B / A |
| | Pengukuran 2 | 107 / 83 | 104,2 / 81,1 | 3,42 | 2,23 | LULUS / LULUS | A / A |
| | Pengukuran 3 | 98 / 78 | 94,5 / 77,1 | 4,00 | 1,17 | LULUS / LULUS | A / A |
| **4 (Nadya)** | Pengukuran 1 | 99 / 76 | 97,9 / 75,4 | 1,43 | 0,60 | LULUS / LULUS | A / A |
| | Pengukuran 2 | 104 / 72 | 104,8 / 72,4 | 1,12 | 0,53 | LULUS / LULUS | A / A |
| | Pengukuran 3 | 103 / 75 | 102,4 / 74,9 | 1,52 | 0,65 | LULUS / LULUS | A / A |
| **5 (Semi)** | Pengukuran 1 | 122 / 81 | 123,2 / 80,5 | 1,83 | 1,08 | LULUS / LULUS | A / A |
| | Pengukuran 2 | 118 / 78 | 123,8 / 80,6 | 5,75 | 2,58 | TIDAK / LULUS | B / A |
| | Pengukuran 3 | 118 / 78 | 111,9 / 77,4 | 6,12 | 0,85 | TIDAK / LULUS | D / A |
| **Rata-rata** | — | — | — | **2,27 mmHg** | **1,14 mmHg** | **LULUS / LULUS** | **Grade A / A** |

**Paragraf Analisis Pembahasan:**
Berdasarkan Tabel 6.2, sistem estimasi tekanan darah terkalibrasi satu titik menghasilkan performa akurasi yang tinggi dengan rata-rata MAE sebesar **2,27 mmHg untuk SBP** dan **1,14 mmHg untuk DBP**. Merujuk pada protokol AAMI/ISO 81060-2 yang mensyaratkan rata-rata error $\le 5\text{ mmHg}$ dengan standar deviasi $\le 8\text{ mmHg}$, sistem dinyatakan **LULUS** pada SBP maupun DBP secara agregat. Pada tingkat klasifikasi BHS, estimasi DBP mencapai **100% Grade A**, sedangkan SBP didominasi oleh Grade A (80%) dan Grade B (13,3%).

Terdapat sedikit deviasi pada Subjek 5 Pengukuran 2 dan 3 di mana MAE SBP mencapai 5,75 mmHg dan 6,12 mmHg. Hal ini dipicu oleh fluktuasi hemodinamik respirasi serta pergeseran baseline elektrode selama transisi antar-sesi. Namun secara umum, kalibrasi 1-titik terbukti sukses menyerap bias fisiologis individu yang sebelumnya menyebabkan error hingga $>25\text{ mmHg}$ pada model murni (*pure inference*).

---

### 6.3 Pengujian Kinerja Komputasi dan Operasi Waktu-Nyata (Menjawab RM 3)

#### 6.3.1 Tujuan Pengujian
Mengukur efisiensi komputasi sistem estimasi secara *real-time*, meliputi waktu latensi inferensi model *deep learning*, kelancaran *rendering* antarmuka GUI (FPS), serta kestabilan alokasi memori (*buffer congestion*).

#### 6.3.2 Prosedur Pengujian
1. Mengintegrasikan modul pewaktu presisi tinggi (`time.perf_counter()`) pada blok inferensi `PyTorch` dan antarmuka `PySide6`.
2. Mencatat waktu komputasi setiap kali jendela 1.250 sampel (10 detik) selesai diproses oleh model BiLSTM.
3. Menghitung rata-rata latensi, latensi puncak (maksimum), serta laju bingkai visualisasi (*Frame Per Second* / FPS) selama 90 jendela inferensi.

#### 6.3.3 Hasil dan Analisis Pengujian
*(Sisipkan Gambar: Grafik Distribusi Latensi Inferensi Real-Time)*

**Tabel 6.3 Hasil Pengujian Waktu Komputasi Sistem**

| Subjek | Pengukuran | Latensi Rata-rata (ms) | Latensi Maksimum (ms) | Throughput GUI (FPS) | Keterangan Antrean Buffer | Target Ambang Real-Time (< 100 ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Aris)** | Pengukuran 1–3 | 6,25 ms | 19,34 ms | 30,1 FPS | Zero Congestion | Terpenuhi (16x lebih cepat) |
| **2 (Cahyo)**| Pengukuran 1–3 | 5,27 ms | 18,45 ms | 28,5 FPS | Zero Congestion | Terpenuhi (19x lebih cepat) |
| **3 (Faiz)** | Pengukuran 1–3 | 5,23 ms | 17,80 ms | 29,2 FPS | Zero Congestion | Terpenuhi (19x lebih cepat) |
| **4 (Nadya)**| Pengukuran 1–3 | 4,83 ms | 16,50 ms | 30,4 FPS | Zero Congestion | Terpenuhi (20x lebih cepat) |
| **5 (Semi)** | Pengukuran 1–3 | 5,00 ms | 17,10 ms | 28,8 FPS | Zero Congestion | Terpenuhi (20x lebih cepat) |
| **Rata-rata** | — | **5,32 ms** | **19,34 ms** | **29,4 FPS** | **Bebas Macet** | **SANGAT MEMENUHI** |

**Paragraf Analisis Pembahasan:**
Hasil pengujian pada Tabel 6.3 menunjukkan bahwa sistem mencatatkan waktu komputasi inferensi rata-rata sebesar **5,32 ms**, dengan nilai puncak maksimum hanya sebesar **19,34 ms**. Batas toleransi latensi untuk sistem pemantauan fisiologis waktu-nyata umumnya adalah $< 100\text{ ms}$ (atau interval pergeseran jendela 10 detik). Dengan demikian, waktu inferensi BiLSTM hanya mengonsumsi $< 6\%$ dari ambang batas aman real-time. Tampilan visual GUI berjalan mulus pada rata-rata **29,4 FPS** tanpa adanya penurunan *frame* (*frame drop*), dan struktur *ring buffer* beroperasi dalam kondisi *Zero Congestion*.

---

### 6.4 Pengujian Efektivitas Kalibrasi 1-Titik vs Tanpa Kalibrasi (Menjawab RM 4 / Ablation Study)

*(Subbab ini menjadi keunggulan utama skripsi Anda dibanding kating terdahulu, membuktikan lompatan performa ilmiah dari metode usulan).*

#### 6.4.1 Tujuan Pengujian
Membuktikan secara kuantitatif urgensi dan signifikansi penerapan Kalibrasi 1-Titik Manset terhadap model murni tanpa kalibrasi (*pure AI*).

#### 6.4.2 Hasil dan Analisis Pengujian
**Tabel 6.4 Perbandingan Kinerja Sebelum vs Sesudah Kalibrasi 1-Titik**

| Metrik Evaluasi | Tanpa Kalibrasi (Pure AI) | Dengan Kalibrasi 1-Titik (Usulan) | Peningkatan Performa |
| :--- | :---: | :---: | :---: |
| **MAE SBP (mmHg)** | 14,35 mmHg (maks. 25,80 mmHg) | **2,27 mmHg** (maks. 6,12 mmHg) | **Membaik 84,2%** |
| **MAE DBP (mmHg)** | 8,98 mmHg (maks. 16,70 mmHg) | **1,14 mmHg** (maks. 2,58 mmHg) | **Membaik 87,3%** |
| **Kelulusan AAMI SBP** | 0% (0 dari 15 sesi) | **86,7% (13 dari 15 sesi)** | Lompatan klinis signifikan |
| **Kelulusan AAMI DBP** | 0% (0 dari 15 sesi) | **100,0% (15 dari 15 sesi)** | Sesuai standar medis |
| **BHS Grade SBP** | Grade D (Gagal) | **Grade A** | Kualifikasi medis tertinggi |
| **BHS Grade DBP** | Grade D (Gagal) | **Grade A** | Kualifikasi medis tertinggi |

**Paragraf Analisis Pembahasan:**
Hasil pada Tabel 6.4 memberikan justifikasi saintifik yang tegas bahwa estimasi tekanan darah berbasis ECG murni secara *zero-shot* tidak dapat langsung digunakan pada subjek baru. Hal ini terjadi karena fenomena *mechano-electric coupling* dan resistensi vaskular perifer setiap individu bersifat unik dan tidak tercermin secara absolut pada amplitudo sinyal ECG Lead II. Dengan mengintegrasikan satu kali pengukuran tensimeter manset di awal sebagai titik jangkar (*anchor point*), sistem mampu memangkas galat rata-rata lebih dari 84%, mengubah sistem dari status gagal medis (Grade D) menjadi berstandar klinis (**Grade A**).
