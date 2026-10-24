# Comparative Review: `run_adamw_all` vs Generasi Model Sebelumnya

> **Sumber data review ini** (semua diverifikasi langsung dari artefak):
> `checkpoints/run_adamw_all/{train.log, evaluation_results.json, training_curves.png, scatter_pred_vs_true.png}`,
> `docs/review_hasil.md`, kode di `src/`, serta dua perhitungan tambahan yang dijalankan saat review
> (baseline prediksi-konstan dan autokorelasi antar-segmen, lihat §6).
>
> **Tidak diperiksa:** `bland_altman.png`, `training_history.json` dan `best_model.pt` (tidak ada di folder lokal).

---

## 1. Ringkasan Eksekutif

| Metrik (Test Set, 26.133 sampel) | SBP | DBP | MAP | **Rata-rata** |
|---|:-:|:-:|:-:|:-:|
| MAE (mmHg) | 8,45 | 4,42 | 4,75 | **5,87** |
| Error STD (mmHg) | 12,59 | 6,84 | 7,19 | – |
| Bias / ME (mmHg) | +0,29 | −0,26 | −0,30 | – |
| BHS Grade | D | **A** | B | – |
| Lolos AAMI (SD ≤ 8, bias kecil) | ❌ | ✅ | ✅ | – |

- **Target dosen (MAE < 10 mmHg) tercapai pada ketiga target**, termasuk SBP (8,45).
- Rata-rata MAE turun dari **11,83 → 5,87** (≈ −50 %) dibanding model terbaik sebelumnya, dan **54 %
  lebih rendah dari baseline "tebak median konstan" (12,67)**.
- **Peringatan utama (baca §6.1):** split data dilakukan acak per segmen, sedangkan segmen berurutan berasal
  dari rekaman yang sama (autokorelasi SBP antar-segmen = 0,96). Angka 5,87 hampir pasti **lebih optimistis
  daripada performa pada pasien yang belum pernah dilihat model**. Angka ini valid sebagai hasil protokol
  *random-split* (sama seperti Fan et al.), tetapi **bukan bukti generalisasi lintas pasien**.

---

## 2. Konfigurasi Run

| Parameter | Nilai |
|---|---|
| Hardware | NVIDIA L40 (44,3 GB), AMP FP16 |
| Data | 12 part, 261.317 segmen termuat (13 berkas, lihat §6.2); Train 209.053 / Val 26.131 / Test 26.133 |
| Fitur statistik | `all25` (SBP head +13, DBP head +12, MAP head +25 fitur) |
| Arsitektur | BiLSTM 2-layer, hidden 256/arah, patch 25 (50 timestep), Feature Projector 64-dim, Gated Fusion, terminal states |
| Parameter terlatih | 3.580.675 |
| Optimizer | **AdamW**, lr 1e-3, weight decay 1e-4 (LSTM sama), grad clip 5,0 |
| Scheduler | **Cosine** 1e-3 → 1e-5 selama 150 epoch |
| Loss | **Smooth L1** (Huber), bobot tugas adaptif (Fan Eq. 8–10, floor 0,15, momentum 0,7), λ_phys = 0,10, tanpa normalisasi skala |
| Dropout | LSTM = 0,0 · FC = 0,0 · projector = 0,1 |
| Durasi | 94,8 menit (~38 dtk/epoch, 1.633 iterasi/epoch) |
| Model terbaik | **Epoch 75** (Val avg MAE 5,866); 53 kali "★ BEST" |

---

## 3. Perbandingan Antar Generasi Model

> ⚠️ **Tidak sepenuhnya apple-to-apple.** Baris Part 1 dilatih pada ±25 ribu sampel; baris 12 Part pada ±261 ribu.
> Perbaikan `run_adamw_all` adalah gabungan **(a) data 10× lebih banyak** dan **(b) perubahan optimizer/loss/arsitektur**.
> Efek (b) saja baru terlihat pada ablasi mini di §5.

| Run | Data | Optimizer / Loss | Arsitektur | SBP | DBP | MAP | **Avg MAE** |
|---|:-:|---|---|:-:|:-:|:-:|:-:|
| *Baseline konstan (median)* | – | – | – | 18,23 | 9,04 | 10,75 | *12,67* |
| `pure_bilstm_root` | 12 Part | SGD wd 0,1 / RMSE | 1250 step, tanpa fitur | 18,25 | 9,17 | 10,87 | 12,76 |
| `run_top12` | Part 1 | SGD wd 0,1 / RMSE | 1250 step + 12 fitur | 17,43 | 9,06 | 10,73 | 12,40 |
| `run_all25` | Part 1 | SGD wd 0,1 / RMSE | 1250 step + 25 fitur | 16,81 | 9,09 | 10,59 | 12,16 |
| `run_patch50x25_gpu0` | Part 1 | SGD wd 0,1 / RMSE | Patch 50×25 + 25 fitur | 16,77 | 9,10 | 10,60 | 12,16 |
| `run_patch_proj_gpu2` | Part 1 | SGD wd 0,1 / RMSE | Patch + Projector | 15,70 | 9,06 | 10,72 | 11,83 |
| `run_full12_*` | Part 1 | SGD / RMSE **dibagi skala** | Patch + Projector/Attn | 21,01 | 11,40 | 10,77 | 14,39 |
| **`run_adamw_all`** | **12 Part** | **AdamW / Smooth L1 + phys** | **Patch + Projector + Gating** | **8,45** | **4,42** | **4,75** | **5,87** |
| *Fan et al. 2021 (acuan)* | 21.422 seg. bersih | SGD / RMSE adaptif | Patch 50×20 | 7,63 | 4,36 | 5,60 | 5,86 |

**Pembacaan kritis:**

1. **Seluruh generasi lama berada di kisaran baseline konstan (12,67).** Rata-rata 11,83–12,76 hanya
   0,8–0,9 mmHg di bawah "menebak median"; DBP bahkan sama/lebih buruk (9,06 vs 9,04). Artinya
   *"batas bawah DBP ≈ 9 mmHg"* di `review_hasil.md` **bukan noise floor data**. DBP kini 4,42, jadi batas itu
   adalah artefak konfigurasi training lama.
2. Penyebab dugaan terkuat: **weight decay 0,1 pada SGD** menyusutkan bobot LSTM (`|W_ih| ≈ 0,002`, ±20× lebih
   kecil dari inisialisasi normal), sehingga model nyaris tidak memakai sinyal ECG. Ini konsisten dengan
   ablasi mini (§5), tetapi *belum dibuktikan* sebagai satu-satunya penyebab.
3. Hasil `run_adamw_all` **berada pada level yang sama dengan Fan et al.** (5,87 vs 5,86 rata-rata), tanpa filter
   skewness. Namun dataset, frekuensi sampling, dan protokol berbeda, jadi kesetaraan ini **jangan diklaim
   sebagai replikasi**. Perbedaan per target: SBP lebih buruk (8,45 vs 7,63), MAP lebih baik (4,75 vs 5,60),
   DBP sama. STD error SBP kita (12,59) juga lebih besar dari Fan (10,32).

---

## 4. Apa yang Berubah: Metode dan Arsitektur

| # | Komponen | Sebelum | Sekarang | Alasan | Status bukti |
|---|---|---|---|---|---|
| 1 | **Optimizer** | SGD mom 0,9, wd 0,1 | AdamW lr 1e-3, wd 1e-4 | wd 0,1 menekan bobot ke ~0 | ✅ Ablasi mini: 14,49 → 12,68 (hanya turunkan wd) |
| 2 | **Loss** | RMSE (L2) | Smooth L1 (≈ L1 untuk error > 1 mmHg) | Target dosen = MAE; tahan outlier ICU | ⚠️ Belum terisolasi |
| 3 | **Scheduler** | Plateau (patience 10) | Cosine 1e-3 → 1e-5 | LR turun mulus, tidak terpotong dini | ⚠️ Belum terisolasi |
| 4 | **Input patching** | 1250 timestep | 50 × 25 (0,2 dtk/patch) | BPTT 25× lebih pendek | ✅ Sudah ada di run lama (gain kecil saat optimizer salah) |
| 5 | **Feature Projector** | Concat fitur mentah | Linear→LN→ReLU→Linear→ReLU (→64) | Cegah feature drowning | ✅ Run lama: SBP 16,77 → 15,70 |
| 6 | **Gated Fusion** *(baru)* | Concat statis | `α=σ(W[h,f])`, `f·α`, bias gate = 1,0 | Model menimbang fitur per sampel | ❌ **Belum ada ablasi** |
| 7 | **Physiological loss** *(baru)* | – | `λ·L(MAP, DBP+(SBP−DBP)/3)`, λ=0,1 | Konsistensi hemodinamik | ❌ **Belum ada ablasi** |
| 8 | **Data** | Part 1 (±25 rb) | 12 Part (±261 rb) | Variasi pasien lebih banyak | ⚠️ Terkonfounding dengan #1–3 |
| 9 | Loss normalization | Bug: dibagi 120/80/93 | Dimatikan (skala absolut) | Gradien 100× mengecil → macet di 21 mmHg | ✅ Run `full12_*` membuktikan jebakannya |

> **Jujur soal atribusi:** yang terbukti hanya #1 (dan #9 sebagai pencegah kegagalan). Kontribusi **gating (#6)**,
> **phys loss (#7)**, **smooth L1 (#2)**, dan **cosine (#3)** terhadap hasil akhir **belum terukur**. Jangan
> menulis di skripsi bahwa "Gated Fusion menurunkan MAE sebesar X" sebelum ablasi (§8, langkah 3) dijalankan.

### Alur arsitektur aktual

```
ECG [B,1250] → reshape [B,50,25] → BiLSTM(2 layer, 256/arah) → fwd_last ⊕ bwd_first → h [512]
fitur statistik (25 → per-head subset) → FeatureProjector → f [64]     (satu projector per head)
        gate_k = σ(W_k[h, f_k] + b_k);  z_k = [h, gate_k ⊙ f_k]  (576)  untuk k ∈ {SBP,DBP,MAP}
z_k → FC(576→512→256→1) → ŷ_k            (bias akhir diinisialisasi 120 / 80 / 93,3)
Loss = Σ θ_k · SmoothL1(ŷ_k, y_k) + 0,1 · SmoothL1(MAP̂, DBP̂ + (SBP̂−DBP̂)/3)
```

Catatan: tiap head punya projector dan gate sendiri (bukan satu gate bersama); SBP memakai 13 fitur,
DBP 12 fitur, MAP 25 fitur.

---

## 5. Bukti Pendukung: Ablasi Mini Lokal

Dijalankan saat audit (CPU/MPS, Part 1, **hanya 8.000 sampel, 12 epoch**, fitur `all25`). Ini indikasi arah,
bukan hasil final.

| Konfigurasi | Avg MAE (test 800) |
|---|:-:|
| A: SGD, wd 0,1 (konfigurasi lama) | 14,49 |
| B: SGD, wd 1e-5 | 12,68 |
| C: AdamW 1e-3, wd 1e-4, Smooth L1 | **9,86** |

B−A mengisolasi efek weight decay (−1,81). C−B mencampur tiga perubahan (AdamW + Smooth L1 + lr), sehingga
**tidak bisa dipisah**. Satu seed, sampel kecil: variansi belum diketahui.

---

## 6. Knooks and Crannies (Temuan Teknis dan Caveat)

### 6.1 🔴 Kebocoran data antar-segmen (paling penting)

- `random_split` memecah **segmen**, bukan pasien/rekaman. Pada data lokal:
  - autokorelasi lag-1 SBP antar-segmen berurutan = **0,96** (DBP 0,95–0,97);
  - selisih |ΔSBP| antar-segmen berurutan = **3,2–3,5 mmHg**, vs **21–25 mmHg** untuk pasangan acak.
- Jadi sebagian besar sampel test punya "tetangga" hampir identik (pasien sama, BP sama, ECG berdekatan)
  di set train. Model dapat menang dengan mengenali *pasien/rekaman*, bukan hanya memetakan ECG → BP.
- **Gejala di log yang konsisten dengan ini:** loss train 1,1 vs val ≈ 5,5 (gap ≈ 5×), val MAE mendatar sejak
  epoch ~50 sementara train loss terus turun hingga epoch 150.
- Dampak: skor 5,87 = batas atas realistis untuk pasien yang sudah "dikenal". Untuk pasien baru kemungkinan
  lebih buruk (besarnya belum diketahui). Fan et al. juga memakai random split, jadi perbandingan di §3 adil
  *secara protokol*, tetapi keduanya sama-sama optimistis.
- **Mitigasi:** split per rekaman/pasien (§8, langkah 1). File `.npz` saat ini **tidak menyimpan ID rekaman**.

### 6.2 🟠 Berkas smoke ikut termuat

Log memuat **13 berkas**, termasuk `part_1_smoke_preprocessed.npz` (0,8 MB). Loader (`--parts all`) melakukan glob
`part_*` sehingga berkas smoke-test ikut masuk. Kemungkinan besar isinya duplikat segmen awal Part 1
(inferensi dari cara pembuatannya; berkasnya tidak ada lokal, jadi tidak diverifikasi). Dampak kecil
(~200 sampel dari 261 ribu) tetapi **merupakan duplikasi train/test** dan perlu dikecualikan.

### 6.3 🟠 Kalibrasi 1-titik di log **tidak valid**

Angka "terkalibrasi" (avg 8,07 > 5,87 tanpa kalibrasi) **lebih buruk** karena tiap blok 10 sampel acak
dianggap "satu pasien". Sudah diberi label *SIMULASI TIDAK VALID* di kode (commit `658745d`). **Jangan dikutip.**
Kalibrasi sungguhan baru bisa dievaluasi setelah ada ID pasien (§8).

### 6.4 🟠 Kriteria AAMI di kode lebih ketat dari definisi standar

Kode menguji `MAE ≤ 5 & STD ≤ 8`. Definisi AAMI/ISO 81060-2 memakai **|mean error| ≤ 5 & SD ≤ 8**. Dengan definisi
standar hasilnya sama (SBP gagal: SD 12,59; DBP & MAP lulus), tetapi untuk penulisan skripsi gunakan definisi
standar. Perlu diingat: AAMI mensyaratkan evaluasi per-subjek dan jumlah subjek tertentu, yang tidak
dipenuhi oleh split acak ini.

### 6.5 🟡 SBP adalah titik lemah

- BHS: ≤5 = 47,7 %, ≤10 = 71,5 %, ≤15 = 83,8 % → **Grade D** (selisih 1,2 poin dari Grade C di kolom ≤15).
- Scatter menunjukkan **regresi ke rata-rata**: SBP > 160 cenderung diprediksi terlalu rendah, SBP < 100 terlalu
  tinggi. Ada sebaran vertikal outlier (prediksi jauh dari kebenaran) pada SBP dan DBP.
- DBP kebenaran memiliki **batas bawah keras ±50 mmHg** (min data = 50, MAP min = 60, SBP min = 67): rentang
  fisiologis disaring saat preprocessing. Konsekuensinya, model belum diuji pada hipotensi berat.

### 6.6 🟡 Bobot tugas adaptif berosilasi liar

Plot θ menunjukkan θ_SBP/DBP/MAP berayun 0,15–0,70 antar-epoch (floor 0,15, momentum 0,7). Dua efek:
(i) `VlLoss` di log naik-turun (5,1–6,9) padahal Val MAE hampir datar. Penyebabnya bobot θ berganti, **bukan
model memburuk**; (ii) `VlLoss` **tidak boleh** dipakai sebagai metrik pemilihan model. Pemilihan checkpoint
sudah memakai Val avg MAE (benar). Belum terbukti adaptif lebih baik daripada bobot tetap.

### 6.7 🟡 Tanpa dropout, tanpa early stopping

`LSTM dropout = 0`, `FC dropout = 0`. Dengan gap train/val di atas, model memorisasi. Val MAE terbaik di
epoch 75 (5,866); epoch 150 = 6,02 (SBP 8,64, DBP 4,52, MAP 4,89), jadi **±75 epoch terakhir (~48 menit) tidak
menambah akurasi** dan sedikit merusak.

### 6.8 🟡 Lintasan konvergensi (Val MAE)

| Epoch | SBP | DBP | MAP | Avg | Catatan |
|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 16,60 | 7,94 | 9,35 | 11,30 | DBP/MAP sudah lebih baik dari semua run lama; SBP setara run lama |
| 20 | 12,61 | 5,54 | 6,14 | 8,10 | fase lambat |
| 30 | 9,92 | 4,90 | 5,33 | 6,72 | fase penurunan SBP tajam (epoch 20–40) |
| 50 | 8,78 | 4,43 | 4,82 | 6,01 | mulai mendatar |
| 75 | 8,49 | 4,39 | 4,72 | **5,87** | **best** |
| 150 | 8,64 | 4,52 | 4,89 | 6,02 | sedikit memburuk |

Test (5,87) ≈ Val terbaik (5,87): tidak ada tanda *test peeking*, dan selisih val–test nol justru
**konsisten dengan kebocoran** (val dan test sama-sama punya tetangga di train).

### 6.9 🟢 Hal yang sudah benar

- `FeatureStandardScaler` di-fit **hanya pada indeks train** (tidak ada leakage statistik fitur).
- Checkpoint terbaik dipilih dari validasi, test dievaluasi sekali di akhir.
- Loss tanpa normalisasi skala (jebakan 21 mmHg dihindari); bias akhir head diinisialisasi ke mean fisiologis.
- AMP FP16 tidak menimbulkan NaN; `last_checkpoint.pt` + auto-resume tersedia untuk koneksi SSH putus.

### 6.10 Perbandingan dengan Fan et al. tidak setara

Fan: 21.422 segmen yang sudah difilter skewness ABP, 100 Hz, patch 50×20. Kita: ±261 ribu segmen **tanpa**
filter skewness (data `*_preprocessed.npz` bertanggal 5 Okt, sebelum filter ditambahkan 22 Okt), 125 Hz,
patch 50×25. Dataset kita lebih besar tetapi lebih bising; filter belum pernah diuji.

---

## 7. Klaim yang Aman vs Tidak Aman untuk Skripsi

| ✅ Aman ditulis | ❌ Jangan ditulis |
|---|---|
| "Pada protokol random-split 80/10/10 (seperti Fan et al.), MAE test SBP/DBP/MAP = 8,45/4,42/4,75 mmHg (rata-rata 5,87), memenuhi target < 10 mmHg." | "Model generalisasi ke pasien baru dengan MAE 5,87." |
| "DBP mencapai BHS Grade A; MAP Grade B; SBP Grade D." | "Model lulus AAMI SP10" (SBP gagal, dan evaluasi bukan per-subjek). |
| "Mengganti SGD(wd 0,1) dengan AdamW(wd 1e-4) menurunkan MAE pada ablasi terkontrol." | "Gated Fusion / physiological loss meningkatkan akurasi" (belum diablasi). |
| "Baseline prediksi konstan MAE 12,67; model terlatih lama mendekati baseline ini." | Angka "terkalibrasi 1-titik" dari log. |
| Keterbatasan: split per segmen, tanpa ID pasien, tanpa filter skewness. | "Hasil setara/mereplikasi Fan et al." |

---

## 8. Langkah Selanjutnya (Urut Prioritas)

1. **Split per rekaman/pasien (validitas, prioritas tertinggi).**
   Simpan `record_id` per segmen di `kachuee_preprocessor.py` (`np.savez_compressed(..., rid=...)`), lalu pada
   `train.py` ganti `random_split` dengan split berbasis grup (mis. `GroupShuffleSplit`). Latih ulang dengan
   konfigurasi yang sama. Selisih skor vs 5,87 = ukuran kebocoran. Ini sekaligus membuka evaluasi
   kalibrasi 1-titik yang valid.
2. **Kecualikan berkas smoke** dari glob `--parts all` (filter nama `*smoke*`).
3. **Ablasi terkontrol (≥ 3 seed) agar klaim kontribusi sah:**
   `baseline AdamW+SmoothL1` → `+gating` → `+phys` → `+gating+phys`; juga `SGD-lowwd vs AdamW` dan
   `RMSE vs SmoothL1`. Gunakan split per pasien dari langkah 1.
4. **Regularisasi + early stopping:** `--lstm-dropout 0.2 --fc-dropout 0.2`, early stopping (patience ~20
   pada Val avg MAE), dan 80–100 epoch cukup. Hemat ±45 menit per run.
5. **Data bersih (skewness filter):** jalankan `kachuee_preprocessor.py --output-dir data/processed_clean`
   lalu `train.py --data-dir data/processed_clean`; bandingkan SBP dan DBP.
6. **Perbaiki SBP:**
   (a) `--use-attention` (sudah ada di kode) sebagai ablasi pooling; (b) stabilkan θ (`--loss-momentum 0.9`)
   atau bobot tetap SBP-prioritas; (c) augmentasi ECG ringan (noise, amplitude scaling); (d) ensemble 3–5 seed /
   SWA (biasanya −3–8 % MAE tanpa mengubah arsitektur).
7. **Pelaporan:** gunakan definisi AAMI standar (mean error), tambahkan Bland-Altman per target dan CI
   (bootstrap) untuk MAE.
8. **Validasi domain (opsional):** uji `best_model.pt` pada data Shimmer3 nyata (`scripts/test_shimmer_ecg.py`);
   pergeseran domain sensor-vs-MIMIC kemungkinan besar menaikkan error.

---

## 9. Reproduksi

```bash
python src/train.py --parts all --epochs 150 --batch-size 128 \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 \
  --scheduler-type cosine --min-lr 1e-5 --loss-fn smooth_l1 --lambda-phys 0.10 \
  --patch-size 25 --use-projection --proj-dim 64 --use-gating \
  --feature-set all25 --checkpoint-dir checkpoints/run_adamw_all
```
