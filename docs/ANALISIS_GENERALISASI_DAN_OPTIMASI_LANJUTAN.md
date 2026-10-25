# Analisis Generalisasi `run_adamw_all` dan Rencana Optimasi Lanjutan

> Melengkapi [COMPARATIVE_REVIEW_RUN_ADAMW_ALL.md](COMPARATIVE_REVIEW_RUN_ADAMW_ALL.md).
> Semua angka di bawah berasal dari eksperimen yang dijalankan langsung (skrip dan log disebutkan).
> Hal yang **belum terbukti** diberi label eksplisit.

---

## 1. Bland-Altman `run_adamw_all` (test set acak, 26.133 sampel)

| Target | Bias | Batas kesepakatan 95 % | Pola |
|---|:-:|:-:|---|
| SBP | +0,29 | −24,4 … +25,0 | Dispersi terbesar di rata-rata 110–140 mmHg; ekor besar condong ke bawah (tekanan tinggi diprediksi terlalu rendah) |
| DBP | −0,26 | −13,7 … +13,1 | Tepi diagonal di kiri bawah = batas bawah data (DBP ≥ 50) |
| MAP | −0,30 | −14,4 … +13,8 | Simetris |

Bias ≈ 0 di semua target. Sisa error adalah **variansi**, bukan pergeseran sistematis.
Konsekuensinya: kalibrasi offset sederhana tidak akan menolong *selama tidak per-pasien*. Menaikkan akurasi
berarti mengurangi variansi prediksi (lihat §3).

---

## 2. Eksperimen Kebocoran Split (hasil terukur)

**Pertanyaan:** seberapa banyak MAE `run_adamw_all` berasal dari "mengenali rekaman yang sudah dilihat"?

**Fakta pendukung (data lokal, Part 1 & 5):** korelasi lag-1 SBP antar-segmen berurutan = 0,96; selisih |ΔSBP|
antar-segmen berurutan 3,2–3,5 mmHg vs 21–25 mmHg untuk pasangan acak.

**Eksperimen:** konfigurasi terbaik (AdamW 1e-3, wd 1e-4, Smooth L1, gating, phys, `all25`, cosine), **Part 1
saja, 20 epoch, 1 seed**, dibandingkan dua cara split yang identik ukurannya (20.156 / 2.519 / 2.521):
split acak per segmen vs split berblok kontigu (train = awal, val = tengah, test = akhir; rekaman test tidak
muncul di train). Skrip: `scratch/run_split_exp.py`.

| Split | SBP | DBP | MAP | **Rata-rata MAE** | Baseline tebak-median-train |
|---|:-:|:-:|:-:|:-:|:-:|
| Acak per segmen | 13,24 | 5,22 | 6,13 | **8,19** | 12,47 |
| **Berblok (lintas-rekaman)** | 20,27 | 6,66 | 11,03 | **12,65** | **10,18** |

(Baseline blok: SBP 15,86 / DBP 6,40 / MAP 8,29.)

### Pembacaan

1. **Pada split lintas-rekaman, model kalah dari baseline konstan** (12,65 vs 10,18), SBP paling parah
   (20,27 vs 15,86). Val SBP memburuk seiring training (18,9 → ±24 pada epoch 10–20) sementara loss train terus turun.
   Itu pola khas *menghafal identitas rekaman*, bukan belajar hubungan ECG → tekanan darah yang berlaku umum.
2. **Pada split acak, model mengalahkan baseline dengan margin besar** (8,19 vs 12,47). Selisih ini sebagian
   besar adalah efek "tetangga bocor", bukan generalisasi.
3. Jika hasil ini berlaku juga pada model penuh, maka **MAE 5,87 pada `run_adamw_all` tidak bisa diklaim sebagai
   kemampuan estimasi untuk pasien baru**.

### Batasan eksperimen (jangan diabaikan)

- Hanya Part 1 (jumlah pasien sedikit), 20 epoch, 1 seed. Variansi antar-seed tidak diketahui.
- **Pergeseran distribusi:** DBP test blok rata-rata 66,6 (train 72,4) dengan STD hanya 5,7, jadi sebagian error
  adalah efek distribusi, bukan hanya generalisasi.
- Batas blok belum tentu tepat di batas rekaman (tidak ada ID rekaman di `.npz`).
- **Model penuh (261 ribu sampel, jauh lebih banyak pasien) bisa menggeneralisasi lebih baik** daripada eksperimen
  kecil ini. Ini **belum terukur** dan menjadi langkah pertama di §4.

---

## 3. Implikasi untuk Strategi Optimasi

Urutan prioritas berubah: **ukur dulu apa yang sebenarnya dicapai model, baru optimalkan.**
Mengoptimalkan MAE di split acak berarti mengoptimalkan kemampuan menghafal.

### Reframing yang jujur dan tetap kuat untuk skripsi

Hasil split acak menunjukkan bahwa **mengetahui rekaman/pasien sangat membantu** (MAE turun ±35 % di
eksperimen kecil, 54 % di run penuh dibanding baseline). Itu justru mendukung desain **estimasi terpersonalisasi**:
model umum + 1–beberapa pengukuran kalibrasi per pasien. Protokol yang sah untuk ini:

1. Latih pada pasien A. Untuk pasien B (tidak dilihat saat training), ambil beberapa menit **awal** rekaman B
   sebagai kalibrasi/fine-tuning, lalu evaluasi pada bagian **setelahnya** dari rekaman B.
2. Laporkan ketiga angka: tanpa kalibrasi lintas-pasien, dengan kalibrasi offset, dengan fine-tuning ringan.

Ini membutuhkan **ID rekaman** di data (belum ada).

---

## 4. Rencana Langkah (Urut Prioritas)

### Langkah 0: Ukur performa lintas-rekaman model penuh (di server, 1 run)

Sudah tersedia flag baru `--split-mode block` (default tetap `random`, perilaku lama tidak berubah) dan log
baseline otomatis. Dengan `--parts all`, blok test = Part 9 (+ sebagian Part 8) = rekaman yang tidak dilihat training.

```bash
python src/train.py --parts all --epochs 80 --split-mode block \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 \
  --scheduler-type cosine --min-lr 1e-5 --loss-fn smooth_l1 --lambda-phys 0.10 \
  --patch-size 25 --use-projection --proj-dim 64 --use-gating --feature-set all25 \
  --checkpoint-dir checkpoints/run_adamw_block
```

Baca di log: baris `📏 Baseline tebak-median (test)` vs `RATA-RATA MAE KESELURUHAN`.
**Keputusan:**
- MAE jauh di bawah baseline → model punya generalisasi nyata; lanjut ke Langkah 2–4.
- MAE ≈ atau di atas baseline → fokus ke Langkah 1 (ID rekaman) dan reframing personalisasi.

> `--max-samples` jangan dipakai dengan mode block (urutan ikut teracak). Berkas smoke kini otomatis
> dikecualikan dari `--parts all`.

### Langkah 1: Simpan ID rekaman di preprocessing

Ubah `kachuee_preprocessor.py` agar menyimpan `rid` per segmen (indeks rekaman asal) di `.npz`, lalu pakai split
berbasis grup (rekaman tidak boleh terbelah antar train/val/test). Menggantikan block split yang kasar,
dan menjadi syarat evaluasi kalibrasi per-pasien yang valid.

### Langkah 2: Regularisasi dan penghentian dini (menurunkan menghafal)

Saat ini `lstm_dropout = fc_dropout = 0`. Coba berurutan, ukur pada split lintas-rekaman:
`--lstm-dropout 0.2 --fc-dropout 0.2`, `--weight-decay 1e-2`, early stopping pada val lintas-rekaman,
80 epoch cukup (best epoch run lama = 75 dari 150).

### Langkah 3: Ablasi terkontrol (≥ 3 seed) pada split yang valid

`baseline` → `+gating` → `+phys` → `+gating+phys`, serta `RMSE vs Smooth L1`. Tanpa ini, kontribusi komponen
**belum bisa diklaim** (lihat §6.9 pada review komparatif).

### Langkah 4: Hipotesis peningkatan model (belum terbukti, uji satu per satu)

| Ide | Alasan | Catatan |
|---|---|---|
| Normalisasi per-rekaman pada fitur/sinyal | Mengurangi perbedaan amplitudo antar-pasien | Perlu ID rekaman |
| Filter skewness ABP (data bersih) | Membuang label bising; satu-satunya hal yang terbukti membedakan data Fan et al. | `kachuee_preprocessor.py --output-dir data/processed_clean` |
| Pooling attention (`--use-attention`) | Alternatif terminal-state | Hanya ablasi |
| Stabilkan bobot adaptif (`--loss-momentum 0.9`) | θ berayun 0,15–0,70 antar-epoch | Belum terbukti lebih baik |
| Augmentasi ECG (noise, skala amplitudo) | Mengurangi menghafal | Pastikan tidak merusak morfologi |
| Ensemble 3–5 seed / SWA | Menurunkan variansi | Biaya komputasi ×N |

### Langkah 5: Literatur (verifikasi sendiri)

Cari dan baca: *subject-independent vs subject-dependent evaluation cuffless blood pressure ECG*, serta
*data leakage random split physiological signals*. Gunakan untuk memposisikan temuan §2 di Bab 4/5
(keterbatasan dan pembahasan). Saya tidak mencantumkan judul spesifik karena belum terverifikasi.

---

## 5. Yang Aman Ditulis Sekarang

| ✅ Aman | ❌ Jangan |
|---|---|
| "Pada protokol random-split 80/10/10 (seperti Fan et al.), MAE rata-rata 5,87 mmHg." | "Model mampu mengestimasi tekanan darah pasien baru dengan MAE 5,87." |
| "Pada eksperimen skala kecil (Part 1), split lintas-rekaman menaikkan MAE dari 8,19 ke 12,65 dan model tidak mengalahkan baseline konstan." | Menyatakan hasil eksperimen kecil itu berlaku untuk model penuh. |
| "Keterbatasan: split per segmen, tanpa ID pasien." | "Model lulus AAMI SP10." |

---

## 6. Reproduksi Eksperimen §2

```bash
# dari root repo; skrip ada di scratch (artefak sesi), config sama dengan run terbaik
SPLIT=random PYTHONPATH=. python run_split_exp.py --parts 1 --epochs 20 --feature-set all25 \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --scheduler-type cosine --min-lr 1e-5 \
  --loss-fn smooth_l1 --lambda-phys 0.1 --use-gating --num-workers 0 --checkpoint-dir /tmp/leak_random
SPLIT=block  PYTHONPATH=. python run_split_exp.py ...  # argumen sama
```

Setara dengan menjalankan `src/train.py --split-mode block` pada run kedua (flag kini ada di `train.py`).
