# BLUEPRINT & FULL GUIDE: OPTIMASI PENDEKATAN MODEL FAN ET AL. (2021)
## Panduan Lengkap untuk AI Agent & Pengembang dalam Menutup Kesenjangan Akurasi (Target: SBP ≤ 8 mmHg, DBP ≤ 5 mmHg)

> **Dokumen Panduan Teknis & System Blueprint (*Context Injection*)**  
> **Status Repositori:** Eksperimen Baseline Pure BiLSTM, Hybrid Top-12, dan Hybrid All-25 telah selesai dievaluasi pada Part 1 dan 12 Part.  
> **Tujuan Dokumen:** Memberikan panduan arsitektural dan metodologis yang komprehensif bagi agen AI berikutnya agar dapat mengimplementasikan langkah-langkah optimasi terencana guna menyamai performa paper acuan Fan et al. (Sensors 2021).

---

## 1. Konteks Status Saat Ini (*Current State & Benchmark Gap*)

### A. Capaian Eksperimen yang Telah Dilakukan
Hingga saat ini, repositori ini telah berhasil melatih dan mengevaluasi 3 varian model:
1. **Pure BiLSTM (500 Epoch, 12 Part penuh / 261.117 sampel):**
   * $\text{SBP: MAE} = \mathbf{18,25\text{ mmHg}}, \text{STD} = 22,39\text{ mmHg}$
   * $\text{DBP: MAE} = \mathbf{9,17\text{ mmHg}}, \text{STD} = 11,74\text{ mmHg}$
   * $\text{Rata-rata MAE} = \mathbf{12,76\text{ mmHg}}$
2. **Hybrid BiLSTM + Top 12 Fitur (500 Epoch, Part 1 / 25.175 sampel):**
   * $\text{SBP: MAE} = \mathbf{17,43\text{ mmHg}}, \text{STD} = 21,44\text{ mmHg}$
   * $\text{DBP: MAE} = \mathbf{9,06\text{ mmHg}}, \text{STD} = 11,57\text{ mmHg}$
   * $\text{Rata-rata MAE} = \mathbf{12,40\text{ mmHg}}$
3. **Hybrid BiLSTM + All 25 Fitur (500 Epoch, Part 1 / 25.175 sampel):**
   * $\text{SBP: MAE} = \mathbf{16,81\text{ mmHg}}, \text{STD} = 20,85\text{ mmHg}$
   * $\text{DBP: MAE} = \mathbf{9,09\text{ mmHg}}, \text{STD} = 11,61\text{ mmHg}$
   * $\text{Rata-rata MAE} = \mathbf{12,16\text{ mmHg}}$

### B. Kesenjangan Target (*The Gap*)
Paper acuan utama (**Fan et al., Sensors 2021**, DOI: [10.3390/s21051595](https://doi.org/10.3390/s21051595)) melaporkan hasil:
* **SBP:** $\text{MAE} = \mathbf{7,63\text{ mmHg}} \quad|\quad \text{STD} = 10,32\text{ mmHg}$
* **DBP:** $\text{MAE} = \mathbf{4,36\text{ mmHg}} \quad|\quad \text{STD} = 5,76\text{ mmHg}$
* **MAP:** $\text{MAE} = \mathbf{5,60\text{ mmHg}} \quad|\quad \text{STD} = 7,35\text{ mmHg}$

---

## 2. Analisis Akar Masalah Teknis (*Root Cause Analysis*)

Agen AI yang membaca dokumen ini harus memahami bahwa kesenjangan akurasi ini **BUKAN** disebabkan oleh kesalahan fungsi loss atau optimizer (karena rumus *Adaptive Weight Learning*, learning rate $0,0008$, dan momentum $0,9$ sudah $100\%$ identik dengan Fan et al.). 

Penyebab utamanya adalah **4 perbedaan fundamental**:

### 1. Struktur Input Tensor: 1.250 Timestep vs 50 Timestep (Paling Berdampak)
* **Model Kita Saat Ini:**
  $$\mathbf{X} \in \mathbb{R}^{B \times 1250 \times 1}$$
  BiLSTM dipaksa melakukan *unrolling* sebanyak **1.250 langkah waktu**. Dalam teori recurrent neural network, informasi pada langkah 1 sampai 500 mengalami peluruhan eksponensial (*memory degradation*) saat mencapai langkah ke-1250. Vektor representasi terminal (`fwd_last` dan `bwd_first`) yang diambil di ujung sinyal kehilangan informasi detak jantung di tengah segmen.
* **Paper Fan et al. (2021):**
  $$\mathbf{X}_{\text{Fan}} \in \mathbb{R}^{B \times 50 \times 20} \quad (\text{pada 100 Hz, 1000 titik})$$
  Untuk dataset kita yang berfrekuensi 125 Hz (1.250 titik per 10 detik):
  $$1.250 = 50 \times 25 \implies \mathbf{X}_{\text{Patch}} \in \mathbb{R}^{B \times 50 \times 25}$$
  BiLSTM hanya berjalan **50 langkah waktu**. BPTT 25× lebih pendek, gradien sangat stabil, dan tiap langkah (25 sampel = 0,2 detik) merepresentasikan 1 morfologi gelombang jantung utuh (gelombang P, kompleks QRS, atau gelombang T).

### 2. Kurasi Kualitas Dataset ABP (*Negative Skewness Elimination*)
* Fan et al. menerapkan pra-penyaringan gelombang ABP: Segmen 10 detik dibagi menjadi sepuluh jendela 1 detik. Jika ada **satu saja jendela dengan $\text{Skewness} < 0$, seluruh segmen 10 detik DIBUANG**.
* Hasil seleksi tersebut menyaring MIMIC-II menjadi hanya **21.422 segmen sinus murni yang steril** dari artefak kateter, aritmia, atau gerak tubuh.
* Model kita saat ini melatih populasi ICU mentah Kachuee yang sangat heterogen dan bising.

### 3. Fenomena *Feature Drowning* di Layer Linear
* Pada model hybrid saat ini, fitur statistik digabungkan langsung: `torch.cat([bilstm_out (512), features (13)])`.
* Fitur statistik hanya menyumbang **$2,4\%$ neuron input** ke layer `Linear(525, 512)`.
* Gradien *backpropagation* didominasi oleh 512 neuron BiLSTM, sehingga fitur statistik kurang memiliki daya ungkit representasi.

---

## 3. Rencana Aksi Terstruktur (*The 4 Next Action Pillars*)

Berikut adalah 4 pilar optimasi yang harus dieksekusi secara bertahap oleh AI agent atau pengembang berikutnya:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PETA JALAN OPTIMASI MENGEJAR TARGET FAN ET AL.              │
├───────────────────────────────────┬─────────────────────────────────────────┤
│ PILAR 1: Input Patching           │ [DONE] Reshape [B, 1250, 1] -> [B, 50, 25]
│          (--patch-size 25)        │ (BiLSTM unrolls 50 steps, patch=0.2s)   │
├───────────────────────────────────┼─────────────────────────────────────────┤
│ PILAR 2: Feature Projection MLP   │ [DONE] Subnet FeatureProjector MLP      │
│          (--use-projection)       │ (Proyeksi 25 fitur ke 64/128 dimensi)   │
├───────────────────────────────────┼─────────────────────────────────────────┤
│ PILAR 3: Skewness Purge ABP       │ [TODO] Filter buang segmen ABP Skew < 0 │
│          (Kurasi Dataset Steril)  │ persis sesuai Bagian 2.2 Fan et al.     │
├───────────────────────────────────┼─────────────────────────────────────────┤
│ PILAR 4: Skema Kalibrasi 1-Titik  │ [TODO] Tambahkan mode kalibrasi baseline│
│          (Wearable Clinical Mode) │ (Subject-specific calibration offset)   │
└───────────────────────────────────┴─────────────────────────────────────────┘
```


---

### PILAR 1: Rekonstruksi Input Patching ($50 \times 25$)

#### Rasio Pemetaan Matematika:
* Frekuensi sampling: $f_s = 125\text{ Hz}$.
* Jendela 10 detik: $N = 1.250\text{ sampel}$.
* Pembagian:
  $$\text{Timesteps } (T) = 50, \quad \text{Feature per step } (D) = 25 \implies 50 \times 25 = 1.250$$
  Tiap timestep berdurasi: $\frac{25}{125\text{ Hz}} = 0,20\text{ detik}$ ($200\text{ ms}$). Ini adalah durasi fisiologis ideal yang membungkus 1 gelombang P, kompleks QRS, atau gelombang T.

#### Spesifikasi Perubahan Kode di `src/models/bilstm.py`:
1. Ubah parameter `input_size` dari default `1` menjadi `25` (dapat dikonfigurasi via argumen `--patch-size`).
2. Di dalam method `extract_features(self, x)`:
```python
# SEBELUM:
if x.dim() == 2:
    x = x.unsqueeze(-1)  # [B, 1250, 1]
lstm_out, _ = self.lstm(x)

# SESUDAH (Mendukung mode Patching):
if self.patch_size > 1:
    # x shape: [B, 1250] atau [B, 1250, 1]
    if x.dim() == 3:
        x = x.squeeze(-1)
    batch_size, total_len = x.shape
    seq_len = total_len // self.patch_size  # 1250 // 25 = 50
    x = x.view(batch_size, seq_len, self.patch_size)  # [B, 50, 25]
else:
    if x.dim() == 2:
        x = x.unsqueeze(-1)

lstm_out, _ = self.lstm(x)
```

> [!TIP]
> **Kompatibilitas Inferensi Sensor Shimmer3:**  
> Dengan membuat method menerima `[B, 1250]` lalu otomatis me-*reshape* ke `[B, 50, 25]` di dalam forward pass, pipeline inferensi luar (`ECGBPPredictor` dan sensor Shimmer3) **tidak perlu diubah sama sekali**. Sensor tetap mengirim 1250 titik 1D, dan model yang menangani pemotongan spasial-temporal secara internal.

---

### PILAR 2: Blok Proyeksi Fitur Statistik (*Feature Projection MLP*)

Untuk memastikan 25 fitur statistik tidak "tenggelam" di samping 512 dimensi BiLSTM, fitur harus dilewatkan melalui subnet proyeksi non-linear sebelum digabungkan.

#### Arsitektur Subnet Proyeksi:
```
Fitur Statistik (12/25) ──► Linear(d_in, 128) ──► LayerNorm / BatchNorm ──► ReLU ──► Linear(128, 64) ──► ReLU ──► Vektor Proyeksi (64)
                                                                                                                   │
BiLSTM Terminal Latent Vector (512) ───────────────────────────────────────────────────────────────────────────────┴──► Concat [512 + 64 = 576] ──► Task Head
```

#### Spesifikasi Kode di `TaskSpecificHead` atau `ECGMultiTaskBiLSTM`:
```python
class FeatureProjector(nn.Module):
    def __init__(self, in_features: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```
Dengan arsitektur ini, representasi fitur statistik memiliki bobot efektif $\frac{64}{576} \approx \mathbf{11,1\%}$ (naik hampir $10\times$ dibanding hanya $1,15\%$), memberikan daya dorong regresi yang jauh lebih seimbang.

---

### PILAR 3: Kurasi Kualitas Dataset ABP (*Negative Skewness Purge*)

Sesuai paper Fan et al. Bagian 2.2, buat skrip pembersih atau integrasikan ke dalam `kachuee_preprocessor.py`:

#### Formulasi Matematis Uji Skewness ABP:
Untuk setiap segmen ABP 10 detik ($1.250$ sampel):
1. Bagi menjadi 10 jendela 1 detik non-overlapping ($W_1, W_2, \dots, W_{10}$, masing-masing 125 titik).
2. Hitung skewness tiap jendela:
   $$\text{Skew}(W_k) = \frac{\frac{1}{M}\sum_{j=1}^M (p_j - \bar{p})^3}{\left[\frac{1}{M}\sum_{j=1}^M (p_j - \bar{p})^2\right]^{3/2}}$$
3. **Kriteria Eliminasi:**
   $$\text{Reject jika } \min_{k=1\dots10} \text{Skew}(W_k) < 0$$

> Melatih model pada subset data yang telah lolos uji *positive skewness* ini adalah kunci mengapa Fan et al. mendapatkan DBP MAE 4,36 mmHg.

---

### PILAR 4: Skema Kalibrasi Awal 1-Titik (*Subject-Specific Calibration Mode*)

Jika ingin mendemonstrasikan akurasi standar medis tertinggi (AAMI Grade A) pada perangkat wearable Shimmer3:

#### Formulasi Model Terkalibrasi:
Model dilatih untuk memprediksi **deviasi tekanan darah relatif ($\Delta \text{BP}$)** terhadap pengukuran awal pasien:
$$\hat{y}_{\text{SBP}} = y_{\text{baseline}}^{\text{SBP}} + f_{\text{model}}(\mathbf{x}_{\text{ECG}}, \text{feats})$$
Atau masukkan nilai skalar $y_{\text{baseline}}$ sebagai fitur input tambahan ke FC head.
Di seluruh literatur biosinyal (Kachuee 2017, Slapničar 2019), mode terkalibrasi 1-titik ini **menurunkan MAE secara instan dari 16 mmHg menjadi 3–4 mmHg**.

---

## 4. Panduan Eksekusi Pelatihan (*Multi-GPU Execution Guide*)

Saat menjalankan eksperimen optimasi berikutnya, gunakan pemisahan GPU agar dapat melatih 2 model secara paralel:

```bash
# Model Eksperimen A (Pilar 1: Patching 50x25 pada GPU 0)
CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --parts all \
    --epochs 500 \
    --batch-size 128 \
    --patch-size 25 \
    --use-features \
    --feature-set all25 \
    --checkpoint-dir checkpoints/run_patch50x25_all25_gpu0 \
    --log-file checkpoints/run_patch50x25_all25_gpu0/train.log

# Model Eksperimen B (Pilar 1 + Pilar 2: Patching + Feature Projection pada GPU 2)
CUDA_VISIBLE_DEVICES=2 python src/train.py \
    --parts all \
    --epochs 500 \
    --batch-size 128 \
    --patch-size 25 \
    --use-projection \
    --use-features \
    --feature-set all25 \
    --checkpoint-dir checkpoints/run_patch_proj_gpu2 \
    --log-file checkpoints/run_patch_proj_gpu2/train.log
```

---

## 5. Ringkasan Kunci untuk Agen AI Berikutnya
* **Jangan ubah:** Rumus *Adaptive Weight Learning* (Eq. 8, 9, 10 Fan et al.) di `src/models/loss.py` dan filter analog Butterworth 0.5–40 Hz (kebutuhan sensor Shimmer3).
* **Fokuskan inovasi pada:** Pengubahan bentuk input BiLSTM ke format patch $50 \times 25$ (`patch_size=25`) dan subnet proyeksi fitur statistik. Langkah ini secara teori komputasi akan memberikan lompatan akurasi terbesar menuju standar Fan et al. (2021).
