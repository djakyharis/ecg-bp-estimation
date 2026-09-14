# Dataset — Cuff-Less Blood Pressure Dataset (Kachuee et al.)

Dataset yang digunakan adalah **Cuff-Less Blood Pressure Dataset** dari Kachuee et al. (2015),
tersedia secara publik di Kaggle. Berisi sinyal PPG, ECG, dan ABP dari 12.000 segmen rekaman ICU.

- **Kaggle:** [mkachuee/BloodPressureDataset](https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset)
- **Format:** `.mat` files (part_1.mat hingga part_N.mat)
- **Sinyal:** `PPG`, `ECG`, `ABP` (Arterial Blood Pressure) @ 125 Hz

---

## Cara Download

### Opsi A — Kaggle CLI (Direkomendasikan)

```bash
# Install Kaggle CLI jika belum ada
pip install kaggle

# Letakkan kaggle.json di ~/.kaggle/ (dari Kaggle → Account → API → Create New Token)
kaggle datasets download -d mkachuee/BloodPressureDataset -p data/raw/ --unzip
```

### Opsi B — kagglehub (otomatis via script)

```bash
pip install kagglehub
python src/download_dataset.py
```

Script akan otomatis mengunduh dan menyimpan ke `data/raw/kachuee_mat/`.

### Opsi C — Download Manual

1. Buka [kaggle.com/datasets/mkachuee/BloodPressureDataset](https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset)
2. Klik **Download** → ekstrak ke `data/raw/kachuee_mat/`

---

## Struktur Setelah Download

```
data/
└── raw/
    └── kachuee_mat/
        ├── part_1.mat
        ├── part_2.mat
        └── ...
```

> Folder `data/raw/` diabaikan oleh `.gitignore` — tidak akan ter-commit ke repo.

---

## Referensi

Kachuee, M., Kiani, M. M., Mohammadzade, H., & Shabany, M. (2015).
Cuffless blood pressure estimation algorithms for continuous health-care monitoring.
*IEEE Transactions on Biomedical Engineering*, 64(4), 859–869.
DOI: [10.1109/TBME.2015.2508812](https://doi.org/10.1109/TBME.2015.2508812)
