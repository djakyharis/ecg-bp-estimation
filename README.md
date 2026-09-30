# ecg-bp-estimation

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)
![License](https://img.shields.io/badge/license-MIT-green)

> Continuous blood pressure estimation from ECG using **Adaptive Weight Learning-Based Multitask BiLSTM**.
>
> Undergraduate Thesis — Computer Engineering, Universitas Brawijaya
> Supervisor: Edita Rosana Widasari, S.T., M.T., M.Eng., Ph.D.

---

## Overview

This repository contains the full experiment pipeline for estimating **Systolic (SBP)**,
**Diastolic (DBP)**, and **Mean Arterial Pressure (MAP)** continuously from ECG signals using a
shared **Bidirectional LSTM** encoder with **Adaptive Weight Learning** across three task heads.

### Model Architecture

```
ECG Signal (1250 samples @ 125 Hz)
         │
         ▼
┌─────────────────────┐
│    Preprocessing    │  Notch Filter (50 Hz) → BPF (0.5–40 Hz) → Robust Scaling
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Shared Encoder    │  BiLSTM · 2 layers · 256 hidden units → 512-dim output
└─────────────────────┘
         │
    ┌────┴────┬────┐
    ▼         ▼    ▼
  [SBP]    [DBP] [MAP]   FC heads: 512 → 256 → 1
                          Adaptive Weight Learning (λ_sbp, λ_dbp, λ_map)
```

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/ecg-bp-estimation.git
cd ecg-bp-estimation

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download dataset (requires PhysioNet account)
python src/download_dataset.py

# 3. Explore data
jupyter notebook notebooks/01_visualisasi_dataset_kachuee.ipynb

# 4. Run tests
pytest tests/ -v
```

---

## Repository Structure

```
ecg-bp-estimation/
├── .github/
│   └── workflows/
│       └── ci.yml              # Auto-run pytest on push
├── src/
│   ├── config.py               # Centralized config (filter, model, paths)
│   ├── download_dataset.py     # Download MIMIC-III via wfdb
│   ├── inspect_dataset.py      # Waveform inspection & validation
│   └── preprocessing/
│       ├── filters.py          # IIR Notch + Butterworth bandpass filter
│       ├── normalization.py    # Robust scaling normalization
│       └── pipeline.py         # End-to-end preprocessing pipeline
├── notebooks/
│   └── 01_visualisasi_dataset_kachuee.ipynb
├── tests/
│   └── test_filters.py
├── data/
│   └── README.md               # Dataset download instructions
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Configuration

All parameters are centralized in [`src/config.py`](src/config.py):

| Parameter               | Value        | Description                                 |
| ----------------------- | ------------ | ------------------------------------------- |
| `shimmer_fs`          | 256 Hz       | Shimmer3 ECG sensor sampling rate           |
| `target_fs`           | 125 Hz       | Resampling target (Kachuee dataset aligned) |
| `notch_freq`          | 50 Hz        | PLN interference frequency                  |
| `lowcut / highcut`    | 0.5 / 40 Hz  | Bandpass filter range                       |
| `seq_length`          | 1250 samples | Input window (10 s @ 125 Hz)                |
| `bilstm_hidden_units` | 256          | BiLSTM hidden units per direction           |

---

## Dataset

Dataset **Cuff-Less Blood Pressure** (Kachuee et al.) tersedia publik di Kaggle.
Unduh via Kaggle CLI:

```bash
pip install kaggle
# Letakkan kaggle.json di ~/.kaggle/ (dari Kaggle → Account → API)
kaggle datasets download -d mkachuee/BloodPressureDataset -p data/raw/ --unzip
```

Atau via script otomatis:

```bash
pip install kagglehub
python src/download_dataset.py
```

Lihat [`data/README.md`](data/README.md) untuk detail instruksi download.

---

## References

- Fan, X., Ren, M., Wang, Y., Zhang, X., & Song, Y. (2021). An Adaptive Weight Learning-Based Multitask Deep Network for Continuous Blood Pressure Estimation Using Electrocardiogram Signals. *Sensors*, 21(5), 1845. DOI: 10.3390/s21051845
- Kachuee, M., Kiani, M. M., Mohammadzade, H., & Shabany, M. (2016). Cuffless Blood Pressure Estimation Algorithms for Continuous Health-Care Monitoring. *IEEE TBME*, 64(4), 859–869. DOI: 10.1109/TBME.2015.2508812
- Li, Y. H., Harfianto, C. Z., Purwandari, K., & Lin, Y. D. (2020). Real-Time Cuffless Continuous Blood Pressure Estimation Using Deep Learning Model. *Sensors*, 20(19), 5606. DOI: 10.3390/s20195606
- Pan, J., & Tompkins, W. J. (1985). A Real-Time QRS Detection Algorithm. *IEEE TBME*, 32(3), 230–236. DOI: 10.1109/TBME.1985.325532

---

## License

Code — [MIT License](LICENSE).
Dataset Kachuee (Kaggle) — [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
