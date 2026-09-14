"""
Modul Pengunduh Robust untuk Dataset Kachuee et al. (Cuff-Less Blood Pressure Dataset).
Mendukung pengunduhan bertahap (resumable chunked download) untuk menghindari Read Timeout.
"""

import os
import sys
import time
import requests
from tqdm import tqdm

# Tambahkan direktori root src ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def download_with_retry(url: str, output_path: str, max_retries: int = 5, chunk_size: int = 1024 * 1024):
    """
    Mengunduh berkas dengan penanganan pembatalan koneksi dan otomatis mencoba ulang (retry).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = output_path + ".tmp"

    for attempt in range(1, max_retries + 1):
        print(f"[Percobaan {attempt}/{max_retries}] Menghubungkan ke server dataset...")
        try:
            # Cek berapa byte yang sudah terunduh
            downloaded_bytes = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            headers = {}
            if downloaded_bytes > 0:
                headers["Range"] = f"bytes={downloaded_bytes}-"
                print(f"Melanjutkan pengunduhan dari byte: {downloaded_bytes:,}")

            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            if response.status_code in [200, 206]:
                total_size = int(response.headers.get("content-length", 0)) + downloaded_bytes
                mode = "ab" if downloaded_bytes > 0 else "wb"
                
                with open(temp_path, mode) as f, tqdm(
                    total=total_size,
                    initial=downloaded_bytes,
                    unit="B",
                    unit_scale=True,
                    desc=os.path.basename(output_path),
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                # Jika sukses, ubah nama temp menjadi nama asli
                os.rename(temp_path, output_path)
                print(f"Pengunduhan selesai: {output_path}")
                return output_path
            else:
                print(f"Status HTTP {response.status_code}. Mencoba ulang...")
        except (requests.exceptions.RequestException, TimeoutError) as e:
            print(f"Koneksi terputus ({e}). Mencoba ulang dalam 5 detik...")
            time.sleep(5)

    raise ConnectionError("Gagal mengunduh berkas setelah beberapa kali percobaan.")


def get_kachuee_dataset_dir():
    """
    Mengembalikan path direktori dataset lokal di 04_Eksperimen_dan_Kode/data/kachuee_mat/
    atau dari kagglehub jika sudah tersedia.
    """
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "kachuee_mat"))
    os.makedirs(local_dir, exist_ok=True)
    
    # Cek kagglehub cache lebih dulu
    try:
        import kagglehub
        cache_dir = kagglehub.dataset_download("mkachuee/BloodPressureDataset")
        if os.path.exists(cache_dir):
            return cache_dir
    except Exception as e:
        print(f"Kagglehub cache tidak siap ({e}). Menggunakan direktori lokal: {local_dir}")
    
    return local_dir


if __name__ == "__main__":
    target_dir = get_kachuee_dataset_dir()
    print(f"Direktori dataset aktif: {target_dir}")
