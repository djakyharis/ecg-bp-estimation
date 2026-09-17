"""
Server Aplikasi GUI Web untuk Estimasi Tekanan Darah Non-Invasif dari Sinyal ECG.
Menyediakan REST API dan Web Dashboard interaktif menggunakan Python standard library HTTP server.
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np
import torch

# Setup path ke direktori proyek
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.filters import iir_notch_filter, butterworth_bandpass_filter
from src.preprocessing.normalization import percentile_clip_scale
from src.models.inference import ECGBPPredictor, classify_blood_pressure_aha
from src.inspect_dataset import KachueeDatasetInspector

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
predictor = ECGBPPredictor(weights_path=BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else None)
inspector = KachueeDatasetInspector()


def generate_synthetic_ecg(duration_sec: float = 10.0, fs: float = 125.0, hr_bpm: float = 75.0) -> np.ndarray:
    """Membuat sinyal ECG sintetis dengan kontaminasi noise realistis (PLN 50 Hz & baseline drift)."""
    t = np.arange(int(duration_sec * fs)) / fs
    ecg = np.zeros_like(t)
    
    # Heartbeat period
    rr = 60.0 / hr_bpm
    beat_times = np.arange(0.2, duration_sec, rr)
    
    for bt in beat_times:
        # P-wave
        ecg += 0.15 * np.exp(-((t - (bt - 0.15)) ** 2) / (2 * (0.025 ** 2)))
        # Q-wave
        ecg -= 0.18 * np.exp(-((t - (bt - 0.04)) ** 2) / (2 * (0.010 ** 2)))
        # R-peak
        ecg += 1.20 * np.exp(-((t - bt) ** 2) / (2 * (0.018 ** 2)))
        # S-wave
        ecg -= 0.35 * np.exp(-((t - (bt + 0.04)) ** 2) / (2 * (0.012 ** 2)))
        # T-wave
        ecg += 0.30 * np.exp(-((t - (bt + 0.22)) ** 2) / (2 * (0.045 ** 2)))

    # Noise kontaminasi: Baseline wander (0.2 Hz) + Powerline 50 Hz + Gaussian EMG
    baseline_drift = 0.35 * np.sin(2 * np.pi * 0.25 * t) + 0.2 * np.cos(2 * np.pi * 0.1 * t)
    noise_50hz = 0.12 * np.sin(2 * np.pi * 50.0 * t)
    emg_noise = 0.05 * np.random.randn(len(t))
    
    raw_signal = ecg + baseline_drift + noise_50hz + emg_noise
    return raw_signal


class ECGGUIRequestHandler(SimpleHTTPRequestHandler):
    """Handler HTTP untuk menyajikan Static Dashboard dan API Inferensi."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        if parsed_path.path == "/api/sample":
            # Endpoint untuk mengambil data sinyal ECG (sintetis atau dataset)
            query = urllib.parse.parse_qs(parsed_path.query)
            sample_type = query.get("type", ["synthetic"])[0]
            
            if sample_type == "synthetic":
                hr = float(query.get("hr", [75])[0])
                raw_sig = generate_synthetic_ecg(duration_sec=10.0, fs=125.0, hr_bpm=hr)
                ground_truth = {"sbp": 120.0 + (hr - 75) * 0.5, "dbp": 80.0 + (hr - 75) * 0.3, "map": 93.3}
            else:
                raw_sig = generate_synthetic_ecg(duration_sec=10.0, fs=125.0, hr_bpm=72.0)
                ground_truth = {"sbp": 118.0, "dbp": 76.0, "map": 90.0}

            # Hitung tahapan pra-pemrosesan
            notch_sig = iir_notch_filter(raw_sig, fs=125.0, f0=50.0, q=30.0)
            bandpass_sig = butterworth_bandpass_filter(notch_sig, fs=125.0, lowcut=0.5, highcut=40.0, order=2)
            normalized_sig = percentile_clip_scale(bandpass_sig, q_low_percentile=1.0, q_high_percentile=99.0)

            payload = {
                "fs": 125.0,
                "duration": 10.0,
                "raw": raw_sig.tolist(),
                "notch": notch_sig.tolist(),
                "bandpass": bandpass_sig.tolist(),
                "normalized": normalized_sig.tolist(),
                "ground_truth": ground_truth,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        elif parsed_path.path == "/":
            self.path = "/index.html"
            return super().do_GET()

        return super().do_GET()

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        if parsed_path.path == "/api/predict":
            # Endpoint inferensi Deep Learning
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            req_json = json.loads(post_data.decode("utf-8"))
            
            raw_ecg = np.array(req_json.get("signal", []), dtype=np.float32)
            if len(raw_ecg) == 0:
                raw_ecg = generate_synthetic_ecg()

            # Jalankan prediksi end-to-end
            result = predictor.predict(raw_ecg, input_fs=125.0)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")


def run_gui_server(port: int = 8080):
    os.makedirs(STATIC_DIR, exist_ok=True)
    server_address = ("", port)
    httpd = HTTPServer(server_address, ECGGUIRequestHandler)
    print(f"================================================================")
    print(f"  Aplikasi GUI Estimasi Tekanan Darah (Deep Learning BiLSTM)")
    print(f"  Server berjalan di: http://localhost:{port}")
    print(f"================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer dihentikan.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_gui_server(port=port)
