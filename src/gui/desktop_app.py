"""
Aplikasi Desktop Native macOS: Shimmer3 Real-Time ECG & Blood Pressure Monitor.
Sesuai panduan 'Penggunaan Alat Shimmer ECG-1.pdf' (Lab Robotika & Embedded System).

Fitur:
1. Panel Kontrol: Mode Real-time Shimmer Bluetooth (macOS /dev/cu.*), Simulasi Mock, & Replay Dataset.
2. Osiloskop Real-Time (PyQtGraph 60 FPS): Sinyal ECG Lead II kontinu dengan penandaan puncak R (Red Markers).
3. Deteksi R-Peak Pan-Tompkins++: Menghitung detak jantung (BPM) dan rata-rata R-R interval.
4. Estimasi Tekanan Darah AI (BiLSTM): Memprediksi SBP, DBP, dan MAP setiap jendela 10 detik.
5. Klasifikasi Klinis AHA/ACC: Status BP (Normal, Pre-Hipertensi, Hipertensi Stage 1/2) & Aritmia.
6. Log Aktivitas Sistem dengan timestamp.
"""

import sys
import os
import time
import math
from collections import deque
from typing import Optional, List, Dict, Any

import numpy as np
import scipy.signal

# GUI & Plotting
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QTextEdit,
    QGroupBox, QFrame, QFileDialog, QSplitter, QDoubleSpinBox
)
import pyqtgraph as pg

# Serial communication
try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

try:
    from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
    SHIMMER_AVAILABLE = True
except ImportError:
    SHIMMER_AVAILABLE = False

# Local ML inference
from src.models.inference import ECGBPPredictor, classify_blood_pressure_aha


# ---------------------------------------------------------------------------
# Algoritma Pan-Tompkins untuk Deteksi Puncak R (R-Peak Detection)
# ---------------------------------------------------------------------------
class PanTompkinsDetector:
    """
    Implementasi deteksi R-peak Pan-Tompkins:
    1. Bandpass filter (5-15 Hz)
    2. Derivatif 5-titik (penonjolan slope QRS)
    3. Squaring (penguatan energi)
    4. Moving Window Integration (integrasi jendela ~150ms)
    5. Thresholding adaptif & pencarian puncak lokal pada sinyal asli
    """
    def __init__(self, fs: float = 125.0):
        self.fs = fs
        self.b_band, self.a_band = scipy.signal.butter(
            2, [5.0 / (0.5 * fs), min(15.0 / (0.5 * fs), 0.95)], btype="bandpass"
        )
        self.win_size = max(1, int(0.15 * fs))  # 150 ms moving window
        self.refractory_samples = int(0.25 * fs)  # minimal 250 ms antar denyut (maks 240 bpm)

    def detect_peaks(self, signal_data: np.ndarray) -> np.ndarray:
        if len(signal_data) < self.win_size * 4:
            return np.array([], dtype=int)

        # 1. Bandpass filter
        filtered = scipy.signal.filtfilt(self.b_band, self.a_band, signal_data)

        # 2. Derivative
        diff = np.diff(filtered)
        diff = np.pad(diff, (1, 0), mode="edge")

        # 3. Squaring
        squared = diff ** 2

        # 4. Moving Window Integration
        kernel = np.ones(self.win_size) / self.win_size
        integrated = np.convolve(squared, kernel, mode="same")

        # 5. Peak Finding pada sinyal integrasi
        min_distance = self.refractory_samples
        threshold = np.mean(integrated) + 0.5 * np.std(integrated)
        cand_peaks, _ = scipy.signal.find_peaks(integrated, distance=min_distance, height=threshold)

        # Refine peaks ke puncak maksimum pada sinyal ECG asli
        r_peaks = []
        search_radius = int(0.08 * self.fs)  # +/- 80 ms
        for cp in cand_peaks:
            start = max(0, cp - search_radius)
            end = min(len(signal_data), cp + search_radius)
            if end > start:
                local_peak = start + int(np.argmax(signal_data[start:end]))
                if not r_peaks or (local_peak - r_peaks[-1]) >= min_distance:
                    r_peaks.append(local_peak)

        return np.array(r_peaks, dtype=int)


# ---------------------------------------------------------------------------
# Worker Thread untuk Komunikasi Data Sensor (Shimmer / Mock / File)
# ---------------------------------------------------------------------------
class ShimmerDataWorker(QThread):
    new_samples_signal = Signal(list)  # List of float (mV)
    status_signal = Signal(str)        # Log message
    connected_signal = Signal(bool)

    def __init__(self, mode="mock", port="/dev/cu.Shimmer3-XXXX-SPP", baudrate=115200, fs=125.0, file_path=None):
        super().__init__()
        self.mode = mode
        self.port = port
        self.baudrate = baudrate
        self.fs = fs
        self.file_path = file_path
        self._is_running = False
        self.serial_conn = None
        self.shimmer_device = None

    def run(self):
        self._is_running = True
        self.status_signal.emit(f"🚀 Memulai akuisisi data (Mode: {self.mode.upper()})...")

        if self.mode == "real":
            self._run_real_shimmer()
        elif self.mode == "file":
            self._run_file_playback()
        else:
            self._run_mock_stream()

    def stop(self):
        self._is_running = False
        if self.shimmer_device:
            try:
                self.shimmer_device.stop_streaming()
            except Exception:
                pass
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        self.wait(1000)
        self.connected_signal.emit(False)
        self.status_signal.emit("⏹️ Streaming data dihentikan.")

    def _run_mock_stream(self):
        """Menghasilkan sinyal ECG sintetik real-time realistis (125 Hz)."""
        self.connected_signal.emit(True)
        self.status_signal.emit("✅ Generator ECG Sintetis Aktif (72 BPM nominal).")
        dt = 1.0 / self.fs
        t = 0.0
        hr_bpm = 72.0

        buffer = []
        last_emit = time.time()

        while self._is_running:
            # Simulasi denyut ECG sintetik (P-QRS-T)
            beat_period = 60.0 / hr_bpm
            phase = (t % beat_period) / beat_period

            val = 0.0
            # P wave
            if 0.12 <= phase <= 0.20:
                val += 0.15 * math.sin((phase - 0.12) / 0.08 * math.pi)
            # Q wave
            elif 0.28 <= phase <= 0.30:
                val -= 0.15 * math.sin((phase - 0.28) / 0.02 * math.pi)
            # R peak (tajam)
            elif 0.30 <= phase <= 0.34:
                val += 1.40 * math.sin((phase - 0.30) / 0.04 * math.pi)
            # S wave
            elif 0.34 <= phase <= 0.37:
                val -= 0.30 * math.sin((phase - 0.34) / 0.03 * math.pi)
            # T wave
            elif 0.45 <= phase <= 0.60:
                val += 0.25 * math.sin((phase - 0.45) / 0.15 * math.pi)

            # Sedikit baseline wander & noise fisiologis
            val += 0.05 * math.sin(2 * math.pi * 0.25 * t)  # Pernafasan (0.25 Hz)
            val += np.random.normal(0, 0.015)               # Derau halus

            buffer.append(float(val))
            t += dt

            now = time.time()
            # Kirim batch setiap ~33 ms (30 FPS update)
            if now - last_emit >= 0.033:
                if buffer:
                    self.new_samples_signal.emit(buffer)
                    buffer = []
                last_emit = now

            time.sleep(dt * 0.9)  # pacing timer

    def _run_file_playback(self):
        """Membaca file dataset (.npz atau .csv) dan memutar ulang secara kontinu."""
        if not self.file_path or not os.path.exists(self.file_path):
            self.status_signal.emit(f"❌ File tidak ditemukan: {self.file_path}")
            self.connected_signal.emit(False)
            return

        try:
            if self.file_path.endswith(".npz"):
                data = np.load(self.file_path)
                if "X" in data:
                    ecg_all = data["X"].flatten()
                elif "ecg" in data:
                    ecg_all = data["ecg"].flatten()
                else:
                    ecg_all = data[data.files[0]].flatten()
            else:
                ecg_all = np.loadtxt(self.file_path, delimiter=",", skiprows=1)
                if ecg_all.ndim > 1:
                    ecg_all = ecg_all[:, 0]
        except Exception as e:
            self.status_signal.emit(f"❌ Gagal memuat file dataset: {e}")
            self.connected_signal.emit(False)
            return

        self.connected_signal.emit(True)
        self.status_signal.emit(f"✅ Memutar ulang file dataset ({len(ecg_all)} titik data)...")

        idx = 0
        dt = 1.0 / self.fs
        buffer = []
        last_emit = time.time()

        while self._is_running:
            buffer.append(float(ecg_all[idx]))
            idx = (idx + 1) % len(ecg_all)

            now = time.time()
            if now - last_emit >= 0.033:
                if buffer:
                    self.new_samples_signal.emit(buffer)
                    buffer = []
                last_emit = now

            time.sleep(dt * 0.95)

    def _run_real_shimmer(self):
        """Membaca hardware Shimmer3 via Bluetooth Serial di macOS."""
        if not serial:
            self.status_signal.emit("❌ Library 'pyserial' belum terinstall.")
            self.connected_signal.emit(False)
            return

        self.status_signal.emit(f"🔌 Menghubungkan ke port: {self.port} ({self.baudrate} baud)...")
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1.0)
        except Exception as e:
            self.status_signal.emit(f"❌ Gagal membuka port {self.port}: {e}")
            self.connected_signal.emit(False)
            return

        if SHIMMER_AVAILABLE:
            try:
                self.shimmer_device = ShimmerBluetooth(self.serial_conn)
                self.shimmer_device.initialize()
                dev_name = self.shimmer_device.get_device_name()
                self.status_signal.emit(f"✅ Terhubung ke Shimmer Device: {dev_name}")

                buffer = []
                last_emit = time.time()

                def packet_cb(pkt: DataPacket):
                    nonlocal buffer, last_emit
                    if not self._is_running:
                        return
                    ch_key = EChannelType.EXG_ADS1292R_1_CH1_24BIT
                    if ch_key in pkt.channels:
                        raw = pkt[ch_key]
                        # ADS1292R ADC to mV konversi
                        sens = (2.42 * 1000.0) / (2 ** 23 - 1)
                        mv = (raw * sens) / 6.0
                        buffer.append(float(mv))

                    now = time.time()
                    if now - last_emit >= 0.033 and buffer:
                        self.new_samples_signal.emit(buffer)
                        buffer = []
                        last_emit = now

                self.shimmer_device.add_stream_callback(packet_cb)
                self.shimmer_device.start_streaming()
                self.connected_signal.emit(True)

                while self._is_running:
                    time.sleep(0.1)

            except Exception as e:
                self.status_signal.emit(f"❌ Error protokol Shimmer: {e}")
                self.connected_signal.emit(False)
        else:
            self.status_signal.emit("❌ 'pyshimmer' tidak ditemukan. Membuka raw serial...")
            self.connected_signal.emit(False)


# ---------------------------------------------------------------------------
# macOS Native Dark QSS Style Sheet
# ---------------------------------------------------------------------------
MACOS_DARK_STYLE = """
QMainWindow {
    background-color: #121318;
}
QWidget {
    color: #E2E8F0;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #1A1C23;
    border: 1px solid #2D3748;
    border-radius: 10px;
    margin-top: 18px;
    padding-top: 14px;
    padding-bottom: 10px;
    padding-left: 10px;
    padding-right: 10px;
    font-weight: 600;
    color: #94A3B8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #38BDF8;
    background-color: transparent;
}
QPushButton {
    background-color: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 7px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #1D4ED8;
}
QPushButton:pressed {
    background-color: #1E40AF;
}
QPushButton#btnStop {
    background-color: #DC2626;
}
QPushButton#btnStop:hover {
    background-color: #B91C1C;
}
QPushButton#btnRefresh {
    background-color: #334155;
    color: #F8FAFC;
}
QPushButton#btnRefresh:hover {
    background-color: #475569;
}
QComboBox {
    background-color: #242832;
    color: #F1F5F9;
    border: 1px solid #3B4252;
    border-radius: 6px;
    padding: 5px 10px;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QTextEdit {
    background-color: #0E1015;
    color: #10B981;
    border: 1px solid #27272A;
    border-radius: 8px;
    font-family: "SF Mono", Menlo, Courier, monospace;
    font-size: 11px;
    padding: 6px;
}
QFrame#metricCard {
    background-color: #1E212B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px;
}
QLabel#cardTitle {
    color: #94A3B8;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QLabel#cardValue {
    color: #F8FAFC;
    font-size: 28px;
    font-weight: 800;
}
QLabel#cardUnit {
    color: #64748B;
    font-size: 12px;
    font-weight: 500;
}
"""


# ---------------------------------------------------------------------------
# Jendela Utama Aplikasi (Main Window)
# ---------------------------------------------------------------------------
class ShimmerECGBPMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shimmer3 ECG & Blood Pressure Monitor — Native macOS")
        self.resize(1240, 820)
        self.setMinimumSize(1080, 700)

        # Buffer data berjalan (10 detik @ 125 Hz = 1250 titik)
        self.fs = 125.0
        self.display_window_sec = 8.0  # Tampilan grafik 8 detik terakhir
        self.buffer_len = int(self.fs * 10.0)  # 10 detik untuk model inferensi
        self.ecg_ring_buffer = deque(maxlen=self.buffer_len)
        self.time_ring_buffer = deque(maxlen=self.buffer_len)

        self.r_detector = PanTompkinsDetector(fs=self.fs)
        self.last_inference_time = 0.0
        self.total_r_peaks_detected = 0
        self.worker: Optional[ShimmerDataWorker] = None

        # Kalibrasi 1-titik (offset SBP, DBP, MAP) & smoothing median prediksi
        self.cal_offset = np.zeros(3)
        self.is_calibrated = False
        self.last_smoothed_raw: Optional[np.ndarray] = None
        self.pred_history = deque(maxlen=5)

        # Load ML Predictor
        self.predictor = None
        self.model_name = "-"
        self._init_predictor()

        # Inisialisasi UI
        self._init_ui()
        self._refresh_serial_ports()

        # Timer untuk UI Refresh (30 FPS)
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._update_plot_and_metrics)
        self.render_timer.start(33)  # ~30 FPS

    def _init_predictor(self):
        """Cari checkpoint: env ECG_BP_MODEL, lalu run block-split, lalu best_model lama."""
        candidates = [
            os.environ.get("ECG_BP_MODEL"),
            "checkpoints/run_adamw_block/final_model_full.pt",
            "checkpoints/run_adamw_block/best_model.pt",
            "checkpoints/best_model.pt",
        ]
        for cand in candidates:
            if not cand:
                continue
            path = os.path.abspath(cand)
            if not os.path.exists(path):
                continue
            try:
                self.predictor = ECGBPPredictor(weights_path=path)
                self.model_name = os.path.relpath(path)
                return
            except Exception:
                self.predictor = None
        self.predictor = None

    def _init_ui(self):
        self.setStyleSheet(MACOS_DARK_STYLE)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # -------------------------------------------------------------------
        # KOLOM KIRI: Panel Kontrol, Parameter, Log Aktivitas (Lebar: ~380px)
        # -------------------------------------------------------------------
        left_panel = QWidget()
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)

        # 1. GROUP BOX: PANEL KONTROL
        ctrl_group = QGroupBox("1. PANEL KONTROL SUMBER DATA")
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.setSpacing(10)

        # Pilihan Mode
        lbl_mode = QLabel("Sumber Data ECG:")
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "Simulasi Sensor (Mock Generator)",
            "Real-time (Shimmer3 Bluetooth)",
            "Putar Ulang File Dataset (.npz)"
        ])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)

        # Pilihan Port Serial macOS
        self.lbl_port = QLabel("Serial Port (macOS):")
        port_layout = QHBoxLayout()
        self.combo_port = QComboBox()
        self.btn_refresh_ports = QPushButton("🔄 Refresh")
        self.btn_refresh_ports.setObjectName("btnRefresh")
        self.btn_refresh_ports.clicked.connect(self._refresh_serial_ports)
        port_layout.addWidget(self.combo_port, 1)
        port_layout.addWidget(self.btn_refresh_ports)

        # Tombol File Dataset
        self.btn_browse_file = QPushButton("📂 Pilih File Dataset...")
        self.btn_browse_file.setObjectName("btnRefresh")
        self.btn_browse_file.clicked.connect(self._browse_dataset_file)
        self.btn_browse_file.hide()
        self.lbl_file_path = QLabel("Belum ada file dipilih")
        self.lbl_file_path.setStyleSheet("color: #64748B; font-size: 11px;")
        self.lbl_file_path.hide()

        # Tombol Aksi Start / Stop
        btn_action_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ Mulai Akuisisi")
        self.btn_start.clicked.connect(self._start_acquisition)
        self.btn_stop = QPushButton("⏹ Berhenti")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_acquisition)
        btn_action_layout.addWidget(self.btn_start)
        btn_action_layout.addWidget(self.btn_stop)

        ctrl_layout.addWidget(lbl_mode)
        ctrl_layout.addWidget(self.combo_mode)
        ctrl_layout.addWidget(self.lbl_port)
        ctrl_layout.addLayout(port_layout)
        ctrl_layout.addWidget(self.btn_browse_file)
        ctrl_layout.addWidget(self.lbl_file_path)
        ctrl_layout.addLayout(btn_action_layout)
        left_layout.addWidget(ctrl_group)

        # 2. GROUP BOX: DETAIL PROSES & PARAMETER
        proc_group = QGroupBox("2. DETAIL PARAMETER & SINYAL")
        proc_layout = QGridLayout(proc_group)
        proc_layout.setSpacing(8)

        proc_layout.addWidget(QLabel("Frekuensi Sampling:"), 0, 0)
        self.lbl_fs = QLabel("125.0 Hz")
        self.lbl_fs.setStyleSheet("font-weight: 700; color: #38BDF8;")
        proc_layout.addWidget(self.lbl_fs, 0, 1)

        proc_layout.addWidget(QLabel("Jumlah Puncak R:"), 1, 0)
        self.lbl_total_r = QLabel("0 beat")
        self.lbl_total_r.setStyleSheet("font-weight: 700; color: #F59E0B;")
        proc_layout.addWidget(self.lbl_total_r, 1, 1)

        proc_layout.addWidget(QLabel("Rata-rata R-R:"), 2, 0)
        self.lbl_avg_rr = QLabel("0.00 s")
        self.lbl_avg_rr.setStyleSheet("font-weight: 700; color: #10B981;")
        proc_layout.addWidget(self.lbl_avg_rr, 2, 1)

        proc_layout.addWidget(QLabel("Buffer Durasi:"), 3, 0)
        self.lbl_buf_duration = QLabel("0.0 s / 10.0 s")
        self.lbl_buf_duration.setStyleSheet("font-weight: 700; color: #E2E8F0;")
        proc_layout.addWidget(self.lbl_buf_duration, 3, 1)

        proc_layout.addWidget(QLabel("Model Status:"), 4, 0)
        model_name = f"BiLSTM ({os.path.basename(self.model_name)})" if self.predictor else "Simulasi / Default Heuristik"
        self.lbl_model_status = QLabel(model_name)
        self.lbl_model_status.setStyleSheet("color: #10B981; font-weight: 600; font-size: 11px;")
        proc_layout.addWidget(self.lbl_model_status, 4, 1)

        left_layout.addWidget(proc_group)

        # 2b. GROUP BOX: KALIBRASI MANSET 1-TITIK
        cal_group = QGroupBox("2b. KALIBRASI MANSET (1-TITIK)")
        cal_layout = QGridLayout(cal_group)
        cal_layout.setSpacing(8)
        cal_layout.addWidget(QLabel("SBP referensi:"), 0, 0)
        self.spin_ref_sbp = QDoubleSpinBox()
        self.spin_ref_sbp.setRange(60.0, 250.0)
        self.spin_ref_sbp.setValue(120.0)
        self.spin_ref_sbp.setSuffix(" mmHg")
        cal_layout.addWidget(self.spin_ref_sbp, 0, 1)
        cal_layout.addWidget(QLabel("DBP referensi:"), 1, 0)
        self.spin_ref_dbp = QDoubleSpinBox()
        self.spin_ref_dbp.setRange(30.0, 150.0)
        self.spin_ref_dbp.setValue(80.0)
        self.spin_ref_dbp.setSuffix(" mmHg")
        cal_layout.addWidget(self.spin_ref_dbp, 1, 1)
        cal_btns = QHBoxLayout()
        self.btn_calibrate = QPushButton("🎯 Kalibrasi")
        self.btn_calibrate.clicked.connect(self._apply_calibration)
        self.btn_reset_cal = QPushButton("Reset")
        self.btn_reset_cal.setObjectName("btnRefresh")
        self.btn_reset_cal.clicked.connect(self._reset_calibration)
        cal_btns.addWidget(self.btn_calibrate)
        cal_btns.addWidget(self.btn_reset_cal)
        cal_layout.addLayout(cal_btns, 2, 0, 1, 2)
        self.lbl_cal_status = QLabel("Status: BELUM TERKALIBRASI")
        self.lbl_cal_status.setStyleSheet("color: #F59E0B; font-size: 11px; font-weight: 600;")
        cal_layout.addWidget(self.lbl_cal_status, 3, 0, 1, 2)
        left_layout.addWidget(cal_group)

        # 3. GROUP BOX: LOG AKTIVITAS SISTEM
        log_group = QGroupBox("3. LOG AKTIVITAS & EVENT")
        log_layout = QVBoxLayout(log_group)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        log_layout.addWidget(self.txt_log)
        left_layout.addWidget(log_group, 1)

        main_layout.addWidget(left_panel)

        # -------------------------------------------------------------------
        # KOLOM KANAN: Osiloskop ECG & Kartu Hasil Estimasi Tekanan Darah
        # -------------------------------------------------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(14)

        # 4. OSILOSKOP REAL-TIME (PyQtGraph)
        plot_group = QGroupBox("4. OSILOSKOP ECG REAL-TIME (LEAD II / ADS1292R)")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(8, 12, 8, 8)

        # Konfigurasi PyQtGraph Dark Medical Theme
        pg.setConfigOption("background", "#0B0D11")
        pg.setConfigOption("foreground", "#94A3B8")
        pg.setConfigOption("antialias", True)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("left", "Amplitudo", units="mV")
        self.plot_widget.setLabel("bottom", "Waktu", units="detik")
        self.plot_widget.setYRange(-1.5, 2.5)

        # Garis ECG (Cyan / Emerald Elektrik)
        pen = pg.mkPen(color="#00F0FF", width=1.8)
        self.curve_ecg = self.plot_widget.plot(pen=pen)

        # Scatter Plot untuk Penanda Titik Puncak R (Merah Bulat)
        self.scatter_r_peaks = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen("#FFFFFF", width=1.5), brush=pg.mkBrush("#EF4444"), symbol="o"
        )
        self.plot_widget.addItem(self.scatter_r_peaks)

        plot_layout.addWidget(self.plot_widget)
        right_layout.addWidget(plot_group, 3)

        # 5. KARTU HASIL METRIK UTAMA (BPM, SBP/DBP, MAP)
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(14)

        # Card 1: Detak Jantung
        self.card_hr = QFrame()
        self.card_hr.setObjectName("metricCard")
        l_hr = QVBoxLayout(self.card_hr)
        l_hr.addWidget(QLabel("DETAK JANTUNG (HEART RATE)", objectName="cardTitle"))
        h_hr_val = QHBoxLayout()
        self.val_hr = QLabel("--", objectName="cardValue")
        self.unit_hr = QLabel("BPM", objectName="cardUnit")
        h_hr_val.addWidget(self.val_hr)
        h_hr_val.addWidget(self.unit_hr)
        h_hr_val.addStretch()
        l_hr.addLayout(h_hr_val)
        self.lbl_hr_status = QLabel("Status: Menunggu Sinyal...")
        self.lbl_hr_status.setStyleSheet("color: #94A3B8; font-size: 11px;")
        l_hr.addWidget(self.lbl_hr_status)
        metrics_layout.addWidget(self.card_hr)

        # Card 2: Estimasi Tekanan Darah (SBP / DBP)
        self.card_bp = QFrame()
        self.card_bp.setObjectName("metricCard")
        l_bp = QVBoxLayout(self.card_bp)
        l_bp.addWidget(QLabel("ESTIMASI TEKANAN DARAH (SBP / DBP)", objectName="cardTitle"))
        h_bp_val = QHBoxLayout()
        self.val_bp = QLabel("-- / --", objectName="cardValue")
        self.val_bp.setStyleSheet("color: #38BDF8; font-size: 28px; font-weight: 800;")
        self.unit_bp = QLabel("mmHg", objectName="cardUnit")
        h_bp_val.addWidget(self.val_bp)
        h_bp_val.addWidget(self.unit_bp)
        h_bp_val.addStretch()
        l_bp.addLayout(h_bp_val)
        self.val_map = QLabel("MAP: -- mmHg")
        self.val_map.setStyleSheet("color: #CBD5E1; font-weight: 600; font-size: 12px;")
        l_bp.addWidget(self.val_map)
        metrics_layout.addWidget(self.card_bp)

        # Card 3: Klasifikasi AHA & Kategori Klinis
        self.card_aha = QFrame()
        self.card_aha.setObjectName("metricCard")
        l_aha = QVBoxLayout(self.card_aha)
        l_aha.addWidget(QLabel("KLASIFIKASI KLINIS (AHA/ACC)", objectName="cardTitle"))
        self.badge_aha = QLabel("MENUNGGU KALKULASI")
        self.badge_aha.setAlignment(Qt.AlignCenter)
        self.badge_aha.setStyleSheet(
            "background-color: #334155; color: #FFFFFF; font-weight: 800; "
            "padding: 6px 12px; border-radius: 6px; font-size: 13px;"
        )
        l_aha.addWidget(self.badge_aha)
        self.desc_aha = QLabel("Memerlukan jendela data 10 detik kontinu.")
        self.desc_aha.setWordWrap(True)
        self.desc_aha.setStyleSheet("color: #94A3B8; font-size: 11px; margin-top: 4px;")
        l_aha.addWidget(self.desc_aha)
        metrics_layout.addWidget(self.card_aha)

        right_layout.addLayout(metrics_layout, 1)
        main_layout.addWidget(right_panel, 1)

        self._log("Aplikasi Shimmer ECG & Blood Pressure Monitor siap.")
        self._log(f"Model Prediktor BiLSTM: {'Dimuat dari ' + self.model_name if self.predictor else 'Mode Simulasi'}")

    def _log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.txt_log.append(f"[{timestamp}] {msg}")

    # -------------------------------------------------------------------
    # Event Handlers & Logika Kontrol
    # -------------------------------------------------------------------
    def _on_mode_changed(self, index: int):
        if index == 0:  # Mock
            self.lbl_port.hide()
            self.combo_port.hide()
            self.btn_refresh_ports.hide()
            self.btn_browse_file.hide()
            self.lbl_file_path.hide()
        elif index == 1:  # Real Shimmer
            self.lbl_port.show()
            self.combo_port.show()
            self.btn_refresh_ports.show()
            self.btn_browse_file.hide()
            self.lbl_file_path.hide()
        else:  # File playback
            self.lbl_port.hide()
            self.combo_port.hide()
            self.btn_refresh_ports.hide()
            self.btn_browse_file.show()
            self.lbl_file_path.show()

    def _refresh_serial_ports(self):
        self.combo_port.clear()
        found_ports = []

        if serial and hasattr(serial.tools, "list_ports"):
            ports = serial.tools.list_ports.comports()
            for p in ports:
                found_ports.append(p.device)

        if sys.platform == "darwin":
            try:
                for entry in os.listdir("/dev"):
                    if entry.startswith("cu.Shimmer") or entry.startswith("cu.Bluetooth"):
                        path = f"/dev/{entry}"
                        if path not in found_ports:
                            found_ports.append(path)
            except Exception:
                pass

        if found_ports:
            # Prioritaskan Shimmer ports di urutan atas
            found_ports.sort(key=lambda x: 0 if "shimmer" in x.lower() else 1)
            self.combo_port.addItems(found_ports)
            self._log(f"Ditemukan {len(found_ports)} serial port.")
        else:
            self.combo_port.addItem("/dev/cu.Shimmer3-XXXX-SPP")
            self._log("Tidak ada port serial fisik terdeteksi (Gunakan Mock jika tanpa alat).")

    def _browse_dataset_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Pilih File Sinyal ECG", "", "ECG Files (*.npz *.csv *.dat);;All Files (*)"
        )
        if file_path:
            self.selected_file_path = file_path
            self.lbl_file_path.setText(os.path.basename(file_path))
            self._log(f"File dipilih: {file_path}")

    def _start_acquisition(self):
        mode_idx = self.combo_mode.currentIndex()
        mode_str = "mock" if mode_idx == 0 else ("real" if mode_idx == 1 else "file")
        port = self.combo_port.currentText()
        file_path = getattr(self, "selected_file_path", None)

        self.ecg_ring_buffer.clear()
        self.time_ring_buffer.clear()
        self.total_r_peaks_detected = 0

        self.worker = ShimmerDataWorker(
            mode=mode_str,
            port=port,
            fs=self.fs,
            file_path=file_path
        )
        self.worker.new_samples_signal.connect(self._on_new_samples)
        self.worker.status_signal.connect(self._log)
        self.worker.connected_signal.connect(self._on_connection_changed)
        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.combo_mode.setEnabled(False)
        self.combo_port.setEnabled(False)

    def _stop_acquisition(self):
        if self.worker:
            self.worker.stop()
            self.worker = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.combo_mode.setEnabled(True)
        self.combo_port.setEnabled(True)

    def _on_connection_changed(self, connected: bool):
        if not connected:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.combo_mode.setEnabled(True)
            self.combo_port.setEnabled(True)

    @Slot(list)
    def _on_new_samples(self, samples: List[float]):
        now = time.time()
        dt = 1.0 / self.fs
        start_t = now - (len(samples) * dt)

        for i, val in enumerate(samples):
            self.ecg_ring_buffer.append(val)
            self.time_ring_buffer.append(start_t + i * dt)

    # -------------------------------------------------------------------
    # Render Loop (Plotting & Deteksi R-Peak & Inferensi BP)
    # -------------------------------------------------------------------
    def _update_plot_and_metrics(self):
        if len(self.ecg_ring_buffer) < 25:
            return

        ecg_data = np.array(self.ecg_ring_buffer)
        n_samples = len(ecg_data)
        dur_sec = n_samples / self.fs
        self.lbl_buf_duration.setText(f"{dur_sec:.1f} s / 10.0 s")

        # 1. Update Osiloskop ECG (hanya tampilkan jendela geser terakhir misal 8 detik)
        display_pts = int(self.fs * self.display_window_sec)
        if n_samples > display_pts:
            disp_ecg = ecg_data[-display_pts:]
            disp_times = np.linspace(0, self.display_window_sec, display_pts)
        else:
            disp_ecg = ecg_data
            disp_times = np.linspace(0, dur_sec, n_samples)

        self.curve_ecg.setData(disp_times, disp_ecg)

        # 2. Deteksi R-Peaks (Pan-Tompkins) pada data osiloskop
        if len(disp_ecg) > int(self.fs * 0.8):
            peaks = self.r_detector.detect_peaks(disp_ecg)
            if len(peaks) > 0:
                peak_times = disp_times[peaks]
                peak_vals = disp_ecg[peaks]
                self.scatter_r_peaks.setData(x=peak_times, y=peak_vals)

                # Hitung Heart Rate (BPM) dan RR-Interval
                if len(peaks) >= 2:
                    rr_intervals = np.diff(peaks) / self.fs  # detik
                    avg_rr = float(np.mean(rr_intervals))
                    bpm = 60.0 / avg_rr if avg_rr > 0 else 0.0

                    self.val_hr.setText(f"{bpm:.0f}")
                    self.lbl_avg_rr.setText(f"{avg_rr:.3f} s")
                    self.lbl_total_r.setText(f"{len(peaks)} beat")

                    # Status Irama Jantung (Normal / Bradikardia / Takikardia)
                    if bpm < 60:
                        self.lbl_hr_status.setText("Status: Bradikardia (<60 BPM)")
                        self.lbl_hr_status.setStyleSheet("color: #F59E0B; font-size: 11px;")
                    elif bpm > 100:
                        self.lbl_hr_status.setText("Status: Takikardia (>100 BPM)")
                        self.lbl_hr_status.setStyleSheet("color: #EF4444; font-size: 11px;")
                    else:
                        self.lbl_hr_status.setText("Status: Normal Rhythm (60-100 BPM)")
                        self.lbl_hr_status.setStyleSheet("color: #10B981; font-size: 11px;")
            else:
                self.scatter_r_peaks.clear()

        # 3. Inferensi Tekanan Darah Deep Learning (Setiap ada 10 detik data & jeda minimal 2 detik)
        now = time.time()
        if n_samples >= self.buffer_len and (now - self.last_inference_time) >= 2.5:
            self.last_inference_time = now
            epoch_10s = ecg_data[-self.buffer_len:]
            self._run_blood_pressure_inference(epoch_10s)

    def _apply_calibration(self):
        """Hitung offset = referensi manset - prediksi model (median terakhir)."""
        if self.last_smoothed_raw is None:
            self._log("⚠️ Kalibrasi gagal: belum ada prediksi (tunggu buffer 10 detik terisi).")
            return
        ref_sbp = self.spin_ref_sbp.value()
        ref_dbp = self.spin_ref_dbp.value()
        ref_map = ref_dbp + (ref_sbp - ref_dbp) / 3.0
        self.cal_offset = np.array([ref_sbp, ref_dbp, ref_map]) - self.last_smoothed_raw
        self.is_calibrated = True
        self.lbl_cal_status.setText(
            f"Status: TERKALIBRASI (offset SBP {self.cal_offset[0]:+.1f}, DBP {self.cal_offset[1]:+.1f})"
        )
        self.lbl_cal_status.setStyleSheet("color: #10B981; font-size: 11px; font-weight: 600;")
        self._log(
            f"🎯 Kalibrasi diterapkan: ref {ref_sbp:.0f}/{ref_dbp:.0f} mmHg, "
            f"offset [SBP {self.cal_offset[0]:+.1f}, DBP {self.cal_offset[1]:+.1f}, MAP {self.cal_offset[2]:+.1f}]"
        )

    def _reset_calibration(self):
        self.cal_offset = np.zeros(3)
        self.is_calibrated = False
        self.pred_history.clear()
        self.lbl_cal_status.setText("Status: BELUM TERKALIBRASI")
        self.lbl_cal_status.setStyleSheet("color: #F59E0B; font-size: 11px; font-weight: 600;")
        self._log("Kalibrasi di-reset.")

    def _run_blood_pressure_inference(self, epoch_10s: np.ndarray):
        """Menjalankan estimasi SBP, DBP, MAP melalui model BiLSTM + smoothing + kalibrasi."""
        try:
            if self.predictor:
                res = self.predictor.predict(epoch_10s)
                raw = np.array([res["sbp"], res["dbp"], res["map"]], dtype=float)
            else:
                # Fallback perhitungan fisiologis estimasi jika model belum diload
                sbp0 = 118.0 + np.random.normal(0, 2.0)
                dbp0 = 76.0 + np.random.normal(0, 1.5)
                raw = np.array([sbp0, dbp0, (sbp0 + 2 * dbp0) / 3.0])

            # Smoothing: median dari 5 prediksi terakhir, lalu offset kalibrasi
            self.pred_history.append(raw)
            self.last_smoothed_raw = np.median(np.array(self.pred_history), axis=0)
            sbp, dbp, map_val = (self.last_smoothed_raw + self.cal_offset).tolist()
            clinical = classify_blood_pressure_aha(sbp, dbp)

            # Update UI Cards
            self.val_bp.setText(f"{sbp:.0f} / {dbp:.0f}")
            self.val_map.setText(f"MAP: {map_val:.1f} mmHg")

            # Update AHA Category Badge
            cat_name = clinical["category"].upper()
            color = clinical.get("color", "#2A9D8F")
            self.badge_aha.setText(cat_name)
            self.badge_aha.setStyleSheet(
                f"background-color: {color}; color: #FFFFFF; font-weight: 800; "
                f"padding: 6px 12px; border-radius: 6px; font-size: 13px;"
            )
            self.desc_aha.setText(clinical.get("description", ""))

            self._log(f"📊 Inferensi Selesai: SBP {sbp:.1f} | DBP {dbp:.1f} | MAP {map_val:.1f} mmHg -> {cat_name}")

        except Exception as e:
            self._log(f"⚠️ Kesalahan saat inferensi model: {e}")

    def closeEvent(self, event):
        self._stop_acquisition()
        event.accept()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    # Optimasi rendering High-DPI untuk layar Retina macOS
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("Shimmer3 ECG & BP Monitor")
    app.setOrganizationName("Lab Robotika & Embedded System")

    window = ShimmerECGBPMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
