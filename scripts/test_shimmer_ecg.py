#!/usr/bin/env python3
"""
Script Pengujian Sensor Shimmer3 ECG untuk macOS / Linux / Windows
Referensi: 'Penggunaan Alat Shimmer ECG.pdf' (Laboratorium Robotika & Embedded System)

Fitur:
1. Deteksi otomatis serial port Shimmer3 (khusus macOS memprioritaskan /dev/cu.*).
2. Panduan koneksi Bluetooth & penempatan elektroda Shimmer3 ECG (RA, LA, LL).
3. Handshake & membaca info perangkat (Device Name, Sampling Rate, Firmware).
4. Streaming real-time channel ECG (ADS1292R CH1 & CH2) dengan konversi ke millivolt (mV).
5. Live visualisasi terminal (ASCII sparkline) & estimasi sampling rate aktual.
6. Opsional visualisasi grafik real-time (--plot) menggunakan Matplotlib.
7. Opsional simpan ke CSV (--save) dan opsi uji coba tanpa alat fisik (--mock).
"""

import sys
import os
import time
import argparse
import signal
from collections import deque
from typing import Optional, List

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("❌ Error: 'pyserial' belum terinstall. Jalankan: pip install pyserial")
    sys.exit(1)

try:
    from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
    SHIMMER_AVAILABLE = True
except ImportError:
    SHIMMER_AVAILABLE = False


def adc_to_millivolts(adc_signal: float, gain: float = 6.0, offset: float = 0.0, vref: float = 2.42, adc_bits: int = 24) -> float:
    """
    Konversi nilai ADC mentah ADS1292R ke millivolt (mV).
    Rumus: ((adc - offset) * ((vref * 1000) / (2^(adc_bits-1) - 1))) / gain
    """
    adc_sensitivity = (vref * 1000.0) / (2 ** (adc_bits - 1) - 1)
    return ((adc_signal - offset) * adc_sensitivity) / gain


def list_available_ports() -> List[str]:
    """Mencari semua port serial yang tersedia di sistem."""
    ports = serial.tools.list_ports.comports()
    result = []
    for p in ports:
        result.append(p.device)
    
    # Di macOS, periksa juga secara langsung pola /dev/cu.* jika list_ports melewatkannya
    if sys.platform == "darwin":
        for entry in os.listdir("/dev"):
            if entry.startswith("cu.Shimmer") or entry.startswith("tty.Shimmer"):
                full_path = os.path.join("/dev", entry)
                if full_path not in result:
                    result.append(full_path)
    return result


def find_shimmer_ports() -> List[str]:
    """
    Mencari port serial yang kemungkinan adalah Shimmer3.
    Pada macOS, port biasanya bernama /dev/cu.Shimmer3-XXXX-SPP
    """
    all_ports = list_available_ports()
    shimmer_ports = []
    
    for port in all_ports:
        port_lower = port.lower()
        if "shimmer" in port_lower:
            shimmer_ports.append(port)
            
    # Di macOS utamakan /dev/cu.* dibanding /dev/tty.* karena tidak blocking saat open
    if sys.platform == "darwin":
        cu_ports = [p for p in shimmer_ports if "/cu." in p]
        if cu_ports:
            return cu_ports
            
    return shimmer_ports


def print_electrode_guide():
    """Menampilkan panduan pemasangan elektroda berdasarkan PDF referensi."""
    print("\n" + "=" * 65)
    print("📍 PANDUAN PEMASANGAN ELEKTRODA SHIMMER3 ECG (3-LEAD / 5-LEAD)")
    print("=" * 65)
    print("1. Bersihkan area kulit dengan alkohol swab / tisu.")
    print("2. Pasang elektroda gel pada titik-titik berikut:")
    print("   • RA (Putih) : Di bawah tulang selangka kanan (Right Arm)")
    print("   • LA (Hitam) : Di bawah tulang selangka kiri (Left Arm)")
    print("   • LL (Merah) : Di sekitar tulang rusuk bawah sisi kiri (Left Leg)")
    print("3. Sambungkan kancing kabel elektroda ke alat Shimmer3 ECG:")
    print("   • Kabel Putih -> Elektroda RA")
    print("   • Kabel Hitam -> Elektroda LA")
    print("   • Kabel Merah -> Elektroda LL")
    print("4. Pasang strap Shimmer3 di tubuh, pastikan sensor posisi vertikal.")
    print("=" * 65 + "\n")


def print_macos_pairing_guide():
    """Menampilkan panduan pairing Bluetooth di macOS."""
    print("\n" + "!" * 65)
    print("ℹ️  PANDUAN PAIRING BLUETOOTH DI MACOS:")
    print("!" * 65)
    print("1. Geser saklar daya Shimmer3 ke posisi 'ON' (LED hijau berkedip).")
    print("2. Buka System Settings > Bluetooth di Mac Anda.")
    print("3. Cari perangkat bernama 'Shimmer3-XXXX'")
    print("   (XXXX = 4 digit terakhir MAC address di belakang casing alat).")
    print("4. Klik 'Connect' / 'Hubungkan' (PIN default jika diminta: 1234).")
    print("5. Setelah terhubung, macOS akan membuat port serial otomatis:")
    print("   contoh: /dev/cu.Shimmer3-XXXX-SPP")
    print("!" * 65 + "\n")


def render_ascii_sparkline(val: float, min_val: float = -1.5, max_val: float = 1.5, width: int = 25) -> str:
    """Membuat representasi grafik mini ASCII sederhana di terminal."""
    clamped = max(min_val, min(max_val, val))
    normalized = (clamped - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    pos = int(normalized * (width - 1))
    mid = width // 2
    bar = [" "] * width
    bar[mid] = "|"
    bar[pos] = "█" if pos != mid else "◈"
    return "[" + "".join(bar) + "]"


class MockShimmerStreamer:
    """Simulator sinyal ECG sintetik untuk menguji script tanpa perangkat fisik."""
    def __init__(self, sample_rate: int = 512):
        self.sample_rate = sample_rate
        self.running = False
        self.callbacks = []

    def add_stream_callback(self, cb):
        self.callbacks.append(cb)

    def start_streaming(self):
        import math
        self.running = True
        t = 0.0
        dt = 1.0 / self.sample_rate
        try:
            while self.running:
                hr = 70.0  # bpm
                beat_period = 60.0 / hr
                phase = (t % beat_period) / beat_period
                
                ecg_val = 0.0
                if 0.15 <= phase <= 0.20:
                    ecg_val += 0.15 * math.sin((phase - 0.15) / 0.05 * math.pi)
                elif 0.30 <= phase <= 0.32:
                    ecg_val -= 0.15 * math.sin((phase - 0.30) / 0.02 * math.pi)
                elif 0.32 <= phase <= 0.36:
                    ecg_val += 1.25 * math.sin((phase - 0.32) / 0.04 * math.pi)
                elif 0.36 <= phase <= 0.38:
                    ecg_val -= 0.25 * math.sin((phase - 0.36) / 0.02 * math.pi)
                elif 0.45 <= phase <= 0.60:
                    ecg_val += 0.25 * math.sin((phase - 0.45) / 0.15 * math.pi)

                class MockPacket:
                    def __init__(self, mv):
                        self.channels = [
                            EChannelType.EXG_ADS1292R_1_CH1_24BIT,
                            EChannelType.EXG_ADS1292R_1_CH2_24BIT
                        ]
                        adc_sens = (2.42 * 1000.0) / (2 ** 23 - 1)
                        raw_adc = int((mv * 6.0) / adc_sens)
                        self._data = {
                            EChannelType.EXG_ADS1292R_1_CH1_24BIT: raw_adc,
                            EChannelType.EXG_ADS1292R_1_CH2_24BIT: int(raw_adc * 0.75)
                        }
                    def __getitem__(self, item):
                        return self._data[item]

                pkt = MockPacket(ecg_val)
                for cb in self.callbacks:
                    cb(pkt)
                
                t += dt
                time.sleep(dt)
        except KeyboardInterrupt:
            pass

    def stop_streaming(self):
        self.running = False


def run_test(
    port: str,
    duration: Optional[float] = None,
    save_csv: Optional[str] = None,
    baudrate: int = DEFAULT_BAUDRATE,
    mock: bool = False,
    enable_plot: bool = False
):
    print("\n" + "=" * 65)
    print(f"🚀 MEMULAI TEST SHIMMER3 ECG (PORT: {port})")
    print("=" * 65)

    csv_file = None
    if save_csv:
        csv_file = open(save_csv, "w", buffering=1)
        csv_file.write("timestamp,sample_count,ch1_raw,ch1_mv,ch2_raw,ch2_mv\n")
        print(f"💾 Menyimpan data ke file: {save_csv}")

    sample_count = 0
    start_time = None
    last_print_time = 0.0
    stop_requested = False

    # Ring buffer untuk live plot (menyimpan 5 detik terakhir ~2500 sampel)
    plot_buffer_len = 2500
    times_buf = deque(maxlen=plot_buffer_len)
    ch1_buf = deque(maxlen=plot_buffer_len)
    ch2_buf = deque(maxlen=plot_buffer_len)

    def sig_handler(sig, frame):
        nonlocal stop_requested
        print("\n\n🛑 Sinyal berhenti diterima (Ctrl+C). Menghentikan streaming...")
        stop_requested = True

    signal.signal(signal.SIGINT, sig_handler)

    def packet_callback(pkt: DataPacket):
        nonlocal sample_count, start_time, last_print_time, stop_requested
        
        now = time.time()
        if start_time is None:
            start_time = now

        sample_count += 1

        ch1_raw = 0
        ch2_raw = 0
        ch1_mv = 0.0
        ch2_mv = 0.0

        if EChannelType.EXG_ADS1292R_1_CH1_24BIT in pkt.channels:
            ch1_raw = pkt[EChannelType.EXG_ADS1292R_1_CH1_24BIT]
            ch1_mv = adc_to_millivolts(ch1_raw, gain=6.0, offset=0.0, adc_bits=24)
        elif EChannelType.EXG_ADS1292R_1_CH1_16BIT in pkt.channels:
            ch1_raw = pkt[EChannelType.EXG_ADS1292R_1_CH1_16BIT]
            ch1_mv = adc_to_millivolts(ch1_raw, gain=6.0, offset=0.0, adc_bits=16)

        if EChannelType.EXG_ADS1292R_1_CH2_24BIT in pkt.channels:
            ch2_raw = pkt[EChannelType.EXG_ADS1292R_1_CH2_24BIT]
            ch2_mv = adc_to_millivolts(ch2_raw, gain=6.0, offset=0.0, adc_bits=24)
        elif EChannelType.EXG_ADS1292R_1_CH2_16BIT in pkt.channels:
            ch2_raw = pkt[EChannelType.EXG_ADS1292R_1_CH2_16BIT]
            ch2_mv = adc_to_millivolts(ch2_raw, gain=6.0, offset=0.0, adc_bits=16)

        t_rel = now - start_time
        if enable_plot:
            times_buf.append(t_rel)
            ch1_buf.append(ch1_mv)
            ch2_buf.append(ch2_mv)

        if csv_file:
            csv_file.write(f"{t_rel:.4f},{sample_count},{ch1_raw},{ch1_mv:.4f},{ch2_raw},{ch2_mv:.4f}\n")

        # Tampilkan progress di terminal setiap 0.05s
        if now - last_print_time >= 0.05:
            last_print_time = now
            elapsed = now - start_time
            actual_fps = sample_count / elapsed if elapsed > 0 else 0.0
            spark = render_ascii_sparkline(ch1_mv, min_val=-1.0, max_val=1.5, width=25)
            sys.stdout.write(
                f"\r⏱️ {elapsed:5.1f}s | Sampel: {sample_count:6d} | Rate: {actual_fps:5.1f} Hz | "
                f"ECG CH1: {ch1_mv:+6.3f} mV {spark} | CH2: {ch2_mv:+6.3f} mV"
            )
            sys.stdout.flush()

    # Siapkan worker thread jika mode riil atau mock
    streamer = None
    shim = None
    ser = None

    if mock:
        print("💡 Berjalan dalam MODE MOCK SIMULATION (Sinyal sintetis).")
        streamer = MockShimmerStreamer(sample_rate=512)
        streamer.add_stream_callback(packet_callback)
        import threading
        stream_thread = threading.Thread(target=streamer.start_streaming, daemon=True)
        stream_thread.start()
    else:
        if not SHIMMER_AVAILABLE:
            print("❌ 'pyshimmer' library belum tersedia. Silakan jalankan: pip install pyshimmer")
            return

        print(f"🔌 Membuka serial port: {port} @ {baudrate} baud...")
        try:
            ser = serial.Serial(port, baudrate, timeout=None)
        except Exception as e:
            print(f"❌ Gagal membuka serial port {port}: {e}")
            print_macos_pairing_guide()
            return

        try:
            print("⏳ Menghubungkan ke Shimmer Bluetooth...")
            shim = ShimmerBluetooth(ser)
            shim.initialize()

            try:
                dev_name = shim.get_device_name()
                print(f"✅ Terhubung ke Perangkat: {dev_name}")
            except Exception as e:
                print(f"ℹ️ Nama perangkat: {e}")

            try:
                sampling_rate = shim.get_sampling_rate()
                print(f"📊 Sampling Rate Terkonfigurasi: {sampling_rate} Hz")
            except Exception as e:
                print(f"ℹ️ Sampling rate default info: {e}")

            shim.add_stream_callback(packet_callback)
            print("\n📡 Memulai akuisisi data streaming...")
            print("Tekan Ctrl+C kapan saja untuk berhenti.\n")
            shim.start_streaming()

        except Exception as e:
            print(f"\n❌ Terjadi kesalahan saat komunikasi dengan Shimmer: {e}")
            if ser:
                ser.close()
            return

    # Loop penanganan UI & durasi
    try:
        if enable_plot:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            fig.canvas.manager.set_window_title(f"Shimmer3 ECG Live Monitor ({port})")
            
            line1, = ax1.plot([], [], color="#1E88E5", lw=1.2, label="ECG Lead I (CH1)")
            line2, = ax2.plot([], [], color="#E53935", lw=1.2, label="ECG Lead II (CH2)")

            ax1.set_ylabel("Amplitudo (mV)")
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right")

            ax2.set_xlabel("Waktu (detik)")
            ax2.set_ylabel("Amplitudo (mV)")
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right")

            def update_chart(frame):
                if stop_requested or (duration and start_time and (time.time() - start_time) >= duration):
                    plt.close(fig)
                    return line1, line2

                if len(times_buf) > 10:
                    t_list = list(times_buf)
                    c1_list = list(ch1_buf)
                    c2_list = list(ch2_buf)

                    line1.set_data(t_list, c1_list)
                    line2.set_data(t_list, c2_list)

                    t_now = t_list[-1]
                    t_min = max(0, t_now - 5.0)
                    t_max = max(5.0, t_now)
                    ax1.set_xlim(t_min, t_max)

                    # Dynamic Y scale
                    all_c1 = c1_list[-min(len(c1_list), 500):]
                    all_c2 = c2_list[-min(len(c2_list), 500):]
                    if all_c1:
                        ax1.set_ylim(min(all_c1) - 0.2, max(all_c1) + 0.2)
                    if all_c2:
                        ax2.set_ylim(min(all_c2) - 0.2, max(all_c2) + 0.2)

                return line1, line2

            ani = animation.FuncAnimation(fig, update_chart, interval=50, blit=False)
            plt.tight_layout()
            plt.show()
        else:
            t0 = time.time()
            while not stop_requested:
                if duration and (time.time() - t0) >= duration:
                    print(f"\n⏱️ Batas waktu durasi pengujian ({duration}s) selesai.")
                    break
                time.sleep(0.05)

    finally:
        print("\n\n🧹 Menghentikan streaming dan menutup koneksi...")
        if mock and streamer:
            streamer.stop_streaming()
        if shim:
            try:
                shim.stop_streaming()
            except Exception:
                pass
            try:
                shim.shutdown()
            except Exception:
                pass
        if ser:
            try:
                ser.close()
            except Exception:
                pass
        if csv_file:
            csv_file.close()

    elapsed = (time.time() - start_time) if start_time else 0.0
    print("=" * 65)
    print("📊 RINGKASAN PENGUJIAN:")
    print(f"   • Total Durasi : {elapsed:.2f} detik")
    print(f"   • Total Sampel : {sample_count} data")
    if elapsed > 0:
        print(f"   • Rata-rata Hz : {sample_count / elapsed:.2f} samples/detik")
    if save_csv:
        print(f"   • File Hasil   : {save_csv}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Script Pengujian & Verifikasi Sensor Shimmer3 ECG (macOS / Windows / Linux)"
    )
    parser.add_argument(
        "--port", "-p",
        type=str,
        default=None,
        help="Path port serial (contoh macOS: /dev/cu.Shimmer3-XXXX-SPP atau Windows: COM11)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=None,
        help="Durasi streaming dalam detik (default: berjalan sampai ditekan Ctrl+C)"
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        default=None,
        help="Nama file CSV untuk menyimpan rekaman pengujian (contoh: ecg_test.csv)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Tampilkan visualisasi grafik ECG real-time di jendela grafis Matplotlib"
    )
    parser.add_argument(
        "--baudrate", "-b",
        type=int,
        default=DEFAULT_BAUDRATE if SHIMMER_AVAILABLE else 115200,
        help="Baudrate koneksi serial (default: DEFAULT_BAUDRATE = 115200)"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Hanya memindai dan menampilkan daftar serial port yang tersedia"
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help="Tampilkan panduan penempatan elektroda dan pairing macOS"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Uji coba menggunakan generator sinyal ECG sintetis tanpa perangkat fisik"
    )

    args = parser.parse_args()

    if args.guide:
        print_electrode_guide()
        print_macos_pairing_guide()
        return

    # Tampilkan panduan pemasangan secara singkat
    print_electrode_guide()

    # Mode scan port
    all_ports = list_available_ports()
    shimmer_ports = find_shimmer_ports()

    if args.scan:
        print("🔍 HASIL PEMINDAIAN PORT SERIAL:")
        if not all_ports:
            print("   (Tidak ada port serial yang terdeteksi)")
        else:
            for p in all_ports:
                indicator = " 🌟 [Kandidat Shimmer3]" if p in shimmer_ports else ""
                print(f"   • {p}{indicator}")
        print_macos_pairing_guide()
        return

    # Mode Mock
    if args.mock:
        run_test(
            port="MOCK_DEVICE",
            duration=args.duration,
            save_csv=args.save,
            mock=True,
            enable_plot=args.plot
        )
        return

    target_port = args.port

    if not target_port:
        if shimmer_ports:
            target_port = shimmer_ports[0]
            print(f"🎯 Port Shimmer terdeteksi otomatis: {target_port}")
            if len(shimmer_ports) > 1:
                print(f"ℹ️ Kandidat port lain yang tersedia: {shimmer_ports[1:]}")
        else:
            print("⚠️ Tidak ditemukan port Shimmer3 otomatis.")
            print("\nPort serial yang saat ini terdeteksi pada Mac Anda:")
            if all_ports:
                for i, p in enumerate(all_ports):
                    print(f"   [{i + 1}] {p}")
            else:
                print("   (Belum ada perangkat serial yang terhubung)")

            print_macos_pairing_guide()
            
            if sys.stdin.isatty() and all_ports:
                val = input("\nMasukkan nomor port atau path lengkap port (atau tekan Enter untuk keluar): ").strip()
                if not val:
                    print("Pengujian dibatalkan.")
                    return
                if val.isdigit() and 1 <= int(val) <= len(all_ports):
                    target_port = all_ports[int(val) - 1]
                else:
                    target_port = val
            else:
                print("💡 Contoh pemanggilan dengan menentukan port manual:")
                print("   python scripts/test_shimmer_ecg.py --port /dev/cu.Shimmer3-XXXX-SPP\n")
                print("💡 Atau uji coba terlebih dahulu menggunakan simulasi mock:")
                print("   python scripts/test_shimmer_ecg.py --mock\n")
                return

    run_test(
        port=target_port,
        duration=args.duration,
        save_csv=args.save,
        baudrate=args.baudrate,
        mock=False,
        enable_plot=args.plot
    )


if __name__ == "__main__":
    main()
