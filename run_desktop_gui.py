#!/usr/bin/env python3
"""
Launcher untuk Aplikasi Desktop Native macOS: Shimmer3 ECG & Blood Pressure Monitor.
Laboratorium Robotika & Embedded System (Skripsi ECG-BP Estimation).
"""

import sys
import os

# Pastikan root direktori repositori ada dalam sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.gui.desktop_app import main

if __name__ == "__main__":
    main()
