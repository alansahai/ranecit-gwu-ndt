"""
AcoustiScan NDT — Acoustic Non-Destructive Testing
===================================================
Hardware: Arduino + 6× KY-038 sound sensors (A0-A5 / D2-D7)
Protocol: 9 space-separated values @ 50 Hz via Serial
  Val 1-6 : raw sensor readings (0-1023)
  Val 7   : weighted-average fusion
  Val 8   : dominant-source index
  Val 9   : majority-vote × 1023

Workflow
--------
1. Connect to Arduino COM port.
2. Set the number of inspection points on the part.
3. TRAINING  — place a known-good part, excite each point with a speaker
               chirp, capture sensor data, compute FFT → store baseline.
4. TESTING   — place the part under test, excite each point, compare FFT
               to baseline → PASS / FAIL per point.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
import os
import serial
import serial.tools.list_ports
import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
import sounddevice as sd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── Constants ────────────────────────────────────────
SERIAL_BAUD       = 115_200
SAMPLE_RATE_HZ    = 50          # Arduino sends data at 50 Hz (delay 20 ms)
RECORD_SECONDS    = 3           # capture window per point
CHIRP_DURATION    = 2.0         # speaker chirp length (s)
CHIRP_F_START     = 200         # Hz
CHIRP_F_END       = 4_000       # Hz
AUDIO_SAMPLE_RATE = 44_100
FREQ_TOLERANCE    = 0.12        # ±12 % tolerance for pass/fail
NUM_SENSORS       = 6
THEME = {
    "bg":       "#1A1D2E",
    "panel":    "#252840",
    "card":     "#2E3250",
    "accent":   "#6C63FF",
    "green":    "#4CAF50",
    "red":      "#F44336",
    "yellow":   "#FFC107",
    "text":     "#E8EAF6",
    "subtext":  "#9E9EC8",
    "border":   "#3D4270",
}

# ─────────────────────────── Audio helpers ────────────────────────────────────

def generate_chirp() -> np.ndarray:
    """Linear sweep CHIRP_F_START → CHIRP_F_END over CHIRP_DURATION seconds."""
    t = np.linspace(0, CHIRP_DURATION, int(AUDIO_SAMPLE_RATE * CHIRP_DURATION))
    chirp = scipy_signal.chirp(t, f0=CHIRP_F_START, f1=CHIRP_F_END,
                               t1=CHIRP_DURATION, method="linear")
    return (chirp * 0.8).astype(np.float32)


def play_chirp_async():
    """Play chirp on default speaker in background thread."""
    wave = generate_chirp()
    sd.play(wave, samplerate=AUDIO_SAMPLE_RATE)


def stop_audio():
    sd.stop()


# ─────────────────────────── Serial reader ────────────────────────────────────

class ArduinoReader:
    """Reads 9-value lines from Arduino and buffers them."""

    def __init__(self):
        self.ser: serial.Serial | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self.latest: list[int] = [0] * 9
        self.buffer: list[list[int]] = []
        self._lock = threading.Lock()

    # ── connection ────────────────────────────────────────────────────────────

    def connect(self, port: str) -> bool:
        try:
            self.ser = serial.Serial(port, SERIAL_BAUD, timeout=1)
            time.sleep(2)          # wait for Arduino reset
            self.ser.flushInput()
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"Serial connect error: {e}")
            return False

    def disconnect(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.ser.close()

    def is_connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    # ── internal loop ─────────────────────────────────────────────────────────

    def _read_loop(self):
        while self._running:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                parts = line.split()
                if len(parts) == 9:
                    vals = [int(p) for p in parts]
                    with self._lock:
                        self.latest = vals
                        self.buffer.append(vals)
            except Exception:
                pass

    # ── public helpers ────────────────────────────────────────────────────────

    def start_capture(self):
        with self._lock:
            self.buffer.clear()

    def stop_capture(self) -> list[list[int]]:
        with self._lock:
            data = list(self.buffer)
            self.buffer.clear()
        return data

    def get_latest(self) -> list[int]:
        with self._lock:
            return list(self.latest)

    @staticmethod
    def list_ports() -> list[str]:
        return [p.device for p in serial.tools.list_ports.comports()]


# ─────────────────────────── Frequency analysis ───────────────────────────────

def extract_dominant_frequency(samples: list[list[int]],
                               channel: int = 6) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Run FFT on the selected channel.
    channel=6 → weighted-average (index 6 in the 9-value row, 0-based).
    Returns (peak_freq_hz, freqs_array, magnitude_array).
    """
    if len(samples) < 8:
        return 0.0, np.array([]), np.array([])

    y = np.array([row[channel] for row in samples], dtype=float)
    y -= y.mean()                       # remove DC offset

    n    = len(y)
    yf   = np.abs(fft(y))[:n // 2]
    xf   = fftfreq(n, d=1.0 / SAMPLE_RATE_HZ)[:n // 2]

    # ignore 0 Hz and very-low-freq bins (structural rumble)
    mask      = xf > 0.5
    xf_masked = xf[mask]
    yf_masked = yf[mask]

    if len(yf_masked) == 0:
        return 0.0, xf, yf

    peak_idx  = np.argmax(yf_masked)
    peak_freq = float(xf_masked[peak_idx])
    return peak_freq, xf, yf


def compare_frequency(measured: float, baseline: float) -> tuple[bool, float]:
    """Returns (pass_bool, deviation_fraction)."""
    if baseline == 0:
        return False, 1.0
    dev = abs(measured - baseline) / baseline
    return dev <= FREQ_TOLERANCE, dev


# ─────────────────────────── Main Application ─────────────────────────────────

class AcoustiScanApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("AcoustiScan NDT — Acoustic Inspection System")
        self.geometry("1280x820")
        self.configure(bg=THEME["bg"])
        self.resizable(True, True)

        self.arduino = ArduinoReader()
        self.num_points: int = 0
        self.current_point: int = 0
        self.baseline: dict[int, dict] = {}      # point_idx → {freq, fft_x, fft_y, raw}
        self.test_results: dict[int, dict] = {}  # point_idx → {freq, pass, dev, fft_x, fft_y}
        self.mode: str = "idle"                  # idle | training | testing
        self._capture_active = False
        self.baseline_file = "acoustiscan_baseline.json"

        self._build_ui()
        self._update_status_loop()

    # ══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=THEME["accent"], height=56)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="🔊  AcoustiScan NDT", font=("Helvetica", 18, "bold"),
                 bg=THEME["accent"], fg="white").pack(side="left", padx=20, pady=10)
        tk.Label(top, text="Acoustic Non-Destructive Testing  |  v2.0 Hardware Edition",
                 font=("Helvetica", 10), bg=THEME["accent"],
                 fg="#D0CEFF").pack(side="left", pady=14)

        # ── main split ───────────────────────────────────────────────────────
        body = tk.Frame(self, bg=THEME["bg"])
        body.pack(fill="both", expand=True)

        self._left  = tk.Frame(body, bg=THEME["panel"], width=320)
        self._left.pack(side="left", fill="y", padx=(12, 6), pady=12)
        self._left.pack_propagate(False)

        self._right = tk.Frame(body, bg=THEME["bg"])
        self._right.pack(side="left", fill="both", expand=True, padx=(6, 12), pady=12)

        self._build_control_panel()
        self._build_right_panel()

    # ── left control panel ───────────────────────────────────────────────────

    def _build_control_panel(self):
        p = self._left

        def section(title):
            f = tk.Frame(p, bg=THEME["card"], bd=0)
            f.pack(fill="x", padx=12, pady=(10, 0))
            tk.Label(f, text=title, font=("Helvetica", 9, "bold"),
                     bg=THEME["card"], fg=THEME["accent"]).pack(anchor="w", padx=10, pady=(8, 2))
            sep = tk.Frame(f, bg=THEME["border"], height=1)
            sep.pack(fill="x", padx=10)
            return f

        # ── Serial connection ─────────────────────────────────────────────
        sec = section("① ARDUINO CONNECTION")
        row = tk.Frame(sec, bg=THEME["card"])
        row.pack(fill="x", padx=10, pady=6)

        self._port_var = tk.StringVar()
        ports = ArduinoReader.list_ports()
        self._port_var.set(ports[0] if ports else "")
        self._port_combo = ttk.Combobox(row, textvariable=self._port_var,
                                        values=ports, width=14)
        self._port_combo.pack(side="left")

        tk.Button(row, text="⟳", command=self._refresh_ports,
                  bg=THEME["card"], fg=THEME["subtext"],
                  relief="flat", font=("Helvetica", 12)).pack(side="left", padx=4)

        self._connect_btn = tk.Button(row, text="Connect",
                                      command=self._toggle_connect,
                                      bg=THEME["accent"], fg="white",
                                      relief="flat", padx=10, pady=4,
                                      font=("Helvetica", 9, "bold"))
        self._connect_btn.pack(side="left", padx=4)

        self._conn_indicator = tk.Label(sec, text="● Disconnected",
                                        font=("Helvetica", 9),
                                        bg=THEME["card"], fg=THEME["red"])
        self._conn_indicator.pack(anchor="w", padx=12, pady=(0, 8))

        # ── Part configuration ────────────────────────────────────────────
        sec2 = section("② PART CONFIGURATION")
        cf = tk.Frame(sec2, bg=THEME["card"])
        cf.pack(fill="x", padx=10, pady=6)

        tk.Label(cf, text="Inspection points:", font=("Helvetica", 9),
                 bg=THEME["card"], fg=THEME["text"]).grid(row=0, column=0,
                 sticky="w", padx=0, pady=2)
        self._pts_var = tk.IntVar(value=5)
        pts_spin = tk.Spinbox(cf, from_=1, to=20, textvariable=self._pts_var,
                              width=5, font=("Helvetica", 10),
                              bg=THEME["bg"], fg=THEME["text"],
                              insertbackground=THEME["text"])
        pts_spin.grid(row=0, column=1, padx=8, pady=2)

        tk.Label(cf, text="Tolerance (%):", font=("Helvetica", 9),
                 bg=THEME["card"], fg=THEME["text"]).grid(row=1, column=0,
                 sticky="w", pady=2)
        self._tol_var = tk.IntVar(value=12)
        tk.Spinbox(cf, from_=1, to=50, textvariable=self._tol_var,
                   width=5, font=("Helvetica", 10),
                   bg=THEME["bg"], fg=THEME["text"],
                   insertbackground=THEME["text"]).grid(row=1, column=1, padx=8)

        tk.Button(sec2, text="Apply Configuration",
                  command=self._apply_config,
                  bg=THEME["accent"], fg="white", relief="flat",
                  padx=10, pady=5, font=("Helvetica", 9, "bold")).pack(
                  padx=10, pady=(4, 10), fill="x")

        # ── Training ──────────────────────────────────────────────────────
        sec3 = section("③ TRAINING MODE  (Good Part)")
        self._train_btn = tk.Button(sec3, text="▶  Start Training",
                                    command=self._start_training,
                                    bg=THEME["green"], fg="white",
                                    relief="flat", padx=10, pady=6,
                                    font=("Helvetica", 10, "bold"))
        self._train_btn.pack(padx=10, pady=(8, 4), fill="x")

        tk.Button(sec3, text="💾  Save Baseline",
                  command=self._save_baseline,
                  bg=THEME["card"], fg=THEME["text"],
                  relief="flat", padx=10, pady=4,
                  font=("Helvetica", 9)).pack(padx=10, pady=2, fill="x")

        tk.Button(sec3, text="📂  Load Baseline",
                  command=self._load_baseline,
                  bg=THEME["card"], fg=THEME["text"],
                  relief="flat", padx=10, pady=4,
                  font=("Helvetica", 9)).pack(padx=10, pady=(2, 10), fill="x")

        # ── Testing ───────────────────────────────────────────────────────
        sec4 = section("④ TESTING MODE  (Unknown Part)")
        self._test_btn = tk.Button(sec4, text="🔍  Start Testing",
                                   command=self._start_testing,
                                   bg="#E91E63", fg="white",
                                   relief="flat", padx=10, pady=6,
                                   font=("Helvetica", 10, "bold"))
        self._test_btn.pack(padx=10, pady=(8, 10), fill="x")

        # ── Live sensor ───────────────────────────────────────────────────
        sec5 = section("LIVE SENSOR READINGS")
        grid = tk.Frame(sec5, bg=THEME["card"])
        grid.pack(fill="x", padx=10, pady=6)
        self._sensor_bars = []
        for i in range(NUM_SENSORS):
            tk.Label(grid, text=f"S{i+1}", width=3, font=("Helvetica", 8, "bold"),
                     bg=THEME["card"], fg=THEME["subtext"]).grid(
                     row=i, column=0, sticky="w", pady=1)
            bar_bg = tk.Frame(grid, bg=THEME["bg"], width=160, height=14)
            bar_bg.grid(row=i, column=1, padx=4, sticky="w")
            bar_bg.pack_propagate(False)
            bar = tk.Frame(bar_bg, bg=THEME["accent"], width=0, height=14)
            bar.place(x=0, y=0, height=14)
            val_lbl = tk.Label(grid, text="0", width=5,
                               font=("Helvetica", 8),
                               bg=THEME["card"], fg=THEME["subtext"])
            val_lbl.grid(row=i, column=2, padx=2)
            self._sensor_bars.append((bar, val_lbl, bar_bg))

        # fusion row
        frow = tk.Frame(sec5, bg=THEME["card"])
        frow.pack(fill="x", padx=10, pady=(0, 10))
        self._fusion_labels = {}
        for col, (key, lbl) in enumerate([("avg", "W.Avg"),
                                          ("dom", "Dominant"),
                                          ("vote", "Vote")]):
            cell = tk.Frame(frow, bg=THEME["bg"], bd=0)
            cell.pack(side="left", expand=True, fill="x", padx=2)
            tk.Label(cell, text=lbl, font=("Helvetica", 7),
                     bg=THEME["bg"], fg=THEME["subtext"]).pack()
            lv = tk.Label(cell, text="—", font=("Helvetica", 11, "bold"),
                          bg=THEME["bg"], fg=THEME["accent"])
            lv.pack()
            self._fusion_labels[key] = lv

    # ── right panel ──────────────────────────────────────────────────────────

    def _build_right_panel(self):
        p = self._right

        # top status bar
        self._status_frame = tk.Frame(p, bg=THEME["card"], height=44)
        self._status_frame.pack(fill="x", pady=(0, 8))
        self._status_frame.pack_propagate(False)
        self._status_lbl = tk.Label(self._status_frame,
                                    text="Ready — Connect Arduino and configure inspection points.",
                                    font=("Helvetica", 10), bg=THEME["card"],
                                    fg=THEME["text"])
        self._status_lbl.pack(side="left", padx=16, pady=8)
        self._mode_badge = tk.Label(self._status_frame, text="  IDLE  ",
                                    font=("Helvetica", 9, "bold"),
                                    bg=THEME["subtext"], fg="white")
        self._mode_badge.pack(side="right", padx=16, pady=8)

        # notebook tabs
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background=THEME["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=THEME["panel"],
                        foreground=THEME["subtext"],
                        padding=[14, 6], font=("Helvetica", 9))
        style.map("TNotebook.Tab",
                  background=[("selected", THEME["accent"])],
                  foreground=[("selected", "white")])

        self._notebook = ttk.Notebook(p)
        self._notebook.pack(fill="both", expand=True)

        self._tab_scan    = tk.Frame(self._notebook, bg=THEME["bg"])
        self._tab_fft     = tk.Frame(self._notebook, bg=THEME["bg"])
        self._tab_results = tk.Frame(self._notebook, bg=THEME["bg"])
        self._tab_log     = tk.Frame(self._notebook, bg=THEME["bg"])

        self._notebook.add(self._tab_scan,    text="  Scan Progress  ")
        self._notebook.add(self._tab_fft,     text="  FFT Spectrum  ")
        self._notebook.add(self._tab_results, text="  Test Results  ")
        self._notebook.add(self._tab_log,     text="  Activity Log  ")

        self._build_scan_tab()
        self._build_fft_tab()
        self._build_results_tab()
        self._build_log_tab()

    # ── scan tab ─────────────────────────────────────────────────────────────

    def _build_scan_tab(self):
        p = self._tab_scan

        instr = tk.Frame(p, bg=THEME["card"])
        instr.pack(fill="x", padx=10, pady=(10, 6))

        self._instr_lbl = tk.Label(instr,
            text="Configure number of inspection points, then click Start Training or Start Testing.",
            font=("Helvetica", 10), bg=THEME["card"], fg=THEME["text"],
            wraplength=720, justify="left")
        self._instr_lbl.pack(anchor="w", padx=14, pady=10)

        # point grid
        grid_frame = tk.Frame(p, bg=THEME["bg"])
        grid_frame.pack(fill="both", expand=True, padx=10, pady=4)

        self._point_canvas = tk.Canvas(grid_frame, bg=THEME["bg"],
                                       highlightthickness=0)
        vsb = ttk.Scrollbar(grid_frame, orient="vertical",
                             command=self._point_canvas.yview)
        self._point_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._point_canvas.pack(fill="both", expand=True)

        self._point_inner = tk.Frame(self._point_canvas, bg=THEME["bg"])
        self._point_canvas.create_window((0, 0), window=self._point_inner,
                                         anchor="nw")
        self._point_inner.bind("<Configure>",
            lambda e: self._point_canvas.configure(
                scrollregion=self._point_canvas.bbox("all")))

        self._point_cells: list[dict] = []

        # capture / skip buttons
        btn_row = tk.Frame(p, bg=THEME["bg"])
        btn_row.pack(fill="x", padx=10, pady=(4, 10))

        self._capture_btn = tk.Button(btn_row, text="🎙  Capture This Point",
                                      command=self._capture_current_point,
                                      bg=THEME["accent"], fg="white",
                                      relief="flat", padx=16, pady=7,
                                      font=("Helvetica", 10, "bold"),
                                      state="disabled")
        self._capture_btn.pack(side="left", padx=(0, 8))

        self._skip_btn = tk.Button(btn_row, text="Skip",
                                   command=self._skip_point,
                                   bg=THEME["card"], fg=THEME["subtext"],
                                   relief="flat", padx=14, pady=7,
                                   font=("Helvetica", 9),
                                   state="disabled")
        self._skip_btn.pack(side="left")

        self._progress_lbl = tk.Label(btn_row, text="",
                                      font=("Helvetica", 9),
                                      bg=THEME["bg"], fg=THEME["subtext"])
        self._progress_lbl.pack(side="right", padx=8)

    def _rebuild_point_grid(self):
        for w in self._point_inner.winfo_children():
            w.destroy()
        self._point_cells.clear()

        cols = 4
        for i in range(self.num_points):
            r, c = divmod(i, cols)
            cell = tk.Frame(self._point_inner, bg=THEME["card"],
                            width=170, height=90, bd=0)
            cell.grid(row=r, column=c, padx=6, pady=6)
            cell.pack_propagate(False)

            tk.Label(cell, text=f"Point {i+1}",
                     font=("Helvetica", 10, "bold"),
                     bg=THEME["card"], fg=THEME["text"]).pack(pady=(10, 2))

            status_lbl = tk.Label(cell, text="Pending",
                                  font=("Helvetica", 9),
                                  bg=THEME["card"], fg=THEME["subtext"])
            status_lbl.pack()

            freq_lbl = tk.Label(cell, text="",
                                font=("Helvetica", 9),
                                bg=THEME["card"], fg=THEME["subtext"])
            freq_lbl.pack()

            self._point_cells.append({
                "frame": cell,
                "status": status_lbl,
                "freq": freq_lbl,
            })

    def _update_point_cell(self, idx: int, status: str, freq: float = None,
                           passed: bool = None):
        if idx >= len(self._point_cells):
            return
        cell = self._point_cells[idx]

        if passed is None:
            color = THEME["yellow"]
            cell["frame"].configure(bg=THEME["card"])
            cell["status"].configure(bg=THEME["card"])
            cell["freq"].configure(bg=THEME["card"])
        elif passed:
            color = THEME["green"]
            cell["frame"].configure(bg="#1B3A1F")
            cell["status"].configure(bg="#1B3A1F")
            cell["freq"].configure(bg="#1B3A1F")
        else:
            color = THEME["red"]
            cell["frame"].configure(bg="#3A1B1B")
            cell["status"].configure(bg="#3A1B1B")
            cell["freq"].configure(bg="#3A1B1B")

        cell["status"].configure(text=status, fg=color)
        if freq is not None:
            cell["freq"].configure(text=f"{freq:.2f} Hz")

    # ── FFT tab ──────────────────────────────────────────────────────────────

    def _build_fft_tab(self):
        p = self._tab_fft
        self._fft_fig = Figure(figsize=(8, 4.5),
                               facecolor=THEME["bg"])
        self._fft_ax  = self._fft_fig.add_subplot(111)
        self._fft_ax.set_facecolor(THEME["panel"])
        self._fft_ax.tick_params(colors=THEME["subtext"])
        self._fft_ax.spines[:].set_color(THEME["border"])
        self._fft_ax.set_xlabel("Frequency (Hz)", color=THEME["subtext"])
        self._fft_ax.set_ylabel("Magnitude", color=THEME["subtext"])
        self._fft_ax.set_title("FFT Spectrum — Waiting for data…",
                               color=THEME["text"])

        self._fft_canvas = FigureCanvasTkAgg(self._fft_fig, master=p)
        self._fft_canvas.get_tk_widget().pack(fill="both", expand=True,
                                              padx=10, pady=10)

    def _plot_fft(self, freqs, mags, title="FFT Spectrum",
                  peak_freq=None, baseline_freq=None):
        ax = self._fft_ax
        ax.clear()
        ax.set_facecolor(THEME["panel"])
        ax.tick_params(colors=THEME["subtext"])
        ax.spines[:].set_color(THEME["border"])

        if len(freqs) > 0:
            ax.plot(freqs, mags, color=THEME["accent"], linewidth=1.2)
            ax.fill_between(freqs, mags, alpha=0.18, color=THEME["accent"])
            if peak_freq:
                ax.axvline(peak_freq, color=THEME["yellow"],
                           linewidth=1.5, linestyle="--",
                           label=f"Measured: {peak_freq:.2f} Hz")
            if baseline_freq:
                ax.axvline(baseline_freq, color=THEME["green"],
                           linewidth=1.5, linestyle=":",
                           label=f"Baseline: {baseline_freq:.2f} Hz")
            if peak_freq or baseline_freq:
                ax.legend(facecolor=THEME["card"], labelcolor=THEME["text"],
                          framealpha=0.8)

        ax.set_xlabel("Frequency (Hz)", color=THEME["subtext"])
        ax.set_ylabel("Magnitude", color=THEME["subtext"])
        ax.set_title(title, color=THEME["text"])
        self._fft_canvas.draw()

    # ── results tab ──────────────────────────────────────────────────────────

    def _build_results_tab(self):
        p = self._tab_results

        self._results_summary = tk.Label(p, text="No test results yet.",
                                         font=("Helvetica", 12),
                                         bg=THEME["bg"], fg=THEME["subtext"])
        self._results_summary.pack(pady=16)

        cols = ("Point", "Baseline Hz", "Measured Hz", "Deviation %", "Result")
        self._results_tree = ttk.Treeview(p, columns=cols, show="headings",
                                          height=14)
        style = ttk.Style()
        style.configure("Treeview", background=THEME["card"],
                        fieldbackground=THEME["card"],
                        foreground=THEME["text"],
                        rowheight=28, font=("Helvetica", 9))
        style.configure("Treeview.Heading", background=THEME["accent"],
                        foreground="white", font=("Helvetica", 9, "bold"))
        style.map("Treeview", background=[("selected", THEME["border"])])

        for c in cols:
            self._results_tree.heading(c, text=c)
            self._results_tree.column(c, width=160, anchor="center")

        self._results_tree.tag_configure("pass", foreground=THEME["green"])
        self._results_tree.tag_configure("fail", foreground=THEME["red"])
        self._results_tree.tag_configure("skip", foreground=THEME["subtext"])

        self._results_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _refresh_results_table(self):
        for row in self._results_tree.get_children():
            self._results_tree.delete(row)

        passed = failed = skipped = 0
        for i, res in self._test_results.items():
            baseline_freq = self.baseline.get(i, {}).get("freq", 0)
            if res.get("skip"):
                self._results_tree.insert("", "end", values=(
                    f"Point {i+1}", f"{baseline_freq:.2f}", "—", "—", "SKIP"
                ), tags=("skip",))
                skipped += 1
            else:
                mf  = res["freq"]
                dev = res["dev"] * 100
                ok  = res["pass"]
                tag = "pass" if ok else "fail"
                self._results_tree.insert("", "end", values=(
                    f"Point {i+1}",
                    f"{baseline_freq:.2f}",
                    f"{mf:.2f}",
                    f"{dev:.1f}",
                    "✅ PASS" if ok else "❌ FAIL"
                ), tags=(tag,))
                if ok: passed += 1
                else:  failed += 1

        total = passed + failed + skipped
        color = THEME["green"] if failed == 0 else THEME["red"]
        verdict = "PART OK ✅" if failed == 0 and total > 0 else \
                  ("PART DEFECTIVE ❌" if failed > 0 else "No results")
        self._results_summary.configure(
            text=f"{verdict}   |   Pass: {passed}  Fail: {failed}  Skip: {skipped}",
            fg=color)

    # ── log tab ──────────────────────────────────────────────────────────────

    def _build_log_tab(self):
        p = self._tab_log
        self._log_text = tk.Text(p, bg=THEME["panel"], fg=THEME["text"],
                                 font=("Consolas", 9),
                                 insertbackground=THEME["text"],
                                 state="disabled", wrap="word")
        vsb = ttk.Scrollbar(p, orient="vertical",
                             command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _log(self, msg: str, level: str = "INFO"):
        ts = time.strftime("%H:%M:%S")
        colors = {"INFO": THEME["text"], "OK": THEME["green"],
                  "FAIL": THEME["red"], "WARN": THEME["yellow"]}
        tag = level
        self._log_text.configure(state="normal")
        self._log_text.insert("end", f"[{ts}] [{level}]  {msg}\n")
        self._log_text.tag_add(tag,
            f"end - {len(msg)+22}c", "end - 1c")
        self._log_text.tag_configure(tag, foreground=colors.get(level, THEME["text"]))
        self._log_text.configure(state="disabled")
        self._log_text.see("end")

    # ══════════════════════════════════════════════════════════════════════════
    # Event handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_ports(self):
        ports = ArduinoReader.list_ports()
        self._port_combo["values"] = ports
        if ports:
            self._port_var.set(ports[0])

    def _toggle_connect(self):
        if self.arduino.is_connected():
            self.arduino.disconnect()
            self._connect_btn.configure(text="Connect", bg=THEME["accent"])
            self._conn_indicator.configure(text="● Disconnected",
                                           fg=THEME["red"])
            self._log("Disconnected from Arduino.")
        else:
            port = self._port_var.get()
            if not port:
                messagebox.showwarning("No port", "Select a COM port first.")
                return
            self._log(f"Connecting to {port}…")
            ok = self.arduino.connect(port)
            if ok:
                self._connect_btn.configure(text="Disconnect",
                                            bg="#607D8B")
                self._conn_indicator.configure(
                    text=f"● Connected  ({port})", fg=THEME["green"])
                self._log(f"Connected to Arduino on {port}.", "OK")
            else:
                messagebox.showerror("Connection failed",
                                     f"Could not open {port}.")
                self._log(f"Failed to connect to {port}.", "FAIL")

    def _apply_config(self):
        n = self._pts_var.get()
        self.num_points = n
        global FREQ_TOLERANCE
        FREQ_TOLERANCE = self._tol_var.get() / 100.0
        self.baseline.clear()
        self.test_results.clear()
        self._rebuild_point_grid()
        self._log(f"Configuration applied: {n} points, ±{self._tol_var.get()}% tolerance.")
        self._set_status("Configuration applied. Ready for Training or Testing.")

    def _set_status(self, msg: str, mode: str = None):
        self._status_lbl.configure(text=msg)
        if mode:
            self.mode = mode
            badge_colors = {
                "idle":     (THEME["subtext"], "white"),
                "training": (THEME["green"],   "white"),
                "testing":  ("#E91E63",         "white"),
                "done":     (THEME["accent"],   "white"),
            }
            bg, fg = badge_colors.get(mode, (THEME["subtext"], "white"))
            self._mode_badge.configure(text=f"  {mode.upper()}  ",
                                       bg=bg, fg=fg)

    # ── training ─────────────────────────────────────────────────────────────

    def _start_training(self):
        if self.num_points == 0:
            messagebox.showwarning("Not configured",
                                   "Apply configuration first.")
            return
        if not self.arduino.is_connected():
            ans = messagebox.askyesno("No Arduino",
                "Arduino is not connected.\n\nRun in DEMO mode (simulated data)?")
            if not ans:
                return
        self.baseline.clear()
        self._rebuild_point_grid()
        self.current_point = 0
        self.mode = "training"
        self._set_status("TRAINING — Place GOOD PART. Position at Point 1 and press Capture.",
                         "training")
        self._instr_lbl.configure(
            text="TRAINING MODE — Place a known-good part.\n"
                 "Position the sensor/speaker at each marked point, then press 'Capture This Point'.\n"
                 "A chirp will be played; sensor data is recorded and analysed.")
        self._capture_btn.configure(state="normal")
        self._skip_btn.configure(state="normal")
        self._update_point_cell(0, "▶ Active", passed=None)
        self._progress_lbl.configure(
            text=f"Point 1 / {self.num_points}")
        self._notebook.select(0)
        self._log("Training started.", "INFO")

    # ── testing ──────────────────────────────────────────────────────────────

    def _start_testing(self):
        if self.num_points == 0:
            messagebox.showwarning("Not configured",
                                   "Apply configuration first.")
            return
        if not self.baseline:
            messagebox.showwarning("No baseline",
                                   "Train on a good part first (or load a baseline).")
            return
        self.test_results.clear()
        self._rebuild_point_grid()
        self.current_point = 0
        self.mode = "testing"
        self._set_status("TESTING — Place UNKNOWN PART. Position at Point 1 and press Capture.",
                         "testing")
        self._instr_lbl.configure(
            text="TESTING MODE — Place the part under test.\n"
                 "Position at each point and press 'Capture This Point'.\n"
                 "Results are compared against the trained baseline automatically.")
        self._capture_btn.configure(state="normal")
        self._skip_btn.configure(state="normal")
        self._update_point_cell(0, "▶ Active", passed=None)
        self._progress_lbl.configure(
            text=f"Point 1 / {self.num_points}")
        self._notebook.select(0)
        self._log("Testing started.", "INFO")

    # ── capture ──────────────────────────────────────────────────────────────

    def _capture_current_point(self):
        if self._capture_active:
            return
        self._capture_active = True
        self._capture_btn.configure(state="disabled", text="⏺  Recording…")
        threading.Thread(target=self._do_capture, daemon=True).start()

    def _do_capture(self):
        idx = self.current_point
        self._log(f"Capturing Point {idx+1}…")
        self.after(0, lambda: self._update_point_cell(idx, "⏺ Recording…"))

        # Play chirp
        play_chirp_async()
        self.after(0, lambda: self._set_status(
            f"Playing chirp for Point {idx+1}… hold steady.", self.mode))

        # Record sensor data
        self.arduino.start_capture()
        time.sleep(RECORD_SECONDS)
        stop_audio()
        raw_data = self.arduino.stop_capture()

        # If no Arduino → generate simulated data
        if not raw_data or len(raw_data) < 10:
            raw_data = self._simulate_data(idx)

        # FFT analysis
        freq, xf, yf = extract_dominant_frequency(raw_data, channel=6)
        self._log(f"  Point {idx+1}: dominant freq = {freq:.2f} Hz  ({len(raw_data)} samples)")

        # Update plot on main thread
        title = f"Point {idx+1} — {'Training' if self.mode=='training' else 'Testing'}"
        baseline_f = self.baseline.get(idx, {}).get("freq")
        self.after(0, lambda: self._plot_fft(xf, yf, title, freq, baseline_f))
        self.after(0, lambda: self._notebook.select(1))

        if self.mode == "training":
            self.baseline[idx] = {"freq": freq, "raw": len(raw_data)}
            self.after(0, lambda: self._update_point_cell(idx, f"✓ {freq:.1f} Hz",
                                                          freq=freq, passed=True))
            self._log(f"  Baseline stored: {freq:.2f} Hz", "OK")
        else:
            base_f = self.baseline.get(idx, {}).get("freq", 0)
            passed, dev = compare_frequency(freq, base_f)
            self.test_results[idx] = {
                "freq": freq, "pass": passed, "dev": dev,
                "fft_x": xf.tolist(), "fft_y": yf.tolist()
            }
            label = "✅ PASS" if passed else "❌ FAIL"
            self.after(0, lambda: self._update_point_cell(
                idx, f"{label}  {freq:.1f} Hz", freq=freq, passed=passed))
            level = "OK" if passed else "FAIL"
            self._log(f"  Point {idx+1}: {label}  measured={freq:.2f}  "
                      f"baseline={base_f:.2f}  dev={dev*100:.1f}%", level)
            self.after(0, self._refresh_results_table)

        self.after(0, self._advance_point)

    def _skip_point(self):
        idx = self.current_point
        self._update_point_cell(idx, "— Skipped")
        if self.mode == "testing":
            self.test_results[idx] = {"skip": True}
            self._refresh_results_table()
        self._log(f"Point {idx+1} skipped.")
        self._advance_point()

    def _advance_point(self):
        self._capture_active = False
        self._capture_btn.configure(state="normal", text="🎙  Capture This Point")

        nxt = self.current_point + 1
        if nxt >= self.num_points:
            self._finish_session()
            return

        self.current_point = nxt
        self._update_point_cell(nxt, "▶ Active")
        self._progress_lbl.configure(
            text=f"Point {nxt+1} / {self.num_points}")
        action = "Training" if self.mode == "training" else "Testing"
        self._set_status(
            f"{action.upper()} — Move to Point {nxt+1} and press Capture.",
            self.mode)
        self._log(f"→ Ready for Point {nxt+1}.")

    def _finish_session(self):
        self._capture_btn.configure(state="disabled")
        self._skip_btn.configure(state="disabled")

        if self.mode == "training":
            self._set_status(
                f"Training complete — baseline stored for {self.num_points} points. "
                "You can now run Testing.", "done")
            self._log("Training session complete.", "OK")
            messagebox.showinfo("Training Complete",
                                f"Baseline captured for {self.num_points} inspection points.\n"
                                "Save the baseline, then switch to Testing Mode.")
        else:
            passed = sum(1 for r in self.test_results.values()
                         if not r.get("skip") and r.get("pass"))
            failed = sum(1 for r in self.test_results.values()
                         if not r.get("skip") and not r.get("pass"))
            verdict = "PART OK ✅" if failed == 0 else "PART DEFECTIVE ❌"
            self._set_status(
                f"Testing complete — {verdict}  (Pass: {passed}  Fail: {failed})",
                "done")
            self._log(f"Testing complete — {verdict}  pass={passed}  fail={failed}",
                      "OK" if failed == 0 else "FAIL")
            self._notebook.select(2)
            messagebox.showinfo("Test Complete",
                                f"Inspection finished.\n\n{verdict}\n"
                                f"Pass: {passed}   Fail: {failed}   "
                                f"Skip: {self.num_points - passed - failed}")

    # ── baseline persistence ──────────────────────────────────────────────────

    def _save_baseline(self):
        if not self.baseline:
            messagebox.showwarning("Nothing to save",
                                   "Complete training first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            initialfile="acoustiscan_baseline.json")
        if not path:
            return
        data = {
            "num_points": self.num_points,
            "tolerance":  self._tol_var.get(),
            "baseline":   {str(k): v for k, v in self.baseline.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._log(f"Baseline saved → {path}", "OK")
        messagebox.showinfo("Saved", f"Baseline saved to\n{path}")

    def _load_baseline(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self.num_points = data["num_points"]
            self._pts_var.set(self.num_points)
            self._tol_var.set(data.get("tolerance", 12))
            self.baseline = {int(k): v for k, v in data["baseline"].items()}
            self._rebuild_point_grid()
            for i, v in self.baseline.items():
                self._update_point_cell(i, f"✓ {v['freq']:.1f} Hz",
                                        freq=v["freq"], passed=True)
            self._log(f"Baseline loaded ← {path}  ({self.num_points} points)", "OK")
            messagebox.showinfo("Loaded",
                                f"Baseline loaded.\n{self.num_points} points ready for Testing.")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    # ── demo / simulation ─────────────────────────────────────────────────────

    def _simulate_data(self, point_idx: int) -> list[list[int]]:
        """Generate realistic fake sensor data for demo / no-Arduino mode."""
        n = RECORD_SECONDS * SAMPLE_RATE_HZ
        t = np.linspace(0, RECORD_SECONDS, n)

        if self.mode == "training":
            # Each point has a fixed 'natural' frequency (good part)
            nat_f = 3.0 + point_idx * 1.7          # Hz in sensor space
        else:
            # Testing: 80 % chance same freq, 20 % chance shifted (defect)
            base_f = 3.0 + point_idx * 1.7
            if np.random.random() < 0.25:
                nat_f = base_f * (1 + np.random.uniform(0.2, 0.5))  # defect
            else:
                nat_f = base_f * (1 + np.random.uniform(-0.05, 0.05))  # ok

        base_amp = 400 + point_idx * 30
        sig = (base_amp * np.sin(2 * np.pi * nat_f * t)
               + np.random.normal(0, 40, n) + 512).clip(0, 1023).astype(int)

        rows = []
        for i in range(n):
            row = [int(sig[i] * (0.85 + 0.3 * np.random.random()))
                   for _ in range(NUM_SENSORS)]
            avg = int(np.mean(row))
            dom = int(np.max(row))
            vote = 1 if sum(v > 512 for v in row) >= 3 else 0
            rows.append(row + [avg, dom, vote * 1023])
        return rows

    # ── live sensor display ───────────────────────────────────────────────────

    def _update_status_loop(self):
        vals = self.arduino.get_latest()
        for i, (bar, lbl, bg) in enumerate(self._sensor_bars):
            v = vals[i] if i < len(vals) else 0
            w = max(1, int(160 * v / 1023))
            bar.place(x=0, y=0, width=w, height=14)
            lbl.configure(text=str(v))

        if len(vals) >= 9:
            self._fusion_labels["avg"].configure(text=str(vals[6]))
            self._fusion_labels["dom"].configure(text=str(vals[7]))
            vote = "YES" if vals[8] > 500 else "NO"
            self._fusion_labels["vote"].configure(text=vote)

        self.after(100, self._update_status_loop)


# ─────────────────────────── Entry point ──────────────────────────────────────

if __name__ == "__main__":
    app = AcoustiScanApp()
    app.mainloop()"""
AcoustiScan NDT — Acoustic Non-Destructive Testing
===================================================
Hardware: Arduino + 6× KY-038 sound sensors (A0-A5 / D2-D7)
Protocol: 9 space-separated values @ 50 Hz via Serial
  Val 1-6 : raw sensor readings (0-1023)
  Val 7   : weighted-average fusion
  Val 8   : dominant-source index
  Val 9   : majority-vote × 1023

Workflow
--------
1. Connect to Arduino COM port.
2. Set the number of inspection points on the part.
3. TRAINING  — place a known-good part, excite each point with a speaker
               chirp, capture sensor data, compute FFT → store baseline.
4. TESTING   — place the part under test, excite each point, compare FFT
               to baseline → PASS / FAIL per point.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
import os
import serial
import serial.tools.list_ports
import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
import sounddevice as sd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── Constants ────────────────────────────────────────
SERIAL_BAUD       = 115_200
SAMPLE_RATE_HZ    = 50          # Arduino sends data at 50 Hz (delay 20 ms)
RECORD_SECONDS    = 3           # capture window per point
CHIRP_DURATION    = 2.0         # speaker chirp length (s)
CHIRP_F_START     = 200         # Hz
CHIRP_F_END       = 4_000       # Hz
AUDIO_SAMPLE_RATE = 44_100
FREQ_TOLERANCE    = 0.12        # ±12 % tolerance for pass/fail
NUM_SENSORS       = 6
THEME = {
    "bg":       "#1A1D2E",
    "panel":    "#252840",
    "card":     "#2E3250",
    "accent":   "#6C63FF",
    "green":    "#4CAF50",
    "red":      "#F44336",
    "yellow":   "#FFC107",
    "text":     "#E8EAF6",
    "subtext":  "#9E9EC8",
    "border":   "#3D4270",
}

# ─────────────────────────── Audio helpers ────────────────────────────────────

def generate_chirp() -> np.ndarray:
    """Linear sweep CHIRP_F_START → CHIRP_F_END over CHIRP_DURATION seconds."""
    t = np.linspace(0, CHIRP_DURATION, int(AUDIO_SAMPLE_RATE * CHIRP_DURATION))
    chirp = scipy_signal.chirp(t, f0=CHIRP_F_START, f1=CHIRP_F_END,
                               t1=CHIRP_DURATION, method="linear")
    return (chirp * 0.8).astype(np.float32)


def play_chirp_async():
    """Play chirp on default speaker in background thread."""
    wave = generate_chirp()
    sd.play(wave, samplerate=AUDIO_SAMPLE_RATE)


def stop_audio():
    sd.stop()


# ─────────────────────────── Serial reader ────────────────────────────────────

class ArduinoReader:
    """Reads 9-value lines from Arduino and buffers them."""

    def __init__(self):
        self.ser: serial.Serial | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self.latest: list[int] = [0] * 9
        self.buffer: list[list[int]] = []
        self._lock = threading.Lock()

    # ── connection ────────────────────────────────────────────────────────────

    def connect(self, port: str) -> bool:
        try:
            self.ser = serial.Serial(port, SERIAL_BAUD, timeout=1)
            time.sleep(2)          # wait for Arduino reset
            self.ser.flushInput()
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"Serial connect error: {e}")
            return False

    def disconnect(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.ser.close()

    def is_connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    # ── internal loop ─────────────────────────────────────────────────────────

    def _read_loop(self):
        while self._running:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                parts = line.split()
                if len(parts) == 9:
                    vals = [int(p) for p in parts]
                    with self._lock:
                        self.latest = vals
                        self.buffer.append(vals)
            except Exception:
                pass

    # ── public helpers ────────────────────────────────────────────────────────

    def start_capture(self):
        with self._lock:
            self.buffer.clear()

    def stop_capture(self) -> list[list[int]]:
        with self._lock:
            data = list(self.buffer)
            self.buffer.clear()
        return data

    def get_latest(self) -> list[int]:
        with self._lock:
            return list(self.latest)

    @staticmethod
    def list_ports() -> list[str]:
        return [p.device for p in serial.tools.list_ports.comports()]


# ─────────────────────────── Frequency analysis ───────────────────────────────

def extract_dominant_frequency(samples: list[list[int]],
                               channel: int = 6) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Run FFT on the selected channel.
    channel=6 → weighted-average (index 6 in the 9-value row, 0-based).
    Returns (peak_freq_hz, freqs_array, magnitude_array).
    """
    if len(samples) < 8:
        return 0.0, np.array([]), np.array([])

    y = np.array([row[channel] for row in samples], dtype=float)
    y -= y.mean()                       # remove DC offset

    n    = len(y)
    yf   = np.abs(fft(y))[:n // 2]
    xf   = fftfreq(n, d=1.0 / SAMPLE_RATE_HZ)[:n // 2]

    # ignore 0 Hz and very-low-freq bins (structural rumble)
    mask      = xf > 0.5
    xf_masked = xf[mask]
    yf_masked = yf[mask]

    if len(yf_masked) == 0:
        return 0.0, xf, yf

    peak_idx  = np.argmax(yf_masked)
    peak_freq = float(xf_masked[peak_idx])
    return peak_freq, xf, yf


def compare_frequency(measured: float, baseline: float) -> tuple[bool, float]:
    """Returns (pass_bool, deviation_fraction)."""
    if baseline == 0:
        return False, 1.0
    dev = abs(measured - baseline) / baseline
    return dev <= FREQ_TOLERANCE, dev


# ─────────────────────────── Main Application ─────────────────────────────────

class AcoustiScanApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("AcoustiScan NDT — Acoustic Inspection System")
        self.geometry("1280x820")
        self.configure(bg=THEME["bg"])
        self.resizable(True, True)

        self.arduino = ArduinoReader()
        self.num_points: int = 0
        self.current_point: int = 0
        self.baseline: dict[int, dict] = {}      # point_idx → {freq, fft_x, fft_y, raw}
        self.test_results: dict[int, dict] = {}  # point_idx → {freq, pass, dev, fft_x, fft_y}
        self.mode: str = "idle"                  # idle | training | testing
        self._capture_active = False
        self.baseline_file = "acoustiscan_baseline.json"

        self._build_ui()
        self._update_status_loop()

    # ══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=THEME["accent"], height=56)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="🔊  AcoustiScan NDT", font=("Helvetica", 18, "bold"),
                 bg=THEME["accent"], fg="white").pack(side="left", padx=20, pady=10)
        tk.Label(top, text="Acoustic Non-Destructive Testing  |  v2.0 Hardware Edition",
                 font=("Helvetica", 10), bg=THEME["accent"],
                 fg="#D0CEFF").pack(side="left", pady=14)

        # ── main split ───────────────────────────────────────────────────────
        body = tk.Frame(self, bg=THEME["bg"])
        body.pack(fill="both", expand=True)

        self._left  = tk.Frame(body, bg=THEME["panel"], width=320)
        self._left.pack(side="left", fill="y", padx=(12, 6), pady=12)
        self._left.pack_propagate(False)

        self._right = tk.Frame(body, bg=THEME["bg"])
        self._right.pack(side="left", fill="both", expand=True, padx=(6, 12), pady=12)

        self._build_control_panel()
        self._build_right_panel()

    # ── left control panel ───────────────────────────────────────────────────

    def _build_control_panel(self):
        p = self._left

        def section(title):
            f = tk.Frame(p, bg=THEME["card"], bd=0)
            f.pack(fill="x", padx=12, pady=(10, 0))
            tk.Label(f, text=title, font=("Helvetica", 9, "bold"),
                     bg=THEME["card"], fg=THEME["accent"]).pack(anchor="w", padx=10, pady=(8, 2))
            sep = tk.Frame(f, bg=THEME["border"], height=1)
            sep.pack(fill="x", padx=10)
            return f

        # ── Serial connection ─────────────────────────────────────────────
        sec = section("① ARDUINO CONNECTION")
        row = tk.Frame(sec, bg=THEME["card"])
        row.pack(fill="x", padx=10, pady=6)

        self._port_var = tk.StringVar()
        ports = ArduinoReader.list_ports()
        self._port_var.set(ports[0] if ports else "")
        self._port_combo = ttk.Combobox(row, textvariable=self._port_var,
                                        values=ports, width=14)
        self._port_combo.pack(side="left")

        tk.Button(row, text="⟳", command=self._refresh_ports,
                  bg=THEME["card"], fg=THEME["subtext"],
                  relief="flat", font=("Helvetica", 12)).pack(side="left", padx=4)

        self._connect_btn = tk.Button(row, text="Connect",
                                      command=self._toggle_connect,
                                      bg=THEME["accent"], fg="white",
                                      relief="flat", padx=10, pady=4,
                                      font=("Helvetica", 9, "bold"))
        self._connect_btn.pack(side="left", padx=4)

        self._conn_indicator = tk.Label(sec, text="● Disconnected",
                                        font=("Helvetica", 9),
                                        bg=THEME["card"], fg=THEME["red"])
        self._conn_indicator.pack(anchor="w", padx=12, pady=(0, 8))

        # ── Part configuration ────────────────────────────────────────────
        sec2 = section("② PART CONFIGURATION")
        cf = tk.Frame(sec2, bg=THEME["card"])
        cf.pack(fill="x", padx=10, pady=6)

        tk.Label(cf, text="Inspection points:", font=("Helvetica", 9),
                 bg=THEME["card"], fg=THEME["text"]).grid(row=0, column=0,
                 sticky="w", padx=0, pady=2)
        self._pts_var = tk.IntVar(value=5)
        pts_spin = tk.Spinbox(cf, from_=1, to=20, textvariable=self._pts_var,
                              width=5, font=("Helvetica", 10),
                              bg=THEME["bg"], fg=THEME["text"],
                              insertbackground=THEME["text"])
        pts_spin.grid(row=0, column=1, padx=8, pady=2)

        tk.Label(cf, text="Tolerance (%):", font=("Helvetica", 9),
                 bg=THEME["card"], fg=THEME["text"]).grid(row=1, column=0,
                 sticky="w", pady=2)
        self._tol_var = tk.IntVar(value=12)
        tk.Spinbox(cf, from_=1, to=50, textvariable=self._tol_var,
                   width=5, font=("Helvetica", 10),
                   bg=THEME["bg"], fg=THEME["text"],
                   insertbackground=THEME["text"]).grid(row=1, column=1, padx=8)

        tk.Button(sec2, text="Apply Configuration",
                  command=self._apply_config,
                  bg=THEME["accent"], fg="white", relief="flat",
                  padx=10, pady=5, font=("Helvetica", 9, "bold")).pack(
                  padx=10, pady=(4, 10), fill="x")

        # ── Training ──────────────────────────────────────────────────────
        sec3 = section("③ TRAINING MODE  (Good Part)")
        self._train_btn = tk.Button(sec3, text="▶  Start Training",
                                    command=self._start_training,
                                    bg=THEME["green"], fg="white",
                                    relief="flat", padx=10, pady=6,
                                    font=("Helvetica", 10, "bold"))
        self._train_btn.pack(padx=10, pady=(8, 4), fill="x")

        tk.Button(sec3, text="💾  Save Baseline",
                  command=self._save_baseline,
                  bg=THEME["card"], fg=THEME["text"],
                  relief="flat", padx=10, pady=4,
                  font=("Helvetica", 9)).pack(padx=10, pady=2, fill="x")

        tk.Button(sec3, text="📂  Load Baseline",
                  command=self._load_baseline,
                  bg=THEME["card"], fg=THEME["text"],
                  relief="flat", padx=10, pady=4,
                  font=("Helvetica", 9)).pack(padx=10, pady=(2, 10), fill="x")

        # ── Testing ───────────────────────────────────────────────────────
        sec4 = section("④ TESTING MODE  (Unknown Part)")
        self._test_btn = tk.Button(sec4, text="🔍  Start Testing",
                                   command=self._start_testing,
                                   bg="#E91E63", fg="white",
                                   relief="flat", padx=10, pady=6,
                                   font=("Helvetica", 10, "bold"))
        self._test_btn.pack(padx=10, pady=(8, 10), fill="x")

        # ── Live sensor ───────────────────────────────────────────────────
        sec5 = section("LIVE SENSOR READINGS")
        grid = tk.Frame(sec5, bg=THEME["card"])
        grid.pack(fill="x", padx=10, pady=6)
        self._sensor_bars = []
        for i in range(NUM_SENSORS):
            tk.Label(grid, text=f"S{i+1}", width=3, font=("Helvetica", 8, "bold"),
                     bg=THEME["card"], fg=THEME["subtext"]).grid(
                     row=i, column=0, sticky="w", pady=1)
            bar_bg = tk.Frame(grid, bg=THEME["bg"], width=160, height=14)
            bar_bg.grid(row=i, column=1, padx=4, sticky="w")
            bar_bg.pack_propagate(False)
            bar = tk.Frame(bar_bg, bg=THEME["accent"], width=0, height=14)
            bar.place(x=0, y=0, height=14)
            val_lbl = tk.Label(grid, text="0", width=5,
                               font=("Helvetica", 8),
                               bg=THEME["card"], fg=THEME["subtext"])
            val_lbl.grid(row=i, column=2, padx=2)
            self._sensor_bars.append((bar, val_lbl, bar_bg))

        # fusion row
        frow = tk.Frame(sec5, bg=THEME["card"])
        frow.pack(fill="x", padx=10, pady=(0, 10))
        self._fusion_labels = {}
        for col, (key, lbl) in enumerate([("avg", "W.Avg"),
                                          ("dom", "Dominant"),
                                          ("vote", "Vote")]):
            cell = tk.Frame(frow, bg=THEME["bg"], bd=0)
            cell.pack(side="left", expand=True, fill="x", padx=2)
            tk.Label(cell, text=lbl, font=("Helvetica", 7),
                     bg=THEME["bg"], fg=THEME["subtext"]).pack()
            lv = tk.Label(cell, text="—", font=("Helvetica", 11, "bold"),
                          bg=THEME["bg"], fg=THEME["accent"])
            lv.pack()
            self._fusion_labels[key] = lv

    # ── right panel ──────────────────────────────────────────────────────────

    def _build_right_panel(self):
        p = self._right

        # top status bar
        self._status_frame = tk.Frame(p, bg=THEME["card"], height=44)
        self._status_frame.pack(fill="x", pady=(0, 8))
        self._status_frame.pack_propagate(False)
        self._status_lbl = tk.Label(self._status_frame,
                                    text="Ready — Connect Arduino and configure inspection points.",
                                    font=("Helvetica", 10), bg=THEME["card"],
                                    fg=THEME["text"])
        self._status_lbl.pack(side="left", padx=16, pady=8)
        self._mode_badge = tk.Label(self._status_frame, text="  IDLE  ",
                                    font=("Helvetica", 9, "bold"),
                                    bg=THEME["subtext"], fg="white")
        self._mode_badge.pack(side="right", padx=16, pady=8)

        # notebook tabs
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background=THEME["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=THEME["panel"],
                        foreground=THEME["subtext"],
                        padding=[14, 6], font=("Helvetica", 9))
        style.map("TNotebook.Tab",
                  background=[("selected", THEME["accent"])],
                  foreground=[("selected", "white")])

        self._notebook = ttk.Notebook(p)
        self._notebook.pack(fill="both", expand=True)

        self._tab_scan    = tk.Frame(self._notebook, bg=THEME["bg"])
        self._tab_fft     = tk.Frame(self._notebook, bg=THEME["bg"])
        self._tab_results = tk.Frame(self._notebook, bg=THEME["bg"])
        self._tab_log     = tk.Frame(self._notebook, bg=THEME["bg"])

        self._notebook.add(self._tab_scan,    text="  Scan Progress  ")
        self._notebook.add(self._tab_fft,     text="  FFT Spectrum  ")
        self._notebook.add(self._tab_results, text="  Test Results  ")
        self._notebook.add(self._tab_log,     text="  Activity Log  ")

        self._build_scan_tab()
        self._build_fft_tab()
        self._build_results_tab()
        self._build_log_tab()

    # ── scan tab ─────────────────────────────────────────────────────────────

    def _build_scan_tab(self):
        p = self._tab_scan

        instr = tk.Frame(p, bg=THEME["card"])
        instr.pack(fill="x", padx=10, pady=(10, 6))

        self._instr_lbl = tk.Label(instr,
            text="Configure number of inspection points, then click Start Training or Start Testing.",
            font=("Helvetica", 10), bg=THEME["card"], fg=THEME["text"],
            wraplength=720, justify="left")
        self._instr_lbl.pack(anchor="w", padx=14, pady=10)

        # point grid
        grid_frame = tk.Frame(p, bg=THEME["bg"])
        grid_frame.pack(fill="both", expand=True, padx=10, pady=4)

        self._point_canvas = tk.Canvas(grid_frame, bg=THEME["bg"],
                                       highlightthickness=0)
        vsb = ttk.Scrollbar(grid_frame, orient="vertical",
                             command=self._point_canvas.yview)
        self._point_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._point_canvas.pack(fill="both", expand=True)

        self._point_inner = tk.Frame(self._point_canvas, bg=THEME["bg"])
        self._point_canvas.create_window((0, 0), window=self._point_inner,
                                         anchor="nw")
        self._point_inner.bind("<Configure>",
            lambda e: self._point_canvas.configure(
                scrollregion=self._point_canvas.bbox("all")))

        self._point_cells: list[dict] = []

        # capture / skip buttons
        btn_row = tk.Frame(p, bg=THEME["bg"])
        btn_row.pack(fill="x", padx=10, pady=(4, 10))

        self._capture_btn = tk.Button(btn_row, text="🎙  Capture This Point",
                                      command=self._capture_current_point,
                                      bg=THEME["accent"], fg="white",
                                      relief="flat", padx=16, pady=7,
                                      font=("Helvetica", 10, "bold"),
                                      state="disabled")
        self._capture_btn.pack(side="left", padx=(0, 8))

        self._skip_btn = tk.Button(btn_row, text="Skip",
                                   command=self._skip_point,
                                   bg=THEME["card"], fg=THEME["subtext"],
                                   relief="flat", padx=14, pady=7,
                                   font=("Helvetica", 9),
                                   state="disabled")
        self._skip_btn.pack(side="left")

        self._progress_lbl = tk.Label(btn_row, text="",
                                      font=("Helvetica", 9),
                                      bg=THEME["bg"], fg=THEME["subtext"])
        self._progress_lbl.pack(side="right", padx=8)

    def _rebuild_point_grid(self):
        for w in self._point_inner.winfo_children():
            w.destroy()
        self._point_cells.clear()

        cols = 4
        for i in range(self.num_points):
            r, c = divmod(i, cols)
            cell = tk.Frame(self._point_inner, bg=THEME["card"],
                            width=170, height=90, bd=0)
            cell.grid(row=r, column=c, padx=6, pady=6)
            cell.pack_propagate(False)

            tk.Label(cell, text=f"Point {i+1}",
                     font=("Helvetica", 10, "bold"),
                     bg=THEME["card"], fg=THEME["text"]).pack(pady=(10, 2))

            status_lbl = tk.Label(cell, text="Pending",
                                  font=("Helvetica", 9),
                                  bg=THEME["card"], fg=THEME["subtext"])
            status_lbl.pack()

            freq_lbl = tk.Label(cell, text="",
                                font=("Helvetica", 9),
                                bg=THEME["card"], fg=THEME["subtext"])
            freq_lbl.pack()

            self._point_cells.append({
                "frame": cell,
                "status": status_lbl,
                "freq": freq_lbl,
            })

    def _update_point_cell(self, idx: int, status: str, freq: float = None,
                           passed: bool = None):
        if idx >= len(self._point_cells):
            return
        cell = self._point_cells[idx]

        if passed is None:
            color = THEME["yellow"]
            cell["frame"].configure(bg=THEME["card"])
            cell["status"].configure(bg=THEME["card"])
            cell["freq"].configure(bg=THEME["card"])
        elif passed:
            color = THEME["green"]
            cell["frame"].configure(bg="#1B3A1F")
            cell["status"].configure(bg="#1B3A1F")
            cell["freq"].configure(bg="#1B3A1F")
        else:
            color = THEME["red"]
            cell["frame"].configure(bg="#3A1B1B")
            cell["status"].configure(bg="#3A1B1B")
            cell["freq"].configure(bg="#3A1B1B")

        cell["status"].configure(text=status, fg=color)
        if freq is not None:
            cell["freq"].configure(text=f"{freq:.2f} Hz")

    # ── FFT tab ──────────────────────────────────────────────────────────────

    def _build_fft_tab(self):
        p = self._tab_fft
        self._fft_fig = Figure(figsize=(8, 4.5),
                               facecolor=THEME["bg"])
        self._fft_ax  = self._fft_fig.add_subplot(111)
        self._fft_ax.set_facecolor(THEME["panel"])
        self._fft_ax.tick_params(colors=THEME["subtext"])
        self._fft_ax.spines[:].set_color(THEME["border"])
        self._fft_ax.set_xlabel("Frequency (Hz)", color=THEME["subtext"])
        self._fft_ax.set_ylabel("Magnitude", color=THEME["subtext"])
        self._fft_ax.set_title("FFT Spectrum — Waiting for data…",
                               color=THEME["text"])

        self._fft_canvas = FigureCanvasTkAgg(self._fft_fig, master=p)
        self._fft_canvas.get_tk_widget().pack(fill="both", expand=True,
                                              padx=10, pady=10)

    def _plot_fft(self, freqs, mags, title="FFT Spectrum",
                  peak_freq=None, baseline_freq=None):
        ax = self._fft_ax
        ax.clear()
        ax.set_facecolor(THEME["panel"])
        ax.tick_params(colors=THEME["subtext"])
        ax.spines[:].set_color(THEME["border"])

        if len(freqs) > 0:
            ax.plot(freqs, mags, color=THEME["accent"], linewidth=1.2)
            ax.fill_between(freqs, mags, alpha=0.18, color=THEME["accent"])
            if peak_freq:
                ax.axvline(peak_freq, color=THEME["yellow"],
                           linewidth=1.5, linestyle="--",
                           label=f"Measured: {peak_freq:.2f} Hz")
            if baseline_freq:
                ax.axvline(baseline_freq, color=THEME["green"],
                           linewidth=1.5, linestyle=":",
                           label=f"Baseline: {baseline_freq:.2f} Hz")
            if peak_freq or baseline_freq:
                ax.legend(facecolor=THEME["card"], labelcolor=THEME["text"],
                          framealpha=0.8)

        ax.set_xlabel("Frequency (Hz)", color=THEME["subtext"])
        ax.set_ylabel("Magnitude", color=THEME["subtext"])
        ax.set_title(title, color=THEME["text"])
        self._fft_canvas.draw()

    # ── results tab ──────────────────────────────────────────────────────────

    def _build_results_tab(self):
        p = self._tab_results

        self._results_summary = tk.Label(p, text="No test results yet.",
                                         font=("Helvetica", 12),
                                         bg=THEME["bg"], fg=THEME["subtext"])
        self._results_summary.pack(pady=16)

        cols = ("Point", "Baseline Hz", "Measured Hz", "Deviation %", "Result")
        self._results_tree = ttk.Treeview(p, columns=cols, show="headings",
                                          height=14)
        style = ttk.Style()
        style.configure("Treeview", background=THEME["card"],
                        fieldbackground=THEME["card"],
                        foreground=THEME["text"],
                        rowheight=28, font=("Helvetica", 9))
        style.configure("Treeview.Heading", background=THEME["accent"],
                        foreground="white", font=("Helvetica", 9, "bold"))
        style.map("Treeview", background=[("selected", THEME["border"])])

        for c in cols:
            self._results_tree.heading(c, text=c)
            self._results_tree.column(c, width=160, anchor="center")

        self._results_tree.tag_configure("pass", foreground=THEME["green"])
        self._results_tree.tag_configure("fail", foreground=THEME["red"])
        self._results_tree.tag_configure("skip", foreground=THEME["subtext"])

        self._results_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _refresh_results_table(self):
        for row in self._results_tree.get_children():
            self._results_tree.delete(row)

        passed = failed = skipped = 0
        for i, res in self._test_results.items():
            baseline_freq = self.baseline.get(i, {}).get("freq", 0)
            if res.get("skip"):
                self._results_tree.insert("", "end", values=(
                    f"Point {i+1}", f"{baseline_freq:.2f}", "—", "—", "SKIP"
                ), tags=("skip",))
                skipped += 1
            else:
                mf  = res["freq"]
                dev = res["dev"] * 100
                ok  = res["pass"]
                tag = "pass" if ok else "fail"
                self._results_tree.insert("", "end", values=(
                    f"Point {i+1}",
                    f"{baseline_freq:.2f}",
                    f"{mf:.2f}",
                    f"{dev:.1f}",
                    "✅ PASS" if ok else "❌ FAIL"
                ), tags=(tag,))
                if ok: passed += 1
                else:  failed += 1

        total = passed + failed + skipped
        color = THEME["green"] if failed == 0 else THEME["red"]
        verdict = "PART OK ✅" if failed == 0 and total > 0 else \
                  ("PART DEFECTIVE ❌" if failed > 0 else "No results")
        self._results_summary.configure(
            text=f"{verdict}   |   Pass: {passed}  Fail: {failed}  Skip: {skipped}",
            fg=color)

    # ── log tab ──────────────────────────────────────────────────────────────

    def _build_log_tab(self):
        p = self._tab_log
        self._log_text = tk.Text(p, bg=THEME["panel"], fg=THEME["text"],
                                 font=("Consolas", 9),
                                 insertbackground=THEME["text"],
                                 state="disabled", wrap="word")
        vsb = ttk.Scrollbar(p, orient="vertical",
                             command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _log(self, msg: str, level: str = "INFO"):
        ts = time.strftime("%H:%M:%S")
        colors = {"INFO": THEME["text"], "OK": THEME["green"],
                  "FAIL": THEME["red"], "WARN": THEME["yellow"]}
        tag = level
        self._log_text.configure(state="normal")
        self._log_text.insert("end", f"[{ts}] [{level}]  {msg}\n")
        self._log_text.tag_add(tag,
            f"end - {len(msg)+22}c", "end - 1c")
        self._log_text.tag_configure(tag, foreground=colors.get(level, THEME["text"]))
        self._log_text.configure(state="disabled")
        self._log_text.see("end")

    # ══════════════════════════════════════════════════════════════════════════
    # Event handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_ports(self):
        ports = ArduinoReader.list_ports()
        self._port_combo["values"] = ports
        if ports:
            self._port_var.set(ports[0])

    def _toggle_connect(self):
        if self.arduino.is_connected():
            self.arduino.disconnect()
            self._connect_btn.configure(text="Connect", bg=THEME["accent"])
            self._conn_indicator.configure(text="● Disconnected",
                                           fg=THEME["red"])
            self._log("Disconnected from Arduino.")
        else:
            port = self._port_var.get()
            if not port:
                messagebox.showwarning("No port", "Select a COM port first.")
                return
            self._log(f"Connecting to {port}…")
            ok = self.arduino.connect(port)
            if ok:
                self._connect_btn.configure(text="Disconnect",
                                            bg="#607D8B")
                self._conn_indicator.configure(
                    text=f"● Connected  ({port})", fg=THEME["green"])
                self._log(f"Connected to Arduino on {port}.", "OK")
            else:
                messagebox.showerror("Connection failed",
                                     f"Could not open {port}.")
                self._log(f"Failed to connect to {port}.", "FAIL")

    def _apply_config(self):
        n = self._pts_var.get()
        self.num_points = n
        global FREQ_TOLERANCE
        FREQ_TOLERANCE = self._tol_var.get() / 100.0
        self.baseline.clear()
        self.test_results.clear()
        self._rebuild_point_grid()
        self._log(f"Configuration applied: {n} points, ±{self._tol_var.get()}% tolerance.")
        self._set_status("Configuration applied. Ready for Training or Testing.")

    def _set_status(self, msg: str, mode: str = None):
        self._status_lbl.configure(text=msg)
        if mode:
            self.mode = mode
            badge_colors = {
                "idle":     (THEME["subtext"], "white"),
                "training": (THEME["green"],   "white"),
                "testing":  ("#E91E63",         "white"),
                "done":     (THEME["accent"],   "white"),
            }
            bg, fg = badge_colors.get(mode, (THEME["subtext"], "white"))
            self._mode_badge.configure(text=f"  {mode.upper()}  ",
                                       bg=bg, fg=fg)

    # ── training ─────────────────────────────────────────────────────────────

    def _start_training(self):
        if self.num_points == 0:
            messagebox.showwarning("Not configured",
                                   "Apply configuration first.")
            return
        if not self.arduino.is_connected():
            ans = messagebox.askyesno("No Arduino",
                "Arduino is not connected.\n\nRun in DEMO mode (simulated data)?")
            if not ans:
                return
        self.baseline.clear()
        self._rebuild_point_grid()
        self.current_point = 0
        self.mode = "training"
        self._set_status("TRAINING — Place GOOD PART. Position at Point 1 and press Capture.",
                         "training")
        self._instr_lbl.configure(
            text="TRAINING MODE — Place a known-good part.\n"
                 "Position the sensor/speaker at each marked point, then press 'Capture This Point'.\n"
                 "A chirp will be played; sensor data is recorded and analysed.")
        self._capture_btn.configure(state="normal")
        self._skip_btn.configure(state="normal")
        self._update_point_cell(0, "▶ Active", passed=None)
        self._progress_lbl.configure(
            text=f"Point 1 / {self.num_points}")
        self._notebook.select(0)
        self._log("Training started.", "INFO")

    # ── testing ──────────────────────────────────────────────────────────────

    def _start_testing(self):
        if self.num_points == 0:
            messagebox.showwarning("Not configured",
                                   "Apply configuration first.")
            return
        if not self.baseline:
            messagebox.showwarning("No baseline",
                                   "Train on a good part first (or load a baseline).")
            return
        self.test_results.clear()
        self._rebuild_point_grid()
        self.current_point = 0
        self.mode = "testing"
        self._set_status("TESTING — Place UNKNOWN PART. Position at Point 1 and press Capture.",
                         "testing")
        self._instr_lbl.configure(
            text="TESTING MODE — Place the part under test.\n"
                 "Position at each point and press 'Capture This Point'.\n"
                 "Results are compared against the trained baseline automatically.")
        self._capture_btn.configure(state="normal")
        self._skip_btn.configure(state="normal")
        self._update_point_cell(0, "▶ Active", passed=None)
        self._progress_lbl.configure(
            text=f"Point 1 / {self.num_points}")
        self._notebook.select(0)
        self._log("Testing started.", "INFO")

    # ── capture ──────────────────────────────────────────────────────────────

    def _capture_current_point(self):
        if self._capture_active:
            return
        self._capture_active = True
        self._capture_btn.configure(state="disabled", text="⏺  Recording…")
        threading.Thread(target=self._do_capture, daemon=True).start()

    def _do_capture(self):
        idx = self.current_point
        self._log(f"Capturing Point {idx+1}…")
        self.after(0, lambda: self._update_point_cell(idx, "⏺ Recording…"))

        # Play chirp
        play_chirp_async()
        self.after(0, lambda: self._set_status(
            f"Playing chirp for Point {idx+1}… hold steady.", self.mode))

        # Record sensor data
        self.arduino.start_capture()
        time.sleep(RECORD_SECONDS)
        stop_audio()
        raw_data = self.arduino.stop_capture()

        # If no Arduino → generate simulated data
        if not raw_data or len(raw_data) < 10:
            raw_data = self._simulate_data(idx)

        # FFT analysis
        freq, xf, yf = extract_dominant_frequency(raw_data, channel=6)
        self._log(f"  Point {idx+1}: dominant freq = {freq:.2f} Hz  ({len(raw_data)} samples)")

        # Update plot on main thread
        title = f"Point {idx+1} — {'Training' if self.mode=='training' else 'Testing'}"
        baseline_f = self.baseline.get(idx, {}).get("freq")
        self.after(0, lambda: self._plot_fft(xf, yf, title, freq, baseline_f))
        self.after(0, lambda: self._notebook.select(1))

        if self.mode == "training":
            self.baseline[idx] = {"freq": freq, "raw": len(raw_data)}
            self.after(0, lambda: self._update_point_cell(idx, f"✓ {freq:.1f} Hz",
                                                          freq=freq, passed=True))
            self._log(f"  Baseline stored: {freq:.2f} Hz", "OK")
        else:
            base_f = self.baseline.get(idx, {}).get("freq", 0)
            passed, dev = compare_frequency(freq, base_f)
            self.test_results[idx] = {
                "freq": freq, "pass": passed, "dev": dev,
                "fft_x": xf.tolist(), "fft_y": yf.tolist()
            }
            label = "✅ PASS" if passed else "❌ FAIL"
            self.after(0, lambda: self._update_point_cell(
                idx, f"{label}  {freq:.1f} Hz", freq=freq, passed=passed))
            level = "OK" if passed else "FAIL"
            self._log(f"  Point {idx+1}: {label}  measured={freq:.2f}  "
                      f"baseline={base_f:.2f}  dev={dev*100:.1f}%", level)
            self.after(0, self._refresh_results_table)

        self.after(0, self._advance_point)

    def _skip_point(self):
        idx = self.current_point
        self._update_point_cell(idx, "— Skipped")
        if self.mode == "testing":
            self.test_results[idx] = {"skip": True}
            self._refresh_results_table()
        self._log(f"Point {idx+1} skipped.")
        self._advance_point()

    def _advance_point(self):
        self._capture_active = False
        self._capture_btn.configure(state="normal", text="🎙  Capture This Point")

        nxt = self.current_point + 1
        if nxt >= self.num_points:
            self._finish_session()
            return

        self.current_point = nxt
        self._update_point_cell(nxt, "▶ Active")
        self._progress_lbl.configure(
            text=f"Point {nxt+1} / {self.num_points}")
        action = "Training" if self.mode == "training" else "Testing"
        self._set_status(
            f"{action.upper()} — Move to Point {nxt+1} and press Capture.",
            self.mode)
        self._log(f"→ Ready for Point {nxt+1}.")

    def _finish_session(self):
        self._capture_btn.configure(state="disabled")
        self._skip_btn.configure(state="disabled")

        if self.mode == "training":
            self._set_status(
                f"Training complete — baseline stored for {self.num_points} points. "
                "You can now run Testing.", "done")
            self._log("Training session complete.", "OK")
            messagebox.showinfo("Training Complete",
                                f"Baseline captured for {self.num_points} inspection points.\n"
                                "Save the baseline, then switch to Testing Mode.")
        else:
            passed = sum(1 for r in self.test_results.values()
                         if not r.get("skip") and r.get("pass"))
            failed = sum(1 for r in self.test_results.values()
                         if not r.get("skip") and not r.get("pass"))
            verdict = "PART OK ✅" if failed == 0 else "PART DEFECTIVE ❌"
            self._set_status(
                f"Testing complete — {verdict}  (Pass: {passed}  Fail: {failed})",
                "done")
            self._log(f"Testing complete — {verdict}  pass={passed}  fail={failed}",
                      "OK" if failed == 0 else "FAIL")
            self._notebook.select(2)
            messagebox.showinfo("Test Complete",
                                f"Inspection finished.\n\n{verdict}\n"
                                f"Pass: {passed}   Fail: {failed}   "
                                f"Skip: {self.num_points - passed - failed}")

    # ── baseline persistence ──────────────────────────────────────────────────

    def _save_baseline(self):
        if not self.baseline:
            messagebox.showwarning("Nothing to save",
                                   "Complete training first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            initialfile="acoustiscan_baseline.json")
        if not path:
            return
        data = {
            "num_points": self.num_points,
            "tolerance":  self._tol_var.get(),
            "baseline":   {str(k): v for k, v in self.baseline.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._log(f"Baseline saved → {path}", "OK")
        messagebox.showinfo("Saved", f"Baseline saved to\n{path}")

    def _load_baseline(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self.num_points = data["num_points"]
            self._pts_var.set(self.num_points)
            self._tol_var.set(data.get("tolerance", 12))
            self.baseline = {int(k): v for k, v in data["baseline"].items()}
            self._rebuild_point_grid()
            for i, v in self.baseline.items():
                self._update_point_cell(i, f"✓ {v['freq']:.1f} Hz",
                                        freq=v["freq"], passed=True)
            self._log(f"Baseline loaded ← {path}  ({self.num_points} points)", "OK")
            messagebox.showinfo("Loaded",
                                f"Baseline loaded.\n{self.num_points} points ready for Testing.")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    # ── demo / simulation ─────────────────────────────────────────────────────

    def _simulate_data(self, point_idx: int) -> list[list[int]]:
        """Generate realistic fake sensor data for demo / no-Arduino mode."""
        n = RECORD_SECONDS * SAMPLE_RATE_HZ
        t = np.linspace(0, RECORD_SECONDS, n)

        if self.mode == "training":
            # Each point has a fixed 'natural' frequency (good part)
            nat_f = 3.0 + point_idx * 1.7          # Hz in sensor space
        else:
            # Testing: 80 % chance same freq, 20 % chance shifted (defect)
            base_f = 3.0 + point_idx * 1.7
            if np.random.random() < 0.25:
                nat_f = base_f * (1 + np.random.uniform(0.2, 0.5))  # defect
            else:
                nat_f = base_f * (1 + np.random.uniform(-0.05, 0.05))  # ok

        base_amp = 400 + point_idx * 30
        sig = (base_amp * np.sin(2 * np.pi * nat_f * t)
               + np.random.normal(0, 40, n) + 512).clip(0, 1023).astype(int)

        rows = []
        for i in range(n):
            row = [int(sig[i] * (0.85 + 0.3 * np.random.random()))
                   for _ in range(NUM_SENSORS)]
            avg = int(np.mean(row))
            dom = int(np.max(row))
            vote = 1 if sum(v > 512 for v in row) >= 3 else 0
            rows.append(row + [avg, dom, vote * 1023])
        return rows

    # ── live sensor display ───────────────────────────────────────────────────

    def _update_status_loop(self):
        vals = self.arduino.get_latest()
        for i, (bar, lbl, bg) in enumerate(self._sensor_bars):
            v = vals[i] if i < len(vals) else 0
            w = max(1, int(160 * v / 1023))
            bar.place(x=0, y=0, width=w, height=14)
            lbl.configure(text=str(v))

        if len(vals) >= 9:
            self._fusion_labels["avg"].configure(text=str(vals[6]))
            self._fusion_labels["dom"].configure(text=str(vals[7]))
            vote = "YES" if vals[8] > 500 else "NO"
            self._fusion_labels["vote"].configure(text=vote)

        self.after(100, self._update_status_loop)


# ─────────────────────────── Entry point ──────────────────────────────────────

if __name__ == "__main__":
    app = AcoustiScanApp()
    app.mainloop()