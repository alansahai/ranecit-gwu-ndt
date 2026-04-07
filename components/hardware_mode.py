"""
hardware_mode.py — Real-Time Hardware NDT Mode (Arduino + KY-038 Sensors)
Fully integrated into Streamlit.  Works in two sub-modes:
  • LIVE MODE  — reads from Arduino via pyserial
  • DEMO MODE  — generates realistic simulated sensor data

Workflow: Connect → Configure → Train (good part) → Test (unknown part)
"""

import time, json, threading
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Optional imports (graceful fallback) ─────────────────────
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import sounddevice as sd
    from scipy import signal as scipy_signal
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

from scipy.fft import fft, fftfreq

# ── Constants ────────────────────────────────────────────────
SERIAL_BAUD       = 115_200
SAMPLE_RATE_HZ    = 50
RECORD_SECONDS    = 3
CHIRP_DURATION    = 2.0
CHIRP_F_START     = 200
CHIRP_F_END       = 4_000
AUDIO_SAMPLE_RATE = 44_100
NUM_SENSORS       = 6

COLORS = {
    "bg":     "#080c18",  "panel":  "#0d1828",  "accent": "#00d4ff",
    "green":  "#00ff88",  "red":    "#ff3366",  "warn":   "#ffaa00",
    "text":   "#c8d8f0",  "grid":   "#1a2d45",
}

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="monospace", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ══════════════════════════════════════════════════════════════
# Audio helpers
# ══════════════════════════════════════════════════════════════

def generate_chirp():
    t = np.linspace(0, CHIRP_DURATION, int(AUDIO_SAMPLE_RATE * CHIRP_DURATION))
    if AUDIO_AVAILABLE:
        chirp = scipy_signal.chirp(t, f0=CHIRP_F_START, f1=CHIRP_F_END,
                                   t1=CHIRP_DURATION, method="linear")
    else:
        chirp = np.sin(2 * np.pi * 1000 * t)
    return (chirp * 0.8).astype(np.float32)


def play_chirp():
    if AUDIO_AVAILABLE:
        wave = generate_chirp()
        sd.play(wave, samplerate=AUDIO_SAMPLE_RATE)


def stop_audio():
    if AUDIO_AVAILABLE:
        sd.stop()


# ══════════════════════════════════════════════════════════════
# Serial reader (thread-safe)
# ══════════════════════════════════════════════════════════════

class ArduinoReader:
    def __init__(self):
        self.ser = None
        self._running = False
        self._thread = None
        self.latest = [0] * 9
        self.buffer = []
        self._lock = threading.Lock()

    def connect(self, port):
        if not SERIAL_AVAILABLE:
            return False
        try:
            self.ser = serial.Serial(port, SERIAL_BAUD, timeout=1)
            time.sleep(2)
            self.ser.flushInput()
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            return True
        except Exception:
            return False

    def disconnect(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.ser.close()

    def is_connected(self):
        return self.ser is not None and self.ser.is_open

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

    def start_capture(self):
        with self._lock:
            self.buffer.clear()

    def stop_capture(self):
        with self._lock:
            data = list(self.buffer)
            self.buffer.clear()
        return data

    def get_latest(self):
        with self._lock:
            return list(self.latest)

    @staticmethod
    def list_ports():
        if not SERIAL_AVAILABLE:
            return []
        return [p.device for p in serial.tools.list_ports.comports()]


# ══════════════════════════════════════════════════════════════
# FFT analysis
# ══════════════════════════════════════════════════════════════

def extract_dominant_frequency(samples, channel=6):
    if len(samples) < 8:
        return 0.0, np.array([]), np.array([])
    y = np.array([row[channel] for row in samples], dtype=float)
    y -= y.mean()
    n = len(y)
    yf = np.abs(fft(y))[:n // 2]
    xf = fftfreq(n, d=1.0 / SAMPLE_RATE_HZ)[:n // 2]
    mask = xf > 0.5
    xf_m, yf_m = xf[mask], yf[mask]
    if len(yf_m) == 0:
        return 0.0, xf, yf
    peak_idx = np.argmax(yf_m)
    return float(xf_m[peak_idx]), xf, yf


def compare_frequency(measured, baseline, tolerance=0.12):
    if baseline == 0:
        return False, 1.0
    dev = abs(measured - baseline) / baseline
    return dev <= tolerance, dev


# ══════════════════════════════════════════════════════════════
# Demo data generator
# ══════════════════════════════════════════════════════════════

def simulate_sensor_data(point_idx, mode="training", num_sensors=6):
    n = RECORD_SECONDS * SAMPLE_RATE_HZ
    t = np.linspace(0, RECORD_SECONDS, n)
    rng = np.random.default_rng(point_idx * 100 + (42 if mode == "training" else 99))

    if mode == "training":
        nat_f = 3.0 + point_idx * 1.7
    else:
        base_f = 3.0 + point_idx * 1.7
        if rng.random() < 0.25:
            nat_f = base_f * (1 + rng.uniform(0.2, 0.5))
        else:
            nat_f = base_f * (1 + rng.uniform(-0.05, 0.05))

    base_amp = 400 + point_idx * 30
    sig = (base_amp * np.sin(2 * np.pi * nat_f * t)
           + rng.normal(0, 40, n) + 512).clip(0, 1023).astype(int)

    rows = []
    for i in range(n):
        row = [int(sig[i] * (0.85 + 0.3 * rng.random())) for _ in range(num_sensors)]
        avg = int(np.mean(row))
        dom = int(np.max(row))
        vote = 1 if sum(v > 512 for v in row) >= 3 else 0
        rows.append(row + [avg, dom, vote * 1023])
    return rows


# ══════════════════════════════════════════════════════════════
# Plotly helpers
# ══════════════════════════════════════════════════════════════

def plot_fft_spectrum(freqs, mags, peak_freq=None, baseline_freq=None, title="FFT Spectrum"):
    fig = go.Figure()
    if len(freqs) > 0:
        fig.add_trace(go.Scatter(x=freqs, y=mags, mode="lines",
                                  line=dict(color=COLORS["accent"], width=1.5),
                                  fill="tozeroy", fillcolor="rgba(0,212,255,0.1)"))
        if peak_freq:
            fig.add_vline(x=peak_freq, line=dict(color=COLORS["warn"], dash="dash", width=1.5),
                          annotation_text=f"Measured: {peak_freq:.2f} Hz",
                          annotation_font_color=COLORS["warn"])
        if baseline_freq:
            fig.add_vline(x=baseline_freq, line=dict(color=COLORS["green"], dash="dot", width=1.5),
                          annotation_text=f"Baseline: {baseline_freq:.2f} Hz",
                          annotation_font_color=COLORS["green"])
    fig.update_layout(**LAYOUT_BASE, height=350,
                      title=dict(text=title, font=dict(color=COLORS["accent"], size=13)),
                      xaxis=dict(title="Frequency (Hz)", gridcolor=COLORS["grid"]),
                      yaxis=dict(title="Magnitude", gridcolor=COLORS["grid"]))
    return fig


def plot_sensor_bars(vals):
    sensors = [f"S{i+1}" for i in range(NUM_SENSORS)]
    colors = [COLORS["accent"] if v < 700 else COLORS["warn"] if v < 900 else COLORS["red"]
              for v in vals[:NUM_SENSORS]]
    fig = go.Figure(go.Bar(x=sensors, y=vals[:NUM_SENSORS],
                           marker_color=colors, opacity=0.9))
    fig.update_layout(**LAYOUT_BASE, height=200,
                      title=dict(text="Live Sensor Readings", font=dict(color=COLORS["accent"], size=11)),
                      yaxis=dict(range=[0, 1023], gridcolor=COLORS["grid"]),
                      xaxis=dict(gridcolor=COLORS["grid"]))
    return fig


# ══════════════════════════════════════════════════════════════
# Main Streamlit page renderer
# ══════════════════════════════════════════════════════════════

def render_hardware_page():
    """Render the complete Real-Time Hardware NDT page."""

    # ── Session state init ──────────────────────────────────
    defaults = {
        "hw_arduino": ArduinoReader(),
        "hw_baseline": {},
        "hw_test_results": {},
        "hw_num_points": 5,
        "hw_tolerance": 12,
        "hw_mode": "idle",
        "hw_current_point": 0,
        "hw_log": [],
        "hw_demo_mode": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    def log(msg, level="INFO"):
        ts = time.strftime("%H:%M:%S")
        st.session_state.hw_log.append(f"[{ts}] [{level}] {msg}")

    # ── Header ──────────────────────────────────────────────
    st.markdown("""<div style="background:linear-gradient(90deg,#6C63FF0d,transparent 80%);
    border-left:4px solid #6C63FF;padding:14px 22px;margin-bottom:18px;border-radius:0 10px 10px 0">
    <h2 style="font-family:'Share Tech Mono',monospace;color:#6C63FF;margin:0;letter-spacing:.06em">
    📡 REAL-TIME HARDWARE MODE</h2>
    <p style="color:#9E9EC8;margin:4px 0 0;font-size:.85rem">
    Arduino + 6× KY-038 Sound Sensors · FFT-Based Acoustic Inspection · Train & Test Workflow</p>
    </div>""", unsafe_allow_html=True)

    # ── Connection & Config Panel ───────────────────────────
    cc1, cc2, cc3 = st.columns([1, 1, 1])

    with cc1:
        st.markdown("""<div style="background:#0a1525;border:1px solid #162a42;
        border-top:2px solid #6C63FF33;border-radius:8px;padding:14px;margin-bottom:12px">
        <h4 style="font-family:monospace;color:#6C63FF;font-size:.82rem;letter-spacing:.1em;
        text-transform:uppercase;margin:0 0 10px">① ARDUINO CONNECTION</h4></div>""",
                    unsafe_allow_html=True)

        ports = ArduinoReader.list_ports()
        use_demo = st.checkbox("🎮 Demo Mode (no Arduino)", value=True,
                               help="Simulate sensor data without physical hardware")
        st.session_state.hw_demo_mode = use_demo

        if not use_demo:
            if ports:
                port = st.selectbox("COM Port", ports)
                c1b, c2b = st.columns(2)
                with c1b:
                    if st.button("🔌 Connect", width="stretch"):
                        if st.session_state.hw_arduino.connect(port):
                            log(f"Connected to {port}", "OK")
                            st.success(f"Connected to {port}")
                        else:
                            log(f"Failed to connect to {port}", "FAIL")
                            st.error("Connection failed")
                with c2b:
                    if st.button("⏏ Disconnect", width="stretch"):
                        st.session_state.hw_arduino.disconnect()
                        log("Disconnected", "INFO")
                        st.info("Disconnected")

                if st.session_state.hw_arduino.is_connected():
                    st.markdown("🟢 **Connected**")
                else:
                    st.markdown("🔴 **Disconnected**")
            else:
                st.warning("No COM ports detected. Use Demo Mode.")
                st.session_state.hw_demo_mode = True
        else:
            st.info("🎮 Running in Demo Mode — simulated sensor data")

    with cc2:
        st.markdown("""<div style="background:#0a1525;border:1px solid #162a42;
        border-top:2px solid #6C63FF33;border-radius:8px;padding:14px;margin-bottom:12px">
        <h4 style="font-family:monospace;color:#6C63FF;font-size:.82rem;letter-spacing:.1em;
        text-transform:uppercase;margin:0 0 10px">② PART CONFIGURATION</h4></div>""",
                    unsafe_allow_html=True)

        num_pts = st.number_input("Inspection Points", 1, 20, 5)
        tolerance = st.number_input("Tolerance (%)", 1, 50, 12)
        if st.button("✅ Apply Configuration", width="stretch"):
            st.session_state.hw_num_points = num_pts
            st.session_state.hw_tolerance = tolerance
            st.session_state.hw_baseline = {}
            st.session_state.hw_test_results = {}
            st.session_state.hw_current_point = 0
            st.session_state.hw_mode = "idle"
            log(f"Config applied: {num_pts} points, ±{tolerance}%")
            st.success(f"Configuration applied: {num_pts} points")

    with cc3:
        st.markdown("""<div style="background:#0a1525;border:1px solid #162a42;
        border-top:2px solid #6C63FF33;border-radius:8px;padding:14px;margin-bottom:12px">
        <h4 style="font-family:monospace;color:#6C63FF;font-size:.82rem;letter-spacing:.1em;
        text-transform:uppercase;margin:0 0 10px">③ WORKFLOW CONTROL</h4></div>""",
                    unsafe_allow_html=True)

        wc1, wc2 = st.columns(2)
        with wc1:
            if st.button("🟢 Start Training", width="stretch", type="primary"):
                st.session_state.hw_mode = "training"
                st.session_state.hw_current_point = 0
                st.session_state.hw_baseline = {}
                log("Training mode started", "INFO")

        with wc2:
            if st.button("🔴 Start Testing", width="stretch"):
                if not st.session_state.hw_baseline:
                    st.error("Train on a good part first!")
                else:
                    st.session_state.hw_mode = "testing"
                    st.session_state.hw_current_point = 0
                    st.session_state.hw_test_results = {}
                    log("Testing mode started", "INFO")

        mode = st.session_state.hw_mode
        mode_colors = {"idle": "#9E9EC8", "training": "#4CAF50", "testing": "#E91E63", "done": "#6C63FF"}
        st.markdown(f'<div style="background:{mode_colors.get(mode,"#9E9EC8")}22;'
                    f'border:1px solid {mode_colors.get(mode,"#9E9EC8")};border-radius:6px;'
                    f'padding:8px;text-align:center;margin-top:8px">'
                    f'<span style="color:{mode_colors.get(mode,"#9E9EC8")};font-family:monospace;'
                    f'font-weight:700;letter-spacing:.1em">{mode.upper()}</span></div>',
                    unsafe_allow_html=True)

        # Baseline save/load
        st.markdown("---")
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.session_state.hw_baseline:
                bl_data = json.dumps({
                    "num_points": st.session_state.hw_num_points,
                    "tolerance": st.session_state.hw_tolerance,
                    "baseline": {str(k): v for k, v in st.session_state.hw_baseline.items()},
                }, indent=2)
                st.download_button("💾 Save Baseline", bl_data,
                                   "acoustiscan_baseline.json", "application/json",
                                   width="stretch")
        with bc2:
            uploaded = st.file_uploader("📂 Load", type=["json"], label_visibility="collapsed")
            if uploaded:
                data = json.load(uploaded)
                st.session_state.hw_num_points = data["num_points"]
                st.session_state.hw_tolerance = data.get("tolerance", 12)
                st.session_state.hw_baseline = {int(k): v for k, v in data["baseline"].items()}
                log(f"Baseline loaded: {data['num_points']} points", "OK")
                st.success("Baseline loaded!")

    st.markdown("---")

    # ── Active Capture Section ──────────────────────────────
    mode = st.session_state.hw_mode
    if mode in ("training", "testing"):
        cp = st.session_state.hw_current_point
        total = st.session_state.hw_num_points

        if cp < total:
            st.markdown(f"""<div style="background:linear-gradient(135deg,#0d1e32,#0a1422);
            border:1px solid #1a3354;border-radius:10px;padding:16px;text-align:center;
            margin-bottom:14px">
            <span style="color:#4a7090;font-size:.78rem;text-transform:uppercase;
            letter-spacing:.1em;font-family:monospace">CURRENT TARGET</span><br>
            <span style="color:#00d4ff;font-size:2rem;font-weight:700;
            font-family:'Share Tech Mono',monospace">Point {cp+1} / {total}</span><br>
            <span style="color:#4a7090;font-size:.82rem">
            {'Place GOOD part at this point and capture' if mode == 'training'
             else 'Place UNKNOWN part at this point and capture'}</span>
            </div>""", unsafe_allow_html=True)

            cap_col, skip_col = st.columns([3, 1])
            with cap_col:
                capture = st.button("🎙 CAPTURE THIS POINT", width="stretch", type="primary")
            with skip_col:
                skip = st.button("⏭ Skip", width="stretch")

            if capture:
                with st.spinner(f"📡 Capturing Point {cp+1}... Playing chirp & recording ({RECORD_SECONDS}s)"):
                    play_chirp()

                    if not st.session_state.hw_demo_mode and st.session_state.hw_arduino.is_connected():
                        st.session_state.hw_arduino.start_capture()
                        time.sleep(RECORD_SECONDS)
                        stop_audio()
                        raw_data = st.session_state.hw_arduino.stop_capture()
                    else:
                        time.sleep(1)
                        stop_audio()
                        raw_data = simulate_sensor_data(cp, mode)

                    freq, xf, yf = extract_dominant_frequency(raw_data, channel=6)
                    log(f"Point {cp+1}: dominant freq = {freq:.2f} Hz ({len(raw_data)} samples)")

                    # Show FFT
                    baseline_f = st.session_state.hw_baseline.get(cp, {}).get("freq")
                    fig = plot_fft_spectrum(xf, yf, peak_freq=freq, baseline_freq=baseline_f,
                                            title=f"Point {cp+1} — {'Training' if mode == 'training' else 'Testing'}")
                    st.plotly_chart(fig, width="stretch")

                    if mode == "training":
                        st.session_state.hw_baseline[cp] = {"freq": freq, "raw": len(raw_data)}
                        log(f"  Baseline stored: {freq:.2f} Hz", "OK")
                        st.success(f"✅ Point {cp+1} baseline: {freq:.2f} Hz")
                    else:
                        base_f = st.session_state.hw_baseline.get(cp, {}).get("freq", 0)
                        tol = st.session_state.hw_tolerance / 100.0
                        passed, dev = compare_frequency(freq, base_f, tol)
                        st.session_state.hw_test_results[cp] = {
                            "freq": freq, "pass": passed, "dev": dev
                        }
                        if passed:
                            st.success(f"✅ PASS — {freq:.2f} Hz (dev: {dev*100:.1f}%)")
                            log(f"  Point {cp+1}: PASS  dev={dev*100:.1f}%", "OK")
                        else:
                            st.error(f"❌ FAIL — {freq:.2f} Hz (dev: {dev*100:.1f}%)")
                            log(f"  Point {cp+1}: FAIL  dev={dev*100:.1f}%", "FAIL")

                    st.session_state.hw_current_point = cp + 1
                    if st.session_state.hw_current_point >= total:
                        st.session_state.hw_mode = "done"
                        log(f"{'Training' if mode == 'training' else 'Testing'} complete!", "OK")
                    st.rerun()

            if skip:
                if mode == "testing":
                    st.session_state.hw_test_results[cp] = {"skip": True}
                log(f"Point {cp+1} skipped")
                st.session_state.hw_current_point = cp + 1
                if st.session_state.hw_current_point >= total:
                    st.session_state.hw_mode = "done"
                st.rerun()
        else:
            st.session_state.hw_mode = "done"

    # ── Point Grid Status ───────────────────────────────────
    st.markdown('<p style="font-family:monospace;color:#00d4ff;font-size:.86rem;'
                'letter-spacing:.13em;text-transform:uppercase;border-bottom:1px solid #162440;'
                'padding-bottom:6px;margin:18px 0 12px">📍 INSPECTION POINT STATUS</p>',
                unsafe_allow_html=True)

    total = st.session_state.hw_num_points
    cols = st.columns(min(total, 5))
    for i in range(total):
        col = cols[i % min(total, 5)]
        with col:
            bl = st.session_state.hw_baseline.get(i)
            tr = st.session_state.hw_test_results.get(i)

            if tr:
                if tr.get("skip"):
                    bg, border, status, color = "#1a1a2e", "#555", "SKIPPED", "#999"
                elif tr.get("pass"):
                    bg, border, status, color = "#0a2a15", "#00ff88", "✅ PASS", "#00ff88"
                else:
                    bg, border, status, color = "#2a0a15", "#ff3366", "❌ FAIL", "#ff3366"
                freq_txt = f"{tr.get('freq', 0):.1f} Hz" if not tr.get("skip") else "—"
            elif bl:
                bg, border, status, color = "#0a1a2a", "#00d4ff", f"✓ {bl['freq']:.1f} Hz", "#00d4ff"
                freq_txt = f"{bl['freq']:.1f} Hz"
            elif i == st.session_state.hw_current_point and st.session_state.hw_mode in ("training", "testing"):
                bg, border, status, color = "#1a1a0a", "#FFC107", "▶ ACTIVE", "#FFC107"
                freq_txt = ""
            else:
                bg, border, status, color = "#111827", "#333", "Pending", "#666"
                freq_txt = ""

            st.markdown(f"""<div style="background:{bg};border:1px solid {border};
            border-radius:8px;padding:10px;text-align:center;margin-bottom:8px">
            <div style="color:#c8d8f0;font-weight:700;font-size:.9rem">Point {i+1}</div>
            <div style="color:{color};font-size:.78rem;font-family:monospace">{status}</div>
            <div style="color:#4a7090;font-size:.72rem">{freq_txt}</div>
            </div>""", unsafe_allow_html=True)

    # ── Test Results Table ──────────────────────────────────
    if st.session_state.hw_test_results:
        st.markdown('<p style="font-family:monospace;color:#00d4ff;font-size:.86rem;'
                    'letter-spacing:.13em;text-transform:uppercase;border-bottom:1px solid #162440;'
                    'padding-bottom:6px;margin:18px 0 12px">📊 TEST RESULTS</p>',
                    unsafe_allow_html=True)

        passed = failed = skipped = 0
        table_rows = []
        for i in range(st.session_state.hw_num_points):
            res = st.session_state.hw_test_results.get(i)
            bl_freq = st.session_state.hw_baseline.get(i, {}).get("freq", 0)
            if res:
                if res.get("skip"):
                    table_rows.append({"Point": f"Point {i+1}", "Baseline Hz": f"{bl_freq:.2f}",
                                       "Measured Hz": "—", "Deviation %": "—", "Result": "⏭ SKIP"})
                    skipped += 1
                else:
                    ok = res["pass"]
                    table_rows.append({
                        "Point": f"Point {i+1}",
                        "Baseline Hz": f"{bl_freq:.2f}",
                        "Measured Hz": f"{res['freq']:.2f}",
                        "Deviation %": f"{res['dev']*100:.1f}",
                        "Result": "✅ PASS" if ok else "❌ FAIL",
                    })
                    if ok: passed += 1
                    else: failed += 1

        import pandas as pd
        df = pd.DataFrame(table_rows)
        st.dataframe(df, width="stretch", hide_index=True)

        verdict_color = "#00ff88" if failed == 0 else "#ff3366"
        verdict_text = "PART OK ✅" if failed == 0 and (passed + skipped) > 0 else \
                       "PART DEFECTIVE ❌" if failed > 0 else "No results"
        st.markdown(f"""<div style="background:linear-gradient(135deg,#0d1e32,#0a1422);
        border:1px solid {verdict_color}44;border-radius:10px;padding:18px;text-align:center;
        margin:12px 0">
        <span style="color:{verdict_color};font-size:1.6rem;font-weight:700;
        font-family:'Share Tech Mono',monospace">{verdict_text}</span><br>
        <span style="color:#4a7090;font-size:.85rem">
        Pass: {passed} &nbsp;|&nbsp; Fail: {failed} &nbsp;|&nbsp; Skip: {skipped}</span>
        </div>""", unsafe_allow_html=True)

    # ── Activity Log ────────────────────────────────────────
    with st.expander("📜 Activity Log", expanded=False):
        if st.session_state.hw_log:
            log_text = "\n".join(reversed(st.session_state.hw_log[-50:]))
            st.code(log_text, language=None)
        else:
            st.info("No activity yet.")

    # ── Hardware Info ────────────────────────────────────────
    with st.expander("ℹ️ Hardware Setup Guide", expanded=False):
        st.markdown("""
### Arduino Hardware Setup
| Component | Specification |
|-----------|--------------|
| **Microcontroller** | Arduino Uno / Mega |
| **Sensors** | 6× KY-038 Sound Sensor Modules |
| **Analog Pins** | A0 – A5 |
| **Digital Pins** | D2 – D7 |
| **Baud Rate** | 115,200 |
| **Protocol** | 9 space-separated values @ 50 Hz |

### Data Protocol
```
Val 1-6 : Raw sensor readings (0-1023)
Val 7   : Weighted-average fusion
Val 8   : Dominant-source index
Val 9   : Majority-vote × 1023
```

### Workflow
1. **Connect** Arduino to COM port
2. **Configure** number of inspection points
3. **Train** — place known-good part, capture baseline FFT
4. **Test** — place unknown part, compare against baseline
        """)
