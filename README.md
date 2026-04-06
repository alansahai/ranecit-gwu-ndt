# 🔊 AcoustiScan Pro — Acoustic NDT Simulator

A Streamlit web application that simulates **Acoustic Wave-based Non-Destructive Testing (NDT)** for aluminium casting components.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate demo 3D models (run once)
```bash
python generate_demo_models.py
```

### 3. Launch the app
```bash
streamlit run main.py
```

---

## 📦 Project Structure

```
ndt_app/
├── main.py                    # Streamlit entry point
├── generate_demo_models.py    # One-time demo OBJ generator
├── requirements.txt
├── models/                    # Demo OBJ files
│   ├── aluminium_rod.obj      # Simple rod → LINEAR scan
│   ├── engine_bracket.obj     # L-bracket → GRID scan
│   └── crankcase_block.obj    # Complex block → MULTI-FACE scan
└── components/
    ├── geometry.py            # Mesh loading, analysis, scan point sampling
    ├── acoustic.py            # Acoustic simulation engine
    ├── visualizer.py          # Plotly visualization functions
    └── report.py              # Report & CSV export
```

---

## 🎯 Features

| Feature | Description |
|---|---|
| 3D Model Loading | Upload OBJ/STL or pick from 3 demo components |
| Geometry Analysis | Bounding box, aspect ratio, surface area, volume |
| Scan Classification | Auto-detects LINEAR / GRID / MULTI-FACE scan type |
| Transducer Path | Ray-cast surface sampling with configurable resolution |
| Baseline Simulation | Defect-free acoustic response (freq, amplitude, TOF) |
| Defect 1: Porosity | Random void clusters → amplitude drop + scatter |
| Defect 2: Cold Shut | Planar discontinuity → reflection spike + signal loss |
| Signal Comparison | 3-panel deviation charts per signal metric |
| Defect Heatmap | 2D severity map (Green → Red) projected on XY |
| Report Export | Text inspection report + full CSV scan data |

---

## 🧪 Demo Components

1. **Aluminium Rod** — Simple cylinder → triggers LINEAR scan
2. **Engine Bracket** — L-shaped bracket with bosses → GRID scan  
3. **Crankcase Block** — Complex block with bores, ribs, flanges → MULTI-FACE scan

---

## ⚙️ Tech Stack

- **Frontend**: Streamlit
- **3D Geometry**: trimesh
- **Simulation**: NumPy, SciPy
- **Visualization**: Plotly

---

## 📌 Notes

- Simulation is conceptual and lightweight — not a certified NDT tool
- Aluminium A380 material properties used (density 2700 kg/m³, E=70 GPa)
- Transducer frequency: 2.5 MHz (standard for casting inspection)
