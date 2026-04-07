# AcoustiScan Pro v3.0

AcoustiScan Pro is a comprehensive Ultrasonic NDT simulator and Real-time Hardware inspection platform for evaluating aluminium die castings. 

## Features (v3.0 Upgrade) 🚀

- **Unified Dashboard:** Seamlessly switch between Simulation Mode and Real-Time Hardware.
- **Real-Time Arduino Integration:** Train and test physical parts using acoustic sensors and FFT analysis.
- **Certification Engine:** Validate simulations against ASTM E127 reference blocks.
- **Deep Physics Simulation:**
  - FEM-based defect geometries with lognormal pore distribution.
  - Granular coupling and surface roughness transmission models.
  - Monte Carlo Probability of Detection (POD) curves per MIL-HDBK-1823A.
  - Phased Array (PAUT) simulation with S-Scan visualization.
  - Thermal expansion and velocity anisotropy mapping.
- **UT vs RT Testing:** Comparison matrix built for engineering decision support.

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Launch execution: `streamlit run main.py`

## Hardware Setup
- Connect Arduino (COM port) with KY-038 sound sensors hooked to A0-A5.
- Enter "Real-Time Hardware" mode inside the app to connect.

## Deployment Ready
Suitable for immediate deployment on Docker, Heroku, Railway, or Streamlit Community Cloud.
*(Note: Hardware COM port features only function locally. Cloud deployments run exclusively in Demo mode).*
