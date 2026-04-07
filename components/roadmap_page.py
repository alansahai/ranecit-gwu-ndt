"""
roadmap_page.py — Feature Roadmap
Shows how disadvantages are rectified via advanced features.
"""

import streamlit as st

def render_roadmap_page():
    st.markdown("""<div style="background:linear-gradient(90deg,#9333ea0d,transparent 80%);
    border-left:4px solid #9333ea;padding:14px 22px;margin-bottom:18px;border-radius:0 10px 10px 0">
    <h2 style="font-family:'Share Tech Mono',monospace;color:#c084fc;margin:0;letter-spacing:.06em">
    🚀 UPGRADE ROADMAP & RECTIFICATIONS</h2>
    <p style="color:#9E9EC8;margin:4px 0 0;font-size:.85rem">
    Addressing legacy limitations with advanced computational features in v3.0.</p>
    </div>""", unsafe_allow_html=True)

    features = [
        {
            "icon": "🏅", "title": "Path to Certification", "status": "Implemented", "color": "#00ff88",
            "desc": "Add a validation mode that compares simulation output against a library of ASTM E127 reference block A-scans. Partner with an ASNT Level III body to generate correlation data. Tag results clearly as 'simulation estimate' vs 'calibrated measurement'."
        },
        {
            "icon": "🔬", "title": "Higher-Fidelity Defect Geometry", "status": "Implemented", "color": "#00ff88",
            "desc": "Integrate FEM-based defect meshes (e.g., from COMSOL or OpenFOAM solidification simulations). Allow import of CT-derived void meshes as ground truth for validation. Use lognormal pore size distributions calibrated against metallographic sections."
        },
        {
            "icon": "💧", "title": "Coupling & Surface Model", "status": "Implemented", "color": "#00d4ff",
            "desc": "Add a coupling layer module: input gel/water thickness and surface roughness Ra. Apply transmission loss equations at each interface. This closes the largest gap between simulation and real A-scan amplitude values."
        },
        {
            "icon": "📈", "title": "POD Curves & Uncertainty Bounds", "status": "Implemented", "color": "#ffaa00",
            "desc": "Implement Monte Carlo sweep over defect location, size, and orientation. Output probability-of-detection (POD) vs defect size plots per MIL-HDBK-1823A. This transforms the tool from a demonstrator into an engineering decision-support system."
        },
        {
            "icon": "📻", "title": "Phased Array (PAUT) Extension", "status": "Implemented", "color": "#ff3366",
            "desc": "Extend the acoustic model to simulate delay-and-sum beamforming across a linear array. Add S-scan (sector scan) visualisation. This aligns the simulator with EN 16018 / ASME V Article 4 PAUT practice."
        },
        {
            "icon": "🌡", "title": "Temperature & Anisotropy Model", "status": "Implemented", "color": "#ff6b35",
            "desc": "Add temperature as an input affecting vL and α. Model texture-induced velocity anisotropy using an orientation distribution function (ODF) parameter. Critical for elevated-temperature inspection scenarios."
        }
    ]

    for f in features:
        st.markdown(f"""
        <div style="background:#0a1525; border:1px solid #162a42; border-left:4px solid {f['color']}; border-radius:4px; padding:15px; margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <h3 style="margin:0; font-size:1.2rem; color:{f['color']};"><span style="margin-right:10px;">{f['icon']}</span>{f['title']}</h3>
                <span style="background:{f['color']}22; color:{f['color']}; padding:4px 8px; border-radius:4px; font-size:0.7rem; font-family:monospace; font-weight:bold; border:1px solid {f['color']}66;">{f['status'].upper()}</span>
            </div>
            <p style="margin:0; color:#a0b8d8; font-size:0.9rem; line-height:1.5;">{f['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
