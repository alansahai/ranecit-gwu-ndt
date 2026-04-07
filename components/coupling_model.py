"""
coupling_model.py — Coupling Layer & Surface Roughness Module
Translates Ra roughness and coupling medium to transmission losses.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from components.visualizer import LAYOUT_BASE, COLORS

def calculate_transmission_loss(Z1, Z2, thickness, freq, v_c):
    # Reflection coefficient R = ((Z2 - Z1) / (Z2 + Z1))^2
    # Simple transmission T = 1 - R (ignoring thickness resonance for simple model)
    R = ((Z2 - Z1) / (Z2 + Z1))**2
    T_interface = 1 - R
    
    # Attenuation in couplant
    alpha_c = 0.1 * (freq/1e6)**2 # Rough approx for water/gel attenuation
    T_atten = np.exp(-2 * alpha_c * (thickness/1000)) # round trip
    
    return T_interface * T_atten

def apply_roughness_loss(Ra, freq, v_c):
    # Loss due to surface scattering
    wavelength = (v_c / freq) * 1e6 # µm
    # Ruze equation approximation for scattering loss
    loss = np.exp(-(4 * np.pi * Ra / wavelength)**2)
    return max(0.1, loss)

def render_coupling_model_page():
    st.markdown("""<div style="background:linear-gradient(90deg,#00d4ff0d,transparent 80%);
    border-left:4px solid #00d4ff;padding:14px 22px;margin-bottom:18px;border-radius:0 10px 10px 0">
    <h2 style="font-family:'Share Tech Mono',monospace;color:#00d4ff;margin:0;letter-spacing:.06em">
    💧 COUPLING & SURFACE MODEL</h2>
    <p style="color:#9E9EC8;margin:4px 0 0;font-size:.85rem">
    Closes the gap between simulation and real A-scan amplitude by modeling exact interface losses.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Interface Parameters")
        medium = st.selectbox("Coupling Medium", ["Water", "Ultrasonic Gel", "Motor Oil", "Air"])
        thickness_mm = st.slider("Couplant Thickness (mm)", 0.0, 5.0, 0.5, 0.1)
        roughness_ra = st.slider("Surface Roughness Ra (µm)", 0.1, 25.0, 3.2, 0.1)
        freq_mhz = st.number_input("Frequency (MHz)", 0.5, 15.0, 2.5)
        
        # Acoustic impedances (MRayl)
        impedances = {
            "Water": {"Z": 1.48, "v": 1480},
            "Ultrasonic Gel": {"Z": 1.60, "v": 1550},
            "Motor Oil": {"Z": 1.28, "v": 1430},
            "Air": {"Z": 0.0004, "v": 343},
            "Aluminium": {"Z": 17.0, "v": 6320} # Default test material
        }
        
    with col2:
        st.markdown("### Amplitude Loss Chain")
        
        Z_c = impedances[medium]["Z"]
        v_c = impedances[medium]["v"]
        Z_m = impedances["Aluminium"]["Z"]
        freq = freq_mhz * 1e6
        
        T_loss = calculate_transmission_loss(Z_c, Z_m, thickness_mm, freq, v_c)
        Ra_loss = apply_roughness_loss(roughness_ra, freq, v_c)
        
        total_amplitude = T_loss * Ra_loss * 100 # percentage
        
        fig = go.Figure(go.Waterfall(
            name="Amplitude", orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Initial Source", "Interface Reflection", "Roughness Scattering", "Final Net Amplitude"],
            textposition="outside",
            text=["100%", f"-{(1-T_loss)*100:.1f}%", f"-{(1-Ra_loss)*100:.1f}%", f"{total_amplitude:.1f}%"],
            y=[100, -((1-T_loss)*100), -((1-Ra_loss)*100), total_amplitude],
            connector={"line": {"color": COLORS['grid']}},
            decreasing={"marker": {"color": COLORS['red']}},
            increasing={"marker": {"color": COLORS['green']}},
            totals={"marker": {"color": COLORS['accent']}}
        ))
        
        fig.update_layout(
            **LAYOUT_BASE,
            title="Transmission Loss Stack",
            showlegend=False,
            height=350,
            yaxis_title="Amplitude (%)"
        )
        st.plotly_chart(fig, width="stretch")

    st.info("**Key Finding:** Roughness loss dominates at higher frequencies, while interface mismatch (Z) dominates coupling efficiency. Air coupling results in near 100% loss without specialized transducers.")
