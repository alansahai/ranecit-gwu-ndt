"""
thermal_model.py — Temperature & Anisotropy Model
Adjusts wave velocity and attenuation based on component temperature.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from components.visualizer import LAYOUT_BASE, COLORS

def render_thermal_page():
    st.markdown("""<div style="background:linear-gradient(90deg,#ff33660d,transparent 80%);
    border-left:4px solid #ff3366;padding:14px 22px;margin-bottom:18px;border-radius:0 10px 10px 0">
    <h2 style="font-family:'Share Tech Mono',monospace;color:#ff3366;margin:0;letter-spacing:.06em">
    🌡 TEMPERATURE & ANISOTROPY MODEL</h2>
    <p style="color:#9E9EC8;margin:4px 0 0;font-size:.85rem">
    Models thermal expansion and texture-induced velocity anisotropy. Critical for elevated-temperature inspection.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 2])
    
    with col1:
        st.markdown("### Environmental Parameters")
        base_temp = 20.0
        temp_c = st.slider("Component Temperature (°C)", 20.0, 400.0, 20.0, 5.0)
        
        # Al Alloy thermal coefficients (approx)
        # Velocity decreases linearly with temp, attenuation increases
        dv_dT = -0.55 # m/s per °C 
        dalpha_dT = 0.005 # Np/m per °C
        
        base_v = 6320.0 # m/s
        base_alpha = 0.02 # Np/m
        
        current_v = base_v + (temp_c - base_temp) * dv_dT
        current_alpha = base_alpha + (temp_c - base_temp) * dalpha_dT
        
        st.markdown("### Texture Anisotropy (ODF)")
        anisotropy_ratio = st.slider("Anisotropy Ratio (%)", 0.0, 5.0, 0.0, 0.5, 
                                     help="Velocity variation depending on grain orientation relative to beam")
        
        st.markdown("---")
        
        st.metric("Adjusted vL (Longitudinal Vel.)", f"{current_v:.1f} m/s", f"{(current_v - base_v):.1f} m/s", delta_color="inverse")
        st.metric("Adjusted Attenuation (α)", f"{current_alpha:.4f} Np/m", f"{(current_alpha - base_alpha):.4f} Np/m", delta_color="inverse")

    with col2:
        # Plot Velocity vs Temp
        temps = np.linspace(20, 400, 100)
        vels = base_v + (temps - base_temp) * dv_dT
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=temps, y=vels, mode='lines', name='Isotropic Base', line=dict(color=COLORS['accent'], width=2)))
        
        # Add bounds for anisotropy
        if anisotropy_ratio > 0:
            v_upper = vels * (1 + anisotropy_ratio/100)
            v_lower = vels * (1 - anisotropy_ratio/100)
            
            fig.add_trace(go.Scatter(
                x=np.concatenate([temps, temps[::-1]]),
                y=np.concatenate([v_upper, v_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 51, 102, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Anisotropy Variance'
            ))
            
        fig.add_vline(x=temp_c, line=dict(color=COLORS['green'], dash='dash'))
        fig.add_annotation(x=temp_c, y=vels.min(), text=f"T = {temp_c}°C", showarrow=False, yshift=10)
        
        fig.update_layout(
            **LAYOUT_BASE,
            title="Acoustic Velocity vs Temperature",
            xaxis_title="Temperature (°C)",
            yaxis_title="Velocity (m/s)",
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, width="stretch")

    st.info("Failing to account for temperature shifts the TOF (Time-Of-Flight), leading to incorrect depth calculations or false rejections in automated systems.")
