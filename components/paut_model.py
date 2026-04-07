"""
paut_model.py — Phased Array Ultrasonic Testing (PAUT) Extension
Simulates delay-and-sum beamforming for sector scans (S-scans).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from components.visualizer import LAYOUT_BASE, COLORS

def simulate_beam_profile(elements, pitch, frequency, steer_angle_deg, c=6320):
    # Wavelength
    wl = c / (frequency * 1e6)
    
    # Array geometry
    x_elements = np.arange(elements) * pitch
    x_elements -= np.mean(x_elements) # center at 0
    
    # Steering angles for S-scan
    angles = np.linspace(-45, 45, 91)
    angles_rad = np.deg2rad(angles)
    steer_rad = np.deg2rad(steer_angle_deg)
    
    # Directivity function (simplified array factor)
    # AF = sin(N * pi * d * (sin(theta) - sin(steer)) / wl) / sin(pi * d * (sin(theta) - sin(steer)) / wl)
    
    AF = np.zeros_like(angles)
    for i, theta in enumerate(angles_rad):
        phase_diff = (2 * np.pi / wl) * pitch * (np.sin(theta) - np.sin(steer_rad))
        if np.abs(phase_diff) < 1e-6:
            AF[i] = elements
        else:
            AF[i] = np.sin(elements * phase_diff / 2) / np.sin(phase_diff / 2)
            
    # Normalize and convert to dB
    AF_norm = np.abs(AF) / elements
    AF_db = 20 * np.log10(AF_norm + 1e-10)
    
    return angles, AF_db

def render_paut_page():
    st.markdown("""<div style="background:linear-gradient(90deg,#00d4ff0d,transparent 80%);
    border-left:4px solid #00d4ff;padding:14px 22px;margin-bottom:18px;border-radius:0 10px 10px 0">
    <h2 style="font-family:'Share Tech Mono',monospace;color:#00d4ff;margin:0;letter-spacing:.06em">
    📻 PHASED ARRAY (PAUT) EXTENSION</h2>
    <p style="color:#9E9EC8;margin:4px 0 0;font-size:.85rem">
    Simulates delay-and-sum beamforming across a linear array. Aligns with EN 16018 / ASME V Article 4.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### PAUT Array Parameters")
        elements = st.slider("Number of Elements", 8, 64, 16, 8)
        frequency = st.slider("Frequency (MHz)", 1.0, 10.0, 5.0, 0.5)
        pitch = st.slider("Element Pitch (mm)", 0.2, 2.0, 0.6, 0.1) / 1000 # convert to m
        steer_angle = st.slider("Steering Angle (deg)", -40, 40, 15, 1)
        
        st.info("PAUT offers dynamic steering and focusing, completely eliminating the primary limitation of conventional UT (blindness to off-axis defects).")

    with col2:
        angles, directivity = simulate_beam_profile(elements, pitch, frequency, steer_angle)
        
        # Plot Array Directivity (Polar)
        fig_polar = go.Figure()
        
        # Convert to a 0-max scale for polar plotting (-40 dB floor)
        r_plot = np.maximum(directivity + 40, 0)
        
        fig_polar.add_trace(go.Scatterpolar(
            r=r_plot,
            theta=angles,
            mode='lines',
            line=dict(color=COLORS['accent'], width=2),
            fill='toself',
            fillcolor='rgba(0, 212, 255, 0.2)'
        ))
        
        fig_polar.update_layout(
            **LAYOUT_BASE,
            title="Beam Directivity Profile",
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 40]),
                sector=[225, 315], # Look downwards
            ),
            showlegend=False,
            height=350,
        )
        
        st.plotly_chart(fig_polar, use_container_width=True)
        
    st.markdown("### Simulated S-Scan (Sector Scan)")
    
    # Generate a fake S-scan image
    theta = np.linspace(-30, 30, 60)
    r = np.linspace(0, 50, 100) # Depth in mm
    T, R = np.meshgrid(theta, r)
    
    # Create an artificial defect in the S-scan field
    defect_r = 30
    defect_theta = 15
    image = np.zeros_like(T)
    
    # Add background noise
    image += np.random.normal(0.1, 0.05, image.shape)
    
    # Add defect response based on steering angle 
    beam_intensity = np.interp(T.flatten(), angles, np.maximum(directivity/40 + 1, 0)).reshape(T.shape)
    
    dist_to_defect = np.sqrt((R - defect_r)**2 + ((T - defect_theta)*R*np.pi/180)**2)
    defect_response = np.exp(-(dist_to_defect**2)/(2*2**2)) * 0.8
    
    # Combine
    image_final = (image + defect_response * beam_intensity)
    image_final = np.clip(image_final, 0, 1)
    
    # Convert polar to cartesian for plotting the S-scan
    X = R * np.sin(np.deg2rad(T))
    Z = R * np.cos(np.deg2rad(T))
    
    fig_sscan = go.Figure(data=go.Heatmap(
        z=image_final, x=theta, y=r,
        colorscale='Viridis', zmin=0, zmax=1
    ))
    
    fig_sscan.update_layout(
        **LAYOUT_BASE,
        title="PAUT S-Scan view (Angle vs Depth)",
        xaxis_title="Steering Angle (deg)",
        yaxis_title="Depth (mm)",
        yaxis=dict(autorange="reversed"),
        height=350
    )
    
    st.plotly_chart(fig_sscan, use_container_width=True)
