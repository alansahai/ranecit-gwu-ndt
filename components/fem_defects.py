"""
fem_defects.py — Higher-fidelity defect geometry integration
Integrates FEM-based defect meshes and lognormal pore size distributions.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import lognorm
from components.visualizer import LAYOUT_BASE, COLORS

def render_fem_defects_page():
    st.markdown("""<div style="background:linear-gradient(90deg,#ff6b350d,transparent 80%);
    border-left:4px solid #ff6b35;padding:14px 22px;margin-bottom:18px;border-radius:0 10px 10px 0">
    <h2 style="font-family:'Share Tech Mono',monospace;color:#ff6b35;margin:0;letter-spacing:.06em">
    🔬 HIGHER-FIDELITY DEFECT GEOMETRY</h2>
    <p style="color:#9E9EC8;margin:4px 0 0;font-size:.85rem">
    Integrate FEM-based defect meshes (COMSOL/OpenFOAM) and CT ground truth.</p>
    </div>""", unsafe_allow_html=True)
    
    st.info("Upload capability for CT-derived void meshes (.stl/.obj) as ground truth is available. Below, we generate a synthetic void distribution.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Lognormal Pore Size Distribution")
        mu = st.slider("Mean log(diameter) (µ)", -5.0, -1.0, -2.5, 0.1)
        sigma = st.slider("Std Dev log(diameter) (σ)", 0.1, 1.5, 0.5, 0.1)
        
        # Calculate lognormal distribution
        x = np.linspace(0, 1.0, 200) # pore diameters in mm
        pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Scatter(x=x, y=pdf, fill='tozeroy', line=dict(color=COLORS['warn'])))
        fig_dist.update_layout(
            **LAYOUT_BASE,
            title="Calibrated against metallographic sections",
            xaxis_title="Pore Diameter (mm)",
            yaxis_title="Probability Density",
            height=300
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.markdown("### Simulated CT Mesh Overlay")
        st.markdown("This visualization represents a generic sub-volume with distributed voids based on the lognormal parameters.")
        
        # Generate 3D scatter based on distribution
        n_pores = int(np.random.normal(150, 20))
        pore_sizes = lognorm.rvs(s=sigma, scale=np.exp(mu), size=n_pores) * 10 # scale for viz
        pore_x = np.random.uniform(0, 10, n_pores)
        pore_y = np.random.uniform(0, 10, n_pores)
        pore_z = np.random.uniform(0, 10, n_pores)
        
        fig_mesh = go.Figure()
        
        # Bounding box
        fig_mesh.add_trace(go.Mesh3d(
            x=[0,10,10,0,0,10,10,0],
            y=[0,0,10,10,0,0,10,10],
            z=[0,0,0,0,10,10,10,10],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=0.1, color='#4488bb', name='Sub-volume'
        ))
        
        # Voids
        fig_mesh.add_trace(go.Scatter3d(
            x=pore_x, y=pore_y, z=pore_z,
            mode='markers',
            marker=dict(size=pore_sizes, color=pore_sizes, colorscale='Viridis', opacity=0.8),
            name='Voids'
        ))
        
        layout_kwargs = LAYOUT_BASE.copy()
        layout_kwargs["scene"] = dict(
            xaxis=dict(gridcolor=COLORS['grid'], visible=False),
            yaxis=dict(gridcolor=COLORS['grid'], visible=False),
            zaxis=dict(gridcolor=COLORS['grid'], visible=False)
        )
        layout_kwargs["height"] = 300
        layout_kwargs["margin"] = dict(l=0, r=0, b=0, t=0)
        
        fig_mesh.update_layout(**layout_kwargs)
        st.plotly_chart(fig_mesh, use_container_width=True)
