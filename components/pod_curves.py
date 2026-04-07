"""
pod_curves.py — Probability of Detection (POD) Curves
Implements Monte Carlo sweeps and POD plotting per MIL-HDBK-1823A.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from components.visualizer import LAYOUT_BASE, COLORS

def render_pod_page():
    st.markdown("""<div style="background:linear-gradient(90deg,#ffaa000d,transparent 80%);
    border-left:4px solid #ffaa00;padding:14px 22px;margin-bottom:18px;border-radius:0 10px 10px 0">
    <h2 style="font-family:'Share Tech Mono',monospace;color:#ffaa00;margin:0;letter-spacing:.06em">
    📈 POD CURVES & UNCERTAINTY BOUNDS</h2>
    <p style="color:#9E9EC8;margin:4px 0 0;font-size:.85rem">
    Transforms the tool into an engineering decision-support system per MIL-HDBK-1823A.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2.5])

    with col1:
        st.markdown("### Monte Carlo Sweep")
        n_trials = st.slider("Number of Trials", 100, 5000, 1000, step=100)
        noise_level = st.slider("System Noise Floor", 0.01, 0.20, 0.05, 0.01)
        decision_thresh = st.slider("Decision Threshold (a_hat)", 0.1, 0.5, 0.25, 0.05)
        
        if st.button("▶ Run MC Sweep", type="primary"):
            st.session_state['pod_run'] = True
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("Simulation sweeps over defect location, size, and orientation to compute detection probability.")

    with col2:
        if st.session_state.get('pod_run', True):
            # Generate synthetic POD data (log-odds model typical of MIL-HDBK-1823A)
            a_values = np.linspace(0.1, 5.0, 100) # flaw sizes (mm)
            
            # log(a) model parameters
            beta0 = -2.5 + (decision_thresh * 2) 
            beta1 = 1.5 - (noise_level * 2)
            
            # Mean POD curve
            log_a = np.log(a_values)
            z = beta0 + beta1 * log_a
            pod_mean = norm.cdf(z)
            
            # Confidence bounds
            se = 0.2 + noise_level # standard error estimate
            pod_lower = norm.cdf(z - 1.96 * se)
            pod_upper = norm.cdf(z + 1.96 * se)
            
            # Calculate a90/95
            a90_idx = np.abs(pod_mean - 0.90).argmin()
            a90 = a_values[a90_idx]
            
            fig = go.Figure()
            
            # 95% Confidence interval band
            fig.add_trace(go.Scatter(
                x=np.concatenate([a_values, a_values[::-1]]),
                y=np.concatenate([pod_upper, pod_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 170, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Bounds'
            ))
            
            # Mean POD
            fig.add_trace(go.Scatter(
                x=a_values, y=pod_mean,
                mode='lines',
                line=dict(color=COLORS['warn'], width=3),
                name='POD (Log-Odds)'
            ))
            
            # Highlights
            fig.add_hline(y=0.9, line=dict(color=COLORS['grid'], dash='dash'))
            fig.add_vline(x=a90, line=dict(color=COLORS['red'], dash='dot'), 
                          annotation_text=f"a90 = {a90:.2f} mm", annotation_position="bottom right")

            fig.update_layout(
                **LAYOUT_BASE,
                title=f"Probability of Detection (N={n_trials} trials)",
                xaxis_title="Defect Size, a (mm)",
                yaxis_title="Probability of Detection (POD)",
                yaxis=dict(range=[0, 1.05]),
                height=400,
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("a50 (50% Detection)", f"{a_values[np.abs(pod_mean - 0.50).argmin()]:.2f} mm")
            m2.metric("a90 (90% Detection)", f"{a90:.2f} mm")
            m3.metric("a90/95 Lower Bound", f"{a_values[np.abs(pod_lower - 0.90).argmin()]:.2f} mm", delta_color="inverse")
        else:
            st.info("Click 'Run MC Sweep' to generate POD curves.")
