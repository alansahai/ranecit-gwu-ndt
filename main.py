"""
main.py — AcoustiScan Pro v3.0
Acoustic NDT Simulation Dashboard + Real-Time Hardware + Advanced Modules

Run: streamlit run main.py
"""

import os, sys, time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from components.geometry       import load_mesh, get_geometry_info, classify_scan_type, sample_scan_points
from components.material_model import build_model_from_inputs, simulate_with_model
from components.acoustic       import compare_signals
from components.gantry         import build_gantry_animation, build_live_signal_chart, build_sensor_array_panel
from components.visualizer     import plot_3d_mesh, plot_heatmap_2d, plot_deviation_bars
from components.report         import generate_text_report, generate_csv_data

# Hardware and Advanced Modules
from components.hardware_mode  import render_hardware_page
from components.certification  import render_certification_page
from components.fem_defects    import render_fem_defects_page
from components.coupling_model import render_coupling_model_page
from components.pod_curves     import render_pod_page
from components.paut_model     import render_paut_page
from components.thermal_model  import render_thermal_page
from components.comparison_page import render_comparison_page
from components.roadmap_page   import render_roadmap_page

DEMO_MODELS = {
    "🔩 Aluminium Rod (Linear)":       os.path.join(ROOT, "models", "aluminium_rod.obj"),
    "🔧 Engine Bracket (Grid)":        os.path.join(ROOT, "models", "engine_bracket.obj"),
    "⚙️ Crankcase Block (Multi-face)": os.path.join(ROOT, "models", "crankcase_block.obj"),
}

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c0d0e8", family="monospace", size=11),
    margin=dict(l=10, r=10, t=44, b=10),
)

st.set_page_config(page_title="AcoustiScan Pro v3.0", page_icon="🔊",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');
html,body,.stApp{background:#06090f!important;color:#c0d0e8!important;font-family:'Barlow',sans-serif}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#0a1220,#080c18)!important;border-right:1px solid #162440}
.ndt-header{background:linear-gradient(90deg,#00d4ff0d,transparent 80%);border-left:4px solid #00d4ff;padding:16px 24px;margin-bottom:20px;border-radius:0 10px 10px 0}
.ndt-header h1{font-family:'Share Tech Mono',monospace;color:#00d4ff;font-size:1.8rem;margin:0;letter-spacing:.07em;text-shadow:0 0 22px #00d4ff44}
.ndt-header p{color:#4a7090;margin:5px 0 0;font-size:.88rem;letter-spacing:.04em}
.badge-v3{display:inline-block;background:#00d4ff18;color:#00d4ff;border:1px solid #00d4ff44;border-radius:4px;padding:1px 8px;font-family:monospace;font-size:.74rem;margin-left:10px;vertical-align:middle}
.kpi-card{background:linear-gradient(135deg,#0d1e32,#0a1422);border:1px solid #1a3354;border-radius:10px;padding:13px 16px;text-align:center;margin-bottom:10px}
.kpi-card .lbl{font-size:.66rem;color:#4a7090;text-transform:uppercase;letter-spacing:.1em;font-family:'Share Tech Mono',monospace}
.kpi-card .val{font-size:1.4rem;font-weight:700;color:#00d4ff;font-family:'Share Tech Mono',monospace}
.kpi-card .unt{font-size:.7rem;color:#4a7090}
.mat-section{background:#0a1525;border:1px solid #162a42;border-top:2px solid #00d4ff33;border-radius:8px;padding:15px;margin-bottom:13px}
.mat-section h4{font-family:'Share Tech Mono',monospace;color:#00d4ff;font-size:.8rem;letter-spacing:.1em;margin:0 0 11px;text-transform:uppercase}
.prop-row{display:flex;justify-content:space-between;border-bottom:1px solid #0f1e30;padding:4px 0;font-size:.8rem}
.prop-row .pk{color:#5888a0;font-family:monospace}.prop-row .pv{color:#00d4ff;font-family:monospace;font-weight:600}
.sh{font-family:'Share Tech Mono',monospace;color:#00d4ff;font-size:.86rem;letter-spacing:.13em;text-transform:uppercase;border-bottom:1px solid #162440;padding-bottom:6px;margin:20px 0 13px}
.info-box{background:#091628;border:1px solid #162a42;border-left:3px solid #00d4ff;border-radius:7px;padding:10px 14px;font-size:.85rem;margin:9px 0}
.defect-box{background:#140a12;border:1px solid #ff336630;border-left:3px solid #ff3366;border-radius:7px;padding:12px 16px;margin:9px 0}
.defect-box h4{color:#ff3366;margin:0 0 5px;font-family:monospace;font-size:.88rem}
.defect-box p{margin:2px 0;font-size:.83rem;color:#b090a0}
.stTabs [data-baseweb="tab"]{font-family:'Share Tech Mono',monospace!important;color:#4a7090!important;font-size:.8rem!important;letter-spacing:.08em!important}
.stTabs [aria-selected="true"]{color:#00d4ff!important;border-bottom:2px solid #00d4ff!important}
.stButton>button{background:linear-gradient(135deg,#00d4ff1a,#0080ff1a)!important;border:1px solid #00d4ff44!important;color:#00d4ff!important;font-family:'Share Tech Mono',monospace!important;letter-spacing:.07em!important;border-radius:6px!important}
.stButton>button:hover{border-color:#00d4ff!important;box-shadow:0 0 14px #00d4ff28!important}
.stProgress>div>div>div>div{background-color:#00d4ff!important}
hr{border-color:#162440!important}
[data-testid="stSlider"] label,[data-testid="stSelectbox"] label,[data-testid="stNumberInput"] label{color:#4a7090!important;font-size:.81rem!important}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔊 ACOUSTISCAN PRO v3.0")
    st.markdown("---")
    
    view_mode = st.radio("Navigation", [
        "Simulation Dashboard",
        "Real-Time Hardware (Arduino)",
        "Advanced Modules",
        "UT vs RT Comparison",
        "Feature Roadmap",
    ], label_visibility="collapsed")
    
    st.markdown("---")
    
    if view_mode == "Simulation Dashboard":
        st.markdown("### ⚙️ SIMULATION CONTROLS")
        model_source = st.radio("Component Source", ["Demo Component", "Upload OBJ/STL"],
                                label_visibility="collapsed")
        mesh_bytes, mesh_name = None, None
        if model_source == "Demo Component":
            sel = st.selectbox("Demo", list(DEMO_MODELS.keys()))
            p   = DEMO_MODELS[sel]
            if os.path.exists(p):
                with open(p,"rb") as f: mesh_bytes = f.read()
                mesh_name = os.path.basename(p)
        else:
            up = st.file_uploader("Upload OBJ/STL", type=["obj","stl"])
            if up: mesh_bytes = up.read(); mesh_name = up.name

        st.markdown("**🔧 Scan**")
        resolution     = st.slider("Grid Resolution", 8, 32, 16)
        anim_step      = st.slider("Animation Density", 1, 8, 4,
                                    help="Every Nth point becomes an animation frame")
        st.markdown("**🧪 Defects**")
        defect_density = st.slider("Porosity Density", 0.05, 0.30, 0.12, 0.01)
        st.markdown("---")
        run_sim = st.button("▶ RUN SIMULATION", use_container_width=True)

        if st.sidebar.button("🗑 Reset Session"):
            st.session_state.clear()
            st.rerun()

# ═══════════════════════════════════════════════════════
# VIEW ROUTER
# ═══════════════════════════════════════════════════════

if view_mode == "Real-Time Hardware (Arduino)":
    render_hardware_page()

elif view_mode == "UT vs RT Comparison":
    render_comparison_page()

elif view_mode == "Feature Roadmap":
    render_roadmap_page()

elif view_mode == "Advanced Modules":
    st.markdown("""<div class="ndt-header">
      <h1>ADVANCED COMPUTATIONAL MODULES</h1>
      <p>Select an advanced feature module below to explore.</p>
    </div>""", unsafe_allow_html=True)
    
    mod_tab1, mod_tab2, mod_tab3, mod_tab4, mod_tab5, mod_tab6 = st.tabs([
        "🏅 Certification", "🔬 FEM Defects", "💧 Coupling Model", 
        "📈 POD Curves", "📻 PAUT Extension", "🌡 Thermal/Anisotropy"
    ])
    
    with mod_tab1: render_certification_page()
    with mod_tab2: render_fem_defects_page()
    with mod_tab3: render_coupling_model_page()
    with mod_tab4: render_pod_page()
    with mod_tab5: render_paut_page()
    with mod_tab6: render_thermal_page()

elif view_mode == "Simulation Dashboard":
    # ── Header ───────────────────────────────────────────────
    st.markdown("""<div class="ndt-header">
      <h1>🔊 ACOUSTISCAN PRO <span class="badge-v3">v3.0</span></h1>
      <p>Acoustic Wave NDT Simulator — Physics Material Model · Gantry Animation · Live Signal Acquisition</p>
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # GUARD
    # ═══════════════════════════════════════════════════════
    if 'mesh_bytes' not in locals() or mesh_bytes is None:
        st.markdown("""<div class="info-box">👈 Select a demo component or upload OBJ/STL, 
        fill in material properties below, then press <b>RUN SIMULATION</b>.</div>""",
                    unsafe_allow_html=True)
        st.stop()

    # ═══════════════════════════════════════════════════════
    # MATERIAL PROPERTIES FORM
    # ═══════════════════════════════════════════════════════
    with st.expander("🔬 ALUMINIUM DIE CAST — MATERIAL PROPERTIES", expanded=True):
        st.markdown("""<div class="info-box">
        Enter the properties of your aluminium die casting alloy. The simulator derives
        <b>wave speed, acoustic impedance, attenuation, and porosity effects</b> from these
        values using elastic wave theory and effective medium approximations.
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="mat-section"><h4>🏗 Mechanical Properties</h4>', unsafe_allow_html=True)
            density_val  = st.number_input("Density (kg/m³)",        200.0, 4000.0, 2720.0, 10.0)
            E_val        = st.number_input("Young's Modulus (GPa)",   10.0,  200.0,   71.0,  0.5)
            nu_val       = st.number_input("Poisson's Ratio",           0.10,  0.48,   0.33, 0.01)
            ys_val       = st.number_input("Yield Strength (MPa)",     50.0, 1000.0, 160.0,  5.0)
            uts_val      = st.number_input("Tensile Strength (MPa)",   50.0, 1500.0, 310.0,  5.0)
            elong_val    = st.number_input("Elongation (%)",            0.1,   30.0,   3.5,  0.1)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="mat-section"><h4>🔥 Thermal & Casting Quality</h4>', unsafe_allow_html=True)
            therm_val    = st.number_input("Thermal Conductivity (W/m·K)", 10.0, 300.0, 96.2, 1.0)
            porosity_val = st.number_input("Casting Porosity (%)", 0.0, 15.0, 0.0, 0.1,
                                           help="Void fraction in casting — reduces wave amplitude directly")
            grain_val    = st.number_input("Grain Size (µm)", 5.0, 500.0, 50.0, 5.0,
                                           help="Controls grain-scattering noise level")
            roughness_val= st.number_input("Surface Roughness Ra (µm)", 0.1, 50.0, 1.6, 0.1)
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="mat-section"><h4>📡 Transducer Setup</h4>', unsafe_allow_html=True)
            freq_val     = st.number_input("Transducer Frequency (MHz)", 0.5, 25.0, 2.5, 0.5)
            coupling_val = st.selectbox("Coupling Medium", ["water","gel","air"])
            st.markdown("""<div style="font-size:.77rem;color:#4a7090;line-height:1.9em;margin-top:10px">
            <b style="color:#00d4ff88">Physics equations used:</b><br>
            v<sub>L</sub> = √(E(1-ν) / ρ(1+ν)(1-2ν))<br>
            v<sub>S</sub> = √(G/ρ)&nbsp; G = E/2(1+ν)<br>
            Z = ρ·v<sub>L</sub>&nbsp; (acoustic impedance)<br>
            α = α<sub>scatter</sub>(f,d) + α<sub>abs</sub>(f)<br>
            f₀ = v<sub>L</sub>/(2·t)&nbsp; (plate resonance)<br>
            δ = 1/(2·α·f)&nbsp; (penetration depth)
            </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Build physics model
    user_props = dict(
        density_kg_m3=density_val, youngs_modulus_gpa=E_val,
        poisson_ratio=nu_val, yield_strength_mpa=ys_val,
        tensile_strength_mpa=uts_val, elongation_pct=elong_val,
        thermal_conductivity=therm_val, porosity_pct=porosity_val,
        grain_size_um=grain_val, surface_roughness_um=roughness_val,
        transducer_freq_mhz=freq_val, coupling_medium=coupling_val,
    )
    mat = build_model_from_inputs(user_props)

    # Derived model display
    st.markdown('<p class="sh">⚛️ DERIVED ACOUSTIC MODEL</p>', unsafe_allow_html=True)
    derived = mat.summary()
    dcols = st.columns(4)
    items = list(derived.items())
    for ci, col in enumerate(dcols):
        with col:
            for k, v in items[ci*(len(items)//4+1):(ci+1)*(len(items)//4+1)]:
                st.markdown(f'<div class="prop-row"><span class="pk">{k}</span>'
                            f'<span class="pv">{v}</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════
    # SIMULATION RUN
    # ═══════════════════════════════════════════════════════
    run_key = f"{mesh_name}|{resolution}|{porosity_val}|{grain_val}|{E_val}|{nu_val}|{freq_val}|{defect_density}"

    if run_sim or "sim_v2" not in st.session_state or st.session_state.get("rk") != run_key:
        prg = st.progress(0); sts = st.empty()

        sts.markdown("⟳ Loading mesh & ray-casting scan grid...")
        mesh     = load_mesh(mesh_bytes, mesh_name)
        geo      = get_geometry_info(mesh)
        stype    = classify_scan_type(geo)
        pts      = sample_scan_points(mesh, resolution)
        prg.progress(22)

        sts.markdown("⟳ Baseline — physics model...")
        base = simulate_with_model(pts, geo["bbox_z"], mat, "none", seed=42)
        prg.progress(40)

        sts.markdown("⟳ Porosity defect injection...")
        d1   = simulate_with_model(pts, geo["bbox_z"], mat, "porosity",
                                   defect_density=defect_density, seed=7)
        prg.progress(56)

        sts.markdown("⟳ Cold shut simulation...")
        d2   = simulate_with_model(pts, geo["bbox_z"], mat, "cold_shut", seed=13)
        prg.progress(70)

        sts.markdown("⟳ Signal deviation analysis...")
        c1   = compare_signals(base, d1)
        c2   = compare_signals(base, d2)
        prg.progress(80)

        sts.markdown("⟳ Building gantry animation frames...")
        ga1  = build_gantry_animation(mesh, pts, d1["amplitude"], d1["frequency"],
                                       d1["tof"], d1["defect_mask"], anim_step,
                                       "Gantry Scan — Porosity Defect")
        ga2  = build_gantry_animation(mesh, pts, d2["amplitude"], d2["frequency"],
                                       d2["tof"], d2["defect_mask"], anim_step,
                                       "Gantry Scan — Cold Shut Defect")
        prg.progress(90)

        sts.markdown("⟳ Building live signal charts...")
        lc1  = build_live_signal_chart(pts, d1["amplitude"], d1["frequency"],
                                        d1["tof"], d1["defect_mask"], anim_step)
        lc2  = build_live_signal_chart(pts, d2["amplitude"], d2["frequency"],
                                        d2["tof"], d2["defect_mask"], anim_step)
        prg.progress(100); prg.empty(); sts.empty()

        st.session_state.update(dict(
            sim_v2=dict(mesh=mesh, geo=geo, stype=stype, pts=pts,
                        base=base, d1=d1, d2=d2, c1=c1, c2=c2,
                        ga1=ga1, ga2=ga2, lc1=lc1, lc2=lc2, mat=mat),
            rk=run_key, mesh_name=mesh_name,
        ))

    R   = st.session_state["sim_v2"]
    mesh= R["mesh"]; geo=R["geo"]; stype=R["stype"]; pts=R["pts"]
    base= R["base"]; d1=R["d1"];   d2=R["d2"]
    c1  = R["c1"];   c2=R["c2"];   mat=R["mat"]
    ga1 = R["ga1"];  ga2=R["ga2"]; lc1=R["lc1"]; lc2=R["lc2"]

    # ═══════════════════════════════════════════════════════
    # KPI BAR
    # ═══════════════════════════════════════════════════════
    st.markdown('<p class="sh">📊 INSPECTION SUMMARY</p>', unsafe_allow_html=True)
    cols = st.columns(7)
    kpis = [
        ("Scan Points",   f"{len(pts):,}",                  ""),
        ("Scan Mode",     stype["type"],                     ""),
        ("v_L",           f"{mat.v_longitudinal:,.0f}",      "m/s"),
        ("Wavelength",    f"{mat.wavelength_mm:.3f}",        "mm"),
        ("Porosity Pts",  f"{c1['defect_mask'].sum():,}",    f"{c1['detection_rate']:.1f}%"),
        ("Cold-Shut Pts", f"{c2['defect_mask'].sum():,}",    f"{c2['detection_rate']:.1f}%"),
        ("Verdict",       "FAIL" if (c1["detection_rate"]>5 or c2["detection_rate"]>5) else "PASS", ""),
    ]
    v_colors = ["#00d4ff","#88ccff","#00d4ff","#00d4ff","#ff6b35","#ff3366",
                "#ff3366" if kpis[6][1]=="FAIL" else "#00ff88"]
    for col, (lbl, val, unt), vc in zip(cols, kpis, v_colors):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="lbl">{lbl}</div>'
                        f'<div class="val" style="color:{vc}">{val}</div>'
                        f'<div class="unt">{unt}</div></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════
    tab_phys, tab_gantry, tab_live, tab_3d, tab_heat, tab_rpt = st.tabs([
        "⚛️ PHYSICS MODEL",
        "🤖 GANTRY ANIMATION",
        "📈 LIVE SIGNAL CHART",
        "🧊 3D DEFECT VIEW",
        "🌡️ HEATMAP",
        "📋 REPORT",
    ])

    # ── PHYSICS MODEL TAB ────────────────────────────────────
    with tab_phys:
        st.markdown('<p class="sh">Derived Acoustic Properties & Material Curves</p>',
                    unsafe_allow_html=True)
        pa, pb = st.columns(2)

        with pa:
            # Wave speed comparison
            ref = {"Al A380":(6320,3100),"Al 6061":(6350,3130),
                   "Cast Iron":(4600,2600),"Steel":(5960,3240)}
            names  = ["Your Alloy"] + list(ref.keys())
            vl_arr = [mat.v_longitudinal] + [v[0] for v in ref.values()]
            vs_arr = [mat.v_shear]        + [v[1] for v in ref.values()]
            fig_ws = go.Figure()
            fig_ws.add_trace(go.Bar(name="Longitudinal",x=names,y=vl_arr,
                                    marker_color="#00d4ff",opacity=0.88))
            fig_ws.add_trace(go.Bar(name="Shear",x=names,y=vs_arr,
                                    marker_color="#0055bb",opacity=0.88))
            fig_ws.update_layout(**LAYOUT_BASE, barmode="group", height=290,
                title=dict(text="Wave Speed Comparison (m/s)",font=dict(color="#00d4ff",size=12)),
                legend=dict(bgcolor="rgba(0,0,0,.4)"),
                xaxis=dict(gridcolor="#162440"),yaxis=dict(gridcolor="#162440",title="m/s"))
            st.plotly_chart(fig_ws, use_container_width=True)

            # Amplitude vs thickness
            t_mm  = np.linspace(1, 50, 300)
            amps  = [mat.compute_amplitude(t/1000) for t in t_mm]
            fig_a = go.Figure(go.Scatter(x=t_mm, y=amps, mode="lines",
                              line=dict(color="#00ff88",width=2),
                              fill="tozeroy",fillcolor="rgba(0,255,136,.07)"))
            fig_a.update_layout(**LAYOUT_BASE, height=270,
                title=dict(text="Amplitude vs Wall Thickness",font=dict(color="#00d4ff",size=12)),
                xaxis=dict(title="Thickness (mm)",gridcolor="#162440"),
                yaxis=dict(title="Normalised Amplitude",gridcolor="#162440"))
            st.plotly_chart(fig_a, use_container_width=True)

        with pb:
            # Attenuation vs frequency
            f_mhz = np.linspace(0.5, 15, 200)
            attens = []
            for fq in f_mhz:
                f_ = fq*1e6; d_ = mat.grain_size_m
                lam_ = mat.v_longitudinal/f_
                as_ = (1e-3*(f_**4)*(d_**3)/(mat.v_longitudinal**3)
                       if d_ < lam_/10 else 1e-1*(f_**2)*d_/mat.v_longitudinal)
                attens.append(as_ + 0.003*fq**1.1)
            fig_att = go.Figure()
            fig_att.add_trace(go.Scatter(x=f_mhz, y=attens, mode="lines",
                              line=dict(color="#ffaa00",width=2),
                              fill="tozeroy",fillcolor="rgba(255,170,0,.07)"))
            fig_att.add_vline(x=freq_val, line=dict(color="#00d4ff",dash="dash",width=1.5))
            fig_att.add_annotation(x=freq_val, y=max(attens)*.55,
                text=f"{freq_val} MHz", font=dict(color="#00d4ff",size=10), showarrow=False)
            fig_att.update_layout(**LAYOUT_BASE, height=270,
                title=dict(text="Attenuation α vs Frequency",font=dict(color="#00d4ff",size=12)),
                xaxis=dict(title="Frequency (MHz)",gridcolor="#162440"),
                yaxis=dict(title="Attenuation (Np/m)",gridcolor="#162440"))
            st.plotly_chart(fig_att, use_container_width=True)

            # TOF vs thickness
            tofs = [mat.compute_tof(t/1000) for t in t_mm]
            fig_tof = go.Figure(go.Scatter(x=t_mm,y=tofs,mode="lines",
                                line=dict(color="#aa88ff",width=2)))
            fig_tof.update_layout(**LAYOUT_BASE, height=270,
                title=dict(text="Time-of-Flight vs Thickness",font=dict(color="#00d4ff",size=12)),
                xaxis=dict(title="Thickness (mm)",gridcolor="#162440"),
                yaxis=dict(title="TOF (µs)",gridcolor="#162440"))
            st.plotly_chart(fig_tof, use_container_width=True)

        # Porosity sensitivity
        por_r = np.linspace(0,15,120)
        por_a = [np.clip(1-2*(p/100),.1,1.) for p in por_r]
        fig_por = go.Figure(go.Scatter(x=por_r,y=por_a,mode="lines",
                            line=dict(color="#ff6b35",width=2),
                            fill="tozeroy",fillcolor="rgba(255,107,53,.07)"))
        fig_por.add_vline(x=porosity_val, line=dict(color="#fff",dash="dot",width=1))
        fig_por.add_annotation(x=porosity_val,y=0.6,
            text=f"Your input: {porosity_val}%",font=dict(color="#fff",size=10),showarrow=True)
        fig_por.update_layout(**LAYOUT_BASE, height=230,
            title=dict(text="Porosity → Amplitude Reduction (Effective Medium Theory)",
                       font=dict(color="#00d4ff",size=12)),
            xaxis=dict(title="Porosity (%)",gridcolor="#162440"),
            yaxis=dict(title="Amplitude Factor",gridcolor="#162440"))
        st.plotly_chart(fig_por, use_container_width=True)

    # ── GANTRY ANIMATION TAB ─────────────────────────────────
    with tab_gantry:
        st.markdown('<p class="sh">🤖 Gantry Transducer — Step-by-Step Scan</p>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
        The CNC gantry moves across the component surface, acquiring one measurement per position.
        <span style="color:#00ff88">■ Green</span> = healthy &nbsp;
        <span style="color:#ffaa00">■ Amber</span> = marginal &nbsp;
        <span style="color:#ff3366">■ Red</span> = defect.<br>
        Press <b>▶ PLAY</b> or drag the scan-point slider to step through manually.
        The annotation shows live amplitude / frequency / TOF at each position.
        </div>""", unsafe_allow_html=True)

        gt1, gt2 = st.tabs(["Porosity Defect Scan", "Cold Shut Defect Scan"])
        with gt1:
            st.plotly_chart(ga1, use_container_width=True)
        with gt2:
            st.plotly_chart(ga2, use_container_width=True)

        st.markdown('<p class="sh">🔬 Sensor Array & A-Scan — Select Position</p>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
        The 4×4 sensor array panel shows per-channel amplitude at the selected scan position.
        The A-scan waveform shows the time-domain echo — ghost reflections appear in defect zones.
        </div>""", unsafe_allow_html=True)

        sel = st.slider("Scan Position", 0, len(pts)-1, len(pts)//2)
        sa, sb = st.columns(2)
        with sa:
            st.markdown("**Porosity Simulation**")
            st.plotly_chart(build_sensor_array_panel(
                sel, d1["amplitude"], d1["frequency"], d1["tof"], d1["defect_mask"]),
                use_container_width=True)
        with sb:
            st.markdown("**Cold Shut Simulation**")
            st.plotly_chart(build_sensor_array_panel(
                sel, d2["amplitude"], d2["frequency"], d2["tof"], d2["defect_mask"]),
                use_container_width=True)

    # ── LIVE SIGNAL CHART TAB ────────────────────────────────
    with tab_live:
        st.markdown('<p class="sh">📈 Live Signal Acquisition — Gantry-Synced</p>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
        Signal traces grow <b>point-by-point in sync with gantry movement</b>.
        Press <b>▶ PLAY</b> — amplitude, frequency, and TOF are plotted as each
        scan point is acquired. <span style="color:#ff3366">Red dots</span> flag defect readings.
        The white dashed cursor line shows the current gantry position.
        </div>""", unsafe_allow_html=True)

        ls1, ls2 = st.tabs(["Porosity Defect", "Cold Shut Defect"])
        with ls1:
            st.plotly_chart(lc1, use_container_width=True)
        with ls2:
            st.plotly_chart(lc2, use_container_width=True)

        # Full static 6-panel comparison
        st.markdown('<p class="sh">📊 Full Signal Comparison — All Metrics</p>',
                    unsafe_allow_html=True)
        idx = np.arange(len(pts))
        fig_cmp = make_subplots(rows=3, cols=2, shared_xaxes=True,
                                 column_titles=["Porosity Defect","Cold Shut Defect"],
                                 row_titles=["Amplitude","Frequency","TOF (µs)"],
                                 vertical_spacing=0.07, horizontal_spacing=0.06)
        for ri, key in enumerate(["amplitude","frequency","tof"], 1):
            for ci, (sim, comp) in enumerate([(d1,c1),(d2,c2)], 1):
                mask = comp["defect_mask"]
                fig_cmp.add_trace(go.Scatter(
                    x=idx, y=base[key], mode="lines",
                    line=dict(color="#2a4060",width=1,dash="dot"),
                    showlegend=(ri==1 and ci==1), name="Baseline"),
                    row=ri, col=ci)
                fig_cmp.add_trace(go.Scatter(
                    x=idx, y=sim[key], mode="lines",
                    line=dict(color="#00ff88" if ci==1 else "#ffaa00",width=1.5),
                    showlegend=False), row=ri, col=ci)
                if mask.any():
                    fig_cmp.add_trace(go.Scatter(
                        x=idx[mask], y=sim[key][mask], mode="markers",
                        marker=dict(size=4,color="#ff3366"),
                        showlegend=False), row=ri, col=ci)
        fig_cmp.update_layout(**LAYOUT_BASE, height=580,
            title=dict(text="Baseline vs Defects — Amplitude / Frequency / TOF",
                       font=dict(color="#00d4ff",size=13)),
            showlegend=True, legend=dict(bgcolor="rgba(0,0,0,.4)"))
        fig_cmp.update_yaxes(gridcolor="#162440")
        fig_cmp.update_xaxes(gridcolor="#162440")
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.plotly_chart(plot_deviation_bars(c1, c2), use_container_width=True)

    # ── 3D DEFECT VIEW TAB ───────────────────────────────────
    with tab_3d:
        st.markdown('<p class="sh">🧊 3D Component — Defect Overlay</p>', unsafe_allow_html=True)
        v1,v2,v3 = st.tabs(["Baseline","Porosity Defects","Cold Shut Defects"])
        with v1: st.plotly_chart(plot_3d_mesh(mesh,pts,title="Baseline"),use_container_width=True)
        with v2: st.plotly_chart(plot_3d_mesh(mesh,pts,c1["defect_mask"],"Porosity Overlay"),use_container_width=True)
        with v3: st.plotly_chart(plot_3d_mesh(mesh,pts,c2["defect_mask"],"Cold Shut Overlay"),use_container_width=True)
        da,db = st.columns(2)
        with da:
            st.markdown(f"""<div class="defect-box"><h4>⚠️ Porosity / Void Clusters</h4>
            <p>Detected : <b>{c1['defect_mask'].sum():,}</b> / {len(pts):,} points</p>
            <p>Rate     : <b>{c1['detection_rate']:.1f}%</b></p>
            <p>Mean Amp Δ: <b>{c1['amplitude_deviation'].mean():.4f}</b></p>
            <p>Mean TOF Δ: <b>{c1['tof_deviation'].mean():.3f} µs</b></p>
            </div>""", unsafe_allow_html=True)
        with db:
            st.markdown(f"""<div class="defect-box"><h4>⚠️ Cold Shut / Unfilled</h4>
            <p>Detected : <b>{c2['defect_mask'].sum():,}</b> / {len(pts):,} points</p>
            <p>Rate     : <b>{c2['detection_rate']:.1f}%</b></p>
            <p>Mean Amp Δ: <b>{c2['amplitude_deviation'].mean():.4f}</b></p>
            <p>Mean TOF Δ: <b>{c2['tof_deviation'].mean():.3f} µs</b></p>
            </div>""", unsafe_allow_html=True)

    # ── HEATMAP TAB ──────────────────────────────────────────
    with tab_heat:
        st.markdown('<p class="sh">🌡️ Defect Severity Heatmaps</p>', unsafe_allow_html=True)
        hc1, hc2 = st.columns(2)
        with hc1: st.plotly_chart(plot_heatmap_2d(pts,c1["combined_severity"],"Porosity Severity Map"),use_container_width=True)
        with hc2: st.plotly_chart(plot_heatmap_2d(pts,c2["combined_severity"],"Cold Shut Severity Map"),use_container_width=True)
        st.markdown("""<div class="info-box">
        Severity = Amplitude Δ × 0.5 + Frequency Δ × 0.3 + TOF Δ(norm) × 0.2 &nbsp;|&nbsp;
        🟢 Green → normal &nbsp; 🔴 Red → defect
        </div>""", unsafe_allow_html=True)

    # ── REPORT TAB ───────────────────────────────────────────
    with tab_rpt:
        st.markdown('<p class="sh">📋 NDT Inspection Report</p>', unsafe_allow_html=True)
        mat_lines = "\n".join(f"  {k:<42}: {v}" for k,v in mat.summary().items())
        rpt = generate_text_report(geo, stype, c1, c2,
                                   component_name=st.session_state.get("mesh_name","Unknown"))
        full = rpt.replace(
            "  [2] SIMULATION RESULTS",
            f"  [1b] PHYSICS-DERIVED ACOUSTIC MODEL\n{'─'*60}\n{mat_lines}\n\n  [2] SIMULATION RESULTS"
        )
        st.code(full, language=None)
        dl1, dl2, _ = st.columns([1,1,2])
        with dl1:
            st.download_button("⬇ Report (.txt)", full, "ndt_report.txt",
                               "text/plain", use_container_width=True)
        with dl2:
            csv = generate_csv_data(pts, base, d1, d2, c1, c2)
            st.download_button("⬇ Scan Data (.csv)", csv, "ndt_scan_data.csv",
                               "text/csv", use_container_width=True)

st.markdown("""<hr><p style="text-align:center;color:#1a3050;font-size:.73rem;
font-family:monospace;letter-spacing:.1em;">
ACOUSTISCAN PRO v3.0 · Physics Material Model · Gantry Animation · Live Acquisition · Phased Array
</p>""", unsafe_allow_html=True)
