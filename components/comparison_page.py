"""
comparison_page.py — UT vs RT Detailed Comparison
A formatted table and detailed breakdown comparing Ultrasonic NDT with X-Ray RT.
"""

import streamlit as st

def render_comparison_page():
    st.markdown("""<div style="background:linear-gradient(90deg,#1a2d45,transparent 80%);
    border-left:4px solid #6c8ebf;padding:14px 22px;margin-bottom:18px;border-radius:0 10px 10px 0">
    <h2 style="font-family:'Share Tech Mono',monospace;color:#a0b8d8;margin:0;letter-spacing:.06em">
    ⚖️ ACOUSTISCAN PRO (UT) vs X-RAY TESTING (RT)</h2>
    <p style="color:#9E9EC8;margin:4px 0 0;font-size:.85rem">
    Comprehensive comparison of Ultrasonic and Radiographic testing methods for Aluminium Die Castings.</p>
    </div>""", unsafe_allow_html=True)

    # Comparison Data
    data = [
        {"crit": "Radiation safety", 
         "ut": "✓ Zero ionising radiation. No shielding, no dosimetry, no regulatory permits required.", 
         "rt": "✗ Ionising radiation. Requires lead shielding, controlled area, RPO oversight."},
        {"crit": "Defect detection — volumetric", 
         "ut": "Excellent for volumetric defects (porosity, inclusions) perpendicular to beam. Blind to defects parallel to axis.", 
         "rt": "Excellent for volumetric defects regardless of orientation. Detects porosity, inclusions, density variations."},
        {"crit": "Defect detection — planar", 
         "ut": "✓ Superior for planar defects (cold shuts, cracks) when beam is perpendicular. High sensitivity to faces.", 
         "rt": "✗ Poor sensitivity to tight planar cracks parallel to beam. Requires specific geometry alignment."},
        {"crit": "Depth location", 
         "ut": "✓ Provides precise depth (Z) information via TOF. Full 3D localisation with gantry scanning.", 
         "rt": "✗ 2D projection only. No depth information without stereo / CT setup. CT adds significant cost."},
        {"crit": "Through-wall thickness", 
         "ut": "Effective up to ~300 mm in aluminum at 2.25 MHz. Limited by grain noise in coarser alloys.", 
         "rt": "Effective up to ~75 mm Al with conventional RT; Megavoltage RT required for thicker sections."},
        {"crit": "Speed (production line)", 
         "ut": "Moderate. Gantry scan time scales with part size and pitch. Typical: 2–15 min per part.", 
         "rt": "Faster per shot for simple geometry. Film processing adds latency; digital RT (DR) is faster but costly."},
        {"crit": "Portability", 
         "ut": "✓ Portable UT flaw detectors available. Full gantry system is bench-top.", 
         "rt": "Portable X-ray sources exist but radiation exclusion zones limit use on production floors."},
        {"crit": "Cost (equipment)", 
         "ut": "Lower — UT flaw detector + transducer: ₹2–10 L. Full gantry: ₹15–50 L.", 
         "rt": "Higher — DR panel + X-ray source + shielded cabinet: ₹30–150 L. Microfocus CT: ₹200 L+."},
        {"crit": "Operator skill", 
         "ut": "High skill required. ASNT Level II UT needed for production. Scan interpretation requires training.", 
         "rt": "High skill required. RT film interpretation is highly subjective. DR/CT reduces subjectivity."},
        {"crit": "Records & traceability", 
         "ut": "A-scan waveform + heatmap stored digitally. AcoustiScan exports TXT report + CSV.", 
         "rt": "Digital radiograph is a permanent, intuitive image record. Easier for non-specialist review."},
        {"crit": "Surface contact required", 
         "ut": "Yes — coupling medium (gel/water) needed. Surface prep required.", 
         "rt": "No contact. Works on parts inside packaging or behind barriers."},
        {"crit": "Simulation/training tool", 
         "ut": "✓ AcoustiScan Pro provides full physics-driven simulation. No instrument needed for training.", 
         "rt": "X-ray simulation tools exist (VGStudio) but are costlier and hardware-dependent."},
        {"crit": "Regulatory use", 
         "ut": "Accepted per EN 4179, NADCAP UT, AMS 2635. Requires qualified personnel.", 
         "rt": "Accepted per EN 4179, ASTM E1742 (RT). Gold standard for weld root inspection."}
    ]

    st.markdown("""<style>
    .comp-table { width:100%; border-collapse:collapse; margin-bottom:20px; font-size:0.85rem; }
    .comp-table th { background:#111827; padding:12px; text-align:left; color:#a0b8d8; border-bottom:2px solid #1a2d45; }
    .comp-table td { padding:10px 12px; border-bottom:1px solid #1a2d45; vertical-align:top; }
    .comp-table tr:hover { background:#0d1420; }
    .adv { color:#00ff88; }
    .disadv { color:#ff3366; }
    </style>""", unsafe_allow_html=True)

    html = "<table class='comp-table'>"
    html += "<tr><th width='20%'>Criterion</th><th width='40%'>Ultrasonic NDT (AcoustiScan)</th><th width='40%'>X-Ray RT</th></tr>"
    
    for row in data:
        ut_fmt = row['ut'].replace("✓", "<span class='adv'>✓</span>").replace("✗", "<span class='disadv'>✗</span>")
        rt_fmt = row['rt'].replace("✓", "<span class='adv'>✓</span>").replace("✗", "<span class='disadv'>✗</span>")
        html += f"<tr><td><strong>{row['crit']}</strong></td><td>{ut_fmt}</td><td>{rt_fmt}</td></tr>"
        
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0a1525;border:1px solid #162a42;border-left:4px solid #00d4ff;border-radius:4px;padding:20px;margin-top:20px;">
        <h3 style="margin-top:0; color:#00d4ff;">Conclusion</h3>
        <p style="margin-bottom:0;"><strong>Ultrasonic NDT (as simulated by AcoustiScan Pro) is the preferred method for depth-resolved, radiation-free inspection of aluminium die castings</strong> — particularly for cold shut and sub-surface porosity detection. X-ray RT is superior for volumetric defect mapping and providing photographic evidence records, but carries radiation risk, higher cost, and no depth resolution without CT. For most production environments, UT is the primary method; RT/CT is reserved for acceptance testing of critical components or failure analysis.</p>
    </div>
    """, unsafe_allow_html=True)
