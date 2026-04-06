"""
gantry.py — Gantry animation engine
Generates Plotly animated figure of:
  1. Gantry arm moving point-by-point over the 3D component
  2. Sensor array reading panel (live bar + waveform)
  3. Sequential signal build-up graph
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trimesh


COLORS = {
    "bg":     "#080c18",
    "panel":  "#0d1828",
    "accent": "#00d4ff",
    "green":  "#00ff88",
    "red":    "#ff3366",
    "warn":   "#ffaa00",
    "yellow": "#ffdd00",
    "text":   "#c8d8f0",
    "grid":   "#1a2d45",
    "gantry": "#888eaa",
    "beam":   "#00d4ff",
}

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="monospace", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
)


def _gantry_geometry(pos: np.ndarray, bb_min: np.ndarray, bb_max: np.ndarray,
                     arm_height: float = 0.06):
    """
    Return x,y,z arrays for gantry structure at given scan position.
    Gantry = horizontal crossbeam + vertical probe arm + transducer head.
    """
    x0, y0, z0 = pos
    span_y = bb_max[1] - bb_min[1]
    z_top  = bb_max[2] + arm_height

    # Horizontal crossbeam (spans Y axis at current X position)
    beam_x = [x0, x0]
    beam_y = [bb_min[1] - 0.01, bb_max[1] + 0.01]
    beam_z = [z_top, z_top]

    # Vertical arm (drops from beam to transducer head)
    arm_x = [x0, x0]
    arm_y = [y0, y0]
    arm_z = [z_top, z0 + 0.01]

    # Transducer head (small sphere marker returned separately)
    return beam_x, beam_y, beam_z, arm_x, arm_y, arm_z


def build_gantry_animation(mesh: trimesh.Trimesh,
                            scan_points: np.ndarray,
                            amplitude: np.ndarray,
                            frequency: np.ndarray,
                            tof: np.ndarray,
                            defect_mask: np.ndarray,
                            step_size: int = 3,
                            title: str = "Gantry Scan Animation") -> go.Figure:
    """
    Build a Plotly animated figure with play/pause controls.

    Every frame shows:
    - Mesh (static)
    - Gantry at current position
    - Visited scan points coloured by defect/ok
    - Current transducer reading as text annotation

    step_size: subsample every N points to reduce frame count for performance.
    """
    pts   = scan_points[::step_size]
    amp   = amplitude[::step_size]
    freq  = frequency[::step_size]
    tof_v = tof[::step_size]
    dmask = defect_mask[::step_size]
    N     = len(pts)

    bb_min = mesh.bounds[0]
    bb_max = mesh.bounds[1]
    verts  = mesh.vertices
    faces  = mesh.faces
    arm_h  = (bb_max[2] - bb_min[2]) * 0.35

    def _color(a, d):
        if d:
            return COLORS["red"]
        if a > 0.75:
            return COLORS["green"]
        if a > 0.45:
            return COLORS["warn"]
        return COLORS["red"]

    # ── Static mesh trace ────────────────────────────────────
    mesh_trace = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.22, color="#3a6080", flatshading=True,
        name="Component", showlegend=False,
        lighting=dict(ambient=0.6, diffuse=0.8),
    )

    # ── Build frames ─────────────────────────────────────────
    frames = []
    for i in range(N):
        pos = pts[i]
        bx, by, bz, ax_, ay_, az_ = _gantry_geometry(pos, bb_min, bb_max, arm_h)

        # Visited points up to i
        visited   = pts[:i+1]
        v_amp     = amp[:i+1]
        v_mask    = dmask[:i+1]
        pt_colors = [_color(a, d) for a, d in zip(v_amp, v_mask)]

        frame_data = [
            mesh_trace,
            # Visited scan points
            go.Scatter3d(
                x=visited[:, 0], y=visited[:, 1], z=visited[:, 2],
                mode="markers",
                marker=dict(size=3.5, color=pt_colors, opacity=0.85),
                name="Scanned", showlegend=False,
            ),
            # Gantry crossbeam
            go.Scatter3d(
                x=bx, y=by, z=bz,
                mode="lines",
                line=dict(color=COLORS["gantry"], width=6),
                name="Beam", showlegend=False,
            ),
            # Probe arm
            go.Scatter3d(
                x=ax_, y=ay_, z=az_,
                mode="lines",
                line=dict(color=COLORS["accent"], width=4),
                name="Arm", showlegend=False,
            ),
            # Transducer head
            go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2] + 0.005],
                mode="markers",
                marker=dict(size=9, color=COLORS["yellow"],
                            symbol="diamond", opacity=1.0,
                            line=dict(color="#fff", width=1)),
                name="Transducer", showlegend=False,
            ),
        ]

        # Reading annotation
        status = "⚠ DEFECT" if dmask[i] else "✓ OK"
        ann_text = (
            f"Point {i+1}/{N} | {status}<br>"
            f"Amp: {amp[i]:.3f} | Freq: {freq[i]:.3f} | TOF: {tof_v[i]:.2f}µs"
        )

        frames.append(go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(
                annotations=[dict(
                    text=ann_text,
                    xref="paper", yref="paper",
                    x=0.01, y=0.97,
                    showarrow=False,
                    font=dict(family="monospace", size=12,
                              color=COLORS["red"] if dmask[i] else COLORS["green"]),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor=COLORS["red"] if dmask[i] else COLORS["green"],
                    borderwidth=1, borderpad=6,
                )]
            ),
        ))

    # ── Initial figure ───────────────────────────────────────
    fig = go.Figure(
        data=[
            mesh_trace,
            go.Scatter3d(x=[], y=[], z=[], mode="markers",
                         marker=dict(size=3, color=COLORS["accent"]),
                         showlegend=False),
            go.Scatter3d(x=[], y=[], z=[], mode="lines",
                         line=dict(color=COLORS["gantry"], width=6),
                         showlegend=False),
            go.Scatter3d(x=[], y=[], z=[], mode="lines",
                         line=dict(color=COLORS["accent"], width=4),
                         showlegend=False),
            go.Scatter3d(x=[], y=[], z=[], mode="markers",
                         marker=dict(size=9, color=COLORS["yellow"]),
                         showlegend=False),
        ],
        frames=frames,
    )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=14, color=COLORS["accent"])),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=0.02, x=0.5, xanchor="center", yanchor="bottom",
            buttons=[
                dict(label="▶  PLAY",
                     method="animate",
                     args=[None, dict(
                         frame=dict(duration=120, redraw=True),
                         fromcurrent=True,
                         transition=dict(duration=0),
                     )]),
                dict(label="⏸  PAUSE",
                     method="animate",
                     args=[[None], dict(
                         frame=dict(duration=0, redraw=False),
                         mode="immediate",
                     )]),
            ],
            bgcolor="rgba(0,20,40,0.8)",
            bordercolor=COLORS["accent"],
            font=dict(color=COLORS["accent"], family="monospace"),
        )],
        sliders=[dict(
            steps=[dict(
                args=[[str(i)],
                      dict(frame=dict(duration=0, redraw=True),
                           mode="immediate")],
                method="animate",
                label=str(i),
            ) for i in range(N)],
            currentvalue=dict(
                prefix="Scan Point: ",
                font=dict(color=COLORS["accent"], family="monospace"),
                visible=True,
            ),
            pad=dict(b=10, t=10),
            bgcolor=COLORS["panel"],
            bordercolor=COLORS["grid"],
            tickcolor=COLORS["accent"],
            font=dict(color=COLORS["text"], size=9),
            len=0.85,
            x=0.075, y=0.0,
        )],
        scene=dict(
            bgcolor="rgba(5,10,20,0.95)",
            xaxis=dict(showbackground=False, gridcolor=COLORS["grid"]),
            yaxis=dict(showbackground=False, gridcolor=COLORS["grid"]),
            zaxis=dict(showbackground=False, gridcolor=COLORS["grid"]),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
        height=580,
    )
    return fig


def build_live_signal_chart(scan_points: np.ndarray,
                             amplitude: np.ndarray,
                             frequency: np.ndarray,
                             tof: np.ndarray,
                             defect_mask: np.ndarray,
                             step_size: int = 3) -> go.Figure:
    """
    Animated chart where signal traces grow point-by-point
    in sync with gantry movement.
    Shows amplitude, frequency, and TOF simultaneously.
    """
    pts   = scan_points[::step_size]
    amp   = amplitude[::step_size]
    freq  = frequency[::step_size]
    tof_v = tof[::step_size]
    dmask = defect_mask[::step_size]
    idx   = np.arange(len(pts))
    N     = len(pts)

    # Colour per point
    pt_colors = ["#ff3366" if d else "#00ff88" for d in dmask]

    frames = []
    for i in range(N):
        # Subset up to current index
        sub_idx   = idx[:i+1]
        sub_amp   = amp[:i+1]
        sub_freq  = freq[:i+1]
        sub_tof   = tof_v[:i+1]
        sub_mask  = dmask[:i+1]

        def_idx_a  = sub_idx[sub_mask]
        def_amp    = sub_amp[sub_mask]
        def_freq   = sub_freq[sub_mask]
        def_tof    = sub_tof[sub_mask]

        frames.append(go.Frame(
            name=str(i),
            data=[
                # Amplitude line (row 1)
                go.Scatter(x=sub_idx, y=sub_amp, mode="lines",
                           line=dict(color="#00ff88", width=1.8),
                           showlegend=False),
                go.Scatter(x=def_idx_a, y=def_amp, mode="markers",
                           marker=dict(size=6, color="#ff3366"),
                           showlegend=False),
                # Frequency line (row 2)
                go.Scatter(x=sub_idx, y=sub_freq, mode="lines",
                           line=dict(color="#00d4ff", width=1.8),
                           showlegend=False),
                go.Scatter(x=def_idx_a, y=def_freq, mode="markers",
                           marker=dict(size=6, color="#ff3366"),
                           showlegend=False),
                # TOF line (row 3)
                go.Scatter(x=sub_idx, y=sub_tof, mode="lines",
                           line=dict(color="#ffaa00", width=1.8),
                           showlegend=False),
                go.Scatter(x=def_idx_a, y=def_tof, mode="markers",
                           marker=dict(size=6, color="#ff3366"),
                           showlegend=False),
                # Cursor lines (current position)
                go.Scatter(x=[i, i], y=[0, 1.1], mode="lines",
                           line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
                           showlegend=False),
                go.Scatter(x=[i, i], y=[0, 1.1], mode="lines",
                           line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
                           showlegend=False),
                go.Scatter(x=[i, i], y=[float(tof_v.min()*0.9), float(tof_v.max()*1.1)],
                           mode="lines",
                           line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
                           showlegend=False),
            ]
        ))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=["Amplitude (normalised)", "Frequency (normalised)", "Time-of-Flight (µs)"],
        vertical_spacing=0.09,
        row_heights=[0.34, 0.33, 0.33],
    )

    # Initial empty traces (6 data + 3 cursor = 9 traces)
    for row in range(1, 4):
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 line=dict(width=1.8)), row=row, col=1)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                                 marker=dict(size=6, color="#ff3366")), row=row, col=1)
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot")),
                      row=row, col=1)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="📈 Live Signal Acquisition — Sync with Gantry",
                   font=dict(size=13, color=COLORS["accent"])),
        showlegend=False,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=-0.07, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶  PLAY",
                     method="animate",
                     args=[None, dict(frame=dict(duration=120, redraw=True),
                                      fromcurrent=True)]),
                dict(label="⏸  PAUSE",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ],
            bgcolor="rgba(0,20,40,0.8)",
            bordercolor=COLORS["accent"],
            font=dict(color=COLORS["accent"], family="monospace"),
        )],
        sliders=[dict(
            steps=[dict(
                args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                method="animate", label="",
            ) for i in range(N)],
            currentvalue=dict(prefix="Position: ",
                              font=dict(color=COLORS["accent"], size=11)),
            pad=dict(b=10, t=10),
            bgcolor=COLORS["panel"], bordercolor=COLORS["grid"],
            len=0.8, x=0.1, y=-0.07,
        )],
        height=520,
    )
    fig.update_yaxes(gridcolor=COLORS["grid"])
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.frames = frames
    return fig


def build_sensor_array_panel(current_idx: int,
                              amplitude: np.ndarray,
                              frequency: np.ndarray,
                              tof: np.ndarray,
                              defect_mask: np.ndarray) -> go.Figure:
    """
    Static snapshot panel showing:
    - 4×4 sensor array heatmap (simulated multi-element array)
    - A-scan waveform at current position
    - Current reading gauges
    """
    i = min(current_idx, len(amplitude) - 1)

    # Simulate a 4×4 sensor array around the current point
    # Spatially jitter 16 channels with correlated noise
    rng = np.random.default_rng(i)
    n_sensors = 16
    base_amp  = amplitude[i]
    is_defect = defect_mask[i]

    sensor_amps = np.clip(
        base_amp + rng.normal(0, 0.06 if not is_defect else 0.18, n_sensors),
        0, 1
    ).reshape(4, 4)

    # A-scan waveform: simulated time-domain pulse
    t_axis = np.linspace(0, tof[i] * 2.5, 200)
    center = tof[i]
    pulse  = (
        np.exp(-((t_axis - center) ** 2) / (2 * (center * 0.08) ** 2))
        * amplitude[i]
        * np.sin(2 * np.pi * frequency[i] * 5 * t_axis / t_axis.max())
    )
    if is_defect:
        # Add ghost reflection from defect boundary
        ghost_pos = center * rng.uniform(0.5, 0.75)
        ghost_amp = amplitude[i] * rng.uniform(0.2, 0.5)
        pulse += (
            np.exp(-((t_axis - ghost_pos) ** 2) / (2 * (ghost_pos * 0.1) ** 2))
            * ghost_amp
            * np.sin(2 * np.pi * frequency[i] * 3 * t_axis / t_axis.max())
        )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Sensor Array (4×4)", "A-Scan Waveform"],
        column_widths=[0.42, 0.58],
    )

    # Heatmap
    fig.add_trace(go.Heatmap(
        z=sensor_amps,
        colorscale=[[0, "#ff3366"], [0.45, "#ffaa00"], [0.75, "#88ff88"], [1, "#00ff88"]],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(len=0.8, thickness=12,
                      title=dict(text="Amp", font=dict(color="#c0d0e8")),
                      tickfont=dict(color=COLORS["text"])),
        text=[[f"{v:.2f}" for v in row] for row in sensor_amps],
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
    ), row=1, col=1)

    # A-scan
    fig.add_trace(go.Scatter(
        x=t_axis, y=pulse,
        mode="lines",
        line=dict(color=COLORS["red"] if is_defect else COLORS["green"], width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba(255,51,102,0.08)" if is_defect else "rgba(0,255,136,0.08)",
    ), row=1, col=2)
    # Mark TOF gate
    fig.add_vline(x=tof[i], line=dict(color=COLORS["accent"], dash="dash", width=1),
                  row=1, col=2)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"Sensor Array Reading — Point {i+1} | "
                        f"{'⚠ DEFECT DETECTED' if is_defect else '✓ NORMAL'}",
                   font=dict(size=12,
                             color=COLORS["red"] if is_defect else COLORS["green"])),
        showlegend=False,
        height=280,
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], row=1, col=2,
                     title_text="Time (µs)")
    fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig
