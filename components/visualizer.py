"""
visualizer.py — All Plotly-based visualizations for the NDT dashboard
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import trimesh


# ──────────────────────────────────────────────
# Color palette (dark industrial NDT theme)
# ──────────────────────────────────────────────
COLORS = {
    "bg": "#0a0e1a",
    "panel": "#111827",
    "accent": "#00d4ff",
    "green": "#00ff88",
    "red": "#ff3366",
    "warn": "#ffaa00",
    "text": "#e0e8ff",
    "grid": "#1e293b",
}

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="monospace"),
    margin=dict(l=10, r=10, t=40, b=10),
)


def plot_3d_mesh(mesh: trimesh.Trimesh, scan_points: np.ndarray = None,
                 defect_mask: np.ndarray = None, title: str = "Component 3D View") -> go.Figure:
    """Render mesh as 3D surface with optional scan path and defect overlay."""
    verts = mesh.vertices
    faces = mesh.faces

    fig = go.Figure()

    # Mesh surface
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.35,
        color="#4488bb",
        flatshading=True,
        name="Component",
        showlegend=True,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3),
    ))

    # Scan points
    if scan_points is not None and len(scan_points) > 0:
        if defect_mask is not None and len(defect_mask) == len(scan_points):
            # Color by defect status
            colors = np.where(defect_mask, COLORS["red"], COLORS["green"])
            # Split for legend
            ok_pts = scan_points[~defect_mask]
            def_pts = scan_points[defect_mask]
            if len(ok_pts) > 0:
                fig.add_trace(go.Scatter3d(
                    x=ok_pts[:, 0], y=ok_pts[:, 1], z=ok_pts[:, 2],
                    mode="markers",
                    marker=dict(size=3.5, color=COLORS["green"], opacity=0.8),
                    name="✅ Normal",
                ))
            if len(def_pts) > 0:
                fig.add_trace(go.Scatter3d(
                    x=def_pts[:, 0], y=def_pts[:, 1], z=def_pts[:, 2],
                    mode="markers",
                    marker=dict(size=5, color=COLORS["red"], opacity=0.95,
                                symbol="diamond"),
                    name="⚠️ Defect",
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=scan_points[:, 0], y=scan_points[:, 1], z=scan_points[:, 2],
                mode="markers",
                marker=dict(size=3, color=COLORS["accent"], opacity=0.7,
                            colorscale="Blues"),
                name="🔵 Scan Points",
            ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=15, color=COLORS["accent"])),
        scene=dict(
            bgcolor="rgba(5,10,20,0.9)",
            xaxis=dict(gridcolor=COLORS["grid"], showbackground=False),
            yaxis=dict(gridcolor=COLORS["grid"], showbackground=False),
            zaxis=dict(gridcolor=COLORS["grid"], showbackground=False),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor=COLORS["grid"],
                    borderwidth=1),
        height=480,
    )
    return fig


def plot_scan_path(scan_points: np.ndarray, title: str = "Transducer Scan Path") -> go.Figure:
    """Show the sequential transducer path as an animated line."""
    fig = go.Figure()

    # Path line
    fig.add_trace(go.Scatter3d(
        x=scan_points[:, 0], y=scan_points[:, 1], z=scan_points[:, 2],
        mode="lines+markers",
        line=dict(color=COLORS["accent"], width=2),
        marker=dict(size=2.5, color=np.arange(len(scan_points)),
                    colorscale="Plasma", opacity=0.8),
        name="Scan Path",
    ))

    # Start & end markers
    fig.add_trace(go.Scatter3d(
        x=[scan_points[0, 0]], y=[scan_points[0, 1]], z=[scan_points[0, 2]],
        mode="markers+text",
        marker=dict(size=8, color=COLORS["green"], symbol="square"),
        text=["START"], textposition="top center",
        name="Start",
    ))
    fig.add_trace(go.Scatter3d(
        x=[scan_points[-1, 0]], y=[scan_points[-1, 1]], z=[scan_points[-1, 2]],
        mode="markers+text",
        marker=dict(size=8, color=COLORS["red"], symbol="x"),
        text=["END"], textposition="top center",
        name="End",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=15, color=COLORS["accent"])),
        scene=dict(bgcolor="rgba(5,10,20,0.9)",
                   xaxis=dict(showbackground=False, gridcolor=COLORS["grid"]),
                   yaxis=dict(showbackground=False, gridcolor=COLORS["grid"]),
                   zaxis=dict(showbackground=False, gridcolor=COLORS["grid"])),
        height=420,
    )
    return fig


def plot_signal_comparison(scan_points: np.ndarray,
                            baseline: dict, defect1: dict, defect2: dict,
                            signal_key: str = "amplitude") -> go.Figure:
    """3-panel comparison of baseline vs defect signals along scan index."""
    idx = np.arange(len(scan_points))
    label_map = {
        "amplitude": ("Amplitude (normalized)", "Signal Amplitude"),
        "frequency": ("Frequency (normalized)", "Acoustic Frequency"),
        "tof": ("Time-of-Flight (µs)", "TOF"),
    }
    y_label, title_str = label_map.get(signal_key, ("Value", signal_key))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["Baseline (Defect-Free)", defect1["type"], defect2["type"]],
        vertical_spacing=0.08,
    )

    for row, sim in enumerate([baseline, defect1, defect2], start=1):
        vals = sim[signal_key]
        mask = sim.get("defect_mask", np.zeros(len(vals), dtype=bool))
        color = COLORS["green"] if row == 1 else COLORS["warn"]

        fig.add_trace(go.Scatter(
            x=idx, y=vals,
            mode="lines",
            line=dict(color=color, width=1.5),
            name=f"{'Baseline' if row == 1 else f'Defect {row-1}'}",
            showlegend=(row == 1),
        ), row=row, col=1)

        # Highlight defect zones
        if mask.any():
            def_idx = np.where(mask)[0]
            fig.add_trace(go.Scatter(
                x=def_idx, y=vals[def_idx],
                mode="markers",
                marker=dict(size=4, color=COLORS["red"], opacity=0.8),
                name="⚠️ Defect" if row == 2 else None,
                showlegend=(row == 2),
            ), row=row, col=1)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"Signal Comparison — {title_str}",
                   font=dict(size=14, color=COLORS["accent"])),
        height=520,
    )
    fig.update_yaxes(gridcolor=COLORS["grid"])
    fig.update_xaxes(gridcolor=COLORS["grid"])
    return fig


def plot_heatmap_2d(scan_points: np.ndarray, values: np.ndarray,
                    title: str = "Defect Severity Heatmap",
                    colorscale: str = "RdYlGn_r") -> go.Figure:
    """2D heatmap of defect severity projected onto XY plane."""
    x = scan_points[:, 0]
    y = scan_points[:, 1]

    fig = go.Figure(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(
            size=10,
            color=values,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="Severity", tickfont=dict(color=COLORS["text"])),
            opacity=0.85,
        ),
        text=[f"Severity: {v:.3f}" for v in values],
        hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>%{text}<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=14, color=COLORS["accent"])),
        xaxis=dict(title="X Position (m)", gridcolor=COLORS["grid"]),
        yaxis=dict(title="Y Position (m)", gridcolor=COLORS["grid"], scaleanchor="x"),
        height=400,
    )
    return fig


def plot_deviation_bars(comparison1: dict, comparison2: dict) -> go.Figure:
    """Bar chart of mean deviations across signal metrics."""
    metrics = ["Amplitude Δ", "Frequency Δ", "TOF Δ (norm.)"]
    vals1 = [
        float(comparison1["amplitude_deviation"].mean()),
        float(comparison1["frequency_deviation"].mean()),
        float(comparison1["tof_deviation"].mean() / comparison1["tof_deviation"].max()),
    ]
    vals2 = [
        float(comparison2["amplitude_deviation"].mean()),
        float(comparison2["frequency_deviation"].mean()),
        float(comparison2["tof_deviation"].mean() / comparison2["tof_deviation"].max()),
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Porosity Defect", x=metrics, y=vals1,
                         marker_color="#ff6b35", opacity=0.85))
    fig.add_trace(go.Bar(name="Cold Shut Defect", x=metrics, y=vals2,
                         marker_color="#ff3366", opacity=0.85))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Mean Signal Deviation vs Baseline",
                   font=dict(size=14, color=COLORS["accent"])),
        barmode="group",
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], title="Mean Deviation"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        height=340,
    )
    return fig
