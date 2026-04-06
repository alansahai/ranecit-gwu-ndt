"""
report.py — NDT defect detection report generator
"""

import io
import datetime
import numpy as np


def generate_text_report(geo_info: dict, scan_info: dict,
                          comparison1: dict, comparison2: dict,
                          component_name: str = "Unknown") -> str:
    """Generate a plain-text NDT inspection report."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "─" * 60

    lines = [
        "═" * 60,
        "   ACOUSTIC NDT INSPECTION REPORT",
        "   AcoustiScan Pro — Simulation Engine v1.0",
        "═" * 60,
        f"  Report Generated : {now}",
        f"  Component        : {component_name}",
        f"  Scan Mode        : {scan_info.get('type', 'N/A')}",
        sep,
        "",
        "  [1] COMPONENT GEOMETRY",
        sep,
        f"  Vertices         : {geo_info['vertices']:,}",
        f"  Faces            : {geo_info['faces']:,}",
        f"  Bounding Box     : {geo_info['bbox_x']*100:.1f} cm × "
        f"{geo_info['bbox_y']*100:.1f} cm × {geo_info['bbox_z']*100:.1f} cm",
        f"  Volume           : {geo_info['volume_cm3']:.2f} cm³",
        f"  Surface Area     : {geo_info['surface_area_cm2']:.2f} cm²",
        f"  Aspect Ratio     : {geo_info['aspect_ratio']:.2f}",
        f"  Watertight       : {'Yes' if geo_info['is_watertight'] else 'No'}",
        "",
        "  [2] SIMULATION RESULTS",
        sep,
        "",
        "  ▶ Defect Simulation 1: Porosity / Void Clusters",
        f"    Defect Points Detected : {comparison1['defect_mask'].sum():,} / "
        f"{len(comparison1['defect_mask']):,}",
        f"    Detection Rate         : {comparison1['detection_rate']:.1f}%",
        f"    Mean Amplitude Δ       : {comparison1['amplitude_deviation'].mean():.4f}",
        f"    Mean Frequency Δ       : {comparison1['frequency_deviation'].mean():.4f}",
        f"    Mean TOF Δ             : {comparison1['tof_deviation'].mean():.4f} µs",
        f"    Severity Assessment    : {'HIGH' if comparison1['detection_rate'] > 15 else 'MODERATE' if comparison1['detection_rate'] > 5 else 'LOW'}",
        "",
        "  ▶ Defect Simulation 2: Cold Shut / Unfilled Region",
        f"    Defect Points Detected : {comparison2['defect_mask'].sum():,} / "
        f"{len(comparison2['defect_mask']):,}",
        f"    Detection Rate         : {comparison2['detection_rate']:.1f}%",
        f"    Mean Amplitude Δ       : {comparison2['amplitude_deviation'].mean():.4f}",
        f"    Mean Frequency Δ       : {comparison2['frequency_deviation'].mean():.4f}",
        f"    Mean TOF Δ             : {comparison2['tof_deviation'].mean():.4f} µs",
        f"    Severity Assessment    : {'HIGH' if comparison2['detection_rate'] > 15 else 'MODERATE' if comparison2['detection_rate'] > 5 else 'LOW'}",
        "",
        "  [3] OVERALL VERDICT",
        sep,
        "  ⚠️  DEFECTS DETECTED — Component requires further inspection.",
        "  Recommend: Physical cross-section analysis at flagged regions.",
        "",
        "  [4] NOTES",
        sep,
        "  • Simulation based on Aluminium alloy A380 properties",
        "  • Transducer frequency: 2.5 MHz (standard casting inspection)",
        "  • This is a simulation report — not a certified NDT result.",
        "═" * 60,
    ]

    return "\n".join(lines)


def generate_csv_data(scan_points: np.ndarray, baseline: dict,
                      defect1: dict, defect2: dict, comp1: dict, comp2: dict) -> str:
    """Export all scan data as CSV."""
    header = ("Point,X,Y,Z,"
              "Base_Amp,Base_Freq,Base_TOF,"
              "D1_Amp,D1_Freq,D1_TOF,D1_Defect,"
              "D2_Amp,D2_Freq,D2_TOF,D2_Defect,"
              "Severity_D1,Severity_D2")

    rows = [header]
    for i in range(len(scan_points)):
        row = (
            f"{i},{scan_points[i,0]:.5f},{scan_points[i,1]:.5f},{scan_points[i,2]:.5f},"
            f"{baseline['amplitude'][i]:.4f},{baseline['frequency'][i]:.4f},{baseline['tof'][i]:.4f},"
            f"{defect1['amplitude'][i]:.4f},{defect1['frequency'][i]:.4f},{defect1['tof'][i]:.4f},"
            f"{int(comp1['defect_mask'][i])},"
            f"{defect2['amplitude'][i]:.4f},{defect2['frequency'][i]:.4f},{defect2['tof'][i]:.4f},"
            f"{int(comp2['defect_mask'][i])},"
            f"{comp1['combined_severity'][i]:.4f},{comp2['combined_severity'][i]:.4f}"
        )
        rows.append(row)

    return "\n".join(rows)
