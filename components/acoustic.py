"""
acoustic.py — Acoustic wave simulation engine for NDT
Models: baseline, porosity defect, cold-shut defect
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ──────────────────────────────────────────────
# Material constants — Aluminium alloy (A380)
# ──────────────────────────────────────────────
AL_DENSITY      = 2700.0   # kg/m³
AL_YOUNGS_MOD   = 70e9     # Pa
AL_SPEED_SOUND  = 6320.0   # m/s (longitudinal)
BASE_FREQ       = 2.5e6    # Hz  (2.5 MHz transducer)
K_STIFFNESS     = 0.85     # coupling coefficient


def _local_thickness(scan_points: np.ndarray, mesh_height: float) -> np.ndarray:
    """Estimate local thickness as fraction of bounding-box height."""
    z_min = scan_points[:, 2].min()
    z_max = scan_points[:, 2].max()
    # Normalised depth below surface → thickness proxy
    thickness = (z_max - scan_points[:, 2]) / max(z_max - z_min, 1e-6)
    return np.clip(thickness, 0.05, 1.0)


def simulate_baseline(scan_points: np.ndarray, mesh_height: float,
                       noise_level: float = 0.04, seed: int = 42) -> dict:
    """
    Simulate ideal (defect-free) acoustic response.
    Returns frequency, amplitude, and time-of-flight arrays.
    """
    rng = np.random.default_rng(seed)
    N = len(scan_points)
    thickness = _local_thickness(scan_points, mesh_height)

    # Frequency: f ∝ √(E/ρ) / thickness
    freq = BASE_FREQ * K_STIFFNESS * np.sqrt(AL_YOUNGS_MOD / AL_DENSITY) / (
        AL_SPEED_SOUND * thickness * 10
    )
    freq = freq / freq.max()  # normalise to [0,1] range for display

    # Amplitude: healthy material → high, uniform amplitude
    amplitude = 0.90 + rng.normal(0, noise_level, N)
    amplitude = np.clip(amplitude, 0, 1)

    # Time-of-flight: proportional to thickness
    tof = (2 * thickness * mesh_height) / AL_SPEED_SOUND * 1e6  # µs
    tof += rng.normal(0, noise_level * tof.mean(), N)

    return {"frequency": freq, "amplitude": amplitude, "tof": tof,
            "type": "Baseline (Defect-Free)"}


def simulate_porosity(scan_points: np.ndarray, mesh_height: float,
                       defect_density: float = 0.12, seed: int = 7) -> dict:
    """
    Defect Simulation 1: Porosity / Void clusters
    - Random spherical void regions
    - Reduced amplitude, frequency shift
    """
    rng = np.random.default_rng(seed)
    base = simulate_baseline(scan_points, mesh_height, seed=seed)
    N = len(scan_points)

    # Inject void clusters
    defect_mask = np.zeros(N, dtype=bool)
    n_clusters = max(2, int(defect_density * 8))
    bb_min = scan_points.min(axis=0)
    bb_max = scan_points.max(axis=0)

    for _ in range(n_clusters):
        center = rng.uniform(bb_min, bb_max)
        radius = rng.uniform(0.02, 0.06) * (bb_max - bb_min).max()
        dists = np.linalg.norm(scan_points - center, axis=1)
        defect_mask |= (dists < radius)

    # Effect: amplitude drops, freq distortion, TOF increases (scatter)
    amplitude = base["amplitude"].copy()
    amplitude[defect_mask] *= rng.uniform(0.3, 0.6, defect_mask.sum())
    amplitude += rng.normal(0, 0.05, N)
    amplitude = np.clip(amplitude, 0, 1)

    freq = base["frequency"].copy()
    freq[defect_mask] *= rng.uniform(0.5, 0.85, defect_mask.sum())

    tof = base["tof"].copy()
    tof[defect_mask] *= rng.uniform(1.1, 1.4, defect_mask.sum())  # scatter delay

    return {
        "frequency": freq, "amplitude": amplitude, "tof": tof,
        "defect_mask": defect_mask,
        "type": "Defect 1 — Porosity / Void Clusters",
        "defect_count": int(defect_mask.sum()),
    }


def simulate_cold_shut(scan_points: np.ndarray, mesh_height: float,
                        seed: int = 13) -> dict:
    """
    Defect Simulation 2: Cold Shut / Unfilled region
    - Linear discontinuity (interface defect)
    - Signal reflection, amplitude drop, TOF spike
    """
    rng = np.random.default_rng(seed)
    base = simulate_baseline(scan_points, mesh_height, seed=seed)
    N = len(scan_points)

    # Cold shut: planar slab defect along a random axis
    bb_min = scan_points.min(axis=0)
    bb_max = scan_points.max(axis=0)
    axis = rng.integers(0, 2)  # X or Y axis
    shut_pos = rng.uniform(0.25, 0.75) * (bb_max[axis] - bb_min[axis]) + bb_min[axis]
    shut_width = rng.uniform(0.04, 0.09) * (bb_max[axis] - bb_min[axis])

    defect_mask = np.abs(scan_points[:, axis] - shut_pos) < shut_width

    # Effect: reflection → amplitude spike then near-zero, TOF jump
    amplitude = base["amplitude"].copy()
    # Approach zone — brief spike
    approach = np.abs(scan_points[:, axis] - shut_pos) < shut_width * 1.5
    amplitude[approach] = np.clip(amplitude[approach] * 1.3, 0, 1)
    # Inside shut — signal loss
    amplitude[defect_mask] *= rng.uniform(0.1, 0.35, defect_mask.sum())
    amplitude = np.clip(amplitude + rng.normal(0, 0.04, N), 0, 1)

    freq = base["frequency"].copy()
    freq[defect_mask] *= rng.uniform(0.2, 0.5, defect_mask.sum())

    tof = base["tof"].copy()
    tof[defect_mask] *= rng.uniform(1.5, 2.5, defect_mask.sum())  # reflection delay

    return {
        "frequency": freq, "amplitude": amplitude, "tof": tof,
        "defect_mask": defect_mask,
        "type": "Defect 2 — Cold Shut / Unfilled Region",
        "defect_count": int(defect_mask.sum()),
    }


def compare_signals(baseline: dict, defect: dict) -> dict:
    """Compute deviation metrics between baseline and defect simulation."""
    amp_dev = np.abs(baseline["amplitude"] - defect["amplitude"])
    freq_dev = np.abs(baseline["frequency"] - defect["frequency"])
    tof_dev  = np.abs(baseline["tof"] - defect["tof"])

    defect_mask = defect.get("defect_mask", amp_dev > 0.15)

    return {
        "amplitude_deviation": amp_dev,
        "frequency_deviation": freq_dev,
        "tof_deviation": tof_dev,
        "combined_severity": (amp_dev * 0.5 + freq_dev * 0.3 + tof_dev / tof_dev.max() * 0.2),
        "defect_mask": defect_mask,
        "detection_rate": float(defect_mask.sum() / len(defect_mask) * 100),
    }
