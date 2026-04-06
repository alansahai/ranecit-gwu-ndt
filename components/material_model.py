"""
material_model.py — Physics-based acoustic model derived from user-supplied
aluminium die cast material properties.

The model computes:
  • Longitudinal wave speed    : v_L = √((E(1-ν)) / (ρ(1+ν)(1-2ν)))
  • Shear wave speed           : v_S = √(G / ρ)
  • Acoustic impedance         : Z   = ρ · v_L
  • Attenuation coefficient    : α   = α₀ · f^n  (frequency-dependent)
  • Natural resonant frequency : f₀  = v_L / (2·t)   per thickness t
  • Reflection coefficient     : R   = (Z2 - Z1)² / (Z2 + Z1)²
  • Skin depth / penetration   : δ   = 1 / (2·α·f)

All inputs are in SI-friendly engineering units (converted internally to SI).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Default reference: Aluminium Alloy A380 die cast
# ─────────────────────────────────────────────────────────────
DEFAULT_PROPS = {
    "density_kg_m3":        2720.0,   # kg/m³
    "youngs_modulus_gpa":   71.0,     # GPa
    "poisson_ratio":        0.33,
    "yield_strength_mpa":   160.0,    # MPa
    "tensile_strength_mpa": 310.0,    # MPa
    "elongation_pct":       3.5,      # %
    "thermal_conductivity": 96.2,     # W/(m·K)
    "porosity_pct":         0.0,      # % - user-specified casting quality
    "grain_size_um":        50.0,     # µm - affects scattering
    "surface_roughness_um": 1.6,      # µm Ra
    "transducer_freq_mhz":  2.5,      # MHz
    "coupling_medium":      "water",  # water | gel | air
}

COUPLING_IMPEDANCE = {
    "water": 1.483e6,   # Pa·s/m  (Z = ρ·v)
    "gel":   1.520e6,
    "air":   413.0,
}


@dataclass
class MaterialModel:
    """
    Complete physics model for a die-cast aluminium component.
    All derived quantities are computed on construction.
    """
    # ── User inputs ──────────────────────────────────────────
    density:           float   # kg/m³
    youngs_modulus:    float   # Pa
    poisson_ratio:     float
    yield_strength:    float   # Pa
    tensile_strength:  float   # Pa
    elongation_pct:    float   # %
    thermal_cond:      float   # W/(m·K)
    porosity_pct:      float   # %
    grain_size_m:      float   # m
    surface_roughness: float   # m  Ra
    transducer_freq:   float   # Hz
    coupling_medium:   str     = "water"

    # ── Derived (filled post-init) ───────────────────────────
    shear_modulus:     float   = field(init=False)
    bulk_modulus:      float   = field(init=False)
    v_longitudinal:    float   = field(init=False)
    v_shear:           float   = field(init=False)
    impedance:         float   = field(init=False)
    impedance_couplant:float   = field(init=False)
    reflection_coeff:  float   = field(init=False)
    transmission_coeff:float   = field(init=False)
    attenuation_coeff: float   = field(init=False)   # Np/m at transducer_freq
    skin_depth_mm:     float   = field(init=False)   # mm
    wavelength_mm:     float   = field(init=False)   # mm at transducer_freq
    near_field_mm:     float   = field(init=False)   # mm  (transducer dia 10mm)
    porosity_factor:   float   = field(init=False)   # amplitude reduction [0,1]
    scattering_noise:  float   = field(init=False)   # grain scattering noise level

    def __post_init__(self):
        E  = self.youngs_modulus
        ν  = self.poisson_ratio
        ρ  = self.density

        # Elastic moduli
        self.shear_modulus = E / (2 * (1 + ν))
        self.bulk_modulus  = E / (3 * (1 - 2*ν))

        # Wave speeds (isotropic elastic solid)
        self.v_longitudinal = np.sqrt(E * (1 - ν) / (ρ * (1 + ν) * (1 - 2*ν)))
        self.v_shear        = np.sqrt(self.shear_modulus / ρ)

        # Acoustic impedance
        self.impedance          = ρ * self.v_longitudinal
        self.impedance_couplant = COUPLING_IMPEDANCE.get(self.coupling_medium, 1.483e6)

        # Reflection / transmission at couplant-metal interface
        Z1, Z2 = self.impedance_couplant, self.impedance
        self.reflection_coeff   = ((Z2 - Z1) / (Z2 + Z1)) ** 2
        self.transmission_coeff = 1.0 - self.reflection_coeff

        # Attenuation: grain scattering (Rayleigh regime: d << λ)
        # α ≈ C_s · f⁴ · d³  (Rayleigh) + C_g · f² · d  (stochastic)
        f   = self.transducer_freq
        d   = self.grain_size_m
        lam = self.v_longitudinal / f
        if d < lam / 10:   # Rayleigh scattering
            alpha_scatter = 1e-3 * (f ** 4) * (d ** 3) / (self.v_longitudinal ** 3)
        else:               # Stochastic scattering
            alpha_scatter = 1e-1 * (f ** 2) * d / self.v_longitudinal

        alpha_abs = 0.003 * (f / 1e6) ** 1.1   # Np/m — absorption loss
        self.attenuation_coeff = alpha_scatter + alpha_abs   # Np/m

        # Skin depth (1/e penetration depth)
        self.skin_depth_mm = (1.0 / max(self.attenuation_coeff, 1e-9)) * 1000

        # Wavelength
        self.wavelength_mm = (self.v_longitudinal / f) * 1000

        # Near-field length (Fresnel zone, assuming 10 mm dia transducer)
        D_transducer = 0.010   # 10 mm
        self.near_field_mm = (D_transducer ** 2 * f) / (4 * self.v_longitudinal) * 1000

        # Porosity: reduces effective modulus → lower wave speed, lower amplitude
        # Effective medium theory (simple linear approximation)
        p = self.porosity_pct / 100.0
        self.porosity_factor = np.clip((1 - 2.0 * p), 0.1, 1.0)

        # Grain scattering noise (normalised 0→1)
        self.scattering_noise = np.clip(
            (d / lam) ** 2 * 0.15, 0.005, 0.25
        )

    # ── Simulation helpers ───────────────────────────────────

    def compute_tof(self, thickness_m: float) -> float:
        """Round-trip time of flight for given thickness (µs)."""
        return (2 * thickness_m / self.v_longitudinal) * 1e6

    def compute_amplitude(self, thickness_m: float) -> float:
        """
        Normalised amplitude after propagating 2×thickness, accounting for:
          - transmission at interface
          - exponential attenuation
          - porosity reduction
        Returns value in [0, 1].
        """
        path = 2 * thickness_m
        atten = np.exp(-self.attenuation_coeff * path)
        return float(np.clip(
            self.transmission_coeff * atten * self.porosity_factor,
            0, 1
        ))

    def frequency_response(self, thickness_m: float) -> float:
        """
        Apparent resonant frequency for a given local thickness.
        f = v_L / (2·t)  — plate resonance model
        Normalised to [0,1] relative to transducer frequency.
        """
        if thickness_m < 1e-6:
            return 0.0
        f_res = self.v_longitudinal / (2 * thickness_m)
        return float(np.clip(f_res / (self.transducer_freq * 10), 0, 1))

    def sensor_reading(self, thickness_m: float, noise_seed: int = 0) -> dict:
        """Return a single-point sensor reading dict."""
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(0, self.scattering_noise)

        amp  = np.clip(self.compute_amplitude(thickness_m) + noise * 0.5, 0, 1)
        freq = np.clip(self.frequency_response(thickness_m) + noise * 0.3, 0, 1)
        tof  = np.clip(self.compute_tof(thickness_m) + noise * 0.2, 0, None)

        return {"amplitude": amp, "frequency": freq, "tof": tof}

    def summary(self) -> dict:
        """Return a flat dict of all derived properties for display."""
        return {
            "Longitudinal Wave Speed (m/s)":  f"{self.v_longitudinal:,.1f}",
            "Shear Wave Speed (m/s)":         f"{self.v_shear:,.1f}",
            "Acoustic Impedance (MRayl)":     f"{self.impedance/1e6:.3f}",
            "Transmission Coeff.":            f"{self.transmission_coeff:.4f}",
            "Reflection Coeff.":              f"{self.reflection_coeff:.4f}",
            "Attenuation (Np/m)":             f"{self.attenuation_coeff:.4f}",
            "Penetration Depth (mm)":         f"{self.skin_depth_mm:.2f}",
            "Wavelength @ freq (mm)":         f"{self.wavelength_mm:.3f}",
            "Near-Field Length (mm)":         f"{self.near_field_mm:.2f}",
            "Shear Modulus (GPa)":            f"{self.shear_modulus/1e9:.2f}",
            "Bulk Modulus (GPa)":             f"{self.bulk_modulus/1e9:.2f}",
            "Porosity Amplitude Factor":      f"{self.porosity_factor:.3f}",
            "Grain Scattering Noise":         f"{self.scattering_noise:.4f}",
        }


def build_model_from_inputs(props: dict) -> "MaterialModel":
    """
    Construct a MaterialModel from the Streamlit UI input dict.
    Handles unit conversion from user-friendly units → SI.
    """
    return MaterialModel(
        density          = props["density_kg_m3"],
        youngs_modulus   = props["youngs_modulus_gpa"] * 1e9,
        poisson_ratio    = props["poisson_ratio"],
        yield_strength   = props["yield_strength_mpa"] * 1e6,
        tensile_strength = props["tensile_strength_mpa"] * 1e6,
        elongation_pct   = props["elongation_pct"],
        thermal_cond     = props["thermal_conductivity"],
        porosity_pct     = props["porosity_pct"],
        grain_size_m     = props["grain_size_um"] * 1e-6,
        surface_roughness= props["surface_roughness_um"] * 1e-6,
        transducer_freq  = props["transducer_freq_mhz"] * 1e6,
        coupling_medium  = props["coupling_medium"],
    )


def simulate_with_model(scan_points: np.ndarray, mesh_height: float,
                         model: "MaterialModel",
                         defect_type: str = "none",
                         defect_density: float = 0.12,
                         seed: int = 42) -> dict:
    """
    Run a full acoustic simulation using the physics-derived material model.

    defect_type: "none" | "porosity" | "cold_shut"
    Returns per-point arrays: amplitude, frequency, tof, defect_mask
    """
    rng   = np.random.default_rng(seed)
    N     = len(scan_points)

    # Local thickness estimate (Z-position proxy)
    z_min = scan_points[:, 2].min()
    z_max = scan_points[:, 2].max()
    thickness = np.clip(
        (z_max - scan_points[:, 2]) / max(z_max - z_min, 1e-6),
        0.05, 1.0
    ) * mesh_height

    # ── Base signal from physics model ──────────────────────
    amplitude = np.array([model.compute_amplitude(t) for t in thickness])
    frequency = np.array([model.frequency_response(t) for t in thickness])
    tof       = np.array([model.compute_tof(t) for t in thickness])

    # Add grain scattering noise
    noise = rng.normal(0, model.scattering_noise, N)
    amplitude = np.clip(amplitude + noise * 0.4, 0, 1)
    frequency = np.clip(frequency + noise * 0.3, 0, 1)
    tof      += np.abs(noise) * 0.05 * tof.mean()

    defect_mask = np.zeros(N, dtype=bool)

    # ── Defect injection ────────────────────────────────────
    if defect_type == "porosity":
        bb_min = scan_points.min(axis=0)
        bb_max = scan_points.max(axis=0)
        n_clusters = max(2, int(defect_density * 10))
        for _ in range(n_clusters):
            center = rng.uniform(bb_min, bb_max)
            radius = rng.uniform(0.025, 0.07) * (bb_max - bb_min).max()
            dists  = np.linalg.norm(scan_points - center, axis=1)
            cluster_mask = dists < radius
            defect_mask |= cluster_mask
            # Physics: porosity reduces effective density & modulus
            sev = rng.uniform(0.25, 0.55, cluster_mask.sum())
            amplitude[cluster_mask] *= sev
            frequency[cluster_mask] *= rng.uniform(0.55, 0.90, cluster_mask.sum())
            tof[cluster_mask]       *= rng.uniform(1.08, 1.35, cluster_mask.sum())

    elif defect_type == "cold_shut":
        bb_min = scan_points.min(axis=0)
        bb_max = scan_points.max(axis=0)
        axis   = rng.integers(0, 2)
        pos    = rng.uniform(0.25, 0.75) * (bb_max[axis] - bb_min[axis]) + bb_min[axis]
        width  = rng.uniform(0.045, 0.10) * (bb_max[axis] - bb_min[axis])
        dist_to_shut = np.abs(scan_points[:, axis] - pos)
        defect_mask  = dist_to_shut < width
        approach     = dist_to_shut < width * 1.6
        # Reflection spike in approach zone
        amplitude[approach] = np.clip(amplitude[approach] * 1.35, 0, 1)
        # Signal collapse inside shut
        amplitude[defect_mask] *= rng.uniform(0.05, 0.30, defect_mask.sum())
        frequency[defect_mask] *= rng.uniform(0.15, 0.45, defect_mask.sum())
        tof[defect_mask]       *= rng.uniform(1.6, 2.8, defect_mask.sum())
        amplitude = np.clip(amplitude + rng.normal(0, 0.03, N), 0, 1)

    return {
        "amplitude":   amplitude,
        "frequency":   frequency,
        "tof":         tof,
        "defect_mask": defect_mask,
        "defect_count": int(defect_mask.sum()),
        "type": {
            "none":      "Baseline (Defect-Free) — Physics Model",
            "porosity":  "Defect 1 — Porosity / Void Clusters",
            "cold_shut": "Defect 2 — Cold Shut / Unfilled Region",
        }[defect_type],
        "model": model,
    }
