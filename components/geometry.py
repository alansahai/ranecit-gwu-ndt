"""
geometry.py — Geometry loading, analysis, and scan-type classification
"""

import numpy as np
import trimesh
import io


def load_mesh(file_bytes: bytes, filename: str) -> trimesh.Trimesh:
    """Load mesh from uploaded file bytes."""
    ext = filename.lower().split(".")[-1]
    mesh = trimesh.load(io.BytesIO(file_bytes), file_type=ext, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh.fix_normals()
    return mesh


def get_geometry_info(mesh: trimesh.Trimesh) -> dict:
    """Extract key geometric properties."""
    bb = mesh.bounding_box.extents  # [lx, ly, lz]
    dims = sorted(bb, reverse=True)  # largest first
    aspect_ratio = dims[0] / max(dims[1], 1e-6)
    volume = float(mesh.volume) if mesh.is_watertight else float(mesh.convex_hull.volume)
    surface_area = float(mesh.area)

    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "bbox_x": float(bb[0]),
        "bbox_y": float(bb[1]),
        "bbox_z": float(bb[2]),
        "dims_sorted": dims,
        "aspect_ratio": float(aspect_ratio),
        "volume_cm3": volume * 1e6,       # m³ → cm³
        "surface_area_cm2": surface_area * 1e4,  # m² → cm²
        "is_watertight": bool(mesh.is_watertight),
        "num_faces": len(mesh.faces),
    }


def classify_scan_type(geo_info: dict) -> dict:
    """
    Classify component geometry into scan strategy.
    Returns scan type and rationale.
    """
    ar = geo_info["aspect_ratio"]
    faces = geo_info["num_faces"]

    if ar >= 4.0 and faces < 300:
        scan_type = "LINEAR"
        description = "Elongated component detected. Transducer will traverse linearly along the primary axis."
        icon = "📏"
    elif faces > 400 or ar < 2.5:
        scan_type = "MULTI-FACE"
        description = "Complex geometry detected. Multi-face surface traversal with normal-guided scan paths."
        icon = "🔄"
    else:
        scan_type = "GRID"
        description = "Medium complexity geometry. Grid-based raster scan over all major surfaces."
        icon = "⊞"

    return {
        "type": scan_type,
        "description": description,
        "icon": icon,
        "aspect_ratio": ar,
    }


def sample_scan_points(mesh: trimesh.Trimesh, resolution: int = 20) -> np.ndarray:
    """
    Generate a grid of scan points projected onto the mesh surface.
    Returns array of shape (N, 3) — 3D coordinates of scan positions.
    """
    bb_min = mesh.bounds[0]
    bb_max = mesh.bounds[1]

    # Build a 2D grid over XY plane (top-down scanning assumption)
    xs = np.linspace(bb_min[0], bb_max[0], resolution)
    ys = np.linspace(bb_min[1], bb_max[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])

    # Ray-cast downward from above to find surface intersection
    ray_origins = np.column_stack([
        grid_xy[:, 0],
        grid_xy[:, 1],
        np.full(len(grid_xy), bb_max[2] + 0.1)
    ])
    ray_directions = np.tile([0, 0, -1], (len(grid_xy), 1)).astype(float)

    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions
    )

    if len(locations) == 0:
        # Fallback: use surface samples
        pts, _ = trimesh.sample.sample_surface(mesh, resolution * resolution)
        return pts

    # Keep only the first hit per ray (topmost surface point)
    unique_rays, first_idx = np.unique(index_ray, return_index=True)
    scan_points = locations[first_idx]
    return scan_points
