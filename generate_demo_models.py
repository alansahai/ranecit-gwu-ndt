"""
Generate 3 demo OBJ files for NDT simulation testing:
1. Simple aluminium rod (linear scan)
2. Engine bracket (medium complexity)
3. Crankcase-like block (complex multi-face)
"""

import numpy as np
import trimesh
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_obj(mesh, name, description):
    path = os.path.join(OUTPUT_DIR, f"{name}.obj")
    mesh.export(path)
    print(f"✅ Saved: {path} | Vertices: {len(mesh.vertices)} | Faces: {len(mesh.faces)}")
    # Save description
    with open(os.path.join(OUTPUT_DIR, f"{name}.txt"), "w") as f:
        f.write(description)


def generate_aluminium_rod():
    """Simple elongated rod — triggers LINEAR scan mode."""
    mesh = trimesh.creation.cylinder(radius=0.05, height=0.6, sections=32)
    mesh.apply_translation([0, 0, 0.3])
    save_obj(mesh, "aluminium_rod",
             "Simple cylindrical aluminium rod (600mm x 50mm diameter). Linear scan mode.")


def generate_engine_bracket():
    """L-shaped bracket — medium complexity."""
    # Main horizontal beam
    beam_h = trimesh.creation.box(extents=[0.4, 0.08, 0.05])
    beam_h.apply_translation([0.2, 0, 0.025])
    # Vertical arm
    beam_v = trimesh.creation.box(extents=[0.05, 0.08, 0.25])
    beam_v.apply_translation([0.025, 0, 0.15])
    # Gusset fillet approximation (small box)
    gusset = trimesh.creation.box(extents=[0.05, 0.08, 0.05])
    gusset.apply_translation([0.025, 0, 0.025])
    # Bolt bosses
    boss1 = trimesh.creation.cylinder(radius=0.018, height=0.02, sections=16)
    boss1.apply_translation([0.35, 0, 0.06])
    boss2 = trimesh.creation.cylinder(radius=0.018, height=0.02, sections=16)
    boss2.apply_translation([0.1, 0, 0.06])
    # Merge all
    combined = trimesh.util.concatenate([beam_h, beam_v, gusset, boss1, boss2])
    save_obj(combined, "engine_bracket",
             "L-shaped aluminium engine bracket with bolt bosses. Multi-face scan mode.")


def generate_crankcase_block():
    """Simplified crankcase-like block — complex multi-face geometry."""
    # Main housing block
    body = trimesh.creation.box(extents=[0.35, 0.20, 0.18])
    body.apply_translation([0.175, 0.10, 0.09])

    # Cylinder bore 1 (subtract by using separate mesh)
    bore1 = trimesh.creation.cylinder(radius=0.045, height=0.20, sections=32)
    bore1.apply_translation([0.10, 0.10, 0.09])

    # Cylinder bore 2
    bore2 = trimesh.creation.cylinder(radius=0.045, height=0.20, sections=32)
    bore2.apply_translation([0.25, 0.10, 0.09])

    # Oil channel ridge
    ridge = trimesh.creation.box(extents=[0.35, 0.015, 0.04])
    ridge.apply_translation([0.175, 0.10, 0.16])

    # Mounting flanges (4 corners)
    flanges = []
    for x, y in [(0.03, 0.03), (0.32, 0.03), (0.03, 0.17), (0.32, 0.17)]:
        f = trimesh.creation.cylinder(radius=0.022, height=0.03, sections=16)
        f.apply_translation([x, y, -0.015])
        flanges.append(f)

    # Rib structures on side
    rib1 = trimesh.creation.box(extents=[0.02, 0.20, 0.08])
    rib1.apply_translation([0.005, 0.10, 0.04])
    rib2 = trimesh.creation.box(extents=[0.02, 0.20, 0.08])
    rib2.apply_translation([0.345, 0.10, 0.04])

    all_parts = [body, bore1, bore2, ridge, rib1, rib2] + flanges
    combined = trimesh.util.concatenate(all_parts)
    save_obj(combined, "crankcase_block",
             "Simplified aluminium crankcase block with bores, oil channels, and mounting flanges. Complex multi-face scan mode.")


if __name__ == "__main__":
    print("Generating demo OBJ models...")
    generate_aluminium_rod()
    generate_engine_bracket()
    generate_crankcase_block()
    print("\nAll models generated successfully!")
