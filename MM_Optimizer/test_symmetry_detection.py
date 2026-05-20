"""
Test _check_rotation_symmetry via analyze_mesh on synthetic primitives.
Run with: python MM_Optimizer/test_symmetry_detection.py
          (or from MM_Optimizer/: python test_symmetry_detection.py)
"""
import sys
import os

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, ".."))
for p in (_DIR, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import open3d as o3d
from mesh_analysis import analyze_mesh, _check_rotation_symmetry, _chamfer_distance


def pcd_from_mesh(mesh: o3d.geometry.TriangleMesh, n: int = 5000) -> o3d.geometry.PointCloud:
    pcd = mesh.sample_points_uniformly(n)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    return pcd


def small_tab_plate() -> o3d.geometry.TriangleMesh:
    """500×70×20mm flat plate with a 20×20×10mm tab on the +X end only.
    The tab covers ~1.7% of the surface. After rotation, the ~85 tab points
    land 5–10mm outside the plate surface → mean Chamfer contribution ≈ 0.1mm
    above the ~2mm sampling noise baseline → detected as symmetric.
    In a bin-picking scene this small feature is frequently occluded, so
    enabling rotation search is the correct behaviour.
    """
    base = o3d.geometry.TriangleMesh.create_box(0.5, 0.07, 0.02)
    tab  = o3d.geometry.TriangleMesh.create_box(0.02, 0.02, 0.01)
    tab.translate([0.49, 0.025, 0.02])
    return base + tab


def asymmetric_step() -> o3d.geometry.TriangleMesh:
    """Step shape: first 250mm is 60mm tall, second 250mm is 10mm tall.
    ~50% of points mismatch by ~25mm after rotation → mean Chamfer ~15mm,
    well above the 5mm threshold. Rotation search is correctly disabled.
    """
    tall  = o3d.geometry.TriangleMesh.create_box(0.25, 0.07, 0.06)
    short = o3d.geometry.TriangleMesh.create_box(0.25, 0.07, 0.01)
    short.translate([0.25, 0.0, 0.0])
    return tall + short


CASES = [
    # ── Clearly symmetric — rotation search must be enabled ────────────
    ("cylinder  r=50mm h=200mm",
     o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=0.2),
     True),
    ("cube 100mm",
     o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1),
     True),
    ("symmetric cuboid 500×70×20mm",
     o3d.geometry.TriangleMesh.create_box(0.5, 0.07, 0.02),
     True),
    # ── Slightly asymmetric — small feature likely occluded in bin, so
    #    rotation search should still be enabled (expect_sym=True) ──────
    ("flat plate 500×70×20mm + small tab one end (1.7% surface area)",
     small_tab_plate(),
     True),
    # ── Clearly asymmetric — rotation search must be disabled ──────────
    ("asymmetric step (tall left half, short right half, 50% mismatch)",
     asymmetric_step(),
     False),
]


def _debug_chamfer(pcd: o3d.geometry.PointCloud) -> None:
    """Print per-axis Chamfer distances for threshold tuning."""
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    obb = pcd.get_minimal_oriented_bounding_box()
    axes = obb.R.T
    for i, axis in enumerate(axes):
        rotated = 2 * (centered @ axis)[:, None] * axis - centered
        d = _chamfer_distance(centered, rotated)
        print(f"       axis[{i}] Chamfer = {d*1e3:.2f} mm")


PASS_COUNT = 0
FAIL_COUNT = 0

for label, mesh, expect_sym in CASES:
    mesh.compute_vertex_normals()
    pcd = pcd_from_mesh(mesh)
    ws  = analyze_mesh(pcd)
    detected = ws.sym_order is not None
    ok = detected == expect_sym
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")
    print(f"       sym_order={ws.sym_order}  sym_axis={ws.sym_axis}  flatness={ws.flatness_ratio:.1f}")
    _debug_chamfer(pcd)
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1

print(f"\n{PASS_COUNT}/{PASS_COUNT + FAIL_COUNT} passed")
