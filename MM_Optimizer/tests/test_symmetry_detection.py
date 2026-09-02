"""
Test _classify_symmetry (and legacy _check_rotation_symmetry) via analyze_mesh
on synthetic primitives.

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
from mesh_analysis import (analyze_mesh, _check_rotation_symmetry, _chamfer_distance,
                            _rotate_points,
                            SYM_ASYMMETRIC, SYM_C2, SYM_C3, SYM_C4, SYM_C6,
                            SYM_SO2, SYM_SO3)


def pcd_from_mesh(mesh: o3d.geometry.TriangleMesh, n: int = 5000) -> o3d.geometry.PointCloud:
    pcd = mesh.sample_points_uniformly(n)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    return pcd


# ---------------------------------------------------------------------------
# Mesh constructors
# ---------------------------------------------------------------------------

def small_tab_plate() -> o3d.geometry.TriangleMesh:
    """500x70x20mm flat plate with a 20x20x10mm tab on the +X end only.
    The tab covers ~1.7% of the surface. After rotation, the ~85 tab points
    land 5-10mm outside the plate surface -- mean Chamfer contribution just
    above the ~2mm sampling noise baseline. In a bin-picking scene this small
    feature is frequently occluded, so enabling rotation search is correct.
    """
    base = o3d.geometry.TriangleMesh.create_box(0.5, 0.07, 0.02)
    tab  = o3d.geometry.TriangleMesh.create_box(0.02, 0.02, 0.01)
    tab.translate([0.49, 0.025, 0.02])
    return base + tab


def asymmetric_step() -> o3d.geometry.TriangleMesh:
    """Step shape: first 250mm is 60mm tall, second 250mm is 10mm tall.
    ~50% of points mismatch by ~25mm after rotation -> mean Chamfer ~15mm,
    well above the 5mm threshold. Rotation search is correctly disabled.
    """
    tall  = o3d.geometry.TriangleMesh.create_box(0.25, 0.07, 0.06)
    short = o3d.geometry.TriangleMesh.create_box(0.25, 0.07, 0.01)
    short.translate([0.25, 0.0, 0.0])
    return tall + short


def hex_bolt_head() -> o3d.geometry.TriangleMesh:
    """Hex prism -- 6-fold rotational symmetry about Z."""
    import math
    r, h = 0.05, 0.02
    top_z, bot_z = h, 0.0
    verts = []
    for k in range(6):
        a = math.radians(k * 60)
        verts.append([r * math.cos(a), r * math.sin(a), top_z])
        verts.append([r * math.cos(a), r * math.sin(a), bot_z])
    verts += [[0, 0, top_z], [0, 0, bot_z]]   # centres
    n = len(verts)
    ct, cb = n - 2, n - 1
    tris = []
    for k in range(6):
        a, b = k * 2, k * 2 + 1
        c, d = ((k + 1) % 6) * 2, ((k + 1) % 6) * 2 + 1
        tris += [[a, c, ct], [b, cb, d], [a, b, d], [a, d, c]]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()
    return mesh


def triangular_prism() -> o3d.geometry.TriangleMesh:
    """Equilateral triangular prism -- 3-fold symmetry about the long Z axis."""
    import math
    r, h = 0.05, 0.15
    verts = []
    for k in range(3):
        a = math.radians(k * 120)
        verts.append([r * math.cos(a), r * math.sin(a), 0.0])
        verts.append([r * math.cos(a), r * math.sin(a), h])
    verts += [[0, 0, 0.0], [0, 0, h]]   # centres
    bt, tp = 6, 7
    tris = []
    for k in range(3):
        a, b = k * 2, k * 2 + 1
        c, d = ((k + 1) % 3) * 2, ((k + 1) % 3) * 2 + 1
        tris += [[a, c, bt], [b, tp, d], [a, b, d], [a, d, c]]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()
    return mesh


def shaft_with_symmetric_bar() -> o3d.geometry.TriangleMesh:
    """Thin cylinder shaft (r=5mm, h=100mm) + wide flat bar (80x30x10mm) centred at top.

    The bar is centred on Z and aligned with X, giving 2-fold (C2) symmetry:
    a 180-degree Z rotation maps the assembly to itself.

    90-degree rotation maps X-bar to Y-bar -- clearly different -- so this is C2, NOT SO2.

    Thin shaft (small surface area ~6.5 cm2) ensures the bar (surface ~8.6 cm2)
    dominates the point cloud, so the Chamfer test is not diluted by the
    near-circular shaft cross-section.
    """
    shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=0.1)
    shaft.translate([0, 0, 0.05])    # shaft along Z: 0..0.1 m
    bar = o3d.geometry.TriangleMesh.create_box(0.08, 0.03, 0.01)
    bar.translate([-0.04, -0.015, 0.095])   # 80x30x10mm centred on Z, top of shaft
    return shaft + bar


# ---------------------------------------------------------------------------
# Test matrix
# (label, mesh_factory, expect_any_sym, expected_symmetry_class_or_None)
# expected_symmetry_class_or_None=None means "any symmetric class is acceptable"
# ---------------------------------------------------------------------------

CASES = [
    # -- Clearly symmetric (broad class check only) ------------------------
    ("cylinder r=50mm h=200mm",
     lambda: o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=0.2),
     True, SYM_SO2),

    ("sphere r=50mm",
     lambda: o3d.geometry.TriangleMesh.create_sphere(radius=0.05),
     True, SYM_SO3),

    ("cube 100mm",
     lambda: o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1),
     True, None),

    ("symmetric cuboid 500x70x20mm",
     lambda: o3d.geometry.TriangleMesh.create_box(0.5, 0.07, 0.02),
     True, SYM_C2),

    ("hex bolt head (6-fold)",
     hex_bolt_head,
     True, None),   # hex approximates a circle well; SO2 is acceptable for bin-picking

    ("triangular prism (3-fold)",
     triangular_prism,
     True, SYM_C3),

    # -- Slightly asymmetric: small occluded feature -> still detect as sym -
    ("flat plate 500x70x20mm + small tab one end (1.7% surface area)",
     small_tab_plate,
     True, None),

    # -- Critical regression: shaft + symmetric bar must be C2, NOT SO2 ---
    ("shaft + symmetric bar: C2 not SO2",
     shaft_with_symmetric_bar,
     True, SYM_C2),

    # -- Clearly asymmetric ------------------------------------------------
    ("asymmetric step (tall left, short right, 50% mismatch)",
     asymmetric_step,
     False, SYM_ASYMMETRIC),
]


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def _debug_chamfer(pcd: o3d.geometry.PointCloud) -> None:
    """Print per-axis Chamfer distances (180 deg) for threshold tuning."""
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    obb  = pcd.get_minimal_oriented_bounding_box()
    axes = obb.R.T
    for i, axis in enumerate(axes):
        rotated = _rotate_points(centered, axis, 180.0)
        d = _chamfer_distance(centered, rotated)
        print(f"       axis[{i}] Chamfer(180deg) = {d*1e3:.2f} mm")


# ---------------------------------------------------------------------------
# Run cases
# ---------------------------------------------------------------------------

PASS_COUNT = 0
FAIL_COUNT = 0

for label, mesh_factory, expect_sym, expect_class in CASES:
    mesh = mesh_factory()
    mesh.compute_vertex_normals()
    pcd = pcd_from_mesh(mesh)
    ws  = analyze_mesh(pcd)
    detected = (ws.sym_order is not None
                or ws.symmetry_class in (SYM_SO2, SYM_SO3))
    ok = detected == expect_sym

    class_ok = True
    if ok and expect_class is not None:
        class_ok = (ws.symmetry_class == expect_class)
        if not class_ok:
            ok = False

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")
    print(f"       symmetry_class={ws.symmetry_class}  sym_order={ws.sym_order}"
          f"  sym_axis={ws.sym_axis}  flatness={ws.flatness_ratio:.1f}")
    if not ok and not class_ok:
        print(f"       EXPECTED class={expect_class}  GOT={ws.symmetry_class}")
    _debug_chamfer(pcd)
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1

# ---------------------------------------------------------------------------
# Explicit SO2 regression: shaft+bar must NOT be classified SO2
# ---------------------------------------------------------------------------
print("\n-- Shaft+bar SO2 regression (8000 pts) --")
mesh_shaft = shaft_with_symmetric_bar()
mesh_shaft.compute_vertex_normals()
pcd_shaft = pcd_from_mesh(mesh_shaft, n=8000)
ws_shaft  = analyze_mesh(pcd_shaft)
shaft_ok  = ws_shaft.symmetry_class != SYM_SO2
print(f"[{'PASS' if shaft_ok else 'FAIL'}] symmetry_class={ws_shaft.symmetry_class}"
      f"  (must not be SO2)")
if shaft_ok:
    PASS_COUNT += 1
else:
    FAIL_COUNT += 1
    print("       FAIL: eigenvalue false-positive not caught by geometric SO2 test")

print(f"\n{PASS_COUNT}/{PASS_COUNT + FAIL_COUNT} passed")
