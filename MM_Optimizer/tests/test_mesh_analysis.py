"""
test_mesh_analysis.py  —  Smoke test for Phase 0 mesh analysis
Requires open3d. No MechVision. Run from project root:
    python MM_Optimizer/tests/test_mesh_analysis.py
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from MM_Optimizer.mesh_analysis import analyze_mesh, load_reference_pcd, WarmStart

MODEL_PATH = os.path.join(
    _ROOT, "MM_Optimizer", "CAD_Match", "resource", "3d_matching",
    "25333MB000_surface", "25333MB000_surface.ply"
)


def test_warm_start_values():
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"

    pcd = load_reference_pcd(MODEL_PATH)
    ws  = analyze_mesh(pcd, n_instances=1)

    print(f"  diameter        = {ws.diameter_m*1e3:.1f} mm")
    print(f"  flatness_ratio  = {ws.flatness_ratio:.2f}")
    print(f"  normal_conc     = {ws.normal_concentration:.2f}")
    print(f"  prefer_edge     = {ws.prefer_edge}")
    print(f"  refStep         = {ws.refStep}")
    print(f"  distQuant       = {ws.distQuantification:.2f}")
    print(f"  dist_ratio_init = {ws.dist_ratio_init:.3f}")
    print(f"  angleQuant      = {ws.angleQuantification}")
    print(f"  maxPairs        = {ws.maxNumOfPointPairsPerFeature}")
    print(f"  sym_order       = {ws.sym_order}")
    print(f"  sym_axis        = {ws.sym_axis}")

    # Basic sanity assertions
    assert ws.diameter_m > 0.01,          "Diameter should be > 10mm"
    assert ws.diameter_m < 1.0,           "Diameter should be < 1m (not absurd)"
    assert ws.refStep >= 1,               "refStep must be >= 1"
    assert ws.distQuantification > 0,     "distQuantification must be > 0"
    assert abs(ws.dist_ratio_init - 1.0) < 0.5, "dist_ratio should be near 1"
    assert ws.angleQuantification in [30, 45, 60, 90], \
        f"Unexpected angleQuantification: {ws.angleQuantification}"
    assert ws.maxNumOfPointPairsPerFeature >= 100, "Too few point pairs"
    assert ws.outputNum == 1,             "outputNum should match n_instances=1"
    assert isinstance(ws.prefer_edge, bool)

    print("  PASS: warm_start_values")


def test_n_instances_sets_output_num():
    pcd = load_reference_pcd(MODEL_PATH)
    for n in [1, 2, 3]:
        ws = analyze_mesh(pcd, n_instances=n)
        assert ws.outputNum == n, f"outputNum={ws.outputNum} != n_instances={n}"
    print("  PASS: n_instances_sets_output_num")


def test_pcd_without_normals():
    """analyze_mesh should not crash on a pcd that lacks normals."""
    import open3d as o3d
    import numpy as np
    pts = np.random.randn(500, 3) * 0.05
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # No normals loaded
    ws = analyze_mesh(pcd, n_instances=1)
    assert ws.diameter_m > 0
    print("  PASS: pcd_without_normals")


if __name__ == "__main__":
    print("test_mesh_analysis.py")
    test_warm_start_values()
    test_n_instances_sets_output_num()
    test_pcd_without_normals()
    print("\nAll mesh_analysis tests PASSED")
