"""
mesh_analysis.py  —  Phase 0: Geometry-derived warm start
----------------------------------------------------------
Derives heuristic starting parameters from the reference point cloud.
Wraps registration/ppf_helpers.py and adds regime + symmetry detection.

Returns a WarmStart dataclass consumed by optimizer.py Phase 0.
No MechVision calls — pure geometry.
"""

import sys
import os
import numpy as np
import open3d as o3d

# Allow running from MM_Optimizer/ or project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from registration.ppf_helpers import compute_model_diameter, estimate_surface_area

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WarmStart:
    """All geometry-derived starting values for the optimizer."""

    # Coarse warm-start values
    distQuantification:            float = 1.0     # MechVision distQ factor (unitless); 1.0 = optimal
    angleQuantification:           int   = 60
    maxNumOfPointPairsPerFeature:  int   = 5000
    outputNum:                     int   = 1       # overridden to N_instances

    # Voxel verification bounds (mm) — geometry-derived, 0.5%/2% of diameter
    minVoxelLength_mm: float = 1.0
    maxVoxelLength_mm: float = 15.0

    # Geometry diagnostics (logged but not directly used as params)
    diameter_m:       float = 0.0
    longest_extent_m: float = 0.0   # largest OBB axis (m); used for adaptive pos threshold
    surface_area_m2:  float = 0.0
    flatness_ratio:   float = 1.0
    normal_concentration: float = 0.0

    # Regime preference
    prefer_edge: bool = False     # True → test edge combos first

    # Symmetry hint
    sym_order: Optional[int]   = None    # 2, 3, 4 if detected, else None
    sym_axis:  Optional[str]   = None    # 'x', 'y', 'z' if detected


def analyze_mesh(ref_pcd: o3d.geometry.PointCloud,
                 n_instances: int = 1) -> WarmStart:
    """Derive warm-start parameters from the reference model point cloud.

    Parameters
    ----------
    ref_pcd     : Reference model point cloud (pre-loaded, with normals).
    n_instances : Expected number of instances per scene. Sets outputNum warm-start.
    """
    ws = WarmStart()
    ws.outputNum = n_instances   # hypotheses per instance; swept in Phase 2b

    # ------------------------------------------------------------------ #
    #  Basic geometry                                                      #
    # ------------------------------------------------------------------ #
    D  = compute_model_diameter(ref_pcd)
    SA = estimate_surface_area(ref_pcd, D)

    ws.diameter_m      = D
    ws.surface_area_m2 = SA
    ws.maxVoxelLength_mm = max(1.0, round(D * 0.02 * 1000, 1))    # 2% of diameter in mm
    ws.minVoxelLength_mm = max(0.5, round(D * 0.005 * 1000, 1))   # 0.5% of diameter in mm (1:4 ratio)

    # ------------------------------------------------------------------ #
    #  PPF warm-start                                                      #
    # ------------------------------------------------------------------ #
    # refStep is swept from 20→1 by the optimizer (MechMind integer constraint);
    # no geometry-derived warm-start needed.
    # distQuantification is a FACTOR: DistanceInterval = distQ × SamplingInterval.
    # MechVision default = 1.0 (optimal bin width ≈ one sampling interval).
    ws.distQuantification  = 1.0
    ws.angleQuantification = 60
    ws.maxNumOfPointPairsPerFeature = 5000 if D < 0.1 else 10000

    # ------------------------------------------------------------------ #
    #  Regime hint: surface vs edge                                        #
    # ------------------------------------------------------------------ #
    normals = np.asarray(ref_pcd.normals) if ref_pcd.has_normals() else None

    obb_ext = np.sort(ref_pcd.get_minimal_oriented_bounding_box().extent)
    ws.longest_extent_m = float(obb_ext[2])
    ws.flatness_ratio   = float(obb_ext[2] / obb_ext[0]) if obb_ext[0] > 1e-9 else 1.0

    if normals is not None and len(normals) > 0:
        dominant_n = normals[np.argmax(np.abs(normals @ normals[0]))]
        dominant_n = dominant_n / (np.linalg.norm(dominant_n) + 1e-9)
        ws.normal_concentration = float(
            np.mean(np.abs(normals @ dominant_n) > 0.85)
        )
    else:
        ws.normal_concentration = 0.0

    ws.prefer_edge = (ws.normal_concentration > 0.50) or (ws.flatness_ratio > 5.0)

    # ------------------------------------------------------------------ #
    #  Symmetry hint — 180° rotation overlap (2-fold)                    #
    # ------------------------------------------------------------------ #
    sym_axis_label = _check_rotation_symmetry(ref_pcd)
    if sym_axis_label is not None:
        ws.sym_order = 2
        ws.sym_axis  = sym_axis_label

    return ws


def _chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean nearest-neighbour distance (bidirectional) using KD-tree."""
    from scipy.spatial import KDTree
    d_ab = KDTree(b).query(a)[0].mean()
    d_ba = KDTree(a).query(b)[0].mean()
    return float((d_ab + d_ba) / 2)


def _check_rotation_symmetry(pcd: "o3d.geometry.PointCloud",
                              threshold_m: float = 0.005) -> Optional[str]:
    """Return OBB axis label ('x','y','z') if 180° 2-fold symmetry detected, else None.

    Uses Rodrigues 180° rotation on each OBB principal axis; low Chamfer
    distance to the original indicates the part maps onto itself under that
    rotation. threshold_m=5mm is conservative — parts with small asymmetric
    features (tabs, holes on one side) will exceed it.
    """
    pts = np.asarray(pcd.points)
    if len(pts) < 10:
        return None
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    obb = pcd.get_minimal_oriented_bounding_box()
    axes = obb.R.T  # rows are principal axes

    for i, axis in enumerate(axes):
        # 180° Rodrigues: R*v = 2*(v·axis)*axis - v
        rotated = 2 * (centered @ axis)[:, None] * axis - centered
        d = _chamfer_distance(centered, rotated)
        if d < threshold_m:
            return ['x', 'y', 'z'][i]
    return None


def load_reference_pcd(model_path: str) -> o3d.geometry.PointCloud:
    """Load a PLY point cloud and estimate normals if missing."""
    pcd = o3d.io.read_point_cloud(model_path)
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)
    return pcd


if __name__ == "__main__":
    import argparse
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Print WarmStart for a reference PLY model.")
    parser.add_argument("ply_path", nargs="?",
                        default=os.path.join(
                            _ROOT, "MM_Optimizer", "CAD_Match", "resource", "3d_matching",
                            "25333MB000_surface", "25333MB000_surface.ply"),
                        help="Path to the surface PLY model (default: 25333MB000_surface)")
    args = parser.parse_args()

    pcd = load_reference_pcd(args.ply_path)
    ws  = analyze_mesh(pcd)

    part = os.path.splitext(os.path.basename(args.ply_path))[0]
    print(f"\n--- WarmStart for {part} ---")
    print(f"  diameter         = {ws.diameter_m*1e3:.1f} mm")
    print(f"  longest_extent   = {ws.longest_extent_m*1e3:.1f} mm")
    print(f"  surface_area     = {ws.surface_area_m2*1e6:.0f} mm²")
    print(f"  flatness_ratio   = {ws.flatness_ratio:.2f}")
    print(f"  normal_conc      = {ws.normal_concentration:.2f}")
    print(f"  prefer_edge      = {ws.prefer_edge}")
    print(f"  distQuant (warm) = {ws.distQuantification:.2f}")
    print(f"  angleQuant       = {ws.angleQuantification}")
    print(f"  maxPairs (warm)  = {ws.maxNumOfPointPairsPerFeature}")
    print(f"  sym_order        = {ws.sym_order}")
    print(f"  sym_axis         = {ws.sym_axis}")
