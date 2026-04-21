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
    refStep:                       int   = 5
    distQuantification:            float = 5.0     # = refStep × dist_ratio_init
    dist_ratio_init:               float = 1.0     # distQuantification / refStep
    angleQuantification:           int   = 60
    maxNumOfPointPairsPerFeature:  int   = 5000
    outputNum:                     int   = 1       # overridden to N_instances

    # Voxel verification bounds (mm) — geometry-derived, 0.5%/2% of diameter
    minVoxelLength_mm: float = 1.0
    maxVoxelLength_mm: float = 15.0

    # Geometry diagnostics (logged but not directly used as params)
    diameter_m:      float = 0.0
    surface_area_m2: float = 0.0
    flatness_ratio:  float = 1.0
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
    RELATIVE_STEP = 0.02   # 2% of diameter per step → refStep≈5 for D=10cm
    ws.refStep             = max(1, int(D / RELATIVE_STEP))
    # distQuantification is a FACTOR: DistanceInterval = distQuantification × SamplingInterval.
    # MechVision default = 1.0 (optimal bin width ≈ one sampling interval).
    # Plan1 coupling "distQ ≈ refStep" refers to OpenCV's unitless relativeDistanceStep;
    # in MechVision's parameterisation the equivalent is distQ ≈ 1.0 for any refStep.
    ws.distQuantification  = 1.0
    ws.dist_ratio_init     = 1.0 / ws.refStep if ws.refStep > 0 else 0.2
    ws.angleQuantification = 60
    ws.maxNumOfPointPairsPerFeature = 5000 if D < 0.1 else 10000

    # ------------------------------------------------------------------ #
    #  Regime hint: surface vs edge                                        #
    # ------------------------------------------------------------------ #
    pts     = np.asarray(ref_pcd.points)
    normals = np.asarray(ref_pcd.normals) if ref_pcd.has_normals() else None

    bb = np.sort(ref_pcd.get_axis_aligned_bounding_box().get_extent())
    ws.flatness_ratio = float(bb[2] / bb[0]) if bb[0] > 1e-9 else 1.0

    if normals is not None and len(normals) > 0:
        # Dominant normal direction via PCA on normals
        cov_n = np.cov(normals.T)
        eigvals = np.linalg.eigvalsh(cov_n)
        dominant_n = normals[np.argmax(np.abs(normals @ normals[0]))]
        dominant_n = dominant_n / (np.linalg.norm(dominant_n) + 1e-9)
        ws.normal_concentration = float(
            np.mean(np.abs(normals @ dominant_n) > 0.85)
        )
    else:
        ws.normal_concentration = 0.0

    ws.prefer_edge = (ws.normal_concentration > 0.50) or (ws.flatness_ratio > 5.0)

    # ------------------------------------------------------------------ #
    #  Symmetry hint via eigenvalue analysis of point cloud               #
    # ------------------------------------------------------------------ #
    if len(pts) >= 10:
        cov_pts = np.cov(pts.T)
        eigvals_pts = np.sort(np.linalg.eigvalsh(cov_pts))   # ascending
        # Rotationally symmetric around one axis: two eigenvalues are nearly equal
        if eigvals_pts[0] > 1e-12 and eigvals_pts[2] > 1e-12:
            ratio_lo_mid = eigvals_pts[0] / eigvals_pts[1]
            ratio_mid_hi = eigvals_pts[1] / eigvals_pts[2]
            # Axial symmetry: lo ≈ mid (two small equal eigen → symmetric around long axis)
            if ratio_lo_mid > 0.80:
                ws.sym_order = 2        # at least 2-fold; can be higher
                ws.sym_axis  = _dominant_axis(pts)
            # Flat symmetry: mid ≈ hi (two large equal eigen → flat object)
            elif ratio_mid_hi > 0.85:
                ws.sym_order = 2
                ws.sym_axis  = _minor_axis(eigvals_pts, cov_pts)

    return ws


def _dominant_axis(pts: np.ndarray) -> str:
    """Return 'x', 'y', or 'z' for the axis with greatest spread."""
    spread = pts.max(axis=0) - pts.min(axis=0)
    return ['x', 'y', 'z'][int(np.argmax(spread))]


def _minor_axis(eigvals: np.ndarray, cov: np.ndarray) -> str:
    """Return the axis label for the smallest eigenvector of cov."""
    _, eigvecs = np.linalg.eigh(cov)
    minor_vec = eigvecs[:, 0]  # smallest eigenvalue column
    return ['x', 'y', 'z'][int(np.argmax(np.abs(minor_vec)))]


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
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model_path = os.path.join(
        _ROOT, "MM_Optimizer", "CAD_Match", "resource", "3d_matching",
        "25333MB000_surface", "25333MB000_surface.ply"
    )
    pcd = load_reference_pcd(model_path)
    ws  = analyze_mesh(pcd)

    print("\n--- WarmStart for 25333MB000 ---")
    print(f"  diameter         = {ws.diameter_m*1e3:.1f} mm")
    print(f"  surface_area     = {ws.surface_area_m2*1e6:.0f} mm²")
    print(f"  flatness_ratio   = {ws.flatness_ratio:.2f}")
    print(f"  normal_conc      = {ws.normal_concentration:.2f}")
    print(f"  prefer_edge      = {ws.prefer_edge}")
    print(f"  refStep (warm)   = {ws.refStep}")
    print(f"  distQuant (warm) = {ws.distQuantification:.2f}")
    print(f"  dist_ratio_init  = {ws.dist_ratio_init:.3f}")
    print(f"  angleQuant       = {ws.angleQuantification}")
    print(f"  maxPairs (warm)  = {ws.maxNumOfPointPairsPerFeature}")
    print(f"  sym_order        = {ws.sym_order}")
    print(f"  sym_axis         = {ws.sym_axis}")
