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
import MM_Optimizer.search_config as SC

from dataclasses import dataclass
from typing import Optional, Tuple


# Symmetry class constants
SYM_ASYMMETRIC   = "ASYMMETRIC"
SYM_C2           = "C2"
SYM_C3           = "C3"
SYM_C4           = "C4"
SYM_C6           = "C6"
SYM_SO2          = "SO2"   # continuous 1-axis (cylinder, disc, plain shaft)
SYM_SO3          = "SO3"   # continuous all-axes (sphere)


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

    # Symmetry — basic hint (preserved for backward compat)
    sym_order: Optional[int]   = None    # 2, 3, 4, 6 if detected, else None
    sym_axis:  Optional[str]   = None    # 'x', 'y', 'z' if detected

    # Symmetry — full classification (new)
    symmetry_class: str = SYM_ASYMMETRIC   # one of the SYM_* constants above

    # Extended geometry features (for experience bank + LLM prompt)
    # All distances in mm (unit-normalised at load time)
    bbox_x_mm:     float = 0.0
    bbox_y_mm:     float = 0.0
    bbox_z_mm:     float = 0.0
    diameter_mm:   float = 0.0
    aspect_ratio:  float = 1.0    # max_dim / min_dim
    surface_area_mm2: float = 0.0
    volume_mm3:    float = 0.0
    convexity:     float = 1.0    # vol / convex_hull_vol; 1.0 = fully convex
    curvature_mean: float = 0.0
    curvature_std:  float = 0.0
    n_flat_clusters: int  = 0     # number of dominant normal-direction clusters
    has_holes:     bool  = False  # genus > 0 heuristic (open mesh boundary)


def analyze_mesh(ref_pcd: o3d.geometry.PointCloud,
                 n_instances: int = 1) -> WarmStart:
    """Derive warm-start parameters from the reference model point cloud.

    Parameters
    ----------
    ref_pcd     : Reference model point cloud (pre-loaded, with normals).
    n_instances : Expected number of instances per scene. Sets outputNum warm-start.
    """
    ws = WarmStart()
    ws.outputNum = n_instances

    # ------------------------------------------------------------------ #
    #  Basic geometry                                                      #
    # ------------------------------------------------------------------ #
    D  = compute_model_diameter(ref_pcd)
    SA = estimate_surface_area(ref_pcd, D)

    ws.diameter_m      = D
    ws.surface_area_m2 = SA
    ws.maxVoxelLength_mm = max(1.0, round(D * 0.02 * 1000, 1))
    ws.minVoxelLength_mm = max(0.5, round(D * 0.005 * 1000, 1))

    # ------------------------------------------------------------------ #
    #  PPF warm-start                                                      #
    # ------------------------------------------------------------------ #
    ws.distQuantification  = 1.0
    ws.angleQuantification = 60
    ws.maxNumOfPointPairsPerFeature = 5000 if D < 0.1 else 10000

    # ------------------------------------------------------------------ #
    #  Regime hint: surface vs edge                                        #
    # ------------------------------------------------------------------ #
    normals = np.asarray(ref_pcd.normals) if ref_pcd.has_normals() else None

    obb     = ref_pcd.get_minimal_oriented_bounding_box()
    obb_ext = np.sort(obb.extent)
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
    #  Extended geometry features (mm, unit-normalised)                   #
    # ------------------------------------------------------------------ #
    _populate_extended_features(ws, ref_pcd, obb, normals)

    # ------------------------------------------------------------------ #
    #  Full symmetry classification (deterministic, pre-optimization)     #
    # ------------------------------------------------------------------ #
    sym_class, sym_order, sym_axis = _classify_symmetry(ref_pcd)
    ws.symmetry_class = sym_class
    ws.sym_order      = sym_order
    ws.sym_axis       = sym_axis

    return ws


# ─────────────────────────────────────────────────────────────────────────────
# Extended feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _populate_extended_features(
    ws: WarmStart,
    pcd: o3d.geometry.PointCloud,
    obb: o3d.geometry.OrientedBoundingBox,
    normals: Optional[np.ndarray],
) -> None:
    """Fill extended geometry fields on ws. All distances converted to mm."""
    m2mm = 1000.0

    # Bounding box dims sorted descending (largest first)
    dims = np.sort(obb.extent)[::-1]  # [max, mid, min]
    ws.bbox_x_mm    = float(dims[0] * m2mm)
    ws.bbox_y_mm    = float(dims[1] * m2mm)
    ws.bbox_z_mm    = float(dims[2] * m2mm)
    ws.diameter_mm  = float(ws.diameter_m * m2mm)
    ws.aspect_ratio = float(dims[0] / dims[2]) if dims[2] > 1e-9 else 1.0
    ws.surface_area_mm2 = float(ws.surface_area_m2 * m2mm * m2mm)

    # Volume estimate via convex hull
    try:
        hull_mesh, _ = pcd.compute_convex_hull()
        hull_vol = hull_mesh.get_volume()  # m³
    except Exception:
        hull_vol = float(dims[0] * dims[1] * dims[2])  # bounding box fallback

    # Dense voxel-grid volume estimate (better for non-convex parts)
    vox_size = max(ws.diameter_m * 0.02, 1e-4)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=vox_size)
    n_voxels   = len(voxel_grid.get_voxels())
    vox_vol    = n_voxels * (vox_size ** 3)

    ws.volume_mm3 = float(vox_vol * m2mm ** 3)
    ws.convexity  = float(np.clip(vox_vol / hull_vol, 0.0, 1.0)) if hull_vol > 1e-12 else 1.0

    # Curvature estimate from normal variation in local neighbourhoods
    if normals is not None and len(normals) >= 10:
        pts = np.asarray(pcd.points)
        # Sample 500 points for efficiency
        idx = np.random.default_rng(0).choice(len(pts), min(500, len(pts)), replace=False)
        from scipy.spatial import KDTree
        tree = KDTree(pts)
        curv_vals = []
        for i in idx:
            nn_idx = tree.query(pts[i], k=min(10, len(pts)))[1]
            nn_normals = normals[nn_idx]
            # Curvature ≈ variance of normals in neighbourhood
            curv_vals.append(float(1.0 - np.abs(nn_normals @ normals[i]).mean()))
        ws.curvature_mean = float(np.mean(curv_vals))
        ws.curvature_std  = float(np.std(curv_vals))
    else:
        ws.curvature_mean = 0.0
        ws.curvature_std  = 0.0

    # Flat-cluster count: count dominant normal directions via histogram
    if normals is not None and len(normals) >= 10:
        ws.n_flat_clusters = _count_normal_clusters(normals, threshold=0.85)
    else:
        ws.n_flat_clusters = 0

    # Hole detection heuristic: check for open boundary edges via voxel surface density
    # Simple proxy: if surface_area / (convex hull surface area) > 1.1, likely has holes
    try:
        hull_sa = hull_mesh.get_surface_area() if hull_vol > 1e-12 else 0.0
        ws.has_holes = bool(hull_sa > 1e-12 and ws.surface_area_m2 / hull_sa > 1.15)
    except Exception:
        ws.has_holes = False


def _count_normal_clusters(normals: np.ndarray, threshold: float = 0.85) -> int:
    """Count dominant normal directions via greedy clustering on unit sphere."""
    n_clusters = 0
    rng = np.random.default_rng(0)
    # Work on a random subsample for speed
    sample = normals[rng.choice(len(normals), min(1000, len(normals)), replace=False)]
    used = np.zeros(len(sample), dtype=bool)
    for i in range(len(sample)):
        if used[i]:
            continue
        dot = np.abs(sample @ sample[i])
        mask = dot > threshold
        if mask.sum() > len(sample) * 0.05:  # at least 5% of points → real cluster
            n_clusters += 1
        used[mask] = True
    return n_clusters


# ─────────────────────────────────────────────────────────────────────────────
# Symmetry classification
# ─────────────────────────────────────────────────────────────────────────────

def _classify_symmetry(
    pcd: "o3d.geometry.PointCloud",
    chamfer_thresh_frac: float = SC.SYM_CHAMFER_THRESH_FRAC,
    chamfer_thresh_min_m: float = SC.SYM_CHAMFER_THRESH_MIN_M,
) -> Tuple[str, Optional[int], Optional[str]]:
    """Full deterministic symmetry classification.

    Returns
    -------
    (symmetry_class, sym_order, sym_axis)
      symmetry_class : one of SYM_* constants
      sym_order      : integer N for C_N (2/3/4/6), None otherwise
      sym_axis       : 'x'/'y'/'z' label of nearest OBB axis, None if SO3 or ASYMMETRIC
    """
    pts = np.asarray(pcd.points)
    if len(pts) < 10:
        return SYM_ASYMMETRIC, None, None

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    diameter = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    thresh   = max(chamfer_thresh_frac * diameter, chamfer_thresh_min_m)

    obb      = pcd.get_minimal_oriented_bounding_box()
    obb_axes = obb.R.T  # rows are OBB principal axes; used only for axis labelling

    # ── Stage 1: PCA eigenvalue analysis — SO2/SO3 candidates ────────────
    # Use eigenvectors (not OBB axes) so the rotation axis is the actual symmetry axis,
    # not the nearest OBB principal axis (they may differ by a few degrees).
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending; eigvecs[:,i] = i-th eigenvector

    # PCA axes sorted descending by eigenvalue (most-spread first).
    # Used for the N-fold test: more reliable than minimal OBB axes — for asymmetric
    # parts the OBB can return diagonal axes that accidentally pass the Chamfer test.
    sort_idx = np.argsort(eigvals)[::-1]
    pca_axes = eigvecs[:, sort_idx].T        # rows are principal axes, descending variance

    so3_candidate = (eigvals[0] > 0
                     and eigvals[2] / eigvals[0] < SC.SYM_SO3_EIGEN_RATIO
                     and eigvals[1] / eigvals[0] < SC.SYM_SO3_EIGEN_RATIO)

    # SO2 candidate: exactly two eigenvalues are near-equal; the OUTLIER eigenvector
    # is the continuous symmetry axis (e.g. long axis of a cylinder).
    so2_candidate_vec = None
    so2_candidate_lbl = None
    if not so3_candidate and eigvals[0] > 0:
        for i in range(3):
            others = np.delete(eigvals, i)
            ratio  = max(others) / min(others) if min(others) > 0 else 999.0
            if ratio < SC.SYM_SO2_EIGEN_RATIO:
                # eigvecs[:,i] is the outlier eigenvector (the symmetry axis)
                so2_candidate_vec = eigvecs[:, i]
                so2_candidate_lbl = _nearest_axis_label(so2_candidate_vec, obb_axes)
                break

    # ── Stage 2: geometric confirmation via Chamfer rotation test ────────

    # SO3 candidate: an arbitrary rotation should leave the surface unchanged
    if so3_candidate:
        test_axis = eigvecs[:, 0] + eigvecs[:, 1]
        test_axis = test_axis / (np.linalg.norm(test_axis) + 1e-9)
        if _chamfer_distance(centered, _rotate_points(centered, test_axis, 73.0)) < thresh:
            return SYM_SO3, None, None

    # SO2 candidate: 90° AND 45° rotations must both pass (continuous = any angle passes)
    if so2_candidate_vec is not None:
        rot90 = _rotate_points(centered, so2_candidate_vec, 90.0)
        rot45 = _rotate_points(centered, so2_candidate_vec, 45.0)
        if (_chamfer_distance(centered, rot90) < thresh and
                _chamfer_distance(centered, rot45) < thresh):
            return SYM_SO2, None, so2_candidate_lbl

    # N-fold test: combine PCA + OBB axes (deduplicated).
    # PCA axes handle asymmetric parts where OBB may be diagonal.
    # OBB axes handle degenerate eigenvalue shapes (cube) where PCA is arbitrary.
    # ALL multiples of 360/n must pass — prevents false positives from rectangular
    # cross-section aliasing (e.g. rotating a box 60° about its long axis maps
    # top-face points near the side wall, but 120° fails and exposes the non-symmetry).
    nfold_candidates = SC.SYM_NFOLD_CANDIDATES
    sym_map = {6: SYM_C6, 4: SYM_C4, 3: SYM_C3, 2: SYM_C2}

    # OBB axes go first so they are preferred over PCA axes during deduplication.
    # PCA axes of an asymmetric part are tilted by the shape asymmetry; that tilt
    # can cause 180° rotation to reflect points near an adjacent face (geometric
    # coincidence, not symmetry), producing false positives even with multi-angle
    # confirmation.  OBB axes are aligned with the dominant geometric extents and
    # avoid this aliasing.  PCA axes are kept for shapes where OBB is diagonal.
    test_axes = []
    for ax in list(obb_axes) + list(pca_axes):
        if not any(abs(float(np.dot(ax, ta))) > 0.99 for ta in test_axes):
            test_axes.append(ax)

    for ax in test_axes:
        ax_lbl = _nearest_axis_label(ax, obb_axes)
        for n in nfold_candidates:
            angle = 360.0 / n
            if all(_chamfer_distance(centered, _rotate_points(centered, ax, k * angle)) < thresh
                   for k in range(1, n)):
                return sym_map[n], n, ax_lbl

    return SYM_ASYMMETRIC, None, None


def _nearest_axis_label(vec: np.ndarray, obb_axes: np.ndarray) -> str:
    """Return 'x'/'y'/'z' for the OBB principal axis most aligned with vec."""
    return ['x', 'y', 'z'][int(np.argmax(np.abs(obb_axes @ vec)))]


def _rotate_points(pts: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation of pts about axis by angle_deg degrees."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    # Rodrigues: v_rot = v*cos + (axis×v)*sin + axis*(axis·v)*(1-cos)
    dot = (pts @ axis)[:, None]
    cross = np.cross(pts, axis)
    return pts * c - cross * s + axis * dot * (1.0 - c)


def _chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean nearest-neighbour distance (bidirectional) using KD-tree."""
    from scipy.spatial import KDTree
    d_ab = KDTree(b).query(a)[0].mean()
    d_ba = KDTree(a).query(b)[0].mean()
    return float((d_ab + d_ba) / 2)


# kept for backward compatibility — wraps _classify_symmetry
def _check_rotation_symmetry(pcd: "o3d.geometry.PointCloud",
                              threshold_m: float = 0.005) -> Optional[str]:  # noqa: ARG001
    """Return OBB axis label if any symmetry detected, else None. (Legacy wrapper.)"""
    sym_class, _, sym_axis = _classify_symmetry(pcd)
    if sym_class in (SYM_C2, SYM_C3, SYM_C4, SYM_C6, SYM_SO2, SYM_SO3):
        return sym_axis
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Natural-language geometry summary (for LLM prompt)
# ─────────────────────────────────────────────────────────────────────────────

def generate_geometry_summary(ws: WarmStart) -> str:
    """Deterministic natural-language description of part geometry for LLM prompt.

    All interpretive clauses are rule-derived — not LLM-generated — so they
    are auditable and stable across runs.
    """
    lines = []

    # Shape classification
    if ws.flatness_ratio > 8.0:
        shape = "very flat/plate-like"
    elif ws.flatness_ratio > 4.0:
        shape = "flat/disc-like"
    elif ws.aspect_ratio > 5.0:
        shape = "elongated/rod-like"
    elif ws.aspect_ratio < 1.5:
        shape = "compact/block-like"
    else:
        shape = "prismatic"

    lines.append(
        f"Part shape: {shape}, bounding box {ws.bbox_x_mm:.0f}×{ws.bbox_y_mm:.0f}×{ws.bbox_z_mm:.0f} mm "
        f"(diameter {ws.diameter_mm:.0f} mm, aspect ratio {ws.aspect_ratio:.1f})."
    )

    # Surface character
    if ws.n_flat_clusters >= 3:
        surf_desc = f"Surface normals form {ws.n_flat_clusters} dominant planes (flat-faced part)"
    elif ws.normal_concentration > 0.7:
        surf_desc = "Surface normals are highly concentrated (mostly one dominant face)"
    elif ws.curvature_mean > 0.3:
        surf_desc = "High surface curvature (curved or complex geometry)"
    else:
        surf_desc = "Mixed surface normals (moderate curvature)"
    lines.append(f"Surface: {surf_desc}.")

    # Regime hint
    if ws.prefer_edge:
        lines.append(
            "Regime hint: edge mode likely effective "
            f"(flatness={ws.flatness_ratio:.1f}, normal_conc={ws.normal_concentration:.2f})."
        )
    else:
        lines.append(
            "Regime hint: surface mode recommended "
            f"(flatness={ws.flatness_ratio:.1f}, normal_conc={ws.normal_concentration:.2f})."
        )

    # Symmetry
    sym_desc = {
        SYM_ASYMMETRIC: "No rotational symmetry — all orientations are distinct.",
        SYM_C2:  f"2-fold (180°) rotational symmetry about the {ws.sym_axis}-axis.",
        SYM_C3:  f"3-fold (120°) rotational symmetry about the {ws.sym_axis}-axis.",
        SYM_C4:  f"4-fold (90°) rotational symmetry about the {ws.sym_axis}-axis.",
        SYM_C6:  f"6-fold (60°) rotational symmetry about the {ws.sym_axis}-axis.",
        SYM_SO2: f"Continuous rotational symmetry about the {ws.sym_axis}-axis (cylinder/disc). "
                 "Angular error about this axis is physically meaningless.",
        SYM_SO3: "Spherical symmetry (all rotations equivalent). "
                 "Only position matters for pose correctness.",
    }.get(ws.symmetry_class, f"Symmetry class: {ws.symmetry_class}.")
    lines.append(f"Symmetry: {sym_desc}")

    # PPF distQ hint based on surface character
    if ws.curvature_mean < 0.1 and ws.n_flat_clusters >= 2:
        lines.append(
            "Feature density: smooth/flat surfaces → few distinctive PPF features → "
            "favor finer distQuantification (lower value, e.g. 0.5–1.0)."
        )
    elif ws.curvature_mean > 0.3:
        lines.append(
            "Feature density: high curvature → rich PPF features → "
            "coarser distQuantification (1.0–2.0) sufficient."
        )

    # Holes
    if ws.has_holes:
        lines.append("Note: mesh has open boundaries / holes — visible surface matching may be unreliable.")

    return " ".join(lines)


def get_feature_vector(ws: WarmStart) -> list:
    """Normalized numeric feature vector for LanceDB similarity search (~12-d)."""
    sym_onehot = {
        SYM_ASYMMETRIC: [1,0,0,0,0,0,0],
        SYM_C2:         [0,1,0,0,0,0,0],
        SYM_C3:         [0,0,1,0,0,0,0],
        SYM_C4:         [0,0,0,1,0,0,0],
        SYM_C6:         [0,0,0,0,1,0,0],
        SYM_SO2:        [0,0,0,0,0,1,0],
        SYM_SO3:        [0,0,0,0,0,0,1],
    }.get(ws.symmetry_class, [0,0,0,0,0,0,0])

    return [
        float(ws.diameter_mm / 1000.0),           # normalised to ~[0,1] for ≤1000mm parts
        float(ws.aspect_ratio / 10.0),
        float(ws.flatness_ratio / 10.0),
        float(ws.normal_concentration),
        float(1.0 if ws.prefer_edge else 0.0),
        float(ws.convexity),
        float(ws.curvature_mean),
        float(ws.curvature_std),
        float(ws.n_flat_clusters / 10.0),
        float(1.0 if ws.has_holes else 0.0),
        float(ws.surface_area_mm2 / 1e6),         # normalised to ~[0,1] for ≤1m² parts
        float(ws.volume_mm3 / 1e6),
    ] + [float(x) for x in sym_onehot]


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    print(f"  symmetry_class   = {ws.symmetry_class}")
    print(f"\n  bbox             = {ws.bbox_x_mm:.0f}×{ws.bbox_y_mm:.0f}×{ws.bbox_z_mm:.0f} mm")
    print(f"  aspect_ratio     = {ws.aspect_ratio:.2f}")
    print(f"  volume           = {ws.volume_mm3:.0f} mm³")
    print(f"  convexity        = {ws.convexity:.3f}")
    print(f"  curvature        = mean={ws.curvature_mean:.3f}  std={ws.curvature_std:.3f}")
    print(f"  n_flat_clusters  = {ws.n_flat_clusters}")
    print(f"  has_holes        = {ws.has_holes}")
    print(f"\n--- Geometry Summary ---")
    print(generate_geometry_summary(ws))
    print(f"\n--- Feature Vector ({len(get_feature_vector(ws))}d) ---")
    print(get_feature_vector(ws))
