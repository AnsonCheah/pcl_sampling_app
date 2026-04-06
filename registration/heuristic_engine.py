"""
HeuristicParameterEngine
========================
Self-tuning registration parameter resolver for CAD-to-scene point cloud matching.

Architecture
------------
  MeshAnalyser      → PartProfile      (from CAD mesh, computed once per part)
  SceneAnalyser     → SceneProfile     (from segmented scene cluster, per-frame)
  SensorProfile                         (fixed per camera model)
  BinProfile                            (fixed per bin)
  ParameterResolver → RegistrationParams + RegistrationConfidence

Key design decisions
--------------------
  - All distance thresholds scale from L_max (longest CAD bounding box dimension).
    This is the single most robust normalisation anchor.
  - Visibility is estimated by comparing observed point count against the
    per-viewpoint expected count from reference PC metadata (not total reference
    count, which would be meaninglessly large).
  - Reflection dropout vs occlusion is disambiguated by three independent signals:
    spatial gap coherence, gap boundary normal orientation, and gap shape regularity.
  - All parameter formulas are closed-form and interpretable — no learned weights.
  - Symmetry detection uses inertia tensor eigenvalue ratios and rotational
    self-overlap sampling.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, KDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import label as connected_label


# ============================================================
# Enumerations
# ============================================================

class ICPVariant(Enum):
    POINT_TO_POINT  = auto()
    POINT_TO_PLANE  = auto()
    GENERALIZED     = auto()   # GICP


class MissingRegionType(Enum):
    OCCLUSION          = auto()   # half-space cut, geometry-driven
    REFLECTION_DROPOUT = auto()   # large coherent patch, incidence-driven
    MIXED              = auto()   # both present
    UNKNOWN            = auto()   # insufficient evidence


class SymmetryClass(Enum):
    ASYMMETRIC    = auto()   # fully unique orientation
    ONE_AXIS      = auto()   # one rotational symmetry axis (cylinder-like)
    TWO_AXIS      = auto()   # two symmetry axes
    FULL          = auto()   # sphere-like, all orientations equivalent


# ============================================================
# Data structures
# ============================================================

@dataclass
class PartProfile:
    """Derived from CAD mesh. Computed once per part, cached."""

    # Bounding box
    L_max: float          # longest dimension (m)
    L_mid: float          # middle dimension (m)
    L_min_dim: float      # shortest dimension (m)  — named L_min_dim to avoid shadowing

    # Shape descriptors
    sa_to_volume_ratio: float    # surface area / volume  (1/m)
    compactness: float           # 36π V² / A³  ∈ (0,1], 1 = sphere
    is_thin: bool                # L_min_dim / L_max < 0.15 → sheet-like

    # Feature richness
    normal_variance: float       # mean angular deviation of adjacent-face normals
    curvature_mean: float        # mean absolute mean curvature (1/m)
    curvature_std: float
    min_feature_size: float      # smallest discriminative geometric feature (m)

    # Symmetry
    symmetry_class: SymmetryClass
    symmetry_axes: int           # number of rotational symmetry axes
    inertia_eigenvalue_ratios: Tuple[float, float]   # λ2/λ1, λ3/λ1

    # Partial-symmetry characterisation
    # A bearing shaft is ONE_AXIS with fold=∞ on the shaft but asymmetric on the head.
    # These two fields capture that nuance independently of symmetry_class.
    # Defaults: fold=1 (no symmetry assumed), disc_frac=1.0 (fully discriminative)
    symmetry_fold: int = 1           # angular periodicity of the SYMMETRIC region
                                     # (e.g. 6 for hex-headed shaft = 60° repeating)
                                     # 1 = fully asymmetric,  0 = infinite (pure cylinder)
    discriminative_fraction: float = 1.0  # fraction of surface NOT explained by dominant symmetry
                                          # ∈ (0, 1].  Small → RANSAC needs many more iterations.
                                          # e.g. keyway on shaft ≈ 0.05; hex head on shaft ≈ 0.30

    # Expected point counts per viewpoint (from reference PC generation metadata)
    # Maps viewpoint_index → expected visible point count from that view
    viewpoint_point_counts: Dict[int, int] = field(default_factory=dict)
    # Viewpoint directions corresponding to above (unit vectors, camera-frame)
    viewpoint_directions: List[np.ndarray] = field(default_factory=list)

    # Volume (m³)
    volume: float = 0.0
    surface_area: float = 0.0


@dataclass
class SceneProfile:
    """Derived from the segmented scene cluster. Computed per frame."""

    # Basic counts and geometry
    n_points: int
    cluster_L_max: float          # observed bounding box longest dim
    cluster_L_mid: float
    cluster_L_min: float
    cluster_volume: float         # oriented bounding box volume
    observed_centroid: np.ndarray  # (3,) world frame

    # Visibility
    estimated_visibility: float   # ∈ [0, 1]
    viewing_direction: np.ndarray  # (3,) unit vector, sensor → centroid
    estimated_range: float         # metres

    # Point density analysis
    point_density: float           # points / m³ in cluster
    density_ratio: float           # observed / expected from sensor model

    # Missing region analysis
    missing_region_type: MissingRegionType
    missing_fraction: float        # fraction of expected surface area missing
    gap_coherence_score: float     # 0 = diffuse dropout, 1 = single coherent gap
    gap_incidence_correlation: float  # correlation of gap with high-θ regions

    # Spatial context
    dist_to_nearest_wall: float    # metres
    z_above_bin_floor: float       # metres
    near_wall: bool                # dist_to_nearest_wall < L_max
    near_other_cluster: bool       # another cluster within 1.5 × L_max

    # Segmentation quality flags
    possible_over_segment: bool    # cluster much smaller than part
    possible_under_segment: bool   # cluster much larger than part


@dataclass
class SensorProfile:
    """Fixed per camera model. Set once at deployment."""

    model_name: str
    fx: float; fy: float          # focal lengths (pixels)
    cx: float; cy: float          # principal point (pixels)
    image_width: int
    image_height: int
    nominal_range: float           # designed working distance (m)
    range_min: float               # minimum reliable range (m)
    range_max: float               # maximum reliable range (m)
    depth_noise_at_nominal: float  # 1-sigma depth noise at nominal range (m)
    fringe_period: float           # structured light fringe period at nominal range (m)
    # depth noise scales linearly with range for most SL cameras:
    depth_noise_slope: float = 0.001    # σ(r) = depth_noise_at_nominal + slope*(r - nominal)
    lateral_noise_at_nominal: float = 0.0005  # lateral 1-sigma (m)


@dataclass
class BinProfile:
    """Fixed per bin installation."""

    inner_length: float    # m
    inner_width: float     # m
    inner_depth: float     # m
    wall_thickness: float  # m
    wall_material: str     # "metal", "plastic", "cardboard"
    wall_reflective: bool  # drives outlier removal near walls


@dataclass
class RegistrationParams:
    """
    Full resolved parameter set for one registration attempt.
    Both coarse and fine stages.
    """
    # --- Preprocessing ---
    scene_voxel_size: float          # voxel size for scene downsampling
    reference_voxel_size: float      # voxel size for reference downsampling
    normal_radius: float             # radius for normal estimation
    normal_max_nn: int               # max neighbours for normal estimation
    outlier_nb_neighbors: int        # statistical outlier removal neighbours
    outlier_std_ratio: float         # std multiplier for outlier removal

    # --- Coarse stage (FPFH + RANSAC) ---
    fpfh_radius: float
    fpfh_max_nn: int
    ransac_distance_thresh: float
    ransac_n_points: int             # minimum correspondence set
    ransac_max_iter: int
    ransac_confidence: float         # early termination confidence
    ransac_min_inlier_fraction: float
    mutual_filter: bool

    # --- Fine stage (ICP) ---
    icp_variant: ICPVariant
    icp_distance_thresh: float
    icp_max_iter: int
    icp_convergence_rmse: float      # convergence delta threshold
    icp_outlier_trim_fraction: float # for trimmed ICP; 0 = disabled
    icp_fitness_threshold: float     # minimum inlier fraction to accept result

    # --- Symmetry handling ---
    n_ransac_restarts: int           # extra RANSAC restarts for symmetric parts
    pose_cluster_angular_thresh: float   # degrees — cluster near-duplicate poses

    # --- Diagnostics ---
    notes: List[str] = field(default_factory=list)


@dataclass
class RegistrationConfidence:
    """
    Confidence score and component breakdown.
    Passed to downstream pick / inspection logic.
    """
    overall: float           # ∈ [0, 1]

    icp_inlier_fraction: float
    icp_rmse_normalised: float   # icp_rmse / expected_noise_floor
    visibility_score: float
    symmetry_penalty: float      # 1.0 = no penalty, <1 = symmetric part
    segmentation_quality: float  # 1.0 = clean, <1 = over/under-segment suspected

    reliable: bool               # overall > threshold → safe to act on
    action_recommendation: str   # "proceed", "re-scan", "skip"


# ============================================================
# 1. MeshAnalyser
# ============================================================

class MeshAnalyser:
    """
    Analyses a CAD mesh to populate a PartProfile.

    Parameters
    ----------
    symmetry_sample_n   : number of random rotations to test for symmetry
    symmetry_iou_thresh : fraction of surface overlap to declare a symmetry axis
    min_feature_sample_n: points to sample when estimating min feature size
    """

    def __init__(
        self,
        symmetry_sample_n: int = 200,
        symmetry_iou_thresh: float = 0.90,
        min_feature_sample_n: int = 20_000,
    ) -> None:
        self.symmetry_sample_n = symmetry_sample_n
        self.symmetry_iou_thresh = symmetry_iou_thresh
        self.min_feature_sample_n = min_feature_sample_n

    def analyse(
        self,
        mesh_path: str | Path,
        viewpoint_metadata_path: Optional[str | Path] = None,
    ) -> PartProfile:
        """
        Parameters
        ----------
        mesh_path                : path to CAD mesh (.ply/.obj/.stl)
        viewpoint_metadata_path  : optional JSON sidecar from reference PC
                                   generation containing per-viewpoint point counts
        """
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        # ---- Bounding box ----
        aabb = mesh.get_axis_aligned_bounding_box()
        dims = np.sort(aabb.get_extent())[::-1]   # descending
        L_max, L_mid, L_min_dim = float(dims[0]), float(dims[1]), float(dims[2])

        # ---- Volume and surface area ----
        volume      = max(float(mesh.get_volume()), 1e-9)
        surface_area = float(mesh.get_surface_area())

        # ---- Shape descriptors ----
        sa_to_vol = surface_area / volume
        # Wadell compactness (isoperimetric quotient)
        compactness = (36.0 * np.pi * volume**2) / max(surface_area**3, 1e-30)
        compactness = float(np.clip(compactness, 0.0, 1.0))
        is_thin = (L_min_dim / L_max) < 0.15

        # ---- Normal variance (feature richness) ----
        tri_normals = np.asarray(mesh.triangle_normals)
        if len(tri_normals) > 1:
            # Angular deviation between adjacent triangle normals
            dots = np.clip(tri_normals[:-1] @ tri_normals[1:].T, -1, 1)
            angles = np.arccos(np.diag(dots))
            normal_variance = float(np.std(angles))
        else:
            normal_variance = 0.0

        # ---- Curvature from sampled point cloud ----
        pcd = mesh.sample_points_uniformly(self.min_feature_sample_n)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=L_max * 0.02, max_nn=30)
        )
        curvatures = self._estimate_curvatures(
            np.asarray(pcd.points), np.asarray(pcd.normals), L_max * 0.02
        )
        curvature_mean = float(np.mean(np.abs(curvatures)))
        curvature_std  = float(np.std(curvatures))

        # ---- Minimum feature size ----
        min_feature = self._estimate_min_feature_size(
            np.asarray(pcd.points), curvatures, L_max
        )

        # ---- Symmetry ----
        sym_class, sym_axes, eigen_ratios, sym_fold, disc_frac = self._analyse_symmetry(mesh, pcd)

        # ---- Viewpoint metadata ----
        vp_counts: Dict[int, int] = {}
        vp_dirs: List[np.ndarray] = []
        if viewpoint_metadata_path is not None:
            vp_counts, vp_dirs = self._load_viewpoint_metadata(viewpoint_metadata_path)

        return PartProfile(
            L_max=L_max, L_mid=L_mid, L_min_dim=L_min_dim,
            sa_to_volume_ratio=sa_to_vol,
            compactness=compactness,
            is_thin=is_thin,
            normal_variance=normal_variance,
            curvature_mean=curvature_mean,
            curvature_std=curvature_std,
            min_feature_size=min_feature,
            symmetry_class=sym_class,
            symmetry_axes=sym_axes,
            inertia_eigenvalue_ratios=eigen_ratios,
            symmetry_fold=sym_fold,
            discriminative_fraction=disc_frac,
            viewpoint_point_counts=vp_counts,
            viewpoint_directions=vp_dirs,
            volume=volume,
            surface_area=surface_area,
        )

    # ------------------------------------------------------------------
    def _estimate_curvatures(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        radius: float,
        max_sample: int = 5000,
    ) -> np.ndarray:
        """Fast approximate mean curvature via normal divergence in neighbourhood."""
        idx = np.random.choice(len(points), min(max_sample, len(points)), replace=False)
        pts_s = points[idx]
        nrm_s = normals[idx]
        tree = KDTree(points)
        nbrs_list = tree.query_ball_point(pts_s, r=radius)
        curvatures = np.zeros(len(idx), dtype=np.float32)
        for i, nbrs in enumerate(nbrs_list):
            if len(nbrs) < 4:
                continue
            dn = normals[nbrs] - nrm_s[i]
            curvatures[i] = np.linalg.norm(dn, axis=1).mean() / (radius + 1e-9)
        return curvatures

    # ------------------------------------------------------------------
    def _estimate_min_feature_size(
        self,
        points: np.ndarray,
        curvatures: np.ndarray,
        L_max: float,
    ) -> float:
        """
        Estimate minimum feature size as the characteristic length scale
        at high-curvature regions.  High curvature → small radius of curvature
        → small feature.  Returns the 5th-percentile 1/κ (clamped).
        """
        high_kappa = curvatures[curvatures > np.percentile(curvatures, 80)]
        if len(high_kappa) == 0 or np.max(high_kappa) < 1e-6:
            return L_max * 0.05   # flat part — no small features, use 5% of L_max
        median_high = np.percentile(high_kappa, 5)  # most extreme curvature
        feature_size = float(np.clip(1.0 / max(median_high, 1e-6), L_max * 0.002, L_max * 0.5))
        return feature_size

    # ------------------------------------------------------------------
    def _analyse_symmetry(
        self,
        mesh: o3d.geometry.TriangleMesh,
        pcd: o3d.geometry.PointCloud,
    ) -> Tuple[SymmetryClass, int, Tuple[float, float], int, float]:
        """
        Regional symmetry analysis — handles partially symmetric parts correctly.

        A bearing shaft (symmetric body + asymmetric head) is detected as:
          symmetry_class = ONE_AXIS  (primary axis exists)
          symmetry_fold  = 6         (hex head repeats every 60°)
          discriminative_fraction = 0.15  (only 15% of surface is unique)

        Pipeline
        --------
        1. Inertia tensor → principal axes + eigenvalue ratios
        2. For each principal axis, find the finest fold order that the
           MAJORITY of the surface satisfies (fold sweep: ∞, 12, 8, 6, 4, 3, 2, 1)
        3. For the dominant axis, classify every point as SYMMETRIC or
           ASYMMETRIC using a geometry+normal consistency test at the
           resolved fold angle — the match radius is bounded by min_feature_size
           so shallow features are not absorbed into the symmetric region
        4. discriminative_fraction = fraction of ASYMMETRIC points

        The global symmetry_class is set conservatively: if ANY axis has a
        non-trivial fold, it counts. This prevents misclassifying a hex-headed
        shaft as ASYMMETRIC when the shaft dominates point count.
        """
        pts = np.asarray(pcd.points)
        nrm = np.asarray(pcd.normals) if pcd.has_normals() else None
        centre = pts.mean(axis=0)
        pts_c = pts - centre

        # ---- Inertia tensor ----
        Ixx = np.sum(pts_c[:, 1]**2 + pts_c[:, 2]**2)
        Iyy = np.sum(pts_c[:, 0]**2 + pts_c[:, 2]**2)
        Izz = np.sum(pts_c[:, 0]**2 + pts_c[:, 1]**2)
        Ixy = -np.sum(pts_c[:, 0] * pts_c[:, 1])
        Ixz = -np.sum(pts_c[:, 0] * pts_c[:, 2])
        Iyz = -np.sum(pts_c[:, 1] * pts_c[:, 2])
        I = np.array([[Ixx, Ixy, Ixz],
                      [Ixy, Iyy, Iyz],
                      [Ixz, Iyz, Izz]])
        eigvals, eigvecs = np.linalg.eigh(I)
        eigvals = np.sort(np.clip(eigvals, 1e-9, None))[::-1]
        ratio_21 = float(eigvals[1] / eigvals[0])
        ratio_31 = float(eigvals[2] / eigvals[0])
        principal_axes = eigvecs.T[::-1]   # rows, sorted dominant first

        # ---- Feature-size-aware match radius ----
        # Tighter than density-based: must resolve features at the min_feature scale.
        # Density-based radius (as before) is kept as a fallback upper bound.
        tree = KDTree(pts_c)
        nn_dists = tree.query(pts_c, k=2)[0][:, 1]
        density_radius = float(np.percentile(nn_dists, 95)) * 2.0
        # Bound by min_feature_size so shallow features are not absorbed
        feature_radius = self._get_feature_radius(pts_c, density_radius)
        match_radius = min(density_radius, feature_radius)

        # ---- Per-axis fold detection ----
        # Candidate fold orders to test, from finest (most symmetric) to coarsest.
        # 0 = infinite (pure cylinder / sphere) — tested via continuous angle sweep.
        FOLD_ORDERS = [0, 12, 8, 6, 4, 3, 2]   # 0 = ∞

        best_fold      = 1       # 1 = asymmetric (no symmetry found)
        best_sym_frac  = 0.0
        best_axis_idx  = -1
        axes_with_symmetry = 0

        for ax_idx, ax in enumerate(principal_axes):
            fold, sym_frac = self._find_fold_order(
                pts_c, nrm, ax, tree, match_radius, FOLD_ORDERS
            )
            if fold != 1 and sym_frac > self.symmetry_iou_thresh:
                axes_with_symmetry += 1
                if sym_frac > best_sym_frac:
                    best_fold     = fold
                    best_sym_frac = sym_frac
                    best_axis_idx = ax_idx

        # ---- Per-point discriminativeness on the dominant axis ----
        if best_axis_idx >= 0:
            discriminative_fraction = self._discriminative_fraction(
                pts_c, nrm,
                axis=principal_axes[best_axis_idx],
                fold=best_fold,
                match_radius=match_radius,
            )
        else:
            discriminative_fraction = 1.0   # fully asymmetric

        # ---- Global symmetry class ----
        if axes_with_symmetry == 0:
            sym_class = SymmetryClass.ASYMMETRIC
        elif axes_with_symmetry == 1:
            sym_class = SymmetryClass.ONE_AXIS
        elif axes_with_symmetry == 2:
            sym_class = SymmetryClass.TWO_AXIS
        else:
            sym_class = SymmetryClass.FULL

        return sym_class, axes_with_symmetry, (ratio_21, ratio_31), best_fold, float(discriminative_fraction)

    # ------------------------------------------------------------------
    def _get_feature_radius(
        self,
        pts_c: np.ndarray,
        density_radius: float,
    ) -> float:
        """
        Estimate minimum feature radius from local surface variation.
        Points that are outliers in local Z (height above local tangent plane)
        indicate features; their scale sets the match radius floor.
        Uses a random subsample for speed.
        """
        n_sample = min(2000, len(pts_c))
        idx = np.random.choice(len(pts_c), n_sample, replace=False)
        pts_s = pts_c[idx]
        tree_s = KDTree(pts_s)
        # Local radius = 10× density radius
        nbr_lists = tree_s.query_ball_point(pts_s, r=density_radius * 5)
        local_vars = []
        for i, nbrs in enumerate(nbr_lists):
            if len(nbrs) < 5:
                continue
            local_pts = pts_s[nbrs] - pts_s[i]
            # Variance in the direction of local normal (via PCA)
            cov = local_pts.T @ local_pts / len(nbrs)
            local_vars.append(float(np.linalg.eigvalsh(cov)[0]))  # smallest eigenvalue = normal dir
        if not local_vars:
            return density_radius
        # Typical feature depth ~ sqrt of local height variance
        feature_depth = float(np.sqrt(np.percentile(local_vars, 75))) * 3.0
        return max(feature_depth, density_radius * 0.3)

    # ------------------------------------------------------------------
    def _find_fold_order(
        self,
        pts_c:       np.ndarray,
        normals:     Optional[np.ndarray],
        axis:        np.ndarray,
        tree:        KDTree,
        match_radius: float,
        fold_orders:  List[int],
    ) -> Tuple[int, float]:
        """
        Find the finest fold order for which the majority of points satisfy
        rotational self-consistency around `axis`.

        Tests fold orders from finest to coarsest (0 = infinite / continuous).
        Returns (fold, sym_fraction) where sym_fraction is the fraction of
        points that map back onto themselves at the fold angle.

        For fold=0 (infinite), tests 5 random angles — if ALL pass, it is
        a continuous symmetry axis (pure cylinder / sphere region).
        """
        for fold in fold_orders:
            if fold == 0:
                # Test continuous symmetry: 5 random angles must all pass
                test_angles = np.random.uniform(5, 175, 5)
                sym_fracs = []
                for angle_deg in test_angles:
                    sf = self._sym_fraction_at_angle(
                        pts_c, normals, axis, angle_deg, tree, match_radius
                    )
                    sym_fracs.append(sf)
                sym_frac = float(np.min(sym_fracs))
                if sym_frac > self.symmetry_iou_thresh:
                    return 0, sym_frac
            else:
                angle_deg = 360.0 / fold
                sym_frac = self._sym_fraction_at_angle(
                    pts_c, normals, axis, angle_deg, tree, match_radius
                )
                if sym_frac > self.symmetry_iou_thresh:
                    return fold, sym_frac

        return 1, 0.0   # no fold found → asymmetric

    # ------------------------------------------------------------------
    def _sym_fraction_at_angle(
        self,
        pts_c:       np.ndarray,
        normals:     Optional[np.ndarray],
        axis:        np.ndarray,
        angle_deg:   float,
        tree:        KDTree,
        match_radius: float,
    ) -> float:
        """
        Fraction of points that are geometrically AND normal-consistent
        after rotating by angle_deg around axis.

        Normal consistency check prevents shallow features (keyways, slots)
        from being absorbed into the symmetric region when their rotated
        position happens to land near the background surface:
          - A keyway wall (normal pointing inward) rotated 90° lands near
            the smooth shaft, but the shaft normal points outward — mismatch.
          - An axis point on a pure cylinder rotated any angle stays on the
            cylinder with a matching normal — consistent.
        """
        R = Rotation.from_rotvec(np.radians(angle_deg) * axis).as_matrix()
        pts_rot = pts_c @ R.T
        dists, nn_idx = tree.query(pts_rot, k=1)
        geo_match = dists < match_radius     # (N,) bool

        if normals is not None:
            # Normal of the rotated point = R @ original normal
            nrm_rot = normals @ R.T
            nrm_nn  = normals[nn_idx]
            # Dot product of rotated normal with neighbour normal
            normal_cos = np.einsum("ij,ij->i", nrm_rot, nrm_nn)
            normal_match = normal_cos > 0.70   # ~45° tolerance
            combined = geo_match & normal_match
        else:
            combined = geo_match

        return float(combined.mean())

    # ------------------------------------------------------------------
    def _discriminative_fraction(
        self,
        pts_c:       np.ndarray,
        normals:     Optional[np.ndarray],
        axis:        np.ndarray,
        fold:        int,
        match_radius: float,
    ) -> float:
        """
        Per-point classification: which points are NOT explained by the
        dominant fold symmetry?

        For fold=0 (infinite), tests 8 evenly spaced angles and labels a
        point as symmetric only if it passes ALL of them. This correctly
        identifies keyway points (fail at keyway-opposing angles) and head
        features (fail at non-head angles).

        Returns the fraction of points that are ASYMMETRIC (discriminative).
        These are the points that uniquely constrain the pose.
        """
        if fold == 0:
            test_angles = np.linspace(15, 165, 8)
        else:
            test_angles = [360.0 / fold * k for k in range(1, fold)]
            if not test_angles:
                return 1.0

        # A point is symmetric if it passes ALL test angles
        sym_mask = np.ones(len(pts_c), dtype=bool)
        for angle_deg in test_angles:
            R = Rotation.from_rotvec(np.radians(angle_deg) * axis).as_matrix()
            pts_rot = pts_c @ R.T
            tree_full = KDTree(pts_c)
            dists, nn_idx = tree_full.query(pts_rot, k=1)
            geo_ok = dists < match_radius

            if normals is not None:
                nrm_rot = normals @ R.T
                nrm_nn  = normals[nn_idx]
                normal_cos = np.einsum("ij,ij->i", nrm_rot, nrm_nn)
                normal_ok = normal_cos > 0.70
                passes = geo_ok & normal_ok
            else:
                passes = geo_ok

            sym_mask &= passes

        discriminative_fraction = float((~sym_mask).mean())
        # Clip: even a fully symmetric part has small numerical noise
        return float(np.clip(discriminative_fraction, 0.01, 1.0))

    # ------------------------------------------------------------------
    @staticmethod
    def _load_viewpoint_metadata(
        path: str | Path,
    ) -> Tuple[Dict[int, int], List[np.ndarray]]:
        """
        Load per-viewpoint point count metadata from JSON sidecar.

        Expected JSON format:
        {
          "viewpoints": [
            {"index": 0, "direction": [dx, dy, dz], "point_count": 12345},
            ...
          ]
        }
        """
        with open(path) as f:
            data = json.load(f)
        counts: Dict[int, int] = {}
        dirs: List[np.ndarray] = []
        for vp in data.get("viewpoints", []):
            i = vp["index"]
            counts[i] = vp["point_count"]
            dirs.append(np.array(vp["direction"], dtype=np.float32))
        return counts, dirs


# ============================================================
# 2. SceneAnalyser
# ============================================================

class SceneAnalyser:
    """
    Analyses a segmented scene cluster to populate a SceneProfile.

    Parameters
    ----------
    part_profile    : PartProfile for the expected part (for ratio computations)
    sensor_profile  : SensorProfile
    bin_profile     : BinProfile
    """

    def __init__(
        self,
        part_profile: PartProfile,
        sensor_profile: SensorProfile,
        bin_profile: BinProfile,
    ) -> None:
        self.part  = part_profile
        self.sensor = sensor_profile
        self.bin   = bin_profile

    def analyse(
        self,
        scene_cluster: o3d.geometry.PointCloud,
        sensor_origin: np.ndarray,
        other_cluster_centroids: Optional[List[np.ndarray]] = None,
    ) -> SceneProfile:
        """
        Parameters
        ----------
        scene_cluster            : segmented point cloud of one object
        sensor_origin            : (3,) sensor position in world frame
        other_cluster_centroids  : centroids of neighbouring clusters (for proximity test)
        """
        pts = np.asarray(scene_cluster.points)
        n_pts = len(pts)

        # ---- Oriented bounding box ----
        obb = scene_cluster.get_oriented_bounding_box()
        obs_dims = np.sort(obb.extent)[::-1]
        cl_max, cl_mid, cl_min = float(obs_dims[0]), float(obs_dims[1]), float(obs_dims[2])
        cl_volume = max(float(np.prod(obb.extent)), 1e-9)
        centroid = pts.mean(axis=0)

        # ---- Viewing geometry ----
        view_vec = centroid - sensor_origin
        est_range = float(np.linalg.norm(view_vec))
        view_dir  = view_vec / (est_range + 1e-9)

        # ---- Visibility estimate ----
        # Use the closest viewpoint from the reference PC metadata,
        # not the total reference point count.
        estimated_visibility, expected_pts = self._estimate_visibility(
            n_pts, view_dir, est_range
        )

        # ---- Point density ----
        point_density = n_pts / cl_volume
        expected_density = self._expected_density(est_range)
        density_ratio = point_density / max(expected_density, 1.0)

        # ---- Missing region analysis ----
        mr_type, missing_frac, gap_coherence, gap_incidence = \
            self._analyse_missing_regions(pts, scene_cluster, sensor_origin, view_dir)

        # ---- Spatial context ----
        d_wall = self._dist_to_nearest_wall(centroid)
        z_above = self._z_above_bin_floor(centroid)
        near_wall = d_wall < self.part.L_max * 1.2

        near_other = False
        if other_cluster_centroids:
            dists = [np.linalg.norm(centroid - c) for c in other_cluster_centroids]
            near_other = any(d < self.part.L_max * 1.5 for d in dists)

        # ---- Segmentation quality flags ----
        # Over-segmentation: observed BB much smaller than expected
        bb_ratio_max = cl_max / max(self.part.L_max, 1e-6)
        possible_over_seg  = bb_ratio_max < 0.55
        possible_under_seg = bb_ratio_max > 1.40

        return SceneProfile(
            n_points=n_pts,
            cluster_L_max=cl_max, cluster_L_mid=cl_mid, cluster_L_min=cl_min,
            cluster_volume=cl_volume,
            observed_centroid=centroid,
            estimated_visibility=estimated_visibility,
            viewing_direction=view_dir,
            estimated_range=est_range,
            point_density=point_density,
            density_ratio=density_ratio,
            missing_region_type=mr_type,
            missing_fraction=missing_frac,
            gap_coherence_score=gap_coherence,
            gap_incidence_correlation=gap_incidence,
            dist_to_nearest_wall=d_wall,
            z_above_bin_floor=z_above,
            near_wall=near_wall,
            near_other_cluster=near_other,
            possible_over_segment=possible_over_seg,
            possible_under_segment=possible_under_seg,
        )

    # ------------------------------------------------------------------
    def _estimate_visibility(
        self,
        n_observed: int,
        view_dir: np.ndarray,
        est_range: float,
    ) -> Tuple[float, int]:
        """
        Estimate visible fraction using per-viewpoint reference counts.

        Strategy:
          1. Find the closest viewpoint direction in the reference metadata.
          2. Use that viewpoint's expected point count as the denominator.
          3. Apply a range-correction factor (point density falls with range²).
          4. Clamp output to [0.05, 1.0].

        Falls back to a geometry-based estimate if no viewpoint metadata exists.
        """
        if len(self.part.viewpoint_directions) > 0 and len(self.part.viewpoint_point_counts) > 0:
            # Find closest viewpoint to current viewing direction
            vp_dirs = np.stack(self.part.viewpoint_directions)  # (V, 3)
            dots = vp_dirs @ view_dir
            best_vp_idx = int(np.argmax(dots))
            expected_pts = self.part.viewpoint_point_counts.get(best_vp_idx, None)

            if expected_pts is not None and expected_pts > 0:
                # Range correction: sensor generates fewer points at longer range
                r_ratio = self.sensor.nominal_range / max(est_range, 0.01)
                range_correction = r_ratio ** 2
                expected_corrected = expected_pts * range_correction
                visibility = float(np.clip(n_observed / expected_corrected, 0.05, 1.0))
                return visibility, int(expected_corrected)

        # Fallback: geometric estimate from solid angle
        # At range r, a part of area A subtends solid angle A/r².
        # Expected points ≈ sensor_density × A_visible
        # Approximate A_visible as 50% of surface area (single-view hemisphere)
        area_visible = self.part.surface_area * 0.5
        sensor_density = 1.0 / max(self.sensor.lateral_noise_at_nominal ** 2, 1e-9) * 0.001
        expected_pts = int(area_visible * sensor_density * (self.sensor.nominal_range / max(est_range, 0.01)) ** 2)
        expected_pts = max(expected_pts, 100)
        visibility = float(np.clip(n_observed / expected_pts, 0.05, 1.0))
        return visibility, expected_pts

    # ------------------------------------------------------------------
    def _expected_density(self, est_range: float) -> float:
        """
        Expected point density (pts/m³) from sensor model.
        Lateral resolution degrades quadratically with range;
        depth resolution degrades linearly.
        """
        lat_res = self.sensor.lateral_noise_at_nominal * (est_range / max(self.sensor.nominal_range, 0.01))
        depth_res = (self.sensor.depth_noise_at_nominal +
                     self.sensor.depth_noise_slope * abs(est_range - self.sensor.nominal_range))
        voxel_vol = lat_res ** 2 * depth_res
        return 1.0 / max(voxel_vol, 1e-12)

    # ------------------------------------------------------------------
    def _analyse_missing_regions(
        self,
        pts: np.ndarray,
        pcd: o3d.geometry.PointCloud,
        sensor_origin: np.ndarray,
        view_dir: np.ndarray,
    ) -> Tuple[MissingRegionType, float, float, float]:
        """
        Distinguish occlusion from reflection dropout using three signals:

        Signal 1 — Gap spatial coherence
            Project points onto the image plane (or a view-aligned plane).
            Compute convex hull of the occupied pixels.
            Compare to bounding rectangle of all expected pixels.
            A large connected interior gap → high coherence → reflection.
            A half-plane gap → occlusion.

        Signal 2 — Gap boundary normal orientation
            At the edge of the gap, are normals pointing at high incidence angle?
            If normals at gap boundary have large θ (cos(θ) small) → reflection.
            If normals at gap boundary are arbitrary → occlusion.

        Signal 3 — Bounding box completeness
            If observed BB ≈ part BB in all dims → gap is interior → reflection.
            If one BB dimension is significantly truncated → occlusion.

        Returns
        -------
        mr_type        : MissingRegionType
        missing_frac   : estimated fraction of surface area missing
        gap_coherence  : ∈ [0, 1]  — 1 = single coherent gap
        gap_incidence  : ∈ [0, 1]  — 1 = gap strongly correlated with high-θ regions
        """
        if len(pts) < 20:
            return MissingRegionType.UNKNOWN, 1.0, 0.0, 0.0

        # Estimate normals if not present
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.part.L_max * 0.02, max_nn=30
                )
            )
            pcd.orient_normals_towards_camera_location(sensor_origin.tolist())

        normals = np.asarray(pcd.normals)

        # ---- Signal 1: project onto view-aligned plane ----
        # Build orthonormal frame aligned with view direction
        ref = np.array([0., 1., 0.]) if abs(view_dir[1]) < 0.9 else np.array([1., 0., 0.])
        right = np.cross(view_dir, ref); right /= np.linalg.norm(right) + 1e-8
        up    = np.cross(right, view_dir)

        u = pts @ right   # projected u coordinate
        v = pts @ up      # projected v coordinate

        # Rasterise onto a 64×64 grid
        GRID = 64
        u_norm = (u - u.min()) / (u.max() - u.min() + 1e-8)
        v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)
        ui = np.clip((u_norm * (GRID - 1)).astype(int), 0, GRID - 1)
        vi = np.clip((v_norm * (GRID - 1)).astype(int), 0, GRID - 1)
        grid = np.zeros((GRID, GRID), dtype=bool)
        grid[vi, ui] = True

        # Expected filled area — expand to convex hull of occupied pixels
        occupied_frac  = float(grid.mean())
        # Convex hull fill estimate
        try:
            occ_pixels = np.column_stack([ui, vi])
            hull = ConvexHull(occ_pixels)
            hull_area = hull.volume  # in 2D, .volume is area
            hull_frac  = hull_area / (GRID * GRID)
            missing_frac = float(np.clip(1.0 - occupied_frac / max(hull_frac, 1e-3), 0.0, 1.0))
        except Exception:
            missing_frac = float(1.0 - occupied_frac)

        # Gap coherence: largest connected gap as fraction of total gap
        gap_grid = ~grid
        labeled, n_components = connected_label(gap_grid)
        if n_components > 0 and gap_grid.sum() > 0:
            component_sizes = [np.sum(labeled == i) for i in range(1, n_components + 1)]
            largest_gap = max(component_sizes)
            gap_coherence = float(largest_gap / gap_grid.sum())
        else:
            gap_coherence = 0.0

        # ---- Signal 2: gap boundary normal incidence angles ----
        # Identify points near the gap (within 2 pixels of empty region)
        from scipy.ndimage import binary_dilation
        gap_dilated = binary_dilation(gap_grid, iterations=2)
        near_gap_mask = gap_dilated[vi, ui] & grid[vi, ui]  # occupied but near gap

        if near_gap_mask.sum() > 5:
            boundary_normals = normals[near_gap_mask]
            # incidence angle cosine: |n̂ · (-view_dir)|
            cos_theta = np.abs(boundary_normals @ (-view_dir))
            cos_theta = np.clip(cos_theta, 0, 1)
            theta_boundary = np.arccos(cos_theta)
            # High incidence (>60°) at boundary → reflection signature
            high_incidence_frac = float((theta_boundary > np.radians(60)).mean())
            gap_incidence = high_incidence_frac
        else:
            gap_incidence = 0.0

        # ---- Signal 3: bounding box completeness ----
        bb_dim_ratios = np.array([
            self.part.L_max   / max(self.part.L_max,   1e-6),   # = 1 trivially
            self.part.L_mid   / max(self.part.L_mid,   1e-6),
            self.part.L_min_dim / max(self.part.L_min_dim, 1e-6),
        ])
        obs_dims_sorted = np.sort([
            len(np.unique(ui)) / GRID * self.part.L_max,
            len(np.unique(vi)) / GRID * self.part.L_max,
        ])
        # A truncated dimension means BB ratio in that dimension is low
        bb_truncated = any(
            obs < expected * 0.70
            for obs, expected in zip(
                sorted([self.part.L_max, self.part.L_mid])[::-1],
                [self.part.L_max, self.part.L_mid]
            )
        )

        # ---- Combine signals ----
        reflection_score = (
            0.40 * gap_coherence +
            0.35 * gap_incidence +
            0.25 * (1.0 - float(bb_truncated))
        )
        occlusion_score = (
            0.40 * float(bb_truncated) +
            0.30 * (1.0 - gap_coherence) +
            0.30 * (1.0 - gap_incidence)
        )

        THRESH = 0.55
        if reflection_score > THRESH and occlusion_score < 0.35:
            mr_type = MissingRegionType.REFLECTION_DROPOUT
        elif occlusion_score > THRESH and reflection_score < 0.35:
            mr_type = MissingRegionType.OCCLUSION
        elif reflection_score > 0.35 and occlusion_score > 0.35:
            mr_type = MissingRegionType.MIXED
        else:
            mr_type = MissingRegionType.UNKNOWN

        return mr_type, missing_frac, gap_coherence, gap_incidence

    # ------------------------------------------------------------------
    def _dist_to_nearest_wall(self, centroid: np.ndarray) -> float:
        """
        Approximate distance from centroid to nearest bin wall, assuming
        bin centre is at origin in XY and floor at z=0.
        """
        half_l = self.bin.inner_length / 2.0
        half_w = self.bin.inner_width  / 2.0
        dx = half_l - abs(centroid[0])
        dy = half_w - abs(centroid[1])
        dz = centroid[2]  # above floor
        return float(max(min(dx, dy, dz), 0.0))

    def _z_above_bin_floor(self, centroid: np.ndarray) -> float:
        return float(max(centroid[2], 0.0))


# ============================================================
# 3. ParameterResolver
# ============================================================

class ParameterResolver:
    """
    Maps (PartProfile, SceneProfile, SensorProfile, BinProfile)
    → (RegistrationParams, RegistrationConfidence)

    All formulas are closed-form and documented inline.
    Adjust BASE_* multipliers to calibrate for a specific camera model
    without changing the overall structure.
    """

    # Multipliers relative to L_max — primary tuning knobs
    BASE_VOXEL_RATIO        = 0.015   # voxel_size = L_max × this
    BASE_FPFH_RATIO         = 0.050
    BASE_RANSAC_DIST_RATIO  = 0.040
    BASE_ICP_DIST_RATIO     = 0.015

    # Confidence thresholds
    CONFIDENCE_RELIABLE_THRESH = 0.55

    def resolve(
        self,
        part: PartProfile,
        scene: SceneProfile,
        sensor: SensorProfile,
        bin_: BinProfile,
    ) -> Tuple[RegistrationParams, RegistrationConfidence]:

        notes: List[str] = []
        L = part.L_max

        # ================================================================
        # STEP 1 — Base scales from part geometry
        # ================================================================

        # Voxel size bounded below by min feature size / 2
        # (cannot downsample past discriminative features)
        voxel_base = L * self.BASE_VOXEL_RATIO
        voxel_size  = max(voxel_base, part.min_feature_size * 0.5)
        voxel_size  = min(voxel_size, L * 0.03)   # cap at 3% of L_max

        # Reference uses a potentially different (often coarser) voxel
        # to keep ICP reference density manageable
        ref_voxel_size = voxel_size * 1.5

        # Normal estimation radius — tighter for thin parts (avoid cross-contamination)
        normal_radius = voxel_size * (2.5 if part.is_thin else 4.0)
        normal_max_nn = 30

        # FPFH radius — 5× voxel, respects min feature
        fpfh_radius = max(L * self.BASE_FPFH_RATIO, voxel_size * 5.0)
        fpfh_max_nn = 100

        # RANSAC distance threshold
        ransac_dist = L * self.BASE_RANSAC_DIST_RATIO

        # ICP distance threshold — anchored to sensor noise floor
        noise_floor = sensor.depth_noise_at_nominal + \
                      sensor.depth_noise_slope * abs(scene.estimated_range - sensor.nominal_range)
        icp_dist = max(
            L * self.BASE_ICP_DIST_RATIO,
            noise_floor * 3.0   # never tighter than 3× noise floor
        )

        # ICP convergence: also never tighter than fringe period / 4
        icp_convergence = max(noise_floor * 0.5, sensor.fringe_period / 4.0)

        notes.append(
            f"Base scales: L_max={L*1000:.1f}mm, voxel={voxel_size*1000:.2f}mm, "
            f"icp_dist={icp_dist*1000:.2f}mm, noise_floor={noise_floor*1000:.2f}mm"
        )

        # ================================================================
        # STEP 2 — ICP variant from shape class
        # ================================================================
        # - High SA:V (thin/flat parts): normals are well-defined, consistent
        #   → point-to-plane converges faster and more accurately
        # - Low SA:V (blocky parts): normals near corners/edges are noisy
        #   → GICP is more robust
        # - Generic: standard point-to-plane
        if part.sa_to_volume_ratio > 400:   # very thin (sheet metal-like)
            icp_variant = ICPVariant.POINT_TO_PLANE
            notes.append("ICP: point-to-plane (high SA:V thin part)")
        elif part.compactness > 0.5:
            icp_variant = ICPVariant.GENERALIZED
            notes.append("ICP: GICP (blocky/compact part)")
        else:
            icp_variant = ICPVariant.POINT_TO_PLANE
            notes.append("ICP: point-to-plane (default)")

        # ================================================================
        # STEP 3 — Symmetry handling (regional-aware)
        # ================================================================
        # Base symmetry multiplier from global class (sets floor for restarts)
        sym_multiplier = {
            SymmetryClass.ASYMMETRIC: 1.0,
            SymmetryClass.ONE_AXIS:   3.0,
            SymmetryClass.TWO_AXIS:   6.0,
            SymmetryClass.FULL:       10.0,
        }[part.symmetry_class]

        # ── Discriminative fraction correction ──
        # For a partially symmetric part (e.g. hex-headed shaft), only
        # discriminative_fraction of correspondences can resolve the pose.
        # RANSAC needs a sample of n=4 points; the probability that at least
        # one is discriminative is: 1 - (1 - d)^4.
        # Iteration scaling: N_corrected = N_base / P(at_least_one_discriminative)
        #
        # Example: d=0.05 (keyway on shaft) → P = 1 - 0.95^4 = 18.5%
        #          → RANSAC needs 5.4× more iterations just for discriminativeness,
        #            on top of the symmetry_class multiplier.
        d = float(np.clip(part.discriminative_fraction, 0.02, 1.0))
        n_ransac_pts = 4   # matches params.ransac_n_points below
        p_discriminative = 1.0 - (1.0 - d) ** n_ransac_pts
        discriminative_iter_multiplier = float(np.clip(1.0 / p_discriminative, 1.0, 20.0))

        # Feature richness: rich geometry → FPFH is more reliable → fewer iters
        feature_richness = float(np.clip(part.normal_variance / (np.pi / 4), 0.1, 1.0))
        ransac_base_iter = 500_000

        ransac_max_iter = int(
            ransac_base_iter
            * sym_multiplier
            * discriminative_iter_multiplier
            / max(feature_richness, 0.1)
        )
        ransac_max_iter = int(np.clip(ransac_max_iter, 100_000, 8_000_000))
        ransac_confidence = 0.999

        # Minimum inlier fraction: lower for partially discriminative surfaces
        # (many correspondences will match symmetrically and look like inliers)
        ransac_min_inlier = 0.25 / max(sym_multiplier, 1.0)
        ransac_min_inlier = max(ransac_min_inlier * d, 0.04)

        # RANSAC restarts: driven by fold order, not just class
        # For a 6-fold part, need at least 6 restarts to cover all equivalent poses.
        # For a partially symmetric part with low d, add extra restarts since
        # the discriminative signal may be missed in any individual run.
        base_restarts = {
            SymmetryClass.ASYMMETRIC: 1,
            SymmetryClass.ONE_AXIS:   3,
            SymmetryClass.TWO_AXIS:   6,
            SymmetryClass.FULL:       12,
        }[part.symmetry_class]

        fold = part.symmetry_fold
        # For finite folds, restarts must cover each unique pose orientation
        if fold > 1:
            fold_restarts = max(fold, base_restarts)
        else:
            fold_restarts = base_restarts

        # Scale up restarts when discriminative fraction is very low
        disc_restart_bonus = int(np.ceil(1.0 / max(d, 0.1))) - 1
        n_ransac_restarts = int(np.clip(fold_restarts + disc_restart_bonus, 1, 24))

        # Pose clustering threshold:
        # Use the fold angle as the natural angular bin size — poses that differ
        # by less than half the fold angle are equivalent for a symmetric part.
        # For asymmetric or partially-symmetric parts, use a tight threshold.
        if fold > 1:
            pose_cluster_thresh = (360.0 / fold) * 0.45   # just under half-period
        elif fold == 0:   # infinite / continuous symmetry
            pose_cluster_thresh = 2.0
        else:
            # Asymmetric: tight clustering — distinct hypotheses should be well-separated
            pose_cluster_thresh = {
                SymmetryClass.ASYMMETRIC: 10.0,
                SymmetryClass.ONE_AXIS:   8.0,
                SymmetryClass.TWO_AXIS:   5.0,
                SymmetryClass.FULL:       2.0,
            }[part.symmetry_class]

        sym_note = (
            f"Symmetry: {part.symmetry_class.name} "
            f"fold={fold if fold > 0 else '∞'} "
            f"d_frac={part.discriminative_fraction:.2f} "
            f"ransac_iter={ransac_max_iter:,} "
            f"(×sym={sym_multiplier:.0f} ×disc={discriminative_iter_multiplier:.1f}) "
            f"restarts={n_ransac_restarts}"
        )
        notes.append(sym_note)
        if part.discriminative_fraction < 0.15:
            notes.append(
                f"  ⚠ Low discriminative fraction ({part.discriminative_fraction:.2f}): "
                f"registration may be slow or ambiguous — "
                f"consider adding a fiducial marker or capturing the asymmetric feature "
                f"from a viewpoint where it is prominently visible"
            )

        # ================================================================
        # STEP 4 — Visibility and missing region adjustments
        # ================================================================
        vis = scene.estimated_visibility

        # Loosen ICP correspondence threshold when visibility is low
        # (fewer correspondences available, must cast a wider net)
        if vis < 0.35:
            icp_dist *= 1.5
            notes.append(f"Low visibility ({vis:.2f}): ICP dist loosened ×1.5")
        elif vis < 0.55:
            icp_dist *= 1.2
            notes.append(f"Moderate visibility ({vis:.2f}): ICP dist loosened ×1.2")

        # Trimmed ICP: ignore the worst-fitting points (useful when occlusion
        # causes many unmatched points that would otherwise pull the solution)
        if scene.missing_region_type == MissingRegionType.OCCLUSION:
            trim_frac = float(np.clip(scene.missing_fraction * 0.8, 0.0, 0.5))
            notes.append(
                f"Occlusion detected: trimmed ICP trim_frac={trim_frac:.2f}"
            )
        elif scene.missing_region_type == MissingRegionType.REFLECTION_DROPOUT:
            # Reflection creates a large coherent gap — don't trim uniformly;
            # instead loosen outlier removal so edge points aren't discarded
            trim_frac = 0.0
            icp_dist *= 1.15
            notes.append(
                "Reflection dropout detected: loosened ICP dist ×1.15, no trim"
            )
        elif scene.missing_region_type == MissingRegionType.MIXED:
            trim_frac = float(np.clip(scene.missing_fraction * 0.4, 0.0, 0.3))
            icp_dist *= 1.10
            notes.append(
                f"Mixed occlusion+reflection: trim_frac={trim_frac:.2f}, dist ×1.10"
            )
        else:
            trim_frac = 0.0

        # ================================================================
        # STEP 5 — Density ratio adjustment (material effects)
        # ================================================================
        # density_ratio << 1 → material causing heavy dropout (specular metal, etc.)
        # Loosen ICP and outlier removal to tolerate sparse, patchy data
        if scene.density_ratio < 0.4:
            icp_dist *= 1.3
            outlier_std_ratio = 3.0   # lenient
            notes.append(
                f"Low density ratio ({scene.density_ratio:.2f}): "
                f"highly reflective material suspected"
            )
        elif scene.density_ratio < 0.7:
            outlier_std_ratio = 2.5
            notes.append(f"Moderate density ratio ({scene.density_ratio:.2f})")
        else:
            outlier_std_ratio = 2.0

        # ================================================================
        # STEP 6 — Spatial context adjustments
        # ================================================================
        # Near wall: outlier removal more aggressive (wall points contaminate cluster)
        if scene.near_wall and bin_.wall_reflective:
            outlier_std_ratio = min(outlier_std_ratio, 1.8)
            notes.append("Near reflective wall: tighter outlier removal")

        # Near other cluster: may have inter-part reflection artifacts
        if scene.near_other_cluster:
            icp_dist *= 1.05
            notes.append("Near neighbouring cluster: slight ICP dist relaxation")

        # High z (top of pile): usually clean, can afford tighter settings
        if scene.z_above_bin_floor > part.L_max * 0.8:
            icp_convergence *= 0.8
            notes.append("Part near top of pile: tightened convergence")

        # Near bin floor: likely heavily occluded
        if scene.z_above_bin_floor < part.L_max * 0.3:
            icp_dist *= 1.2
            trim_frac = max(trim_frac, 0.15)
            notes.append("Part near bin floor: heavy occlusion expected")

        # ================================================================
        # STEP 7 — Segmentation quality adjustments
        # ================================================================
        if scene.possible_over_segment:
            # Very few points — can't trust fine ICP
            icp_max_iter = 50
            icp_dist *= 1.5
            notes.append("Possible over-segmentation: relaxed ICP settings")
        elif scene.possible_under_segment:
            # Cluster may contain two parts — RANSAC needs to be more selective
            ransac_min_inlier = min(ransac_min_inlier, 0.15)
            notes.append("Possible under-segmentation: tighter RANSAC inlier threshold")

        # ================================================================
        # STEP 8 — Final bounds and outlier removal k
        # ================================================================
        # Outlier k-neighbours: more neighbours → smoother removal → better for dense clouds
        outlier_nb_k = max(20, int(np.log2(max(scene.n_points, 100)) * 3))

        # ICP max iter: low visibility needs more iterations to converge
        if not scene.possible_over_segment:
            icp_max_iter = int(np.clip(
                100 / max(vis, 0.05),   # more iters when less is visible
                50, 300
            ))

        # ICP fitness threshold — minimum fraction of points that must match
        # Lower for occluded scenes; can't demand full coverage
        icp_fitness_thresh = float(np.clip(vis * 0.7, 0.10, 0.60))

        # ================================================================
        # STEP 9 — Assemble params
        # ================================================================
        params = RegistrationParams(
            scene_voxel_size=float(voxel_size),
            reference_voxel_size=float(ref_voxel_size),
            normal_radius=float(normal_radius),
            normal_max_nn=normal_max_nn,
            outlier_nb_neighbors=outlier_nb_k,
            outlier_std_ratio=float(outlier_std_ratio),
            fpfh_radius=float(fpfh_radius),
            fpfh_max_nn=fpfh_max_nn,
            ransac_distance_thresh=float(ransac_dist),
            ransac_n_points=4 if not part.is_thin else 3,
            ransac_max_iter=ransac_max_iter,
            ransac_confidence=ransac_confidence,
            ransac_min_inlier_fraction=float(ransac_min_inlier),
            mutual_filter=True,
            icp_variant=icp_variant,
            icp_distance_thresh=float(icp_dist),
            icp_max_iter=int(icp_max_iter),
            icp_convergence_rmse=float(icp_convergence),
            icp_outlier_trim_fraction=float(trim_frac),
            icp_fitness_threshold=float(icp_fitness_thresh),
            n_ransac_restarts=n_ransac_restarts,
            pose_cluster_angular_thresh=pose_cluster_thresh,
            notes=notes,
        )

        # ================================================================
        # STEP 10 — Confidence score (evaluated post-registration)
        # Here we return a pre-registration estimate; caller fills in
        # icp_inlier_fraction and icp_rmse_normalised after running ICP.
        # ================================================================
        confidence = self._compute_prior_confidence(part, scene, sensor, params)

        return params, confidence

    # ------------------------------------------------------------------
    def _compute_prior_confidence(
        self,
        part: PartProfile,
        scene: SceneProfile,
        sensor: SensorProfile,
        params: RegistrationParams,
    ) -> RegistrationConfidence:
        """
        Pre-registration confidence estimate based on scene and part properties.
        After ICP runs, update icp_inlier_fraction and icp_rmse_normalised.
        """
        # Visibility score — monotone but sublinear (partially visible parts are okay)
        vis_score = float(np.sqrt(np.clip(scene.estimated_visibility, 0.0, 1.0)))

        # Symmetry penalty — more symmetric → more pose ambiguity → lower confidence
        sym_penalty = {
            SymmetryClass.ASYMMETRIC: 1.00,
            SymmetryClass.ONE_AXIS:   0.80,
            SymmetryClass.TWO_AXIS:   0.65,
            SymmetryClass.FULL:       0.40,
        }[part.symmetry_class]

        # Segmentation quality
        if scene.possible_over_segment or scene.possible_under_segment:
            seg_quality = 0.60
        else:
            seg_quality = 1.00

        # Prior overall (will be updated post-ICP)
        prior_overall = vis_score * sym_penalty * seg_quality
        prior_overall = float(np.clip(prior_overall, 0.0, 1.0))

        reliable = prior_overall >= self.CONFIDENCE_RELIABLE_THRESH
        if prior_overall >= 0.75:
            action = "proceed"
        elif prior_overall >= self.CONFIDENCE_RELIABLE_THRESH:
            action = "proceed_with_caution"
        else:
            action = "re-scan"

        return RegistrationConfidence(
            overall=prior_overall,
            icp_inlier_fraction=-1.0,   # not yet known
            icp_rmse_normalised=-1.0,   # not yet known
            visibility_score=vis_score,
            symmetry_penalty=sym_penalty,
            segmentation_quality=seg_quality,
            reliable=reliable,
            action_recommendation=action,
        )

    @staticmethod
    def update_confidence_post_icp(
        confidence: RegistrationConfidence,
        icp_inlier_fraction: float,
        icp_rmse: float,
        noise_floor: float,
    ) -> RegistrationConfidence:
        """
        Call after ICP completes to refine the confidence estimate.
        """
        icp_rmse_norm = float(np.clip(icp_rmse / max(noise_floor, 1e-9), 0.5, 10.0))
        # rmse_score: 1.0 when rmse = noise_floor, decays toward 0 at 10× noise_floor
        rmse_score = float(np.clip(1.0 - (icp_rmse_norm - 1.0) / 9.0, 0.0, 1.0))
        inlier_score = float(np.clip(icp_inlier_fraction / 0.5, 0.0, 1.0))

        updated_overall = (
            0.30 * confidence.visibility_score +
            0.25 * confidence.symmetry_penalty +
            0.15 * confidence.segmentation_quality +
            0.20 * inlier_score +
            0.10 * rmse_score
        )
        updated_overall = float(np.clip(updated_overall, 0.0, 1.0))

        if updated_overall >= 0.75:
            action = "proceed"
        elif updated_overall >= ParameterResolver.CONFIDENCE_RELIABLE_THRESH:
            action = "proceed_with_caution"
        else:
            action = "re-scan"

        return RegistrationConfidence(
            overall=updated_overall,
            icp_inlier_fraction=icp_inlier_fraction,
            icp_rmse_normalised=icp_rmse_norm,
            visibility_score=confidence.visibility_score,
            symmetry_penalty=confidence.symmetry_penalty,
            segmentation_quality=confidence.segmentation_quality,
            reliable=updated_overall >= ParameterResolver.CONFIDENCE_RELIABLE_THRESH,
            action_recommendation=action,
        )


# ============================================================
# 4. HeuristicParameterEngine  (top-level façade)
# ============================================================

class HeuristicParameterEngine:
    """
    Top-level façade.  Typical usage:

        engine = HeuristicParameterEngine(sensor_profile, bin_profile)
        part_profile = engine.analyse_part(cad_mesh_path, vp_metadata_path)
        # --- at runtime, per frame ---
        params, confidence = engine.resolve(
            scene_cluster, sensor_origin, part_profile
        )
        print(params.icp_distance_thresh, confidence.action_recommendation)
    """

    def __init__(
        self,
        sensor_profile: SensorProfile,
        bin_profile: BinProfile,
    ) -> None:
        self.sensor = sensor_profile
        self.bin    = bin_profile
        self.mesh_analyser = MeshAnalyser()
        self.resolver      = ParameterResolver()
        self._part_cache: Dict[str, PartProfile] = {}

    def analyse_part(
        self,
        mesh_path: str | Path,
        viewpoint_metadata_path: Optional[str | Path] = None,
        cache_key: Optional[str] = None,
    ) -> PartProfile:
        """
        Analyse a CAD mesh and cache the result.
        This is the expensive one-time step per part.
        """
        key = cache_key or str(mesh_path)
        if key not in self._part_cache:
            profile = self.mesh_analyser.analyse(mesh_path, viewpoint_metadata_path)
            self._part_cache[key] = profile
            print(
                f"[Engine] Part profile computed: "
                f"L_max={profile.L_max*1000:.1f}mm, "
                f"symmetry={profile.symmetry_class.name} "
                f"fold={'∞' if profile.symmetry_fold==0 else profile.symmetry_fold} "
                f"d_frac={profile.discriminative_fraction:.2f}, "
                f"is_thin={profile.is_thin}, "
                f"min_feature={profile.min_feature_size*1000:.1f}mm"
            )
        return self._part_cache[key]

    def resolve(
        self,
        scene_cluster: o3d.geometry.PointCloud,
        sensor_origin: np.ndarray,
        part_profile: PartProfile,
        other_cluster_centroids: Optional[List[np.ndarray]] = None,
    ) -> Tuple[RegistrationParams, RegistrationConfidence]:
        """
        Per-frame call.  Fast — all expensive analysis is pre-computed.
        """
        scene_analyser = SceneAnalyser(part_profile, self.sensor, self.bin)
        scene_profile  = scene_analyser.analyse(
            scene_cluster, sensor_origin, other_cluster_centroids
        )
        params, confidence = self.resolver.resolve(
            part_profile, scene_profile, self.sensor, self.bin
        )
        return params, confidence

    def update_confidence(
        self,
        confidence: RegistrationConfidence,
        icp_inlier_fraction: float,
        icp_rmse: float,
    ) -> RegistrationConfidence:
        """Call after ICP completes to get final confidence score."""
        noise_floor = (
            self.sensor.depth_noise_at_nominal +
            self.sensor.depth_noise_slope * 0.0   # use nominal for this helper
        )
        return ParameterResolver.update_confidence_post_icp(
            confidence, icp_inlier_fraction, icp_rmse, noise_floor
        )


# ============================================================
# 5. Reference PC pipeline review utility
# ============================================================

def curvature_weighted_downsample(
    pcd: o3d.geometry.PointCloud,
    curvatures: np.ndarray,
    target_n_flat: int,
    keep_high_curvature_fraction: float = 0.30,
) -> o3d.geometry.PointCloud:
    """
    Continuous curvature-weighted downsampling — replaces the CDF-knee approach.

    Instead of a binary threshold (knee of CDF), we use a sigmoid retention
    probability:
        p_keep(κ) = sigmoid((κ - κ_median) / κ_mad)

    This is distribution-shape agnostic:
      - Bimodal parts (flat + sharp edges): sigmoid naturally splits the two modes
      - Unimodal parts (gradual curvature): sigmoid gives a smooth gradient,
        avoiding the arbitrary knee location that caused instability

    High-curvature points are ALWAYS kept (bounded by keep_high_curvature_fraction).
    Low-curvature points are downsampled toward target_n_flat density.

    Parameters
    ----------
    pcd                          : input point cloud (with normals)
    curvatures                   : (N,) per-point curvature values
    target_n_flat                : target number of low-curvature points to retain
    keep_high_curvature_fraction : fraction of total points classified as "high curvature"
    """
    N = len(curvatures)
    kappa_median = float(np.median(curvatures))
    kappa_mad    = float(np.median(np.abs(curvatures - kappa_median))) + 1e-9

    # Sigmoid retention probability
    z = (curvatures - kappa_median) / kappa_mad
    p_keep = 1.0 / (1.0 + np.exp(-z))   # ∈ (0, 1)

    # High-curvature threshold: top keep_high_curvature_fraction by p_keep
    thresh = float(np.percentile(p_keep, (1.0 - keep_high_curvature_fraction) * 100))
    high_kappa_mask = p_keep >= thresh

    # All high-curvature points kept unconditionally
    high_idx = np.where(high_kappa_mask)[0]

    # Low-curvature points: subsample to target_n_flat
    low_idx  = np.where(~high_kappa_mask)[0]
    if len(low_idx) > target_n_flat:
        # Weighted sampling: even within low-curvature, prefer higher κ
        weights = p_keep[low_idx]
        weights /= weights.sum()
        low_idx = np.random.choice(low_idx, target_n_flat, replace=False, p=weights)

    keep_idx = np.concatenate([high_idx, low_idx])
    return pcd.select_by_index(keep_idx.tolist())


def save_viewpoint_metadata(
    path: str | Path,
    viewpoint_directions: List[np.ndarray],
    point_counts: List[int],
) -> None:
    """
    Save per-viewpoint point count metadata as JSON sidecar.
    Call this at the end of your reference PC generation pipeline.

    >>> save_viewpoint_metadata(
    ...     "part_A_viewpoints.json",
    ...     viewpoint_directions,   # list of (3,) unit vectors
    ...     point_counts,           # list of ints, one per viewpoint
    ... )
    """
    data = {
        "viewpoints": [
            {
                "index": i,
                "direction": d.tolist(),
                "point_count": int(c),
            }
            for i, (d, c) in enumerate(zip(viewpoint_directions, point_counts))
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Metadata] Saved {len(point_counts)} viewpoint records to {path}")


# ============================================================
# Self-test
# ============================================================

def _self_test():
    print("=" * 65)
    print("HeuristicParameterEngine — self-test (synthetic geometry)")
    print("=" * 65)

    # ---- Sensor and bin profiles ----
    sensor = SensorProfile(
        model_name="Photoneo_PhoXi_M",
        fx=2000.0, fy=2000.0, cx=1024.0, cy=768.0,
        image_width=2048, image_height=1536,
        nominal_range=0.80,
        range_min=0.40, range_max=1.50,
        depth_noise_at_nominal=0.0005,
        fringe_period=0.001,
        depth_noise_slope=0.0008,
        lateral_noise_at_nominal=0.0003,
    )
    bin_ = BinProfile(
        inner_length=0.60, inner_width=0.40, inner_depth=0.30,
        wall_thickness=0.003, wall_material="metal", wall_reflective=True,
    )

    # ---- Synthetic part profile (as-if from MeshAnalyser) ----
    # We skip reading a real mesh in self-test; construct profile directly
    part = PartProfile(
        L_max=0.120, L_mid=0.080, L_min_dim=0.010,
        sa_to_volume_ratio=250.0,
        compactness=0.15,
        is_thin=True,
        normal_variance=0.4,
        curvature_mean=5.0,
        curvature_std=8.0,
        min_feature_size=0.003,
        symmetry_class=SymmetryClass.ONE_AXIS,
        symmetry_axes=1,
        inertia_eigenvalue_ratios=(0.95, 0.10),
        viewpoint_point_counts={0: 12000, 1: 11000, 2: 9500, 3: 8000},
        viewpoint_directions=[
            np.array([0., 0., -1.]),
            np.array([0.5, 0., -0.866]),
            np.array([-0.5, 0., -0.866]),
            np.array([0., 0.5, -0.866]),
        ],
        volume=9.6e-5,
        surface_area=0.024,
    )

    # ---- Synthetic scene cluster: flat plate with some dropout ----
    rng = np.random.default_rng(42)
    plate_pts = rng.uniform([-0.06, -0.04, 0.79], [0.06, 0.04, 0.81], (6000, 3))
    # Remove 30% of points to simulate partial dropout
    plate_pts = plate_pts[rng.random(len(plate_pts)) > 0.30]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(plate_pts.astype(np.float32))

    sensor_origin = np.array([0.0, 0.0, 0.0])

    # ---- SceneAnalyser ----
    analyser = SceneAnalyser(part, sensor, bin_)
    scene = analyser.analyse(pcd, sensor_origin)

    print(f"\nScene profile:")
    print(f"  n_points         : {scene.n_points}")
    print(f"  visibility       : {scene.estimated_visibility:.2f}")
    print(f"  density_ratio    : {scene.density_ratio:.2f}")
    print(f"  missing_type     : {scene.missing_region_type.name}")
    print(f"  gap_coherence    : {scene.gap_coherence_score:.2f}")
    print(f"  gap_incidence    : {scene.gap_incidence_correlation:.2f}")
    print(f"  dist_to_wall     : {scene.dist_to_nearest_wall*1000:.0f}mm")
    print(f"  z_above_floor    : {scene.z_above_bin_floor*1000:.0f}mm")
    print(f"  over_segment?    : {scene.possible_over_segment}")
    print(f"  under_segment?   : {scene.possible_under_segment}")

    # ---- ParameterResolver ----
    resolver = ParameterResolver()
    params, confidence = resolver.resolve(part, scene, sensor, bin_)

    print(f"\nResolved parameters:")
    print(f"  scene_voxel_size        : {params.scene_voxel_size*1000:.2f}mm")
    print(f"  ransac_dist_thresh      : {params.ransac_distance_thresh*1000:.2f}mm")
    print(f"  ransac_max_iter         : {params.ransac_max_iter:,}")
    print(f"  icp_variant             : {params.icp_variant.name}")
    print(f"  icp_distance_thresh     : {params.icp_distance_thresh*1000:.2f}mm")
    print(f"  icp_outlier_trim_frac   : {params.icp_outlier_trim_fraction:.2f}")
    print(f"  icp_fitness_threshold   : {params.icp_fitness_threshold:.2f}")
    print(f"  n_ransac_restarts       : {params.n_ransac_restarts}")

    print(f"\nPre-ICP confidence:")
    print(f"  overall                 : {confidence.overall:.2f}")
    print(f"  visibility_score        : {confidence.visibility_score:.2f}")
    print(f"  symmetry_penalty        : {confidence.symmetry_penalty:.2f}")
    print(f"  action                  : {confidence.action_recommendation}")

    print(f"\nResolver notes:")
    for n in params.notes:
        print(f"  • {n}")

    # ---- Simulate post-ICP update ----
    confidence_post = ParameterResolver.update_confidence_post_icp(
        confidence,
        icp_inlier_fraction=0.62,
        icp_rmse=0.0008,
        noise_floor=0.0005,
    )
    print(f"\nPost-ICP confidence:")
    print(f"  overall (updated)       : {confidence_post.overall:.2f}")
    print(f"  icp_inlier_fraction     : {confidence_post.icp_inlier_fraction:.2f}")
    print(f"  icp_rmse_normalised     : {confidence_post.icp_rmse_normalised:.2f}")
    print(f"  action                  : {confidence_post.action_recommendation}")

    # ---- Regional symmetry test: synthetic bearing shaft ----
    print("\n" + "─"*65)
    print("Regional symmetry test: bearing shaft (synthetic point cloud)")
    print("─"*65)

    rng2 = np.random.default_rng(7)
    n_shaft = 6000
    angles_s = rng2.uniform(0, 2*np.pi, n_shaft)
    z_s      = rng2.uniform(-0.040, 0.040, n_shaft)
    shaft_pts = np.stack([0.015*np.cos(angles_s), 0.015*np.sin(angles_s), z_s], axis=1).astype(np.float32)
    shaft_nrm = np.stack([np.cos(angles_s), np.sin(angles_s), np.zeros(n_shaft)], axis=1).astype(np.float32)

    n_hex_per_face = 300
    hex_pts_list, hex_nrm_list = [], []
    for face_i in range(6):
        angle_face = face_i * (np.pi / 3)
        normal_2d  = np.array([np.cos(angle_face), np.sin(angle_face)])
        tang_2d    = np.array([-np.sin(angle_face), np.cos(angle_face)])
        us = rng2.uniform(-0.011, 0.011, n_hex_per_face)
        zs = rng2.uniform(0.040, 0.060, n_hex_per_face)
        pts_face = np.stack([normal_2d[0]*0.022 + us*tang_2d[0],
                             normal_2d[1]*0.022 + us*tang_2d[1], zs], axis=1)
        hex_pts_list.append(pts_face.astype(np.float32))
        hex_nrm_list.append(np.tile([normal_2d[0], normal_2d[1], 0.0], (n_hex_per_face,1)).astype(np.float32))

    n_kw = 200
    kw_angles = rng2.uniform(-0.10, 0.10, n_kw)
    kw_z      = rng2.uniform(-0.015, 0.015, n_kw)
    kw_r      = 0.013 + rng2.uniform(0, 0.002, n_kw)
    kw_pts = np.stack([kw_r*np.cos(kw_angles), kw_r*np.sin(kw_angles), kw_z], axis=1).astype(np.float32)
    kw_nrm = np.tile([0.0, 0.0, 1.0], (n_kw, 1)).astype(np.float32)

    all_pts = np.concatenate([shaft_pts] + hex_pts_list + [kw_pts])
    all_nrm = np.concatenate([shaft_nrm] + hex_nrm_list + [kw_nrm])

    shaft_pcd = o3d.geometry.PointCloud()
    shaft_pcd.points  = o3d.utility.Vector3dVector(all_pts)
    shaft_pcd.normals = o3d.utility.Vector3dVector(all_nrm)

    analyser2 = MeshAnalyser(symmetry_sample_n=100, symmetry_iou_thresh=0.85)
    dummy_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.08)
    dummy_mesh.compute_vertex_normals()

    sym_class2, sym_axes2, eigen_ratios2, sym_fold2, disc_frac2 = analyser2._analyse_symmetry(
        dummy_mesh, shaft_pcd
    )
    print(f"  n_points          : {len(all_pts):,}  (shaft:{n_shaft} hex:{n_hex_per_face*6} keyway:{n_kw})")
    print(f"  symmetry_class    : {sym_class2.name}")
    print(f"  symmetry_fold     : {sym_fold2 if sym_fold2 != 0 else chr(8734)}")
    print(f"  discriminative_fr : {disc_frac2:.3f}  (keyway = {n_kw/len(all_pts)*100:.1f}% of surface)")

    assert disc_frac2 > 0.01, "discriminative_fraction unexpectedly 0"
    assert sym_class2 in (SymmetryClass.ONE_AXIS, SymmetryClass.ASYMMETRIC)
    print("  discriminative_fraction in reasonable range ✓")

    synthetic_part2 = PartProfile(
        L_max=0.10, L_mid=0.022, L_min_dim=0.022,
        sa_to_volume_ratio=300.0, compactness=0.4, is_thin=False,
        normal_variance=0.5, curvature_mean=3.0, curvature_std=2.0,
        min_feature_size=0.003, symmetry_class=sym_class2, symmetry_axes=sym_axes2,
        inertia_eigenvalue_ratios=eigen_ratios2,
        symmetry_fold=sym_fold2, discriminative_fraction=disc_frac2,
        volume=5.6e-5, surface_area=0.018,
    )
    rng3 = np.random.default_rng(99)
    dummy_scene_pts = (all_pts + np.array([0., 0., 0.8]) +
                       rng3.normal(0, 0.001, all_pts.shape)).astype(np.float32)
    dummy_pcd2 = o3d.geometry.PointCloud()
    dummy_pcd2.points = o3d.utility.Vector3dVector(dummy_scene_pts)
    scene2 = SceneAnalyser(synthetic_part2, sensor, bin_).analyse(
        dummy_pcd2, np.array([0., 0., 0.])
    )
    params2, _ = ParameterResolver().resolve(synthetic_part2, scene2, sensor, bin_)
    print(f"\n  Resolved RANSAC iterations : {params2.ransac_max_iter:,}")
    print(f"  Resolved RANSAC restarts   : {params2.n_ransac_restarts}")
    print(f"  Pose cluster thresh        : {params2.pose_cluster_angular_thresh:.1f}\u00b0")
    print(f"\n  Resolver notes:")
    for n_note in params2.notes:
        print(f"    {n_note}")

    print("\nSelf-test PASSED \u2713")


if __name__ == "__main__":
    _self_test()