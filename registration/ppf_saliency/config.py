"""Parameter derivation for the PPF matcher.

The point of writing this matcher from scratch is that we own the parameters, so the design
goal is to **eliminate** them rather than expose them for tuning.  A part catalogue that
needs per-part configuration does not scale to part #5000, and a knob that has to be tuned
is a knob that has to be tuned *per part*.

Three groups:

  * **Eliminated by design.**  The scene reference-point step disappears because we match
    one segmented instance at a time and the cluster is small enough to use every point.
    The hypothesis count disappears because one cluster yields one pose.
    (The per-bucket cap does *not* disappear -- see ``max_bucket_entries``. Vote dedup fixes
    the bias that cap was patching, but not the cost, so the cap survives as a derived
    compute budget rather than a tuned number.)
  * **Derived from part geometry.**  Sampling distance, pair-distance bounds, and the pose
    clustering tolerances.
  * **Derived from a one-time sensor calibration.**  The angular binning and the
    verification tolerance follow from depth noise.  This is a per-*camera* constant, not a
    per-part one, so it does not break the scale constraint -- it is the same kind of
    absolute sensor-physics floor that ``AmbiguityConfig`` already documents ("The absolute
    floors exist for sensor-physics reasons, not as size rules").

What is left is two *application* policies -- ``model_target_points`` (a compute budget) and
the acceptance threshold -- both shared across every part.

``PPFConfig.derive`` records which bound actually bound each value in ``.provenance``, so
"why is tau 4.6 mm" has an answer without re-deriving it by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import numpy as np

__all__ = ["SensorProfile", "PPFConfig", "MECHVISION_NAMES"]


# Mapping to MechVision's 3D coarse matching parameters. Kept so that findings here
# transfer to the MM_Optimizer tuning work, which drives a black box we cannot change.
MECHVISION_NAMES = {
    "tau": "distQuantification (x model diameter)",
    "n_alpha": "angleQuantification",
    "scene_step": "referredStep  (eliminated here -- see module docstring)",
    "vote_cap": "maxNumOfPointPairsPerFeature  (kept, but derived -- see max_bucket_entries)",
    "verify_tol": "voxelLengthRange",
}


@dataclass(frozen=True)
class SensorProfile:
    """Fixed per camera model, measured once at deployment.

    Defaults mirror the simulated sensor in ``sensor/scene_render.py`` (``add_sensor_noise``
    with ``sigma_z_ref=0.2 mm`` at ``z_ref=2.0 m``, quadratic range model) so that anything
    derived here is consistent with the synthetic data the benchmark runs on.
    """

    sigma_z_ref: float = 0.0002      # 1-sigma depth noise at z_ref (m)
    z_ref: float = 2.0               # range at which sigma_z_ref was measured (m)
    sigma_lateral: float = 0.0005    # 1-sigma lateral noise (m)
    working_distance: float = 1.5    # actual stand-off; matches MujocoBinScene.camera_distance

    def sigma_z(self, distance: Optional[float] = None) -> float:
        """Depth noise at ``distance``. Structured-light depth error grows ~quadratically."""
        d = self.working_distance if distance is None else float(distance)
        return float(self.sigma_z_ref * (d / self.z_ref) ** 2)


@dataclass
class PPFConfig:
    """Resolved parameters for one part. Build with :meth:`derive`, not by hand."""

    # --- geometry ---
    diameter: float                  # true max pairwise distance (m)
    tau: float                       # sampling distance == distance bin width (m)
    min_pair_dist: float             # pairs closer than this carry no pose information
    max_pair_dist: float             # == diameter; no pair on the model can exceed it

    # --- quantisation ---
    n_angle: int                     # bins over [0, pi] for the three feature angles
    n_alpha: int                     # bins over [0, 2pi) for the accumulator's alpha axis

    # --- pose clustering / verification ---
    cluster_pos_tol: float           # (m)
    cluster_ang_tol_deg: float
    verify_tol: float                # inlier distance for scoring a pose (m)
    # Verification also requires normal agreement. Proximity alone cannot rank hypotheses on
    # a rounded or symmetric part -- a pose with the axis tilted wrong still drapes surface
    # near every scene point. Same value and same reason as
    # ``AmbiguityConfig.normal_cos_tol``: ~45 degrees, loose enough to survive normal
    # estimation noise, tight enough to reject a face pointing the other way.
    verify_cos_tol: float = 0.70

    # --- algorithm toggles, each individually ablatable ---
    vote_dedup: bool = True          # Hinterstoisser: one vote per scene point per cell
    spread_angle_bins: bool = True   # tolerate a correspondence falling in an adjacent bin
    steep_pair_readmit_deg: float = 30.0   # re-admit sub-min_pair_dist pairs this divergent

    # Cap on entries stored per feature bin. NOT redundant with `vote_dedup`: dedup fixes
    # planar-region bias but runs after expansion, so it does nothing about cost. Entries are
    # strided, never truncated. See registration/CLAUDE.md.
    max_bucket_entries: int = 256

    # --- policy (shared across all parts, not per-part) ---
    model_target_points: int = 500
    accept_score: float = 0.0        # minimum pose score to report; 0 = report the best

    provenance: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def derive(cls,
               model,
               sensor: Optional[SensorProfile] = None,
               model_target_points: int = 500,
               min_feature_size: Optional[float] = None,
               spacing_factor: float = 3.0,
               max_tau_frac: float = 0.15,
               min_angle_bin_deg: float = 5.0,
               work_budget: int = 40_000_000,
               **overrides) -> "PPFConfig":
        """Derive every parameter from the model cloud plus a sensor calibration.

        ``tau`` is the one that matters, because cost scales as its inverse *square* on both
        the model and the scene side.  Measured on the bunny at 100 instances: 3.6 M scene
        pair-evaluations at tau = 8% of diameter, but 305 M at 2%.  So tau cannot simply be
        "as fine as the sensor allows" -- it is bracketed:

            lower   sensor/cloud resolution; below it the bins quantise noise
            upper   the smallest discriminative feature (if known), else a fraction of D
            target  whatever yields ``model_target_points``, i.e. the compute budget

        The target is found by **bisecting on the actual voxel-downsampled point count**,
        not from a surface-area estimate.  The obvious closed form, ``tau = sqrt(SA/M)`` with
        ``SA ~ N*s^2``, is wrong by a large constant: for a Poisson-sampled cloud the median
        nearest-neighbour distance is ``0.4697/sqrt(density)``, not ``1/sqrt(density)``, so
        that formula understates area by ~4.5x -- measured 4.63x on a box whose true area is
        known.  Since ``M`` scales as ``tau^-2`` and cost as ``M^2``, a 2x error in tau is a
        ~20x error in work.
        Bisecting on the real count is regime-independent -- it does not care whether the
        cloud is Poisson-sampled, gridded, or already decimated.

        In practice the compute budget binds and the bounds are slack, which is the honest
        situation: a good sensor over-resolves these parts, so the limit is what we can
        afford to vote with, not what we can see.  ``provenance`` records which one bound.

        Parameters
        ----------
        model : PointCloud / TriangleMesh / (N,3) array -- the reference model
        min_feature_size : smallest feature that must stay resolved (m).  Optional; when it
            is not supplied the upper bound falls back to ``max_tau_frac * diameter``, which
            is a coverage bound rather than a feature bound.
        """
        from geometry.geom_utils import median_spacing, model_diameter

        sensor = sensor or SensorProfile()
        pts = np.asarray(getattr(model, "points", model), dtype=float).reshape(-1, 3)

        diameter = model_diameter(pts)
        spacing = median_spacing(pts)
        prov: Dict[str, str] = {}

        # ---- tau ------------------------------------------------------
        lower = max(spacing_factor * spacing, spacing_factor * sensor.sigma_lateral)
        upper = float(min_feature_size) if min_feature_size else max_tau_frac * diameter
        target = _tau_for_point_count(pts, model_target_points,
                                      max(lower, 1e-6), max(upper, 2e-6))
        if upper < lower:
            # The part's discriminative detail is finer than the sensor can see. Clamping to
            # the sensor floor is the only honest option, but the caller needs to know that
            # this part is at the edge of what the hardware can resolve at all.
            prov["tau"] = (f"UNDER-RESOLVED: min feature {upper * 1e3:.2f}mm is below the "
                           f"sensor floor {lower * 1e3:.2f}mm; clamped to the floor")
            tau = lower
        else:
            tau = float(np.clip(target, lower, upper))
            prov["tau"] = ("compute budget" if lower <= target <= upper else
                           "sensor/resolution floor" if target < lower else "feature-size cap")

        # ---- angular binning ------------------------------------------
        # A normal estimated over radius r from depth with noise sigma_z tilts by about
        # atan(sigma_z / r); the bin has to be wider than that or a correct correspondence
        # falls outside its own bin.
        normal_radius = 2.0 * tau
        sigma_theta_deg = float(np.degrees(np.arctan2(sensor.sigma_z(), normal_radius)))
        noise_bin = 2.0 * sigma_theta_deg
        angle_bin_deg = max(noise_bin, min_angle_bin_deg)
        prov["angle_bin"] = (f"noise-implied {noise_bin:.2f}deg"
                             if noise_bin >= min_angle_bin_deg else
                             f"cost floor {min_angle_bin_deg:.1f}deg "
                             f"(noise implied only {noise_bin:.2f}deg)")

        cfg = cls(
            diameter=diameter,
            tau=tau,
            min_pair_dist=tau,
            max_pair_dist=diameter,
            n_angle=int(round(180.0 / angle_bin_deg)),
            n_alpha=int(round(360.0 / angle_bin_deg)),
            cluster_pos_tol=tau,
            cluster_ang_tol_deg=2.0 * angle_bin_deg,
            # Half tau, floored at 3 sigma (the same convention AmbiguityConfig.
            # epsilon_floor_m uses). A full tau is the *binning* resolution, which is far
            # too loose to rank hypotheses: at that tolerance a wrongly-tilted pose covers
            # the scene points about as well as the right one. Halving it, together with the
            # normal test, is what moved top-1 selection from 0.75 to 1.00 on T-LESS
            # obj_000017 and 0.28 to 0.40 on obj_000013.
            verify_tol=max(3.0 * sensor.sigma_z(), 0.5 * tau),
            # Work per instance is roughly (scene pairs) x (mean bucket) ~ M^2 * bucket, so
            # the cap follows from a per-instance vote budget. Hardware policy, shared by
            # every part -- not something to tune per part.
            max_bucket_entries=int(np.clip(
                work_budget // max(model_target_points ** 2, 1), 64, 1024)),
            model_target_points=model_target_points,
            provenance=prov,
        )
        return replace(cfg, **overrides) if overrides else cfg

    # ------------------------------------------------------------------
    def describe(self) -> str:
        return "\n".join([
            f"  diameter      {self.diameter * 1e3:8.2f} mm",
            f"  tau           {self.tau * 1e3:8.2f} mm   ({self.tau / self.diameter:.3%} of D)"
            f"   [{self.provenance.get('tau', '-')}]",
            f"  pair dist     {self.min_pair_dist * 1e3:.2f} .. {self.max_pair_dist * 1e3:.2f} mm",
            f"  n_angle       {self.n_angle:8d}     ({180.0 / self.n_angle:.2f} deg/bin)"
            f"   [{self.provenance.get('angle_bin', '-')}]",
            f"  n_alpha       {self.n_alpha:8d}     ({360.0 / self.n_alpha:.2f} deg/bin)",
            f"  cluster tol   {self.cluster_pos_tol * 1e3:.2f} mm / {self.cluster_ang_tol_deg:.1f} deg",
            f"  verify tol    {self.verify_tol * 1e3:8.2f} mm",
            f"  max bucket    {self.max_bucket_entries:8d}",
            f"  dedup={self.vote_dedup}  spread={self.spread_angle_bins}",
        ])


def _voxel_count(pts: np.ndarray, tau: float) -> int:
    """Points surviving a voxel downsample at ``tau``. Cheap enough to bisect on.

    Uses Open3D's binning so the bisection converges on the count the matcher will actually
    get -- ``np.unique(axis=0)`` on floor-divided keys anchors the grid at the world origin
    while Open3D anchors it at the cloud's own bounding box, and those disagree by a couple
    of points. Small, but it is free to just measure the real thing.
    """
    import open3d as o3d

    if tau <= 0:
        return len(pts)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    return len(pcd.voxel_down_sample(tau).points)


def _tau_for_point_count(pts: np.ndarray, target: int, lo: float, hi: float,
                         iters: int = 20) -> float:
    """Bisect ``tau`` so a voxel downsample yields about ``target`` points.

    The count is monotonically decreasing in ``tau``, so plain bisection converges. If even
    the coarsest allowed ``tau`` still leaves more than ``target`` points the part simply
    cannot be represented that compactly within its bounds, and the caller's clamp reports
    which bound bound.
    """
    if _voxel_count(pts, hi) >= target:
        return hi
    if _voxel_count(pts, lo) <= target:
        return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _voxel_count(pts, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
