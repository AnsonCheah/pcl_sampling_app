"""View-dependent ambiguity analysis for CAD matching.

A part flips during matching when the surface patch visible from one viewpoint fits the
model in more than one place.  Global symmetry is the special case where *every* viewpoint
is ambiguous, so this module detects both with one mechanism — there is no separate
"symmetric part" code path.

The search is restricted to **rotations about an arbitrary axis** (direction ``a``, point
``p``, angle ``theta``) rather than general SE(3) — exactly the family MechVision's
``rotationStrategy`` + ``angleStep`` can act on.  It is Mitra/Guibas/Pauly
transformation-space voting (SIGGRAPH 2006) specialised to rotations, in three stages:

  A. propose axis *directions* (``_candidate_directions``)
  B. vote the axis *point* in the plane perpendicular to each direction, with the angle
     fixed, where ``(I - R_theta)`` is invertible and the centre follows in closed form
     (``_vote_centres``)
  C. verify every candidate against the cloud, then locally refine direction and position
     against that same measure (``_explains``, ``_refine_axis``)

Because ``p`` is voted for rather than assumed, the recovered axis need not pass through
the centroid — which is the whole point, since a feature's symmetry axis generally does
not.  On a cylinder joined to an off-axis block, the axis is recovered to 0.4 degrees and
0.11 mm while sitting 7.8 mm from the centroid and 19 mm from the AABB centre.

Two formulations were tried and rejected; both fail on precisely the parts this exists for.

  * Deriving the axis from a single oriented point pair via ``a ~ (n_v - n_w) x (v - w)``.
    One oriented pair gives 5 constraints on a 5-DOF rotation, so *every* pair yields a
    valid rotation and no pair-level filter exists.  Worse, on a surface of revolution the
    radial displacement is parallel to the normal difference, so the cross product
    vanishes exactly on the true correspondences: the construction is degenerate on the
    symmetric case.
  * Ranking directions by the autocorrelation of the projected normal-angle histogram.
    Vacuous for 2-fold symmetry — any closed body has its faces in +/- normal pairs, so
    that histogram is 180-degree symmetric about every direction.  Measured on a
    100x30x20 box, an arbitrary tilted direction scored 0.9997 against 0.9989 / 0.9894 /
    0.9992 for the three true axes.

Hence stage A generates rather than ranks, and verification does the filtering.

All tolerances are relative to the part so a 20 mm and a 500 mm part behave the same; the
only absolute floors are sensor-physics ones (see ``AmbiguityConfig``).

Base layer: imports nothing from sensor/, physics/ or registration/.  Visibility is raycast
directly against a ``RaycastingScene`` built once and reused, which also avoids the
per-call BVH rebuild in ``sensor.scene_render()``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

from geometry.geom_utils import (
    camera_view_matrix,
    fibonacci_sphere,
    rotation_aligning_vector_to_axis,
)

__all__ = [
    "AmbiguityConfig",
    "AmbiguityAxis",
    "ViewAmbiguity",
    "AmbiguityProfile",
    "analyse_ambiguity",
    "rank_axes",
    "save_ambiguity_profile",
    "load_ambiguity_profile",
    "per_point_path",
    "reclassify_global",
    "heat_colour",
    "discriminative_colours",
    "ambiguity_geometries",
]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AmbiguityConfig:
    """Tolerances for the ambiguity search.

    Everything that scales with the part is expressed as a fraction of the part diameter
    or of the visible point count.  The absolute floors exist for sensor-physics reasons,
    not as size rules: geometric agreement finer than sensor noise is not meaningful, and
    a feature the sensor cannot resolve cannot disambiguate however large it is relative
    to the part.
    """

    # Viewpoint sweep
    n_views: int = 200
    res: int = 256

    # Direction candidates (stage A)
    n_directions: int = 200         # Fibonacci hemisphere probes, used to fill remaining slots
    max_directions: int = 48        # directions handed to stage B
    max_normal_clusters: int = 16   # dominant surface normals used to seed structured axes
    dir_dedup_dot: float = 0.995    # ~5.7 deg
    # Angles stage B votes with: the union over folds 2,3,4,5,6,8 plus intermediate probes
    # so a continuous axis is recognised too.  Verification does the filtering, so recall
    # matters far more than the cost of a few extra angles.
    probe_angles: Tuple[float, ...] = (
        180.0, 120.0, 240.0, 90.0, 270.0, 72.0, 144.0, 216.0, 288.0,
        60.0, 300.0, 45.0, 135.0, 225.0, 315.0,
    )

    # Axis refinement (and the screen that decides what is worth refining)
    screen_min_area: float = 0.30   # pre-refinement bar on best-per-view coverage
    max_refine: int = 24            # cap on how many candidates get polished
    refine_iters: int = 40
    refine_dir_step_deg: float = 4.0
    refine_pt_step_frac: float = 0.02
    refine_subsample: int = 3000

    # Centre voting (stage B)
    vote_subsample: int = 1200
    curvature_k: int = 12           # kNN for the local descriptor
    curvature_tol: float = 0.40     # relative descriptor agreement required to pair
    theta_bin_deg: float = 6.0
    point_bin_frac: float = 0.02    # axis-point bin, fraction of diameter
    max_candidates: int = 50
    min_votes: int = 8

    # Verification.  The tolerance is set by the cloud's own resolution, not by the part
    # size: a rotated point lands *between* samples, so anything below the sampling pitch
    # reports missing symmetry that is really missing resolution.  The factor is the
    # covering radius of the sample, not its nearest-neighbour distance -- calibrated on
    # exact symmetries of cylinder/torus/box/sphere, which explain 0.998-0.999 at 3.0
    # while non-symmetry rotations of the same parts stay at 0.20-0.31.
    epsilon_spacing_factor: float = 3.0   # multiples of the median nearest-neighbour spacing
    epsilon_floor_m: float = 0.0005       # sensor-physics floor (~3*sigma_depth)
    normal_cos_tol: float = 0.70          # a point must also agree in normal direction
    global_area_frac: float = 0.95        # coverage above which an axis preserves the whole model
    # ...OR the axis is ambiguous from essentially every viewpoint.
    #
    # Surface coverage alone is too strict for real manufactured parts.  Validated against
    # BOP's published symmetry annotations for the 30 T-LESS objects: 10 parts BOP calls
    # symmetric were missed on coverage alone, their best area_fraction spanning 0.81-0.95
    # (a symmetric body with a small boss, hole or chamfer breaking exact surface agreement
    # at the 5-20% level).  Every one of those 10 had view_fraction == 1.00 and the correct
    # fold, so the axis was recovered perfectly and only the *label* was wrong.
    #
    # view_fraction is the more direct measure of what "global" has to mean operationally:
    # an axis that makes every viewpoint ambiguous will flip the matcher from anywhere,
    # whether or not the last few percent of the surface agrees.
    global_view_frac: float = 0.98
    # ...but that clause on its own is too permissive, so it carries a coverage floor.
    #
    # Without one, the three BOP-asymmetric T-LESS objects all get promoted to global: they
    # too have an axis ambiguous from every viewpoint, explaining 0.71-0.80 of the surface.
    # They are *nearly* symmetric, and a matcher looking at partial views really will flip
    # them -- BOP calls them asymmetric because with the whole model in hand the poses are
    # separable.
    #
    # Be aware how thin the separation is: across the 30 objects, symmetric parts bottom out
    # at area_fraction 0.809 and asymmetric ones top out at 0.803. A 0.006 gap is not a
    # natural boundary, it is a continuum of "how nearly symmetric", and this threshold is
    # calibrated on 30 samples sitting either side of it. It gets 29/30 here; do not read it
    # as a law. Where a published annotation exists, prefer it -- `bench/metrics.py` already
    # does, and only falls back to this classification for parts BOP has never seen.
    global_view_area_floor: float = 0.80

    # Per-view survival.
    #
    # This asks "would the matcher plausibly land on this pose", NOT BOP-Distrib's "is
    # this pose provably indistinguishable".  Their tau (~28 points, well under 1% of the
    # model) answers the second question; using it here rejects every real case, because
    # ambiguity on a manufactured part is partial -- a large chunk of the visible patch
    # aligns elsewhere and the remainder does not, which is exactly the pose a matcher
    # scoring by inlier fraction will happily return.
    #
    # Calibrated on 25333MB000: 300 random rotations reach a best-per-view explained
    # fraction of p50 0.033, p99 0.273, max 0.315, and NONE reach 0.60; the genuine
    # ambiguity axes reach 0.709-0.735.  Requiring 60% explained sits in the gap.
    f_tau: float = 0.40             # unexplained fraction of the visible patch allowed
    n_floor: int = 4                # sensor-resolution floor on disambiguating evidence

    # Axis grouping / fold fitting
    axis_dir_tol_deg: float = 6.0
    axis_pt_tol_frac: float = 0.03
    fold_candidates: Tuple[int, ...] = (2, 3, 4, 6, 8, 12)
    continuous_probes: int = 5
    # An axis is "significant" if it affects enough viewpoints to matter. This is a
    # frequency test on view_fraction, deliberately independent of the ranking score, so
    # retuning the exponent below cannot silently change how many axes a part is said to
    # have.
    significant_score: float = 0.05

    # Ranking.  score = view_fraction * area_fraction ** rank_area_exponent, estimating
    # P(the matcher returns this wrong pose):
    #     view_fraction  -- how often the ambiguity is geometrically available
    #     area_fraction  -- how much of the WHOLE model still coincides, i.e. whether the
    #                       hypothesis survives verification against the unseen part
    # The exponent models the shape of that second term.  k=0 means verification ignores
    # the unseen model (equivalent to onlyConsiderVisibleSurfaceOfModel=True); k=1 means
    # a linear fit score; k>1 reflects verification being a *threshold*, which collapses
    # acceptance faster than linearly once overlap drops below it.  A logistic centred on
    # the effective threshold would be the honest form; this is a one-knob approximation.
    #
    # Calibrated against observed behaviour on 25333MB000, where only the off-centroid
    # disc axis produces flips in real scenes: it leads for k > 1.40, and at k = 2.0 leads
    # by 32%.  Ranking is a pure function of the stored per-axis metrics, so `rank_axes`
    # can retune this without re-running the analysis.
    rank_area_exponent: float = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AmbiguityAxis:
    direction: np.ndarray            # (3,) unit
    point: np.ndarray                # (3,) a point ON the axis — NOT a centroid
    fold: int                        # 2,3,4,6,...; 0 = continuous
    angles_deg: List[float]          # supported rotation angles
    is_global: bool                  # preserves the whole model, not just some views
    view_fraction: float             # fraction of viewpoints made ambiguous
    area_fraction: float             # fraction of the whole surface involved
    # How completely the visible patch is explained where this axis bites.  Reported so
    # ambiguity strength is visible rather than collapsed into the survival flag: the
    # random-rotation noise ceiling is ~0.32, genuine ambiguity on 25333MB000 is ~0.73.
    patch_fraction: float = 0.0      # median over the views where it survives
    best_patch_fraction: float = 0.0 # best single view
    # = view_fraction * area_fraction ** rank_area_exponent; set by `rank_axes`, which is
    # also what orders `AmbiguityProfile.axes`. Do not read it as a frequency.
    score: float = 0.0

    def angle_step_deg(self) -> float:
        """The MechVision ``angleStep`` implied by this axis."""
        if self.fold == 0:
            return 0.0                       # continuous: caller picks from a small-step set
        return 360.0 / self.fold

    def transformed(self, T: np.ndarray) -> "AmbiguityAxis":
        """Express this axis in a frame related by the 4x4 ``T``.

        Needed because the profile is computed before the cloud is recentred but consumed
        after: an axis left in the pre-recentre frame would send MechVision's rotation
        search to the wrong line.
        """
        R = np.asarray(T)[:3, :3]
        t = np.asarray(T)[:3, 3]
        d = R @ self.direction
        d = d / (np.linalg.norm(d) + 1e-12)
        p = R @ self.point + t
        return replace(self, direction=d, point=p - float(p @ d) * d)


@dataclass
class ViewAmbiguity:
    direction: np.ndarray
    n_visible: int
    n_transforms: int                # surviving non-identity transforms for this view
    discriminative_fraction: float


@dataclass
class AmbiguityProfile:
    axes: List[AmbiguityAxis] = field(default_factory=list)
    dominant: Optional[AmbiguityAxis] = None
    n_significant_axes: int = 0
    discriminative_fraction: float = 1.0
    per_point_discriminative: np.ndarray = field(default_factory=lambda: np.empty(0))
    per_view: List[ViewAmbiguity] = field(default_factory=list)
    frame_changed: bool = False
    ppf_degeneracy: Dict[str, float] = field(default_factory=dict)
    epsilon_m: float = 0.0
    f_tau: float = 0.0
    diameter_m: float = 0.0
    rank_area_exponent: float = 2.0

    @property
    def is_ambiguous(self) -> bool:
        return self.dominant is not None

    def transformed(self, T: np.ndarray) -> "AmbiguityProfile":
        """Re-express the axes and viewpoint directions in a frame related by ``T``.

        Per-point scores are attached to points and travel with the cloud unchanged; the
        axes and the view directions are geometric and must be rotated.
        """
        R = np.asarray(T)[:3, :3]
        moved = [ax.transformed(T) for ax in self.axes]
        # Identity, not equality: AmbiguityAxis holds arrays, so `==` is ambiguous.
        idx = next((i for i, a in enumerate(self.axes) if a is self.dominant), None)
        views = [replace(v, direction=R @ v.direction) for v in self.per_view]
        return replace(self, axes=moved,
                       dominant=moved[idx] if idx is not None else None,
                       per_view=views)


# ─────────────────────────────────────────────────────────────────────────────
# Visibility sweep
# ─────────────────────────────────────────────────────────────────────────────

def _visibility_masks(mesh: o3d.geometry.TriangleMesh,
                      pts: np.ndarray,
                      cfg: AmbiguityConfig,
                      progress_cb: Optional[Callable[[float], None]] = None,
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Raycast the mesh from ``cfg.n_views`` directions; report which cloud points are seen.

    Orthographic rays are used rather than a pinhole frustum: visibility analysis wants
    uniform surface sampling and no field-of-view coverage tuning, and orthographic
    projection makes the sweep deterministic and scale-free.  ``camera_view_matrix`` still
    supplies the right/up basis so the view frames match the rest of the pipeline.

    Returns
    -------
    (visible, view_dirs) : (n_views, N) bool array and (n_views, 3) directions
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    centre = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    radius = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) * 0.5
    span = radius * 1.05
    distance = radius * 4.0

    view_dirs = fibonacci_sphere(cfg.n_views)
    tree = cKDTree(pts)
    # A hit only counts for the nearest cloud point if it is genuinely that point's
    # surface; scale with the cloud's own spacing so this is resolution-independent.
    snap_radius = max(2.0 * radius / cfg.res, 1e-6)

    grid = np.linspace(-span, span, cfg.res)
    uu, vv = np.meshgrid(grid, grid, indexing="ij")
    uu = uu.ravel()
    vv = vv.ravel()

    visible = np.zeros((cfg.n_views, len(pts)), dtype=bool)
    for k, d in enumerate(view_dirs):
        T = camera_view_matrix(centre + d * distance, centre)
        right, up = T[:3, 0], T[:3, 1]
        origins = (centre + d * distance)[None, :] + uu[:, None] * right + vv[:, None] * up
        dirs = np.repeat((-d)[None, :], len(origins), axis=0)

        rays = o3d.core.Tensor(np.hstack([origins, dirs]).astype(np.float32))
        ans = scene.cast_rays(rays)
        t_hit = ans["t_hit"].numpy()
        hit = np.isfinite(t_hit)
        if not hit.any():
            continue
        xyz = origins[hit] + t_hit[hit, None] * dirs[hit]

        dist, idx = tree.query(xyz, k=1)
        visible[k, idx[dist < snap_radius]] = True

        if progress_cb is not None:
            progress_cb(0.35 * (k + 1) / cfg.n_views)

    return visible, view_dirs


# ─────────────────────────────────────────────────────────────────────────────
# Rotation-axis voting
# ─────────────────────────────────────────────────────────────────────────────

def _local_descriptor(pts: np.ndarray, k: int) -> np.ndarray:
    """PCA surface variation lambda0 / sum(lambda) per point — cheap rotation-invariant signature."""
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)))
    nbrs = pts[idx]                                    # (N, k, 3)
    centred = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centred, centred) / max(idx.shape[1] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)                  # ascending
    total = eigvals.sum(axis=1)
    return np.where(total > 1e-18, eigvals[:, 0] / np.maximum(total, 1e-18), 0.0)


def _basis(d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal pair spanning the plane perpendicular to ``d``."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ref, d)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


def _candidate_directions(normals: np.ndarray,
                          pts: np.ndarray,
                          cfg: AmbiguityConfig) -> np.ndarray:
    """Propose rotation-axis *directions* to hand to the centre vote.

    This deliberately does **not** try to rank directions by a normal-histogram
    criterion.  That test is vacuous for 2-fold symmetry: any closed body has its faces
    in +/- normal pairs, so the projected normal-angle histogram is 180-degree symmetric
    about *every* direction.  Measured on a 100x30x20 box, the exact axes score 0.9989 /
    0.9894 / 0.9992 while an arbitrary tilted direction scores 0.9997 — the criterion
    ranks a bogus direction above all three true ones.

    So generate, don't rank.  Structured directions come first because the rotation axes
    of manufactured parts lie along dominant surface normals, their cross products, or
    the PCA axes; the Fibonacci grid fills the remaining slots for freeform shapes.
    Verification is what filters, and ``_refine_axis`` cleans up the residual tilt.
    """
    out: List[np.ndarray] = []

    def add(v: np.ndarray) -> None:
        n = float(np.linalg.norm(v))
        if n < 1e-9 or len(out) >= cfg.max_directions:
            return
        v = v / n
        if v[np.argmax(np.abs(v))] < 0:            # d and -d are the same axis
            v = -v
        for k in out:
            if abs(float(np.dot(v, k))) > cfg.dir_dedup_dot:
                return
        out.append(v)

    for v in np.linalg.eigh(np.cov((pts - pts.mean(axis=0)).T))[1].T:
        add(v)

    clusters: List[np.ndarray] = []
    for n in normals:
        if all(abs(float(np.dot(c, n))) <= 0.985 for c in clusters):
            clusters.append(n / (np.linalg.norm(n) + 1e-12))
        if len(clusters) >= cfg.max_normal_clusters:
            break
    for c in clusters:
        add(c)
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            add(np.cross(clusters[i], clusters[j]))

    for d in fibonacci_sphere(cfg.n_directions):
        add(d)

    return np.asarray(out)


def _refine_axis(pts: np.ndarray,
                 normals: np.ndarray,
                 tree: cKDTree,
                 direction: np.ndarray,
                 point: np.ndarray,
                 angle_deg,
                 epsilon: float,
                 diameter: float,
                 cfg: AmbiguityConfig,
                 rng: np.random.Generator,
                 ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Local search on (direction, point) maximising the explained fraction.

    The candidate generator only has to get within its grid resolution; this is what
    turns "roughly the right axis" into an axis accurate enough to build a model frame
    from.  Both the direction and the axis *position* are refined, so an axis that misses
    the centroid converges just as tightly as one through it.

    ``angle_deg`` may be a sequence, in which case the score is the **worst** angle rather
    than the mean.  That matters for continuous axes: optimising at one angle leaves a
    residual tilt that still scores well at that angle but degrades at others, and the
    fold fit then reads the degradation as a finite fold.  Measured on a cylinder joined
    to an off-axis block, the exact axis explains 0.76 at every angle (spread 0.002) while
    a 3.6 degree tilt swings 0.51-0.76 and is fitted as a spurious C3.
    """
    angles = [float(angle_deg)] if np.isscalar(angle_deg) else [float(a) for a in angle_deg]

    if len(pts) > cfg.refine_subsample:
        sel = rng.choice(len(pts), cfg.refine_subsample, replace=False)
        sub, sub_n = pts[sel], normals[sel]
    else:
        sub, sub_n = pts, normals

    def score(d: np.ndarray, p: np.ndarray) -> float:
        worst = 1.0
        for a in angles:
            R, t = _rotation_about(d, p, a)
            worst = min(worst, float(_explains(sub, sub_n, tree, normals, R, t,
                                               epsilon, cfg.normal_cos_tol).mean()))
        return worst

    d = direction / np.linalg.norm(direction)
    p = point - float(point @ d) * d
    best = score(d, p)

    ang_step = np.deg2rad(cfg.refine_dir_step_deg)
    pt_step = cfg.refine_pt_step_frac * diameter
    for _ in range(cfg.refine_iters):
        improved = False
        e1, e2 = _basis(d)
        for axis in (e1, e2):
            for sgn in (1.0, -1.0):
                cand = Rot.from_rotvec(axis * (sgn * ang_step)).as_matrix() @ d
                cand /= np.linalg.norm(cand)
                s = score(cand, p - float(p @ cand) * cand)
                if s > best + 1e-9:
                    best, d = s, cand
                    p = p - float(p @ d) * d
                    improved = True
        for axis in (e1, e2):
            for sgn in (1.0, -1.0):
                cand = p + axis * (sgn * pt_step)
                cand = cand - float(cand @ d) * d
                s = score(d, cand)
                if s > best + 1e-9:
                    best, p = s, cand
                    improved = True
        if not improved:
            ang_step *= 0.5
            pt_step *= 0.5
            if ang_step < np.deg2rad(0.05) and pt_step < diameter * 1e-4:
                break
    return d, p, best


def _vote_centres(u: np.ndarray,
                  z: np.ndarray,
                  psi: np.ndarray,
                  desc: np.ndarray,
                  angles_deg: List[float],
                  diameter: float,
                  cfg: AmbiguityConfig,
                  ) -> List[Tuple[np.ndarray, float, int]]:
    """Vote for the axis *point* in the plane perpendicular to a known direction.

    With the rotation angle already fixed by stage A, ``(I - R_theta)`` is invertible in
    2D, so each candidate correspondence yields the rotation centre in closed form:

        p = (I - R_theta)^-1 (u_j - R_theta u_i)

    There is no degeneracy here — which is the point of splitting direction from centre.
    The centre is voted, never assumed, so an axis that misses the centroid is found
    exactly as easily as one through it.

    Returns ``(centre_2d, angle_deg, n_votes)``.
    """
    m = len(u)
    ii, jj = np.triu_indices(m, k=1)

    # A correspondence must sit at the same height along the axis and have matching
    # local geometry; both are angle-independent so they are computed once.
    axial_tol = max(cfg.point_bin_frac * diameter, 1e-9)
    ok = np.abs(z[ii] - z[jj]) < axial_tol
    dd = np.abs(desc[ii] - desc[jj])
    sc = np.maximum(np.maximum(desc[ii], desc[jj]), 1e-6)
    ok &= dd / sc < cfg.curvature_tol
    if not ok.any():
        return []
    ii, jj = ii[ok], jj[ok]

    dpsi = np.rad2deg(psi[jj] - psi[ii]) % 360.0
    bin_m = max(cfg.point_bin_frac * diameter, 1e-9)
    out: List[Tuple[np.ndarray, float, int]] = []

    for ang in angles_deg:
        # The in-plane normal rotation must equal the rotation angle — a strong, cheap
        # correspondence test that costs one comparison per pair.
        sel = np.abs((dpsi - ang + 180.0) % 360.0 - 180.0) < cfg.theta_bin_deg
        if sel.sum() < cfg.min_votes:
            continue
        a_i, a_j = ii[sel], jj[sel]
        th = np.deg2rad(ang)
        c, s = np.cos(th), np.sin(th)
        denom = 2.0 - 2.0 * c
        if denom < 1e-9:
            continue
        rx = c * u[a_i, 0] - s * u[a_i, 1]
        ry = s * u[a_i, 0] + c * u[a_i, 1]
        bx, by = u[a_j, 0] - rx, u[a_j, 1] - ry
        px = ((1.0 - c) * bx - s * by) / denom
        py = (s * bx + (1.0 - c) * by) / denom

        keys = np.stack([np.rint(px / bin_m), np.rint(py / bin_m)], axis=1).astype(np.int64)
        uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        for b in np.argsort(-counts)[:3]:
            if counts[b] < cfg.min_votes:
                break
            mem = inv == b
            out.append((np.array([px[mem].mean(), py[mem].mean()]), ang, int(counts[b])))
    return out


def _vote_rotation_axes(pts: np.ndarray,
                        normals: np.ndarray,
                        diameter: float,
                        cfg: AmbiguityConfig,
                        rng: np.random.Generator,
                        progress_cb: Optional[Callable[[float], None]] = None,
                        ) -> List[Tuple[np.ndarray, np.ndarray, float, int]]:
    """Two-stage candidate generation.

    Stage A fixes the axis *direction* from the normal distribution (centre-free);
    stage B votes the axis *point* with the angle known.  Recall matters far more than
    precision here — every candidate is verified against the full cloud afterwards.

    Returns ``(direction, point, angle_deg, n_votes)`` candidates, best first.
    """
    n = len(pts)
    if n > cfg.vote_subsample:
        sel = rng.choice(n, cfg.vote_subsample, replace=False)
        sel.sort()
    else:
        sel = np.arange(n)
    v = pts[sel]
    nv = normals[sel]
    desc = _local_descriptor(v, cfg.curvature_k)
    m = len(v)

    dirs = _candidate_directions(normals, pts, cfg)
    if progress_cb is not None:
        progress_cb(0.45)
    if len(dirs) == 0:
        return []

    angles = list(cfg.probe_angles)
    out: List[Tuple[np.ndarray, np.ndarray, float, int]] = []
    for k, d in enumerate(dirs):
        e1, e2 = _basis(d)
        u = np.stack([v @ e1, v @ e2], axis=1)
        z = v @ d
        perp = nv - np.outer(nv @ d, d)
        psi = np.arctan2(perp @ e2, perp @ e1)

        for centre2d, ang, votes in _vote_centres(u, z, psi, desc, angles, diameter, cfg):
            out.append((d, centre2d[0] * e1 + centre2d[1] * e2, ang, votes))

        if progress_cb is not None:
            progress_cb(0.45 + 0.15 * (k + 1) / len(dirs))

    out.sort(key=lambda c: -c[3])
    return out[: cfg.max_candidates]


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_about(direction: np.ndarray, point: np.ndarray, angle_deg: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid rotation about an arbitrary axis -> (R, t) with x' = R x + t."""
    R = Rot.from_rotvec(np.asarray(direction) * np.deg2rad(angle_deg)).as_matrix()
    return R, point - R @ point


def _explains(pts: np.ndarray,
              normals: np.ndarray,
              tree: cKDTree,
              tree_normals: np.ndarray,
              R: np.ndarray,
              t: np.ndarray,
              epsilon: float,
              normal_cos_tol: float) -> np.ndarray:
    """Per-point mask: does the transform map this query point back onto the surface?

    ``pts``/``normals`` are the query set (which may be a subsample); ``tree`` and
    ``tree_normals`` describe the full surface being matched against, so the two must be
    kept separate — indices returned by the tree address ``tree_normals``.

    Geometry alone is not enough: a point can land near the surface with an inverted
    normal (a thin plate maps onto its own back face).  Normal agreement is required too.
    """
    moved = pts @ R.T + t
    dist, idx = tree.query(moved, k=1)
    ok = dist < epsilon
    moved_n = normals @ R.T
    cos = np.einsum("ij,ij->i", moved_n, tree_normals[idx])
    return ok & (cos > normal_cos_tol)


# ─────────────────────────────────────────────────────────────────────────────
# Fold fitting and grouping
# ─────────────────────────────────────────────────────────────────────────────

def _is_global(view_fraction: float, area_fraction: float, cfg: AmbiguityConfig) -> bool:
    """Does this axis map the whole model onto itself, for practical purposes?

    Either it preserves nearly all the surface, **or** it is ambiguous from essentially
    every viewpoint. See ``AmbiguityConfig.global_view_frac`` for why the second clause is
    needed — coverage alone missed 10 of the 27 symmetric T-LESS parts.
    """
    return bool(area_fraction > cfg.global_area_frac
                or (view_fraction >= cfg.global_view_frac
                    and area_fraction >= cfg.global_view_area_floor))


def reclassify_global(profile: "AmbiguityProfile",
                      cfg: Optional[AmbiguityConfig] = None) -> "AmbiguityProfile":
    """Recompute ``is_global`` in place from each axis's stored metrics.

    Both inputs are already persisted per axis, so the classification can be revised on a
    saved profile without re-running the analysis — which costs minutes per part. Same
    reasoning as ``rank_axes``: keep anything derivable from stored metrics re-derivable.
    """
    cfg = cfg or AmbiguityConfig()
    for ax in profile.axes:
        ax.is_global = _is_global(ax.view_fraction, ax.area_fraction, cfg)
    return profile


def rank_axes(profile: "AmbiguityProfile",
              area_exponent: float = 2.0,
              significant_score: float = 0.05,
              tie_frac: float = 0.05) -> "AmbiguityProfile":
    """(Re)rank a profile's axes in place and pick the dominant one.

    Ranking is a pure function of three already-stored per-axis numbers, so the exponent
    can be retuned on a saved profile without re-running the analysis — which takes
    minutes on a large cloud.

        score = view_fraction * area_fraction ** area_exponent

    See ``AmbiguityConfig.rank_area_exponent`` for what the exponent means.  Significance
    stays a frequency test on ``view_fraction`` so changing the exponent cannot alter how
    many axes a part is reported to have.
    """
    for ax in profile.axes:
        ax.score = float(ax.view_fraction * ax.area_fraction ** area_exponent)

    profile.axes.sort(key=lambda ax: -ax.score)

    # Score decides the order.  The fold preference is a *tie-break only*, and only among
    # GLOBAL axes -- it says "of two axes that are equally likely to be returned, prefer
    # the one that costs more to get wrong", which is a statement about a symmetry group:
    # a hex prism's C2 and C6 axes are both global and score within a thousandth of each
    # other, and reporting C2/180deg for a part that needs C6/60deg leaves two thirds of
    # the ambiguity unmitigated.  A cylinder's continuous axis beats its perpendicular C2s
    # the same way.
    #
    # It must NOT reach view-dependent axes.  Their scores are two orders of magnitude
    # smaller (0.01-0.03, not ~1.0), so the old absolute `round(score, 2)` tie window put
    # every one of them in a single bucket and silently promoted fold to the primary sort
    # key.  Measured on 25333MB000: six axes scoring 0.0017-0.0172, the highest being the
    # off-centroid disc axis at fold 1 -- which was demoted to rank 2 behind two C2 axes
    # scoring LESS, because 0.0172 and 0.0170 both round to 0.02 and fold 2 > fold 1.
    # That is the ranking the exponent was calibrated to avoid.
    #
    # The window is a fraction of the leading score rather than an absolute step, so it
    # means the same thing whether scores sit near 1.0 or near 0.01.
    if profile.axes:
        lead = profile.axes[0].score
        tied = [i for i, ax in enumerate(profile.axes)
                if ax.score >= lead * (1.0 - tie_frac) and ax.is_global]
        if len(tied) > 1:
            best = max(tied, key=lambda i: 1000 if profile.axes[i].fold == 0
                       else profile.axes[i].fold)
            profile.axes.insert(0, profile.axes.pop(best))
    profile.dominant = profile.axes[0] if profile.axes else None
    profile.n_significant_axes = sum(1 for ax in profile.axes
                                     if ax.view_fraction >= significant_score)
    return profile


def _fit_fold(angles: List[float], cfg: AmbiguityConfig,
              probe: Callable[[float], bool]) -> Tuple[int, List[float]]:
    """Determine the rotational order supported by an axis.

    Every fold is *probed* against the geometry rather than inferred from which vote
    angles happened to survive.  Inferring from the survivors under-reports: a hex prism
    whose 60/120 degree candidates were pruned earlier would be recorded as C2, and the
    resulting MechVision ``angleStep`` of 180 would leave two thirds of the ambiguity
    unmitigated.

    Continuous is tested by requiring several *arbitrary* angles to pass, not just the ones
    the vote happened to find — otherwise an axis is called continuous whenever the vote
    was thorough rather than whenever the geometry is.
    """
    rng = np.random.default_rng(0)
    if all(probe(float(x)) for x in rng.uniform(5.0, 175.0, cfg.continuous_probes)):
        return 0, []

    # Highest fold first, so the *smallest* angle step that still fully explains the axis
    # wins: a hex prism must come back C6 (step 60), not C2 (step 180), even though 180
    # also passes.
    for n in sorted(cfg.fold_candidates, reverse=True):
        wanted = [360.0 * k / n for k in range(1, n)]
        if all(probe(w) for w in wanted):
            return n, wanted

    tol = cfg.theta_bin_deg
    for n in sorted(cfg.fold_candidates, reverse=True):
        wanted = [360.0 * k / n for k in range(1, n)]
        if all(any(abs((w - a + 180.0) % 360.0 - 180.0) < tol for a in angles) for w in wanted):
            return n, wanted
    return 1, sorted(angles)


def _same_axis(d1, p1, d2, p2, cfg: AmbiguityConfig, diameter: float) -> bool:
    if abs(float(np.dot(d1, d2))) < np.cos(np.deg2rad(cfg.axis_dir_tol_deg)):
        return False
    return float(np.linalg.norm(p1 - p2)) < cfg.axis_pt_tol_frac * diameter


# ─────────────────────────────────────────────────────────────────────────────
# PPF degeneracy diagnostic (reported, not acted on)
# ─────────────────────────────────────────────────────────────────────────────

def _ppf_degeneracy(pts: np.ndarray, normals: np.ndarray, diameter: float,
                    rng: np.random.Generator, n_sample: int = 600) -> Dict[str, float]:
    """Entropy of the PPF feature histogram.

    PPF's 4-tuple is near-constant over a planar patch, so all pairs hash to a few buckets
    and the vote argmax becomes noise-determined — a failure mode that occurs *even when
    the patch is geometrically unambiguous*.  Diagnostic only; nothing consumes this.
    """
    if len(pts) < 16:
        return {"entropy": 1.0, "max_bucket_frac": 0.0}
    sel = rng.choice(len(pts), min(n_sample, len(pts)), replace=False)
    v, nv = pts[sel], normals[sel]
    ii, jj = np.triu_indices(len(v), k=1)
    d = v[jj] - v[ii]
    dist = np.linalg.norm(d, axis=1)
    ok = dist > 1e-9
    ii, jj, d, dist = ii[ok], jj[ok], d[ok], dist[ok]
    u = d / dist[:, None]
    f1 = np.rint(dist / max(diameter * 0.05, 1e-9)).astype(np.int32)
    f2 = np.rint(np.arccos(np.clip(np.einsum("ij,ij->i", nv[ii], u), -1, 1)) / (np.pi / 12)).astype(np.int32)
    f3 = np.rint(np.arccos(np.clip(np.einsum("ij,ij->i", nv[jj], u), -1, 1)) / (np.pi / 12)).astype(np.int32)
    f4 = np.rint(np.arccos(np.clip(np.einsum("ij,ij->i", nv[ii], nv[jj]), -1, 1)) / (np.pi / 12)).astype(np.int32)
    counts = np.array(list(Counter(map(tuple, np.stack([f1, f2, f3, f4], axis=1))).values()), dtype=float)
    p = counts / counts.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    return {"entropy": entropy, "max_bucket_frac": float(p.max()), "n_buckets": float(len(p))}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyse_ambiguity(mesh: o3d.geometry.TriangleMesh,
                      pcd: o3d.geometry.PointCloud,
                      cfg: Optional[AmbiguityConfig] = None,
                      seed: int = 0,
                      progress_cb: Optional[Callable[[float], None]] = None,
                      ) -> AmbiguityProfile:
    """Detect view-dependent and global ambiguity for one part.

    Parameters
    ----------
    mesh : the part mesh, used only for visibility raycasting
    pcd  : the reference point cloud with normals (the thing that actually gets matched)
    """
    cfg = cfg or AmbiguityConfig()
    rng = np.random.default_rng(seed)

    pts = np.asarray(pcd.points, dtype=float)
    normals = np.asarray(pcd.normals, dtype=float)
    if len(pts) < 16 or len(normals) != len(pts):
        return AmbiguityProfile(per_point_discriminative=np.ones(len(pts)))
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    diameter = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    tree = cKDTree(pts)
    # Median nearest-neighbour spacing = the cloud's own resolution.  A rotated point
    # generally lands between samples, so a tolerance below this measures sampling pitch
    # rather than symmetry.
    spacing = float(np.median(tree.query(pts, k=2)[0][:, 1]))
    epsilon = max(cfg.epsilon_spacing_factor * spacing, cfg.epsilon_floor_m)

    # rank_area_exponent is recorded, not left at the dataclass default: the sidecar is the
    # only record of how the stored scores were produced, and `load_ambiguity_profile`
    # re-ranks from it. Left unset, a profile built with `--rank-exp 0` claimed 2.0.
    profile = AmbiguityProfile(epsilon_m=epsilon, f_tau=cfg.f_tau, diameter_m=diameter,
                               rank_area_exponent=cfg.rank_area_exponent)

    visible, view_dirs = _visibility_masks(mesh, pts, cfg, progress_cb)
    candidates = _vote_rotation_axes(pts, normals, diameter, cfg, rng, progress_cb)

    if not candidates:
        profile.per_point_discriminative = np.ones(len(pts))
        profile.discriminative_fraction = 1.0
        profile.per_view = [ViewAmbiguity(d, int(v.sum()), 0, 1.0)
                            for d, v in zip(view_dirs, visible)]
        profile.ppf_degeneracy = _ppf_degeneracy(pts, normals, diameter, rng)
        return profile

    # ── verify, screen by per-view survival, then refine the survivors ────────
    # The screen MUST be per-view, not global.  A view-dependent ambiguity maps the
    # patch visible from one viewpoint onto the model; it need not map the model onto
    # itself, so it can explain an arbitrarily small fraction of the whole surface.
    # Screening on global coverage would discard exactly the case this module exists for.
    def verify(cands):
        ex = np.zeros((len(cands), len(pts)), dtype=bool)
        for i, (d, p, ang, _) in enumerate(cands):
            R, t = _rotation_about(d, p, ang)
            ex[i] = _explains(pts, normals, tree, normals, R, t, epsilon, cfg.normal_cos_tol)
        return ex

    def per_view_survival(ex):
        surv = np.zeros((len(ex), len(view_dirs)), dtype=bool)
        frac = np.zeros((len(ex), len(view_dirs)))
        for k in range(len(view_dirs)):
            vis = visible[k]
            n_vis = int(vis.sum())
            if n_vis == 0:
                continue
            hits = ex[:, vis].sum(axis=1)
            frac[:, k] = hits / n_vis
            n_tau = max(int(np.ceil(cfg.f_tau * n_vis)), cfg.n_floor)
            surv[:, k] = (n_vis - hits) < n_tau
        return surv, frac

    # Two-tier screen.  Refinement typically lifts a candidate's coverage substantially,
    # so the pre-refinement bar must be well below the survival threshold or good axes
    # are discarded before they are ever polished; it sits just above the measured
    # random-rotation ceiling (~0.32 best-per-view) purely to bound refinement cost.
    explained = verify(candidates)
    best_view = per_view_survival(explained)[1].max(axis=1)
    keep = np.flatnonzero(best_view >= cfg.screen_min_area)
    keep = keep[np.argsort(-best_view[keep])][: cfg.max_refine]
    if progress_cb is not None:
        progress_cb(0.68)

    def empty_profile():
        profile.per_point_discriminative = np.ones(len(pts))
        profile.discriminative_fraction = 1.0
        profile.per_view = [ViewAmbiguity(d, int(v.sum()), 0, 1.0)
                            for d, v in zip(view_dirs, visible)]
        profile.ppf_degeneracy = _ppf_degeneracy(pts, normals, diameter, rng)
        return profile

    if len(keep) == 0:
        return empty_profile()

    candidates = [
        (*_refine_axis(pts, normals, tree, *candidates[i][:3],
                       epsilon, diameter, cfg, rng)[:2], candidates[i][2], candidates[i][3])
        for i in keep
    ]
    explained = verify(candidates)
    if progress_cb is not None:
        progress_cb(0.85)

    # ── per-view intersection with the soft tau threshold ─────────────────────
    survives, patch_frac = per_view_survival(explained)
    per_view: List[ViewAmbiguity] = []
    for k in range(len(view_dirs)):
        vis = visible[k]
        n_vis = int(vis.sum())
        n_surv = int(survives[:, k].sum())
        if n_vis and n_surv:
            covered = explained[survives[:, k]][:, vis].any(axis=0)
            discrim = 1.0 - float(covered.mean())
        else:
            discrim = 1.0
        per_view.append(ViewAmbiguity(view_dirs[k], n_vis, n_surv, discrim))
    profile.per_view = per_view

    view_fraction = survives.mean(axis=1)
    area_fraction = explained.mean(axis=1)
    alive = view_fraction > 0
    if not alive.any():
        return empty_profile()

    # ── group surviving candidates into axes, fit fold order ──────────────────
    groups: List[List[int]] = []
    for i in np.flatnonzero(alive):
        d, p, _, _ = candidates[i]
        for g in groups:
            d0, p0, _, _ = candidates[g[0]]
            if _same_axis(d, p, d0, p0, cfg, diameter):
                g.append(i)
                break
        else:
            groups.append([i])

    axes: List[AmbiguityAxis] = []
    for g in groups:
        weights = np.array([view_fraction[i] for i in g])
        lead = g[int(np.argmax(weights))]
        d, p, _, _ = candidates[lead]
        angles = [candidates[i][2] for i in g]
        bar = area_fraction[lead] * 0.9

        def probe_at(_d, _p, angle_deg: float) -> bool:
            R, t = _rotation_about(_d, _p, angle_deg)
            mask = _explains(pts, normals, tree, normals, R, t, epsilon, cfg.normal_cos_tol)
            return bool(mask.mean() >= bar)

        # Continuity attempt.  The axis was refined at a single angle, which leaves a
        # residual tilt that scores well there and worse elsewhere -- read naively that
        # looks like a finite fold. Re-refining against the worst of several spread
        # angles removes the tilt, so a genuinely continuous axis is reported as
        # continuous instead of shipping a too-coarse angleStep.
        spread = [40.0, 100.0, 160.0, 220.0, 280.0]
        cd, cp, _ = _refine_axis(pts, normals, tree, d, p, spread,
                                 epsilon, diameter, cfg, rng)
        if all(probe_at(cd, cp, a) for a in spread):
            d, p = cd, cp

        fold, fold_angles = _fit_fold(angles, cfg,
                                      lambda a, _d=d, _p=p: probe_at(_d, _p, a))
        vf = float(view_fraction[g].max())
        af = float(area_fraction[g].max())
        # Median over the (member, view) cells where that member actually survives.
        # Selecting whole columns instead -- views where ANY member survived -- mixes in
        # the non-surviving members' fractions and can report a patch fraction below the
        # survival floor, which is impossible by construction.
        surviving = patch_frac[g][survives[g]]
        pf = float(np.median(surviving)) if surviving.size else 0.0
        bpf = float(patch_frac[g].max())
        axes.append(AmbiguityAxis(
            direction=d, point=p, fold=fold,
            angles_deg=fold_angles or sorted(angles),
            is_global=_is_global(vf, af, cfg),
            view_fraction=vf, area_fraction=af,
            patch_fraction=pf, best_patch_fraction=bpf,
        ))

    profile.axes = axes
    rank_axes(profile, cfg.rank_area_exponent, cfg.significant_score)

    # ── discriminative scoring ────────────────────────────────────────────────
    live = np.flatnonzero(alive)
    weights = view_fraction[live]
    covered = (explained[live] * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-12)
    profile.per_point_discriminative = np.clip(1.0 - covered, 0.0, 1.0)
    profile.discriminative_fraction = float(np.mean([v.discriminative_fraction for v in per_view]))
    profile.ppf_degeneracy = _ppf_degeneracy(pts, normals, diameter, rng)

    if progress_cb is not None:
        progress_cb(1.0)
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

# Perceptually-ordered dark -> hot ramp (inferno-like), defined inline so the base
# geometry layer stays free of a matplotlib dependency.
_HEAT = np.array([
    [0.001, 0.000, 0.014],   # near-black
    [0.259, 0.039, 0.408],   # purple
    [0.576, 0.149, 0.404],   # magenta
    [0.867, 0.318, 0.227],   # orange-red
    [0.988, 0.647, 0.040],   # amber
    [0.988, 0.998, 0.645],   # pale yellow
])


def heat_colour(t) -> np.ndarray:
    """Sample the heat ramp at ``t`` in [0, 1]; 0 is coolest, 1 is hottest."""
    t = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), 0.0, 1.0)
    pos = t * (len(_HEAT) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(_HEAT) - 1)
    frac = (pos - lo)[:, None]
    out = _HEAT[lo] * (1.0 - frac) + _HEAT[hi] * frac
    return out[0] if np.ndim(t) == 0 or len(out) == 1 else out


def discriminative_colours(profile: "AmbiguityProfile") -> np.ndarray:
    """Per-point colours for the discriminative map.

    Hot = the point is explained by no ambiguity transform, so it is what actually pins
    the pose down; cool = the point is interchangeable with somewhere else on the model
    and contributes nothing to disambiguation.
    """
    disc = profile.per_point_discriminative
    if disc.size == 0:
        return np.zeros((0, 3))
    # Lift the cool end off the floor of the ramp.  A fully-ambiguous point maps to 0,
    # and the ramp's 0 is near-black, which disappears against a dark viewer background
    # -- the most ambiguous region would be the one you cannot see.
    return heat_colour(0.18 + 0.82 * disc)


def ambiguity_geometries(pcd: "o3d.geometry.PointCloud",
                         profile: "AmbiguityProfile",
                         max_axes: int = 6,
                         show_markers: bool = True,
                         ) -> Tuple[List, List[dict]]:
    """Build Open3D geometry for a profile: heat-mapped cloud plus ranked axes.

    Axes are drawn as solid rods spanning the part, coloured hot-to-cool by rank, each
    with a sphere at the axis's closest point to the part centre.  That marker is the
    thing worth looking at: if it does not sit on the centroid, a centroid-based analysis
    could not have produced this axis.

    Returns ``(geometries, legend)`` where legend rows describe what each colour means.
    """
    geoms: List = []
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return geoms, []

    # Points only — deliberately no normals.  Open3D shades a cloud whenever it carries
    # normals, which modulates every colour by the surface orientation and destroys the
    # colour-to-score mapping the heat map exists to convey: two points with the same
    # discriminative score would render differently just for facing different ways.
    # Meshes (the rods and markers) keep their normals and stay lit, so the axes still
    # read as solid objects.
    coloured = o3d.geometry.PointCloud()
    coloured.points = o3d.utility.Vector3dVector(pts)
    if profile.per_point_discriminative.size == len(pts):
        coloured.colors = o3d.utility.Vector3dVector(discriminative_colours(profile))
    else:
        coloured.paint_uniform_color([0.45, 0.45, 0.45])
    geoms.append(coloured)

    centre = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    centroid = pts.mean(axis=0)
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    rod_r = max(extent * 0.0028, 1e-5)
    marker_r = rod_r * 3.2

    axes = profile.axes[:max_axes]
    legend: List[dict] = []
    for rank, ax in enumerate(axes):
        # Hottest = rank 0. With one axis, use the hot end rather than the cold one.
        t = 1.0 - (rank / max(len(axes) - 1, 1)) * 0.8 if len(axes) > 1 else 1.0
        colour = heat_colour(t)

        d = ax.direction / np.linalg.norm(ax.direction)
        anchor = ax.point + float((centre - ax.point) @ d) * d
        rod = o3d.geometry.TriangleMesh.create_cylinder(radius=rod_r, height=extent * 1.15)
        rod.rotate(rotation_aligning_vector_to_axis([0.0, 0.0, 1.0], d), center=(0, 0, 0))
        rod.translate(anchor)
        rod.compute_vertex_normals()
        rod.paint_uniform_color(colour)
        geoms.append(rod)

        if show_markers:
            knob = o3d.geometry.TriangleMesh.create_sphere(radius=marker_r)
            knob.translate(anchor)
            knob.compute_vertex_normals()
            knob.paint_uniform_color(colour)
            geoms.append(knob)

        legend.append({
            "rank": rank,
            "colour": colour,
            "fold": "continuous" if ax.fold == 0 else (f"C{ax.fold}" if ax.fold > 1 else "none"),
            "angle_step_deg": ax.angle_step_deg(),
            "direction": d,
            "point": anchor,
            "is_global": ax.is_global,
            "view_fraction": ax.view_fraction,
            "patch_fraction": ax.patch_fraction,
            "offset_from_centroid_mm": _point_line_distance(centroid, d, ax.point) * 1000.0,
            "offset_from_aabb_centre_mm": _point_line_distance(centre, d, ax.point) * 1000.0,
        })

    if show_markers:
        # Reference markers, so an off-centroid axis is obvious rather than asserted.
        for pos, rgb in ((centroid, [1.0, 1.0, 1.0]), (centre, [0.35, 0.55, 1.0])):
            m = o3d.geometry.TriangleMesh.create_sphere(radius=marker_r * 0.8)
            m.translate(pos)
            m.compute_vertex_normals()
            m.paint_uniform_color(rgb)
            geoms.append(m)

    return geoms, legend


def _point_line_distance(query: np.ndarray, direction: np.ndarray, point: np.ndarray) -> float:
    v = np.asarray(query, float) - np.asarray(point, float)
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    return float(np.linalg.norm(v - float(v @ d) * d))


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def _axis_to_dict(ax: AmbiguityAxis) -> dict:
    d = asdict(ax)
    d["direction"] = [float(x) for x in ax.direction]
    d["point"] = [float(x) for x in ax.point]
    d["angles_deg"] = [float(x) for x in ax.angles_deg]
    d["angle_step_deg"] = ax.angle_step_deg()
    return d


def per_point_path(path) -> "Path":
    """Companion ``.npy`` holding the full per-point heat map for a profile JSON."""
    from pathlib import Path
    p = Path(path)
    return p.with_name(p.stem + "_per_point.npy")


def save_ambiguity_profile(profile: AmbiguityProfile, path, per_point: bool = False) -> None:
    """Write the profile sidecar.

    The JSON carries the axes and a *summary* of ``per_point_discriminative``, because a
    20 000-element array in JSON is neither readable nor compact.

    With ``per_point=True`` the full array is additionally written to a companion ``.npy``
    (see :func:`per_point_path`), which is what a matcher needs in order to weight votes by
    discriminability — the summary cannot be used for that.

    **It is off by default on purpose.** The array is index-aligned with the cloud the
    analysis ran on, which is the *surface* cloud. The same profile is written next to the
    edge / feature / flat variants too, where the axes remain valid but the per-point array
    does not correspond to those clouds' points at all. Passing ``per_point=True`` there
    would produce a file that looks usable and silently misattributes every score.
    """
    if per_point and profile.per_point_discriminative.size:
        np.save(per_point_path(path), profile.per_point_discriminative.astype(np.float32))
    disc = profile.per_point_discriminative
    payload = {
        "axes": [_axis_to_dict(a) for a in profile.axes],
        "dominant": _axis_to_dict(profile.dominant) if profile.dominant else None,
        "n_significant_axes": profile.n_significant_axes,
        "discriminative_fraction": profile.discriminative_fraction,
        "frame_changed": profile.frame_changed,
        "ppf_degeneracy": profile.ppf_degeneracy,
        "epsilon_m": profile.epsilon_m,
        "f_tau": profile.f_tau,
        "rank_area_exponent": profile.rank_area_exponent,
        "diameter_m": profile.diameter_m,
        "n_views": len(profile.per_view),
        "per_view": [
            {"direction": [float(x) for x in v.direction],
             "n_visible": v.n_visible,
             "n_transforms": v.n_transforms,
             "discriminative_fraction": v.discriminative_fraction}
            for v in profile.per_view
        ],
        "per_point_discriminative_summary": {
            "n": int(disc.size),
            "mean": float(disc.mean()) if disc.size else 1.0,
            "p10": float(np.percentile(disc, 10)) if disc.size else 1.0,
            "p90": float(np.percentile(disc, 90)) if disc.size else 1.0,
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def _axis_from_dict(d: dict) -> AmbiguityAxis:
    return AmbiguityAxis(
        direction=np.asarray(d["direction"], dtype=float),
        point=np.asarray(d["point"], dtype=float),
        fold=int(d["fold"]),
        angles_deg=[float(x) for x in d["angles_deg"]],
        is_global=bool(d["is_global"]),
        view_fraction=float(d["view_fraction"]),
        area_fraction=float(d["area_fraction"]),
        score=float(d["score"]),
    )


def load_ambiguity_profile(path) -> AmbiguityProfile:
    """Read a profile sidecar, including the per-point heat map when it was written.

    Profiles saved without ``per_point=True`` (and every sidecar written before the
    companion ``.npy`` existed) load fine with an empty ``per_point_discriminative`` — the
    axes are the part that most consumers want, and callers already have to handle the
    empty case because a part with no recovered axes has no heat map either.
    """
    with open(path) as f:
        d = json.load(f)
    pp = per_point_path(path)
    disc = np.load(pp).astype(np.float64) if pp.exists() else np.empty(0)
    # `is_global` is re-derived rather than trusted from the file. It is a pure function of
    # `view_fraction` and `area_fraction`, both of which are stored, so a sidecar written
    # under an older classification rule is judged by the current one instead of silently
    # carrying a stale label. (Same principle as `rank_axes`.)
    return reclassify_global(AmbiguityProfile(
        per_point_discriminative=disc,
        axes=[_axis_from_dict(a) for a in d.get("axes", [])],
        dominant=_axis_from_dict(d["dominant"]) if d.get("dominant") else None,
        n_significant_axes=int(d.get("n_significant_axes", 0)),
        discriminative_fraction=float(d.get("discriminative_fraction", 1.0)),
        per_view=[ViewAmbiguity(np.asarray(v["direction"], dtype=float),
                                int(v["n_visible"]), int(v["n_transforms"]),
                                float(v["discriminative_fraction"]))
                  for v in d.get("per_view", [])],
        frame_changed=bool(d.get("frame_changed", False)),
        ppf_degeneracy=d.get("ppf_degeneracy", {}),
        epsilon_m=float(d.get("epsilon_m", 0.0)),
        f_tau=float(d.get("f_tau", 0.0)),
        rank_area_exponent=float(d.get("rank_area_exponent", 2.0)),
        diameter_m=float(d.get("diameter_m", 0.0)),
    ))
