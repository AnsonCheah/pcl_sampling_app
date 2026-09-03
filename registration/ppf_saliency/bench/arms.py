"""The reference-cloud / vote-weight variants under test.

Each arm answers the same question differently: *which model points should carry the pose
vote, and how much should each one count?*

The split that matters is between arms that change the **cloud** and arms that change the
**weights**:

* Changing the cloud (B, C, D, H) removes points, and PPF builds all O(N^2) pairs -- so
  dropping half the points removes ~75% of the votes, not half. Those arms need their own
  trained table.
* Changing the weights (E, F, G) keeps every point and every pair, and reduces exactly to
  the baseline at uniform weights. They share arm A's table, which also makes the comparison
  clean: any measured difference is the weights and nothing else.

The literature is lopsided on which should win. Birdal & Ilic (IROS 2017) built a sampler
specifically for PPF whose stated design property is *even spacing, not* keypoint
concentration; K-PPF (Sensors 2022) prunes and reports a ~30% speedup with under 1 point of
accuracy change. Weighting has the better prior art -- Tuzel et al. (ECCV 2014) put learned
per-pair weights straight into the accumulator, and Cur-PPF (Sensors 2022) measured +2.65
points of matching rate for +33 ms while *deliberately not* subsampling. So B/C/D are here
as controls we expect to lose, and the informative outcome would be them winning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

__all__ = ["ArmSpec", "ARMS", "build_arm", "ArmContext"]


@dataclass
class ArmContext:
    """Everything an arm needs to build its model cloud and weights."""

    points: np.ndarray               # reference cloud, full resolution (m)
    normals: np.ndarray
    tau: float                       # PPF sampling distance (m)
    heat: Optional[np.ndarray] = None    # per-point discriminative score, or None
    part: str = ""


@dataclass
class ArmSpec:
    name: str
    description: str
    retrains: bool                   # True = its own table; False = reweights arm A's
    needs_heat: bool = False


ARMS: Dict[str, ArmSpec] = {
    "A_uniform": ArmSpec("A_uniform", "baseline Drost, uniform cloud", retrains=True),
    "B_heat_prune": ArmSpec("B_heat_prune", "keep discriminative points (CDF knee)",
                            retrains=True, needs_heat=True),
    "C_curvature": ArmSpec("C_curvature", "high-curvature points only", retrains=True),
    "D_curv_heat": ArmSpec("D_curv_heat", "curvature union heat map", retrains=True,
                           needs_heat=True),
    "E_heat_weight": ArmSpec("E_heat_weight", "uniform cloud + heat-map vote weights",
                             retrains=False, needs_heat=True),
    "F_ppf_weight": ArmSpec("F_ppf_weight", "uniform cloud + PPF-saliency vote weights",
                            retrains=False),
    "G_combined": ArmSpec("G_combined", "heat map x PPF saliency weights", retrains=False,
                          needs_heat=True),
    "H_edge": ArmSpec("H_edge", "edge/boundary points only", retrains=True),
}


def _knee_mask(values: np.ndarray) -> np.ndarray:
    """Keep the upper tail of ``values``, split at the CDF knee.

    A knee rather than a fixed percentile, and this is the whole reason the arm can be run
    unattended over a catalogue. The heat map's usable range is wildly part-dependent --
    measured: the Stanford bunny scores 1.000 everywhere (no ambiguity axes, so nothing to
    rank), a 100x30x20 box scores ~0.03 everywhere (all of it ambiguous), and only a part
    with real structure spreads out. "Keep the top half" is therefore undefined on the first,
    meaningless on the second, and a different operation on every part -- which is exactly the
    per-part manual configuration this project rejects.

    ``find_cdf_knee`` is the same splitter already used for curvature and edge extraction.
    """
    from .._utils import find_cdf_knee

    v = np.asarray(values, dtype=np.float64)
    if v.size < 10 or float(v.max() - v.min()) < 1e-9:
        return np.ones(v.size, dtype=bool)         # no structure to split on: keep everything

    # Drop the zero-score atom before fitting the knee, or the knee lands on it, the
    # threshold comes back 0.0, and the pruning arm silently becomes the baseline.
    # See registration/README.md.
    atom = v <= v.min() + 1e-9
    rest = v[~atom]
    if rest.size < 10:
        return ~atom if (~atom).sum() >= 50 else np.ones(v.size, dtype=bool)

    threshold, _, _ = find_cdf_knee(rest)
    mask = (~atom) & (v >= threshold)
    # A knee can still land pathologically near the top of a skewed distribution; a model of
    # a handful of points cannot constrain a pose at all, so rather than emit an arm that
    # fails for the wrong reason, fall back to "everything above the atom".
    if mask.sum() < 50:
        mask = ~atom
    return mask if mask.sum() >= 50 else np.ones(v.size, dtype=bool)


def build_arm(name: str, ctx: ArmContext):
    """Return ``(points, normals, weights)`` for one arm. ``weights`` may be ``None``.

    Points are at full resolution; the caller voxel-downsamples at ``tau`` so every arm goes
    through identical preprocessing.
    """
    from .. import downsample
    from ..saliency import combine, ppf_saliency, transfer_weights

    spec = ARMS[name]
    if spec.needs_heat and ctx.heat is None:
        raise ValueError(f"arm {name} needs the ambiguity heat map, which was not found")

    pts, nrm, heat = ctx.points, ctx.normals, ctx.heat

    if name == "A_uniform":
        return pts, nrm, None

    if name == "B_heat_prune":
        m = _knee_mask(heat)
        return pts[m], nrm[m], None

    if name == "C_curvature":
        from .._utils import mean_curvature
        m = _knee_mask(mean_curvature(pts, nrm))
        return pts[m], nrm[m], None

    if name == "D_curv_heat":
        from .._utils import mean_curvature
        m = _knee_mask(mean_curvature(pts, nrm)) | _knee_mask(heat)
        return pts[m], nrm[m], None

    if name == "H_edge":
        # Boundary points instead of surface points: on a flat part the surface normals are
        # near-constant, so surface PPF features collapse into a few bins and the accumulator
        # peak becomes noise-determined. Misc3D exposes this as a separate VotingMode and the
        # edge-PPF literature (Choi & Christensen 2012, PPF-MEAM 2018) reports it as the fix.
        import open3d as o3d
        from .._utils import extract_edge_points
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.normals = o3d.utility.Vector3dVector(nrm)
        m = extract_edge_points(pcd, ctx.tau)
        return (pts[m], nrm[m], None) if m.sum() >= 50 else (pts, nrm, None)

    # --- weight-only arms: same cloud, same table, different vote values ---
    m_pts, m_nrm = downsample(pts, nrm, ctx.tau)
    if name == "E_heat_weight":
        return pts, nrm, transfer_weights(pts, heat, m_pts)
    if name == "F_ppf_weight":
        return pts, nrm, "ppf_saliency"            # resolved after training; needs the table
    if name == "G_combined":
        return pts, nrm, ("combine", transfer_weights(pts, heat, m_pts))
    raise ValueError(f"unknown arm {name!r}")


def resolve_weights(name: str, model, prebuilt) -> Optional[np.ndarray]:
    """Finish the weight arms that need the trained table (PPF saliency reads its buckets)."""
    from ..saliency import combine, ppf_saliency

    if prebuilt is None:
        return None
    if isinstance(prebuilt, str) and prebuilt == "ppf_saliency":
        return ppf_saliency(model)
    if isinstance(prebuilt, tuple) and prebuilt[0] == "combine":
        return combine(prebuilt[1], ppf_saliency(model))
    return prebuilt
