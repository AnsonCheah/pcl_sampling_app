"""Curvature estimation and curvature-driven resampling.

Two things live here that the pipeline needs in more than one place: a curvature estimate
with honest *units*, and the curvature-weighted downsampling used to build "feature cloud"
reference models.

On units — this is the trap.  The cheap and common curvature proxy is PCA surface variation,
``lambda_0 / sum(lambda)``, which is *dimensionless*: it says how non-planar a neighbourhood
is, not how tightly it curves.  Inverting it does not give a radius, so any "minimum feature
size" derived as ``1 / surface_variation`` is not a length at all — it silently changes
meaning when the cloud is resampled, because surface variation depends on the neighbourhood
size.  (The estimator this replaces, ``registration/heuristic_engine._estimate_min_feature_size``,
did exactly that.)

``mean_curvature`` here instead measures how fast the normal turns per unit distance, which
genuinely carries units of 1/length, so its reciprocal is a radius of curvature and
``min_feature_size`` is a length that survives resampling.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["mean_curvature", "surface_variation", "min_feature_size",
           "curvature_weighted_downsample"]


def mean_curvature(points: np.ndarray, normals: np.ndarray, k: int = 12) -> np.ndarray:
    """Per-point curvature in 1/metres, from how fast the normal turns with distance.

    For neighbours ``j`` of point ``i``, ``|n_i - n_j| / |p_i - p_j|`` approximates the
    normal's rate of change along the surface, which is the curvature magnitude.  Averaged
    over the neighbourhood for stability.

    Unlike :func:`surface_variation` this has units, so ``1/kappa`` is a radius and can be
    compared against sensor resolution or part size.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    if len(pts) < 4:
        return np.zeros(len(pts))

    kk = min(k + 1, len(pts))
    dist, idx = cKDTree(pts).query(pts, k=kk)
    dist, idx = dist[:, 1:], idx[:, 1:]                     # drop self

    # Normals are only oriented consistently up to sign on some clouds; comparing against
    # the flipped neighbour when that is closer keeps a consistently-oriented cloud
    # unchanged while stopping an inconsistent one from reporting spurious huge curvature.
    dn = np.linalg.norm(nrm[idx] - nrm[:, None, :], axis=2)
    dn = np.minimum(dn, np.linalg.norm(nrm[idx] + nrm[:, None, :], axis=2))

    valid = dist > 1e-12
    kappa = np.zeros_like(dn)
    kappa[valid] = dn[valid] / dist[valid]
    n_valid = np.maximum(valid.sum(axis=1), 1)
    return kappa.sum(axis=1) / n_valid


def surface_variation(points: np.ndarray, k: int = 12) -> np.ndarray:
    """PCA surface variation ``lambda_0 / sum(lambda)`` — dimensionless, in [0, 1/3].

    Useful for *ranking* points by how non-planar their neighbourhood is (which is all the
    adaptive-downsampling path needs).  Not a curvature: see the module docstring.

    Neighbourhood covariances come from Open3D's ``estimate_covariances`` — 3x faster than
    assembling them here with a KD-tree query and an einsum, and identical to 1.8e-12.
    """
    import open3d as o3d

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 4:
        return np.zeros(len(pts))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.estimate_covariances(o3d.geometry.KDTreeSearchParamKNN(min(k, len(pts))))
    eig = np.linalg.eigvalsh(np.asarray(pcd.covariances))
    total = eig.sum(axis=1)
    return np.where(total > 1e-18, eig[:, 0] / np.maximum(total, 1e-18), 0.0)


def min_feature_size(points: np.ndarray, normals: np.ndarray, diameter: float,
                     k: int = 12, percentile: float = 95.0) -> float:
    """Smallest resolvable geometric feature, as a radius of curvature (metres).

    Taken as the reciprocal of a high percentile of the curvature distribution: the most
    tightly curved places on the part are its smallest features.  Clamped into
    ``[0.002, 0.5] * diameter`` because both ends are meaningless — below the lower bound
    the estimate is dominated by sampling noise, and above the upper bound the "feature" is
    the whole part.

    Feeds ``PPFConfig.derive`` as the upper bound on the sampling distance: sampling coarser
    than this erases the geometry that distinguishes one pose from another.
    """
    kappa = mean_curvature(points, normals, k=k)
    kappa = kappa[np.isfinite(kappa)]
    if len(kappa) == 0:
        return 0.05 * diameter
    hi = float(np.percentile(kappa, percentile))
    if hi <= 1e-9:
        return 0.05 * diameter                              # flat part: no small features
    return float(np.clip(1.0 / hi, 0.002 * diameter, 0.5 * diameter))


def curvature_weighted_downsample(points: np.ndarray,
                                  normals: np.ndarray,
                                  curvatures: Optional[np.ndarray] = None,
                                  target_flat: int = 500,
                                  high_fraction: float = 0.30,
                                  seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Keep every high-curvature point, thin the flat remainder toward ``target_flat``.

    A sigmoid on the median/MAD-normalised curvature rather than a hard CDF knee, so the
    behaviour degrades smoothly on unimodal parts.  A knee is well defined for a bimodal
    distribution (flat faces plus sharp edges) but arbitrary for a part whose curvature
    varies gradually, and its location then moves under small changes in sampling.

    Returns ``(points, normals)`` of the retained subset.  Seeded, because the reference
    cloud it produces is baked into exported bundles and has to be reproducible.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    kappa = mean_curvature(pts, nrm) if curvatures is None else np.asarray(curvatures, float)

    med = float(np.median(kappa))
    mad = float(np.median(np.abs(kappa - med))) + 1e-12
    p_keep = 1.0 / (1.0 + np.exp(-(kappa - med) / mad))

    thresh = float(np.percentile(p_keep, (1.0 - high_fraction) * 100.0))
    high = np.flatnonzero(p_keep >= thresh)
    low = np.flatnonzero(p_keep < thresh)

    if len(low) > target_flat:
        rng = np.random.default_rng(seed)
        prob = p_keep[low] / max(p_keep[low].sum(), 1e-12)
        low = rng.choice(low, size=target_flat, replace=False, p=prob)

    idx = np.sort(np.concatenate([high, low]))
    return pts[idx], nrm[idx]
