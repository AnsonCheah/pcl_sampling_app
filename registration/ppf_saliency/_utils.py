"""Cloud-analysis helpers the ablation arms need, inlined so this package owns them.

``find_cdf_knee`` and ``extract_edge_points`` mirror ``geometry.math_utils`` and
``geometry.geom_utils``; ``mean_curvature`` is owned outright here (the former
``geometry/curvature.py`` was unused and has been removed).  They live here rather than being
imported so that the weighting ablation moves as one unit with the package it measures.

One deliberate behavioural change from the originals: ``find_cdf_knee`` no longer prints.
It is called once per arm per part inside a sweep over thousands of instances, where two
lines of stdout per call is noise that buries the run's actual output.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

__all__ = ["find_cdf_knee", "mean_curvature", "extract_edge_points"]


def find_cdf_knee(values):
    """Threshold at the knee of the sorted-value CDF, by max distance to the chord.

    The knee is where the distribution stops being "most points, slowly increasing" and
    starts being "a tail of large values" -- which is the split the pruning arms want, with
    no per-part threshold to pick.

    Returns ``(threshold, percentile, knee_index)``.
    """
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        raise ValueError("Not enough points for knee detection")

    v_sorted = np.sort(v)
    n = len(v_sorted)
    v_min = v_sorted[0]
    v_range = v_sorted[-1] - v_min
    x = (v_sorted - v_min) / (v_range + 1e-12)
    y = np.linspace(0, 1, n)

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len <= 1e-12:
        # Degenerate: every value identical, so there is no knee. The median is the only
        # defensible answer, and the caller's fallbacks handle the empty split.
        return v_sorted[n // 2], 50, n // 2
    line_vec = line_vec / line_len

    points = np.column_stack([x, y])
    vec_to_points = points - p1
    proj_points = p1 + np.dot(vec_to_points, line_vec)[:, None] * line_vec
    knee_idx = int(np.argmax(np.linalg.norm(points - proj_points, axis=1)))
    return v_sorted[knee_idx], int(np.floor(100.0 * knee_idx / (n - 1))), knee_idx


def mean_curvature(points: np.ndarray, normals: np.ndarray, k: int = 12) -> np.ndarray:
    """Per-point curvature in 1/metres, from how fast the normal turns with distance.

    For neighbours ``j`` of point ``i``, ``|n_i - n_j| / |p_i - p_j|`` approximates the
    normal's rate of change along the surface, which is the curvature magnitude.  Averaged
    over the neighbourhood for stability.

    Has units, so ``1/kappa`` is a radius and can be compared against sensor resolution or
    part size.
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
    return kappa.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)


def extract_edge_points(pcd, voxel_size):
    """Boolean mask marking boundary/crease points, via angular gap analysis.

    For each point the neighbour vectors are projected onto the plane perpendicular to its
    normal; the largest gap between consecutive azimuthal angles is the edge score.  A point
    in the middle of a surface is surrounded on all sides and has a small largest gap; a
    point on a boundary has neighbours only on one side and a large one.  The threshold comes
    from :func:`find_cdf_knee`, so there is no per-part tuning.
    """
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    n_points = len(pts)

    tree = o3d.geometry.KDTreeFlann(pcd)
    radius = voxel_size * 5.0
    max_gaps = np.zeros(n_points, dtype=np.float64)

    for i in range(n_points):
        _, idx, _ = tree.search_radius_vector_3d(pts[i], radius)
        idx = np.asarray(idx)
        if len(idx) < 4:                      # need self + at least 3 neighbours
            continue

        n = nrm[i]
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm

        neighbors = pts[idx[1:]] - pts[i]     # vectors to neighbours (exclude self)
        projected = neighbors - (neighbors @ n)[:, None] * n

        arb = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, arb)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

        cu = projected @ u
        cv = projected @ v
        valid = np.sqrt(cu ** 2 + cv ** 2) > 1e-10
        if valid.sum() < 2:
            continue

        angles = np.sort(np.arctan2(cv[valid], cu[valid]))
        gaps = np.diff(angles)
        wrap_gap = (angles[0] + 2.0 * np.pi) - angles[-1]
        max_gaps[i] = np.max(np.append(gaps, wrap_gap))

    threshold, _, _ = find_cdf_knee(max_gaps)
    return max_gaps >= threshold
