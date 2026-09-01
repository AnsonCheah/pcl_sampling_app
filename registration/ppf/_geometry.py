"""The three geometry helpers this package needs, inlined to keep it standalone.

These are copies of ``geometry.geom_utils.{model_diameter, median_spacing, project_to_so3}``
from the repository this package was extracted from.  They are duplicated rather than
imported on purpose: the point of the extraction is that ``registration.ppf`` depends on
nothing but NumPy, SciPy and Open3D, so it can be lifted into another project unchanged.

If you are editing this file because the shared version changed, the definitions are small
and self-contained -- the only thing that must not drift is ``model_diameter``, because
``PPFConfig`` normalises every part-relative tolerance against it.  Two different diameter
conventions were in use before it was settled (longest minimal-OBB extent vs AABB diagonal,
94.4 mm vs 146.5 mm on the Stanford bunny), so "5% of diameter" meant two different things
depending on which module you were standing in.
"""

from __future__ import annotations

import numpy as np

__all__ = ["model_diameter", "median_spacing", "project_to_so3", "as_points"]


def as_points(obj) -> np.ndarray:
    """Accept a PointCloud, a TriangleMesh, or an (N,3) array and return the points."""
    if isinstance(obj, np.ndarray):
        return np.asarray(obj, dtype=float).reshape(-1, 3)
    if hasattr(obj, "points"):
        return np.asarray(obj.points, dtype=float)
    if hasattr(obj, "vertices"):
        return np.asarray(obj.vertices, dtype=float)
    raise TypeError(f"cannot extract points from {type(obj)}")


def model_diameter(obj, max_hull_points: int = 3000) -> float:
    """Diameter -- the largest distance between any two points.

    The diameter is always realised by a pair of convex-hull vertices, so the search runs
    over the hull.  Exact whenever the hull has at most ``max_hull_points`` vertices, which
    covers most parts (the Stanford bunny's 21 668-point cloud has a 1 416-vertex hull).

    Above that the hull is strided down and the result is a tight lower bound rather than
    exact.  The subsample always retains the six axis-extreme points, so it can never come
    back shorter than the longest AABB edge.  This matters more than it sounds: a sphere
    sampled at 20 000 points has a 19 172-vertex hull -- nearly every point is a vertex --
    and an unguarded all-pairs distance matrix over that would ask for 8.8 GB.

    PPF needs this value specifically: it is the upper bound on the point-pair distance, so
    anything smaller silently discards long pairs (the ones with the best lever arm on
    rotation) and anything larger wastes distance bins on pairs that cannot occur.
    """
    from scipy.spatial.distance import pdist

    pts = as_points(obj)
    if len(pts) < 2:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        # Degenerate (coplanar/collinear) clouds have no 3D hull; the answer is still the
        # max pairwise distance, just over every point.
        hull = pts
    if len(hull) > max_hull_points:
        extremes = np.concatenate([hull.argmin(axis=0), hull.argmax(axis=0)])
        stride = np.linspace(0, len(hull) - 1, max_hull_points).astype(np.int64)
        hull = hull[np.unique(np.concatenate([stride, extremes]))]
    return float(pdist(hull).max())


def median_spacing(obj) -> float:
    """Median nearest-neighbour distance -- the cloud's own resolution.

    This is the floor on any geometric tolerance: agreement asserted below the sampling
    pitch is measuring the sampling, not the geometry.
    """
    from scipy.spatial import cKDTree

    pts = as_points(obj)
    if len(pts) < 2:
        return 0.0
    return float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1]))


def project_to_so3(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix to ``R``.  Accepts ``(3,3)`` or a batch ``(..., 3, 3)``.

    Needed wherever rotations are averaged -- pose clustering averages the rotations of the
    hypotheses in a cluster, and the mean of several rotation matrices is not itself one.
    Feeding an unprojected mean downstream produces a transform that quietly scales and
    shears the model, which shows up as a plausible-looking pose that fails verification.

    (This is also the failure OpenCV's ``ppf_match_3d`` ships with -- opencv_contrib #3223,
    an unnormalised quaternion in ``clusterPoses`` -- so the same guard is needed whether the
    clustering is ours or theirs.)

    Delegates to ``scipy.spatial.transform.Rotation.from_matrix``, which orthogonalises a
    non-proper input via Markley's quaternion method rather than raising.  That agrees with a
    hand-written SVD projection to 1.3e-15 over 200 perturbed rotations.
    """
    from scipy.spatial.transform import Rotation

    arr = np.asarray(R, dtype=float)
    return Rotation.from_matrix(arr).as_matrix().reshape(arr.shape)
