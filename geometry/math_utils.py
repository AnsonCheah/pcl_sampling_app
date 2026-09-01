import numpy as np


def find_cdf_knee(curvature):
    """Split a value distribution at the knee of its CDF.

    curvature : (N,) values; non-finite entries are dropped. Needs >= 10 finite values.
    Returns (threshold, percentile, knee_idx) where percentile is a floored int in [0, 100].
    """
    curv = np.asarray(curvature)
    curv = curv[np.isfinite(curv)]
    if len(curv) < 10:
        raise ValueError("Not enough points for knee detection")

    curv_sorted = np.sort(curv)
    n = len(curv_sorted)
    curv_min = curv_sorted[0]
    curv_range = curv_sorted[-1] - curv_min
    x = (curv_sorted - curv_min) / (curv_range + 1e-12)
    y = np.linspace(0, 1, n)

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_vec_norm = np.linalg.norm(line_vec)
    if line_vec_norm <= 1e-12:      # all points collinear: no knee to find
        return curv_sorted[n // 2], 50, n // 2
    line_vec = line_vec / line_vec_norm

    points = np.column_stack([x, y])
    projections = np.dot(points - p1, line_vec)[:, None] * line_vec
    distances = np.linalg.norm(points - (p1 + projections), axis=1)
    knee_idx = int(np.argmax(distances))

    return curv_sorted[knee_idx], int(np.floor(100.0 * knee_idx / (n - 1))), knee_idx
