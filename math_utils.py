
import matplotlib.pyplot as plt
import numpy as np

def plot_curvature_cdf(curvature, threshold=None, percentile=None):
    """
    Plot cumulative distribution function (CDF) of curvature.

    Args:
        curvature (np.ndarray): curvature values
        threshold (float, optional): curvature threshold to annotate
        percentile (float, optional): percentile of threshold (0-100)
    """

    title="Curvature Cumulative Distribution"
    curvature = np.asarray(curvature)
    curvature = curvature[np.isfinite(curvature)]

    if len(curvature) == 0:
        print("[WARN] No valid curvature values to plot.")
        return

    curv_sorted = np.sort(curvature)
    cdf = np.linspace(0, 1, len(curv_sorted))

    plt.figure(figsize=(7, 5))
    plt.plot(curv_sorted, cdf, linewidth=2)

    if threshold is not None:
        plt.axvline(threshold, linestyle="--", linewidth=2)
        label = f"Threshold = {threshold:.2e}"
        if percentile is not None:
            label += f"\nPercentile = {percentile:.1f}%"
        plt.text(
            threshold,
            0.05,
            label,
            rotation=90,
            verticalalignment="bottom"
        )

    plt.xlabel("Curvature")
    plt.ylabel("Cumulative probability")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def find_cdf_knee(curvature):
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
    
    if line_vec_norm > 1e-12:
        line_vec = line_vec / line_vec_norm
    else:
        # Degenerate case: all points on a vertical or horizontal line
        return curv_sorted[n // 2], 50, n // 2
    
    points = np.column_stack([x, y])
    vec_to_points = points - p1
    projections = np.dot(vec_to_points, line_vec)[:, None] * line_vec
    proj_points = p1 + projections
    distances = np.linalg.norm(points - proj_points, axis=1)
    knee_idx = np.argmax(distances)
    threshold = curv_sorted[knee_idx]
    percentile = 100.0 * knee_idx / (n - 1)
    print(f"Calculated percentile: {percentile:.2f}")
    print(f"Rounded percentile: {int(np.floor(percentile))}")
    
    return threshold, int(np.floor(percentile)), knee_idx