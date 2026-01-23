import numpy as np
import struct
import open3d as o3d

def center_pointcloud_to_geometric_center(pcd):
    """
    Aligns the pointcloud so that its geometric center (mean of all points)
    lies at the origin (0,0,0).

    Returns:
        pcd_centered : Open3D PointCloud
        T_center     : 4x4 transform that was applied
        centroid     : original centroid
    """
    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty")
    pts = np.asarray(pcd.points, dtype=np.float64)
    # True geometric center of sampled surface
    centroid = pts.mean(axis=0)

    T = np.eye(4)
    T[:3, 3] = -centroid
    pcd_centered = pcd.transform(T)
    return pcd_centered #, T, centroid

def normalize_normals(pcd):
    """
    Normalize all normals in a point cloud to unit length.
    Removes any points with zero-magnitude normals.
    
    Args:
        pcd: Open3D PointCloud object
    
    Returns:
        The same point cloud object (modified in-place)
    
    Raises:
        ValueError: If point cloud has no normals
    """
    if not pcd.has_normals():
        raise ValueError("Point cloud has no normals")
    
    normals = np.asarray(pcd.normals, dtype=np.float64)
    points = np.asarray(pcd.points, dtype=np.float64)
    magnitudes = np.linalg.norm(normals, axis=1)
    valid_mask = magnitudes > 1e-10
    num_removed = np.sum(~valid_mask)
    
    if num_removed > 0:
        print(f"Removing {num_removed} points with zero-magnitude normals")
        points = points[valid_mask]
        normals = normals[valid_mask]
        magnitudes = magnitudes[valid_mask]
    
    normalized = normals / magnitudes[:, np.newaxis]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normalized)
    
    return pcd

def validate_normals(pcd, tolerance=1e-5):
    """
    Check if all normals in the point cloud are normalized (magnitude = 1.0).
    Raises ValueError if any normal is not normalized beyond the tolerance.
    
    Args:
        pcd: Open3D PointCloud object
        tolerance: Allowed deviation from magnitude 1.0 (default: 1e-5)
    
    Raises:
        ValueError: If normals are missing or not normalized
    """
    if not pcd.has_normals():
        raise ValueError("Point cloud has no normals")
    
    normals = np.asarray(pcd.normals, dtype=np.float64)
    if normals.shape[0] == 0:
        raise ValueError("Normals array is empty")
    
    magnitudes = np.linalg.norm(normals, axis=1)
    deviations = np.abs(magnitudes - 1.0)
    
    if np.any(deviations > tolerance):
        invalid_count = np.sum(deviations > tolerance)
        worst_idx = np.argmax(deviations)
        raise ValueError(
            f"Found {invalid_count} unnormalized normals. "
            f"Max deviation: {deviations[worst_idx]:.2e} at index {worst_idx} "
            f"(magnitude: {magnitudes[worst_idx]:.8f}). "
            f"Min/Max magnitudes: {np.min(magnitudes):.8f}/{np.max(magnitudes):.8f}"
        )
    print(f"All {len(magnitudes)} normals passed validation.")

def fibonacci_sphere(samples):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def mask_point_cloud(pcd, mask):
    masked_pcd = o3d.geometry.PointCloud()
    masked_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask])
    masked_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
    return masked_pcd

def orient_normals_using_cameras(pcd, cam_positions):
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    
    view_vecs = pts - cam_positions   # from camera → point
    view_vecs /= np.linalg.norm(view_vecs, axis=1, keepdims=True)

    dots = np.sum(nrm * view_vecs, axis=1)

    # If dot > 0, normal points *away* from camera → flip it
    flip = dots > 0
    nrm[flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(nrm)

def plot_curvature_cdf(curvature,
                        threshold=None,
                        percentile=None,
                        title="Curvature Cumulative Distribution"):
        """
        Plot cumulative distribution function (CDF) of curvature.

        Args:
            curvature (np.ndarray): curvature values
            threshold (float, optional): curvature threshold to annotate
            percentile (float, optional): percentile of threshold (0-100)
        """
        import matplotlib.pyplot as plt
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
    curv_sorted = np.sort(curv)

    n = len(curv_sorted)
    if n < 10:
        raise ValueError("Not enough points for knee detection")

    x = (curv_sorted - curv_sorted.min()) / (np.ptp(curv_sorted) + 1e-12)
    y = np.linspace(0, 1, n)

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_vec /= np.linalg.norm(line_vec)

    distances = np.zeros(n)
    for i in range(n):
        p = np.array([x[i], y[i]])
        proj = p1 + np.dot(p - p1, line_vec) * line_vec
        distances[i] = np.linalg.norm(p - proj)

    knee_idx = np.argmax(distances)

    threshold = curv_sorted[knee_idx]
    percentile = 100.0 * knee_idx / (n - 1)
    print(f"calculated percentile: {percentile}")
    print(f"rounded percentile: {int(np.floor(percentile))}")
    return threshold, int(np.floor(percentile)), knee_idx

def pointcloud_to_ply(pcd, ply_path):
    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)

    if len(points) == 0:
        raise ValueError("Empty point cloud")
    if normals.shape[0] != points.shape[0]:
        raise ValueError("Normals missing or size mismatch")

    validate_normals(pcd)

    curvature = np.zeros((points.shape[0], 1), dtype=np.float32)
    vertex_data = np.hstack([points, normals, curvature])

    with open(ply_path, "wb") as f:
        header = f"""ply
format binary_little_endian 1.0
comment PCL generated
element vertex {len(vertex_data)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float curvature
element face 0
element camera 1
property float view_px
property float view_py
property float view_pz
property float x_axisx
property float x_axisy
property float x_axisz
property float y_axisx
property float y_axisy
property float y_axisz
property float z_axisx
property float z_axisy
property float z_axisz
property float focal
property float scalex
property float scaley
property float centerx
property float centery
property int viewportx
property int viewporty
property float k1
property float k2
end_header
"""
        f.write(header.encode("ascii"))

        # --- Vertex block ---
        for row in vertex_data:
            f.write(struct.pack("<7f", *row))

        # --- Camera block ---
        camera_floats = [
            0.0, 0.0, 1.0,     # view point
            1.0, 0.0, 0.0,     # x axis
            0.0, 1.0, 0.0,     # y axis
            0.0, 0.0, 1.0,     # z axis
            525.0,            # focal
            1.0, 1.0,         # scale
            320.0, 240.0      # center
        ]

        for v in camera_floats:
            f.write(struct.pack("<f", v))

        f.write(struct.pack("<i", 640))  # viewportx
        f.write(struct.pack("<i", 480))  # viewporty
        f.write(struct.pack("<f", 0.0))  # k1
        f.write(struct.pack("<f", 0.0))  # k2


def make_camera_frustum(cam_pos, look_at, fov_deg=60, aspect=1.0, depth=0.5):
    forward = look_at - cam_pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, [0,0,1])
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, [0,1,0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    h = np.tan(np.deg2rad(fov_deg/2)) * depth
    w = h * aspect

    center = cam_pos + forward * depth
    corners = [
        center + up*h + right*w,
        center + up*h - right*w,
        center - up*h - right*w,
        center - up*h + right*w,
    ]

    points = [cam_pos] + corners
    lines = [
        [0,1],[0,2],[0,3],[0,4],
        [1,2],[2,3],[3,4],[4,1]
    ]

    frustum = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector2iVector(lines)
    )
    frustum.paint_uniform_color([1,0,0])
    return frustum

def make_grid(center, normal, size=1.0, step=0.1):
    normal = normal / np.linalg.norm(normal)

    # find two orthogonal axes on plane
    tmp = np.array([1,0,0]) if abs(normal[0]) < 0.9 else np.array([0,1,0])
    axis1 = np.cross(normal, tmp)
    axis1 /= np.linalg.norm(axis1)
    axis2 = np.cross(normal, axis1)

    lines = []
    points = []
    n = int(size / step)

    for i in range(-n, n+1):
        p1 = center + axis1 * i * step + axis2 * size
        p2 = center + axis1 * i * step - axis2 * size
        p3 = center + axis2 * i * step + axis1 * size
        p4 = center + axis2 * i * step - axis1 * size

        points.append(p1); points.append(p2)
        lines.append([len(points)-2, len(points)-1])

        points.append(p3); points.append(p4)
        lines.append([len(points)-2, len(points)-1])

    grid = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector2iVector(lines)
    )
    grid.paint_uniform_color([0.3,0.3,0.3])
    return grid
