import numpy as np
import struct
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from pathlib import Path

def random_rotation_matrix():
    return R.random().as_matrix()

def pcd_geocenter(pcd):
    """
    Returns transformation matrix with consistent orientation.
    """
    points = np.asarray(pcd.points)
    center = points.mean(axis=0)
    
    centered_points = points - center
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    rotation_matrix = eigenvectors[:, idx]
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1

    for i in range(3):
        axis = rotation_matrix[:, i]
        # Find the component with largest absolute value
        max_idx = np.argmax(np.abs(axis))
        # If that component is negative, flip the entire axis
        if axis[max_idx] < 0:
            rotation_matrix[:, i] *= -1
    
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1
    
    rotation_matrix = np.round(rotation_matrix, decimals=3)
    tf = np.eye(4)
    tf[:3, 3] = center
    tf[:3, :3] = rotation_matrix
    tf = np.linalg.inv(tf)
    
    return tf

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
    
    # Vectorized magnitude calculation
    magnitudes = np.linalg.norm(normals, axis=1)
    valid_mask = magnitudes > 1e-10
    num_removed = np.sum(~valid_mask)
    
    if num_removed > 0:
        print(f"Removing {num_removed} points with zero-magnitude normals")
        points = points[valid_mask]
        normals = normals[valid_mask]
        magnitudes = magnitudes[valid_mask]
    
    # Vectorized normalization - no need for np.newaxis, broadcasting handles it
    normalized = normals / magnitudes[:, None]
    
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

def import_ply(file_path):
    """
    Import a PLY file into Open3D pointcloud.
    
    Args:
        file_path: Path to the .ply file
        
    Returns:
        o3d.geometry.PointCloud object
    """
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError(f"Failed to load pointcloud from {file_path}")
    print(f"Loaded {file_path} with {len(pcd.points)}")
    return pcd

def pointcloud_to_ply(pcd, ply_path):
    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)

    if len(points) == 0:
        raise ValueError("Empty point cloud")
    if normals.shape[0] != points.shape[0]:
        raise ValueError("Normals missing or size mismatch")

    # validate_normals(pcd)

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

# ===============================
# File dialogs
# ===============================

def save_ply_dialog(default_name=None):
    Tk().withdraw()
    default_name = f"{'output' if not default_name else default_name}.ply"

    path = filedialog.asksaveasfilename(
        defaultextension=".ply",
        initialdir=Path.cwd(), 
        initialfile=default_name,
        filetypes=[("PLY files", "*.ply")]
    )

    return Path(path) if path else None

def open_source_folder_dialog():
    Tk().withdraw()
    path = filedialog.askdirectory(initialdir=Path.cwd(), title="Select source folder (STL files)")
    return Path(path) if path else None


def meshes_intersect(mesh1, mesh2):
    aabb1 = mesh1.get_axis_aligned_bounding_box()
    aabb2 = mesh2.get_axis_aligned_bounding_box()

    min1 = aabb1.get_min_bound()
    max1 = aabb1.get_max_bound()
    min2 = aabb2.get_min_bound()
    max2 = aabb2.get_max_bound()
    if not np.all(max1 >= min2) and np.all(max2 >= min1):
        return False

    scene = o3d.t.geometry.RaycastingScene()
    m1 = o3d.t.geometry.TriangleMesh.from_legacy(mesh1)
    scene.add_triangles(m1)
    pts = np.asarray(mesh2.sample_points_uniformly(500).points)
    query = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
    sdf = scene.compute_signed_distance(query).numpy()
    return np.any(sdf < 0)

def random_camera(viewpoint, distance, jitter=0.05):
    cam_pos = viewpoint * distance
    look_at = np.zeros(3)
    # camera frame
    forward = (look_at - cam_pos)
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, [0,0,1])
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, [0,1,0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # jitter in camera local frame
    cam_pos += (
        right * np.random.uniform(-jitter, jitter) +
        up    * np.random.uniform(-jitter, jitter) +
        forward * np.random.uniform(-jitter, jitter)
    )

    return cam_pos, look_at, up

def camera_frame(cam_pos, look_at, up=np.array([0, 0, 1])):
    """
    Returns rotation matrix R_cam (camera → world)
    Columns: [right, up, forward]
    """
    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    R_cam = np.stack([right, true_up, forward], axis=1)
    return R_cam

def projector_from_camera(cam_pos, look_at, baseline=0.25, vertical_offset=0.0, forward_offset=0.0):
    """
    Create a projector rigidly connected to the camera.

    baseline: lateral offset (meters, camera-right direction)
    vertical_offset: vertical offset (meters)
    forward_offset: forward offset (meters)
    """
    R_cam = camera_frame(cam_pos, look_at)
    t_cp = np.array([baseline, vertical_offset, forward_offset])
    projector_pos = cam_pos + R_cam @ t_cp
    return projector_pos

if __name__=="__main__":
    pcd = import_ply("reference_pcd/25333MB000.ply")
    pcd.paint_uniform_color([0.0,1.0,0.0])
    print(pcd_geocenter(pcd))
    pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fibonacci_sphere(1000)))
    o3d.visualization.draw_geometries([pcd, pcd1], width=1080, height=720, zoom=1.0)
