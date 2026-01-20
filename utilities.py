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
    
    # Identify valid points (non-zero magnitude normals)
    valid_mask = magnitudes > 1e-10
    num_removed = np.sum(~valid_mask)
    
    if num_removed > 0:
        print(f"Removing {num_removed} points with zero-magnitude normals")
        points = points[valid_mask]
        normals = normals[valid_mask]
        magnitudes = magnitudes[valid_mask]
    
    # Normalize
    normalized = normals / magnitudes[:, np.newaxis]
    
    # Update point cloud
    o3d = __import__('open3d')
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

def write_mechmind_ply(pcd, ply_path):
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
