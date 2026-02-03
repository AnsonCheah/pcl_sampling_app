import numpy as np
import struct
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import copy

def random_rotation_matrix():
    return R.random().as_matrix()

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

def get_tf_to_origin(pcd):
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
    centroid = pts.mean(axis=0) # translation only
    T = np.eye(4)
    T[:3, 3] = centroid
    T[:3, :3] = pcd.get_minimal_oriented_bounding_box().R
    T = np.linalg.inv(T)
    return T

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

def add_surface_noise(pcd:o3d.geometry.PointCloud, sigma=0.001):
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    noise = np.random.normal(0, sigma, (len(pts), 1))
    pcd.points = o3d.utility.Vector3dVector(pts + nrm * noise)
    return pcd

def add_outliers(pcd:o3d.geometry.PointCloud):
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    diag = np.linalg.norm(extent)/2
    pts = np.asarray(pcd.points)
    center = pcd.get_center()
    outlier_count = int(len(pcd.points)/3)
    out = center + np.random.uniform(-diag, diag, (outlier_count,3))
    pcd.points = o3d.utility.Vector3dVector(np.vstack([pts, out]))
    return pcd

def jitter_ray_direction(dirs, sigma_angle=0.001):
    noise = np.random.normal(0, sigma_angle, dirs.shape)
    dirs_noisy = dirs + noise
    dirs_noisy /= np.linalg.norm(dirs_noisy, axis=1, keepdims=True)
    return dirs_noisy

def edge_dropout(edge_strength, p_base=0.02, p_edge=0.4, gamma=2.0):
    """
    edge_strength: (N,) in [0,1]
    returns: keep_mask (True = keep point)
    """
    edge_strength = np.clip(edge_strength, 0.0, 1.0)

    p_drop = p_base + p_edge * (edge_strength ** gamma)
    p_drop = np.clip(p_drop, 0.0, 0.95)  # safety

    rand = np.random.rand(len(edge_strength))
    keep = rand > p_drop
    return keep, p_drop

def generate_ray_outliers(origins, dirs, edge_strength, depth_low, depth_high, outlier_prob=0.15):
    """
    Generate exactly ONE false depth per selected ray.
    Depths are sampled independently per ray within object depth span.
    """

    N = len(origins)

    # Edge-biased selection probability
    p = outlier_prob * (0.75 + 0.25 * edge_strength)
    use = np.random.rand(N) < p

    if not np.any(use):
        return None, use

    # --- Independent depth sampling per ray ---
    # Use a mixture: uniform + object-biased Gaussian
    num = np.sum(use)

    # Uniform component
    t_uniform = np.random.uniform(depth_low, depth_high, size=num)

    # Object-centered Gaussian
    mu = 0.5 * (depth_low + depth_high)
    sigma = 0.35 * (depth_high - depth_low)
    t_gauss = np.random.normal(mu, sigma, size=num)

    # Mixture selection per ray
    mix = np.random.rand(num) < 0.6
    t_out = np.where(mix, t_gauss, t_uniform)
    # Final clamp
    t_out = np.clip(t_out, depth_low, depth_high)

    # Back-project
    pts_out = origins[use] + t_out[:, None] * dirs[use]

    return pts_out, use

def scene_render(meshes, cam_pos, proj_pos, look_at, fov, res_width, res_height,
    depth_sigma=0.0005, angular_noise_sigma=0.00005, dropout_prob=0.1,
    cam_grazing_cos_thresh=0.5, proj_grazing_cos_thresh=0.5,
    ):

    # ---------------- Scene ----------------
    scene = o3d.t.geometry.RaycastingScene()
    for m in meshes:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))

    # ---------------- Camera rays (1 ray per pixel) ----------------
    rays = scene.create_rays_pinhole(fov_deg=fov, center=look_at, eye=cam_pos, up=[0, 0, 1],
                                    width_px=res_width, height_px=res_height)

    rays_np = rays.numpy().reshape(-1, 6)
    ray_origins = rays_np[:, :3]
    ray_dirs = rays_np[:, 3:]

    # ---------------- Exact raycast ----------------
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy().reshape(-1)
    hit_mask = np.isfinite(t_hit)
    geom_ids_all = ans["geometry_ids"].numpy().reshape(-1)
    geom_ids_hit = geom_ids_all[hit_mask]
    # Early exit
    if not np.any(hit_mask):
        return None

    # Active pixels
    origins = ray_origins[hit_mask]
    dirs = ray_dirs[hit_mask]
    t = t_hit[hit_mask]
    points_exact = origins + t[:, None] * dirs

    num_rays = res_width * res_height # 
    ray_indices = np.arange(num_rays) # 
    hit_idx = ray_indices[hit_mask] # 
    nohit_idx = ray_indices[~hit_mask] # 

    # ---------------- Depth image (measurement space) ----------------
    depth_img = np.full(res_width * res_height, np.nan)
    depth_img[hit_mask] = t_hit[hit_mask]
    depth_img = depth_img.reshape(res_height, res_width)
    dz_dx = np.zeros_like(depth_img)
    dz_dy = np.zeros_like(depth_img)
    dz_dx[:, 1:-1] = np.abs(depth_img[:, 2:] - depth_img[:, :-2])
    dz_dy[1:-1, :] = np.abs(depth_img[2:, :] - depth_img[:-2, :])

    edge_strength_img = np.sqrt(dz_dx**2 + dz_dy**2)
    edge_strength_img /= (np.nanpercentile(edge_strength_img, 95) + 1e-6)
    edge_strength_img = np.clip(edge_strength_img, 0.0, 1.0)
    edge_strength_flat = edge_strength_img.reshape(-1)
    edge_strength_img = np.nan_to_num(
        edge_strength_img,
        nan=0.0,      # interior pixels → low noise
        posinf=1.0,
        neginf=0.0,
    )

    # ---------------- Surface normals (exact geometry) ----------------
    pcd_exact = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_exact))
    pcd_exact.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.005, max_nn=50
        )
    )
    normals = np.asarray(pcd_exact.normals)

    # ---------------- Camera grazing test ----------------
    v_cam = -dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    cam_cos = np.abs(np.sum(v_cam * normals, axis=1))
    cam_keep = cam_cos > cam_grazing_cos_thresh

    # ---------------- Projector visibility test (exact surface) ----------------
    proj_dirs = points_exact - proj_pos
    proj_dist = np.linalg.norm(proj_dirs, axis=1)
    proj_dirs /= proj_dist[:, None]
    proj_rays = o3d.core.Tensor(
        np.hstack([np.repeat(proj_pos[None, :], len(points_exact), axis=0), proj_dirs]),
        dtype=o3d.core.Dtype.Float32,
    )
    proj_hits = scene.cast_rays(proj_rays)
    proj_t = proj_hits["t_hit"].numpy()
    proj_visible = np.abs(proj_t - proj_dist) < 1e-3

    # ---------------- Projector grazing test ----------------
    v_proj = -proj_dirs
    proj_cos = np.abs(np.sum(v_proj * normals, axis=1))
    proj_keep = proj_cos > proj_grazing_cos_thresh

    # ---------------- Random pixel dropout ----------------
    rand_keep = np.random.rand(len(t)) > dropout_prob

    # ---------------- Final visibility mask (measurement space) ----------------
    visibility_mask = cam_keep & proj_visible & proj_keep & rand_keep
    visible_hit_idx = hit_idx[visibility_mask]        # A
    invalid_hit_idx = hit_idx[~visibility_mask]       # B

    # ---------------- Apply mask ----------------
    origins = origins[visibility_mask]
    dirs = dirs[visibility_mask]
    t = t[visibility_mask]
    normals = normals[visibility_mask]
    geom_ids_hit = geom_ids_hit[visibility_mask]
    edge_strength = edge_strength_flat[hit_mask][visibility_mask]
    
    edge_keep, p_drop = edge_dropout(
        edge_strength,
        p_base=dropout_prob,   # reuse existing param
        p_edge=0.4,
        gamma=2.5
    )

    origins = origins[edge_keep]
    dirs = dirs[edge_keep]
    t = t[edge_keep]
    normals = normals[edge_keep]
    geom_ids_hit = geom_ids_hit[edge_keep]
    edge_strength = edge_strength[edge_keep]
    # ---------------- Measurement-space noise ----------------
    # ---------------- Edge-aware depth noise ----------------
    edge_gain = 2.0  # tune this
    sigma_depth = depth_sigma * (1.0 + edge_gain * edge_strength)
    sigma_depth = np.nan_to_num(sigma_depth, nan=depth_sigma)
    if not np.all(np.isfinite(sigma_depth)):
        raise RuntimeError("Non-finite sigma_depth detected")
    t_noisy = t + np.random.normal(0.0, sigma_depth)
    t_noisy = np.clip(t_noisy, 0.0, None)
    if angular_noise_sigma > 0:
        noise = np.random.normal(0.0, angular_noise_sigma, size=dirs.shape)
        dirs_noisy = dirs + noise
        dirs_noisy /= np.linalg.norm(dirs_noisy, axis=1, keepdims=True)
    else:
        dirs_noisy = dirs

    # ---------------- Single back-projection ----------------
    points_final = origins + t_noisy[:, None] * dirs_noisy

    t_min = np.min(t)
    t_max = np.max(t)
    depth_margin = 0.15 * (t_max - t_min)
    depth_low = max(0.0, t_min - depth_margin)
    depth_high = t_max + depth_margin

    # ---------------- Lift visibility mask back to pixel space ----------------
    visible_pixel_mask = np.zeros_like(hit_mask, dtype=bool)
    visible_pixel_mask[np.where(hit_mask)[0][visibility_mask]] = True

    orig_B = ray_origins[invalid_hit_idx]
    dirs_B = ray_dirs[invalid_hit_idx]
    edge_B = edge_strength_flat[invalid_hit_idx]
    print(f"Generating outliers for {len(hit_idx)} invalid hit rays")
    pts_B, use_B = generate_ray_outliers(orig_B, dirs_B, edge_B, depth_low, depth_high, outlier_prob=0.15)
    print(f"Generated {len(pts_B)} outliers for invalid rays")

    orig_C = ray_origins[nohit_idx]
    dirs_C = ray_dirs[nohit_idx]
    edge_C = edge_strength_flat[nohit_idx]
    print(f"Generating outliers for {len(nohit_idx)} no-hit rays")
    pts_C, use_C = generate_ray_outliers(orig_C, dirs_C, edge_C, depth_low, depth_high, outlier_prob=0.05)
    # print(f"Generated {len(pts_C)} outliers for no-hit rays")

    points_all = [points_final]
    if pts_B is not None:
        points_all.append(pts_B)
    if pts_C is not None:
        points_all.append(pts_C)

    points_with_outliers = np.vstack(points_all)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_with_outliers))

    pcd.normals = o3d.utility.Vector3dVector(
        np.vstack([
            normals,
            np.zeros((len(points_with_outliers) - len(normals), 3))
        ])
    )

    return {
        "pcd": pcd,
        "geom_ids_hit": geom_ids_hit,
        "ray_origins": origins,
        "ray_hits": points_final,
    }

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
    exit()

    file_path = "mesh_raw/972703T000.STL"
    mesh = o3d.io.read_triangle_mesh(file_path)
    if mesh.is_empty():
        print("[WARN] Empty mesh")
        exit()

    bbox = mesh.get_axis_aligned_bounding_box()
    extent_max = bbox.get_extent().max()
    unit_conversion = 1.0
    if extent_max > 5 and extent_max < 5000.0:
        # Likely in millimeters -> convert to meters
        print(f"[INFO] Converting units from mm to m for: {file_path}")
        unit_conversion = 0.001
        mesh.scale(unit_conversion, center=(0, 0, 0))

    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())
    target_mesh = mesh

    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_corners = np.asarray(bbox.get_box_points())
    extent_min = bbox.get_extent().min()   # (dx, dy, dz) in world units

    num_targets = 6
    view_sphere = fibonacci_sphere(num_targets)
    visible_target_pcd = o3d.geometry.PointCloud()
    occluders_pcd = o3d.geometry.PointCloud()
    dropout = 0.1

    fov_deg = 25
    res_width = 1920
    res_height = 1200
    synthetic_occlusion = False
    min_occlusion_ratio = 0.1
    max_occlusion_ratio = 0.3
    synthetic_targets = []
    for i in range(num_targets):
        cam_pos, look_at, up = random_camera(view_sphere[i], 1.5)
        proj_pos = projector_from_camera(cam_pos, look_at, baseline=0.27)
        target_center = target_mesh.get_center()
        view_dir = (target_center - cam_pos)
        view_dir = view_dir / np.linalg.norm(view_dir)
        scene_meshes = [target_mesh]

        # orthonormal basis around view dir
        right = np.cross(view_dir, [0,0,1])
        right = np.cross(view_dir, [0,1,0]) if np.linalg.norm(right) < 1e-6 else right
        right /= np.linalg.norm(right)
        up = np.cross(right, view_dir)

        # --- Target only ---
        initial_res = scene_render(scene_meshes, cam_pos, proj_pos, look_at, fov_deg, res_width, res_height)

        target_hit_ids = initial_res["geom_ids_hit"]
        target_geom_ids = [int(i) for i in np.unique(target_hit_ids) if i != 4294967295 and i != -1]
        if len(target_geom_ids) != 1 or target_geom_ids[0] != 0:
            raise RuntimeError(f"Foreign id in target generation: {target_geom_ids}")
        target_geom_id = target_geom_ids[0]
        target_pixels = len(target_hit_ids)
        print(f"target pixels: {target_pixels}")

        occluders = []
        view_dir = (target_center - cam_pos)
        view_dir = view_dir / np.linalg.norm(view_dir)
        target_extent = target_mesh.get_axis_aligned_bounding_box().get_extent()
        target_radius = 0.5 * np.linalg.norm(target_extent)
        occlusion_ratio = 0
        synthetic_occlusion = i>=(num_targets>>1)
        num_occluders = (0 if not synthetic_occlusion else 1)
        visible_target_pcd = initial_res["pcd"]
        visible_target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = 0.005, max_nn = 60))

        # self.visible_target_pcd = add_outliers(self.visible_target_pcd)
        print(f"num point bef ds: {len(visible_target_pcd.points)}")
        visible_target_pcd = visible_target_pcd.voxel_down_sample(0.001)
        print(f"num point aft ds: {len(visible_target_pcd.points)}")
        visible_target_pcd.estimate_normals()
        orient_normals_using_cameras(visible_target_pcd, cam_pos)
        normalize_normals(visible_target_pcd)
        validate_normals(visible_target_pcd)
        visible_target_pcd.paint_uniform_color([0.0,1.0,0.0])
        synthetic_targets.append(visible_target_pcd)

    for i in synthetic_targets:
        o3d.visualization.draw_geometries([i], width=1080, height=720, zoom=0.5)

