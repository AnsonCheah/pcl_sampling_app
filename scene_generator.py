import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy

# ----------------------------
# Utility
# ----------------------------

def random_pose():
    rot = R.random().as_matrix()
    return rot

def look_at_matrix(eye, target, up):
    z = (eye - target)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.vstack([x, y, z])
    T[:3, 3] = eye
    return T

# ----------------------------
# Mesh preparation
# ----------------------------

def prepare_target_mesh(mesh_path):
    print(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        print("[WARN] Empty mesh")
        return

    bbox = mesh.get_axis_aligned_bounding_box()
    extent_max = bbox.get_extent().max()
    unit_conversion = 1.0
    if extent_max > 5 and extent_max < 5000.0:
        # Likely in millimeters -> convert to meters
        print(f"[INFO] Converting units from mm to m for: {mesh_path}")
        unit_conversion = 0.001
        mesh.scale(unit_conversion, center=(0, 0, 0))

    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())

    rot = random_pose()
    mesh.rotate(rot, center=(0,0,0))

    return mesh, rot

# ----------------------------
# Camera
# ----------------------------

def random_camera(base_distance=1.5, jitter=0.2):
    cam_pos = np.array([
        base_distance,
        np.random.uniform(-jitter, jitter),
        np.random.uniform(-jitter, jitter)
    ])
    look_at = np.zeros(3)
    up = np.array([0,0,1])
    return cam_pos, look_at, up

# ----------------------------
# Intersection check
# ----------------------------

def meshes_intersect(mesh1, mesh2):
    aabb1 = mesh1.get_axis_aligned_bounding_box()
    aabb2 = mesh2.get_axis_aligned_bounding_box()
    if not aabb_intersect(aabb1, aabb2):
        return False

    # Accurate SDF check
    scene = o3d.t.geometry.RaycastingScene()
    m1 = o3d.t.geometry.TriangleMesh.from_legacy(mesh1)
    m2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh2)
    scene.add_triangles(m1)
    pts = np.asarray(mesh2.sample_points_uniformly(500).points)
    query = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
    sdf = scene.compute_signed_distance(query).numpy()
    return np.any(sdf < 0)

def aabb_intersect(aabb1, aabb2):
    min1 = aabb1.get_min_bound()
    max1 = aabb1.get_max_bound()
    min2 = aabb2.get_min_bound()
    max2 = aabb2.get_max_bound()

    return np.all(max1 >= min2) and np.all(max2 >= min1)
# ----------------------------
# Occluder generation
# ----------------------------

def generate_occluder(target_mesh, max_trials=50):
    for _ in range(max_trials):
        occ = copy.deepcopy(target_mesh)
        occ.rotate(random_pose(), center=(0,0,0))
        occ.translate([
            np.random.uniform(0.15, 0.4),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.2, 0.2)
        ])
        if not meshes_intersect(target_mesh, occ):
            return occ
    raise RuntimeError("Failed to generate non-intersecting occluder")

# ----------------------------
# Raycasting
# ----------------------------

def render_scene(meshes, cam_pos, look_at, width=5000, height=5000, fov=60):
    scene = o3d.t.geometry.RaycastingScene()
    for m in meshes:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))

    rays = scene.create_rays_pinhole(
        fov_deg=fov,
        center=look_at,
        eye=cam_pos,
        up=[0,0,1],
        width_px=width,
        height_px=height
    )

    ans = scene.cast_rays(rays)

    hit = ans['t_hit'].isfinite().numpy()
    rays_np = rays.numpy()
    t_hit = ans['t_hit'].numpy()

    origins = rays_np[hit][:, :3]
    dirs    = rays_np[hit][:, 3:]
    t       = t_hit[hit][:, None]

    points = origins + dirs * t
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    return pcd, origins, points


# ----------------------------
# Noise models
# ----------------------------

def add_surface_noise(pcd, sigma=0.0001):
    pcd.estimate_normals()
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    noise = np.random.normal(0, sigma, (len(pts), 1))
    pcd.points = o3d.utility.Vector3dVector(pts + nrm * noise)
    return pcd

def add_outliers(pcd, n=300, radius=0.4):
    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    out = center + np.random.uniform(-radius, radius, (n,3))
    pcd.points = o3d.utility.Vector3dVector(np.vstack([pts, out]))
    return pcd

# ----------------------------
# Visualization
# ----------------------------

def visualize_scene(target, occluder, pcd, cam_pos, ray_origins, ray_hits):
    target.paint_uniform_color([0,1,0])
    occluder.paint_uniform_color([1,0,0])
    pcd.paint_uniform_color([0,0,1])

    cam = o3d.geometry.TriangleMesh.create_sphere(0.02)
    cam.translate(cam_pos)
    cam.paint_uniform_color([1,1,0])

    # Rays (subsample for sanity)
    idx = np.random.choice(len(ray_hits), size=min(1000, len(ray_hits)), replace=False)
    lines = [[i, i+len(idx)] for i in range(len(idx))]
    points = np.vstack([ray_origins[idx], ray_hits[idx]])
    colors = [[1,0,0] for _ in lines]
    ray_lines = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector2iVector(lines)
    )
    ray_lines.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([target, occluder, pcd, cam, ray_lines], width=1440, height=900)

# ----------------------------
# Full pipeline
# ----------------------------

def generate_sample(mesh_path):
    target, gt_rot = prepare_target_mesh(mesh_path)
    cam_pos, look_at, up = random_camera()
    occluder = generate_occluder(target)
    pcd, ray_origins, ray_hits = render_scene([target, occluder], cam_pos, look_at)
    pcd = add_surface_noise(pcd)
    pcd = add_outliers(pcd)
    visualize_scene(target, occluder, pcd, cam_pos, ray_origins, ray_hits)

    return {"pcd": pcd,"gt_rot": gt_rot,"camera": cam_pos}

# ----------------------------
# Run
# ----------------------------

if __name__ == "__main__":
    mesh_path = "input/25333MB000.STL"
    # for _ in range(100):
    sample = generate_sample(mesh_path)
