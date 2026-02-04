import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from utilities import fibonacci_sphere, projector_from_camera, random_camera, random_rotation_matrix, meshes_intersect, mask_point_cloud, orient_normals_using_cameras, normalize_normals, validate_normals
import copy
# def generate_ray_outliers(origins, dirs, edge_strength,
#                           depth_low, depth_high,
#                           outlier_prob=0.15,
#                           geom_id=-1):

#     N = len(origins)
#     use = np.random.rand(N) < outlier_prob

#     if not np.any(use):
#         return None, None

#     num = np.sum(use)

#     t_uniform = np.random.uniform(depth_low, depth_high, size=num)
#     mu = 0.5 * (depth_low + depth_high)
#     sigma = 0.35 * (depth_high - depth_low)
#     t_gauss = np.random.normal(mu, sigma, size=num)

#     mix = np.random.rand(num) < 0.6
#     t_out = np.where(mix, t_gauss, t_uniform)
#     t_out = np.clip(t_out, depth_low, depth_high)

#     pts = origins[use] + t_out[:, None] * dirs[use]
#     geom_ids = np.full(num, geom_id)

#     return pts, geom_ids

def generate_edge_outliers(
    origins,
    dirs,
    edge_strength,
    geom_ids,
    geom_depth_ranges,
    global_depth_range,
    outlier_prob=0.15,
    geom_id_outlier=-1,
    target_geom_id = None
):
    """
    Edge-aware outliers:
    - Depth range sampled from target geometry when available
    - Falls back to global depth range otherwise
    """

    N = len(origins)
    p = outlier_prob * (0.75 + 0.25 * edge_strength)
    use = np.random.rand(N) < p

    if not np.any(use):
        return None, None

    origins_u = origins[use]
    dirs_u = dirs[use]
    geom_u = geom_ids[use]

    if target_geom_id is not None:
        mask = geom_u == target_geom_id
        origins_u = origins_u[mask]
        dirs_u = dirs_u[mask]
        geom_u = geom_u[mask]

    t_out = np.zeros(len(origins_u))
    for i, gid in enumerate(geom_u):
        # if target_geom_id is not None and gid != target_geom_id:
        #     continue1
        if gid in geom_depth_ranges:
            d_low, d_high = geom_depth_ranges[gid]
        # else:
        #     d_low, d_high = global_depth_range

        # Mixture sampling
        if np.random.rand() < 0.6:
            mu = 0.5 * (d_low + d_high)
            sigma = 0.35 * (d_high - d_low)
            t = np.random.normal(mu, sigma)
        else:
            t = np.random.uniform(d_low, d_high)

        t_out[i] = np.clip(t, d_low, d_high)

    pts_out = origins_u + t_out[:, None] * dirs_u
    geom_out = np.full(len(pts_out), geom_id_outlier)
    print(f"Generated {len(pts_out)} edge outliers")
    return pts_out, geom_out

def apply_dropout(edge_strength, p_base=0.1, p_edge=0.4, gamma=2.5):
    p = p_base + p_edge * (edge_strength ** gamma)
    keep = np.random.rand(len(p)) > p
    return keep

def apply_depth_noise(t, edge_strength, depth_sigma, edge_gain=2.0):
    sigma = depth_sigma * (1.0 + edge_gain * edge_strength)
    sigma = np.nan_to_num(sigma, nan=depth_sigma)
    t_noisy = t + np.random.normal(0.0, sigma)
    return np.clip(t_noisy, 0.0, None)

def compute_edge_strength(depth_img):
    dz_dx = np.zeros_like(depth_img)
    dz_dy = np.zeros_like(depth_img)

    dz_dx[:, 1:-1] = np.abs(depth_img[:, 2:] - depth_img[:, :-2])
    dz_dy[1:-1, :] = np.abs(depth_img[2:, :] - depth_img[:-2, :])

    edge = np.sqrt(dz_dx**2 + dz_dy**2)
    edge /= (np.nanpercentile(edge, 95) + 1e-6)
    edge = np.clip(edge, 0.0, 1.0)

    return np.nan_to_num(edge, nan=0.0)

def compute_geom_depth_ranges(t_visible, geom_ids_visible, margin_ratio=0.15):
    """
    Returns dict: geom_id -> (depth_low, depth_high)
    """
    depth_ranges = {}

    for gid in np.unique(geom_ids_visible):
        if gid < 0:
            continue

        depths = t_visible[geom_ids_visible == gid]
        if len(depths) < 5:
            continue

        d_min = depths.min()
        d_max = depths.max()
        margin = margin_ratio * (d_max - d_min)

        depth_ranges[int(gid)] = (
            max(0.0, d_min - margin),
            d_max + margin
        )

    return depth_ranges

def scene_render(meshes, cam_pos, proj_pos, look_at, fov,
                 res_width, res_height,
                 cam_grazing_cos_thresh=0.3,
                 proj_grazing_cos_thresh=0.3):

    # ---------- Scene ----------
    scene = o3d.t.geometry.RaycastingScene()
    for m in meshes:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))

    # ---------- Camera rays ----------
    rays = scene.create_rays_pinhole(
        fov_deg=fov,
        center=look_at,
        eye=cam_pos,
        up=[0, 0, 1],
        width_px=res_width,
        height_px=res_height,
    )

    rays_np = rays.numpy().reshape(-1, 6)
    ray_origins = rays_np[:, :3]
    ray_dirs = rays_np[:, 3:]

    # ---------- Raycast ----------
    ans = scene.cast_rays(rays)
    t_hit_all = ans["t_hit"].numpy().reshape(-1)
    geom_ids_all = ans["geometry_ids"].numpy().reshape(-1)

    hit_mask = np.isfinite(t_hit_all)
    if not np.any(hit_mask):
        return None

    t_hit = t_hit_all[hit_mask]
    geom_ids_hit = geom_ids_all[hit_mask]
    origins = ray_origins[hit_mask]
    dirs = ray_dirs[hit_mask]
    points_exact = origins + t_hit[:, None] * dirs

    # ---------- Depth image ----------
    depth_img = np.full(res_width * res_height, np.nan)
    depth_img[hit_mask] = t_hit_all[hit_mask]
    depth_img = depth_img.reshape(res_height, res_width)
    edge_strength_img = compute_edge_strength(depth_img)

    # ---------- Normals ----------
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_exact))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=50))
    normals = np.asarray(pcd.normals)

    # ---------- Camera grazing ----------
    v_cam = -dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    cam_keep = np.abs(np.sum(v_cam * normals, axis=1)) > cam_grazing_cos_thresh

    # ---------- Projector visibility ----------
    proj_dirs = points_exact - proj_pos
    proj_dist = np.linalg.norm(proj_dirs, axis=1)
    proj_dirs /= proj_dist[:, None]

    proj_rays = o3d.core.Tensor(
        np.hstack([np.repeat(proj_pos[None, :], len(points_exact), axis=0), proj_dirs]),
        dtype=o3d.core.Dtype.Float32,
    )

    proj_hits = scene.cast_rays(proj_rays)
    proj_visible = np.abs(proj_hits["t_hit"].numpy() - proj_dist) < 1e-3

    # ---------- Projector grazing ----------
    v_proj = -proj_dirs
    proj_keep = np.abs(np.sum(v_proj * normals, axis=1)) > proj_grazing_cos_thresh
    visibility_mask = cam_keep & proj_visible & proj_keep

    # ---------- Pixel index bookkeeping ----------
    num_rays = res_width * res_height
    ray_indices = np.arange(num_rays)
    hit_idx = ray_indices[hit_mask]
    visible_hit_idx = hit_idx[visibility_mask]
    invalid_hit_idx = hit_idx[~visibility_mask]

    print(f"Generated {len(visible_hit_idx)} visible points out of {num_rays} rays.")
    print(f"Generated {len(hit_mask)} total hit out of {num_rays} rays.")
    print(f"geom id hit length {len(geom_ids_hit)}")

    return {
        "ray_origins": ray_origins,
        "ray_dirs": ray_dirs,
        "hit_mask": hit_mask,
        "points_exact": points_exact[visibility_mask],
        "t_hit": t_hit[visibility_mask],
        "normals": normals[visibility_mask],
        "geom_ids_hit": geom_ids_hit,
        "depth_img": depth_img,
        "edge_strength_img": edge_strength_img,
        "visible_hit_idx": visible_hit_idx,
        "invalid_hit_idx": invalid_hit_idx,
        "visibility_mask": visibility_mask
    }

if __name__ == "__main__":
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

    # num_targets = 6
    # view_sphere = fibonacci_sphere(num_targets)
    # visible_target_pcd = o3d.geometry.PointCloud()
    # occluders_pcd = o3d.geometry.PointCloud()

    # cam_pos, look_at, up = random_camera(view_sphere[1], 1.5)
    # proj_pos = projector_from_camera(cam_pos, look_at, baseline=0.27)
    # target_center = target_mesh.get_center()
    # view_dir = (target_center - cam_pos)
    # view_dir = view_dir / np.linalg.norm(view_dir)
    # scene_meshes = [target_mesh]
    # render = scene_render(scene_meshes, cam_pos, proj_pos, look_at, fov=25, res_width=1920, res_height=1200)
    # if render is None:
    #     raise RuntimeError("No rays hit geometry")
    
    # ############ Extract Info ################
    # points_exact = render["points_exact"]
    # t_hit = render["t_hit"]
    # normals = render["normals"]
    # geom_ids = render["geom_ids_hit"]
    # ray_origins = render["ray_origins"]
    # ray_dirs = render["ray_dirs"]
    # hit_mask = render["hit_mask"]
    # depth_img = render["depth_img"]
    # edge_img = render["edge_strength_img"]
    # visible_hit_idx = render["visible_hit_idx"]
    # invalid_hit_idx = render["invalid_hit_idx"]
    # visibility_mask = render["visibility_mask"]
    # edge_flat = edge_img.reshape(-1)
    # edge_visible = edge_flat[visible_hit_idx]

    # ############ Depth Noise ################
    # depth_sigma = 0.0005
    # t_noisy = apply_depth_noise(t_hit, edge_visible, depth_sigma=depth_sigma, edge_gain=2.0)

    # ############ Angular Noise ################
    # angular_sigma = 0.00005
    # dirs_visible = ray_dirs[visible_hit_idx]
    # dirs_noisy = dirs_visible + np.random.normal(0.0, angular_sigma, size=dirs_visible.shape)
    # dirs_noisy /= np.linalg.norm(dirs_noisy, axis=1, keepdims=True)

    # ############ Dropouts ################
    # keep_mask = apply_dropout(edge_visible, p_base=0.05, p_edge=0.4, gamma=2.5)
    # origins_kept = ray_origins[visible_hit_idx][keep_mask]
    # dirs_kept = dirs_noisy[keep_mask]
    # t_kept = t_noisy[keep_mask]
    # normals_kept = normals[keep_mask]
    # geom_ids_kept = geom_ids[visibility_mask][keep_mask]
    # points_final = origins_kept + t_kept[:, None] * dirs_kept

    # ############ Edge Outliers ################
    # t_min = np.min(t_kept)
    # t_max = np.max(t_kept)
    # depth_margin = 0.15 * (t_max - t_min)
    # depth_low = max(0.0, t_min - depth_margin)
    # depth_high = t_max + depth_margin
    # edge_outlier_origins = ray_origins[invalid_hit_idx]
    # edge_outlier_dirs = ray_dirs[invalid_hit_idx]
    # edge_outlier_edge = edge_flat[invalid_hit_idx]
    # global_depth_range = (depth_low, depth_high)

    # print(f"Geomid kept: \n{geom_ids_kept}")
    # geom_depth_ranges = compute_geom_depth_ranges(t_kept, geom_ids_kept)
    
    # geom_ids_img = np.full_like(hit_mask, fill_value=-1, dtype=int)
    # geom_ids_img[hit_mask] = geom_ids
    # pts_out, geom_out = generate_edge_outliers(
    #     origins=ray_origins[invalid_hit_idx],
    #     dirs=ray_dirs[invalid_hit_idx],
    #     edge_strength=edge_flat[invalid_hit_idx],
    #     geom_ids=geom_ids_img[invalid_hit_idx],  # FIXED
    #     geom_depth_ranges=geom_depth_ranges,
    #     global_depth_range=global_depth_range,
    #     outlier_prob=0.15,
    #     geom_id_outlier=-1,
    # )

    # points_all = [points_final]
    # geom_all = [geom_ids_kept]

    # if pts_out is not None:
    #     points_all.append(pts_out)
    #     geom_all.append(geom_out)

    # ############ Final PCD ################

    # points_all = np.vstack(points_all)
    # geom_all = np.concatenate(geom_all)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_all)

    # # Optional: color outliers red
    # colors = np.zeros((len(points_all), 3))
    # colors[geom_all == -1] = [1.0, 0.0, 0.0]   # outliers
    # colors[geom_all != -1] = [0.0, 0.0, 1.0]

    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd, mesh], point_show_normal=False)
    # exit()
    
    # import matplotlib.pyplot as plt
    # plt.imshow(edge_img, cmap="hot")
    # plt.colorbar()
    # plt.title("Edge Strength")
    # plt.show()

    num_targets =  6
    view_sphere = fibonacci_sphere(num_targets)
    visible_target_pcd = o3d.geometry.PointCloud()
    occluders_pcd = o3d.geometry.PointCloud()
    dropout = 0.1
    fov_deg = 25
    res_height = 1200
    res_width = 1920
    min_occlusion_ratio = 0.1
    max_occlusion_ratio = 0.3
    synthetic_targets = []
    synthetic_scene = []

    for i in range(num_targets):
        cam_pos, look_at, up = random_camera(view_sphere[i], 1.5)
        proj_pos = projector_from_camera(cam_pos, look_at, baseline=0.3)
        target_center = target_mesh.get_center()
        view_dir = (target_center - cam_pos)
        view_dir = view_dir / np.linalg.norm(view_dir)
        scene_meshes = [target_mesh]
        synthetic_scene.append(scene_meshes)

        # orthonormal basis around view dir
        right = np.cross(view_dir, [0,0,1])
        right = np.cross(view_dir, [0,1,0]) if np.linalg.norm(right) < 1e-6 else right
        right /= np.linalg.norm(right)
        up = np.cross(right, view_dir)

        # --- Target only ---
        initial_render = scene_render(scene_meshes, cam_pos, proj_pos, look_at, fov_deg, res_width, res_height)
        points_exact = initial_render["points_exact"]
        t_hit = initial_render["t_hit"]
        normals = initial_render["normals"]
        geom_ids = initial_render["geom_ids_hit"]
        ray_origins = initial_render["ray_origins"]
        ray_dirs = initial_render["ray_dirs"]
        hit_mask = initial_render["hit_mask"]
        depth_img = initial_render["depth_img"]
        edge_img = initial_render["edge_strength_img"]
        visible_hit_idx = initial_render["visible_hit_idx"]
        invalid_hit_idx = initial_render["invalid_hit_idx"]
        visibility_mask = initial_render["visibility_mask"]
        edge_flat = edge_img.reshape(-1)
        edge_visible = edge_flat[visible_hit_idx]
        target_geom_ids = [int(i) for i in np.unique(geom_ids) if i != 4294967295 and i != -1]
        if len(target_geom_ids) != 1 or target_geom_ids[0] != 0:
            raise RuntimeError("Foreign id in target generation")
        target_geom_id = target_geom_ids[0]
        target_pixels = len(geom_ids[geom_ids == target_geom_id])
        print(f"target pixels: {target_pixels}")

        occluders = []
        view_dir = (target_center - cam_pos)
        view_dir = view_dir / np.linalg.norm(view_dir)
        target_extent = target_mesh.get_axis_aligned_bounding_box().get_extent()
        target_radius = 0.5 * np.linalg.norm(target_extent)
        occlusion_ratio = 0
        synthetic_occlusion = i>=(num_targets>>1)
        num_occluders = (0 if not synthetic_occlusion else 1)
        max_trials = 1000
        for occ_idx in range(num_occluders):
            success = False

            for trial in range(max_trials):
                occ = copy.deepcopy(target_mesh)
                occ.rotate(random_rotation_matrix(), center=(0,0,0))
                right_offset = right * np.random.uniform(-1.5*target_radius, 1.5*target_radius)
                up_offset = up * np.random.uniform(-1.5*target_radius, 1.5*target_radius)
                depth_offset = np.random.uniform(1, 3) * target_radius
                base_pos = target_center - view_dir * depth_offset
                occ.translate(base_pos + right_offset + up_offset)

                if any(meshes_intersect(m, occ) for m in scene_meshes):
                    print(f"intersection detected, regenerating")
                    continue

                test_scene = scene_meshes + [occ]
                synthetic_scene[i] = test_scene

                render = scene_render(test_scene, cam_pos, proj_pos, look_at, fov_deg, res_width, res_height)
                points_exact = render["points_exact"]
                t_hit = render["t_hit"]
                normals = render["normals"]
                geom_ids = render["geom_ids_hit"]
                ray_origins = render["ray_origins"]
                ray_dirs = render["ray_dirs"]
                hit_mask = render["hit_mask"]
                depth_img = render["depth_img"]
                edge_img = render["edge_strength_img"]
                visible_hit_idx = render["visible_hit_idx"]
                invalid_hit_idx = render["invalid_hit_idx"]
                visibility_mask = render["visibility_mask"]
                edge_flat = edge_img.reshape(-1)
                edge_visible = edge_flat[visible_hit_idx]

                target_hit_mask = (geom_ids[visibility_mask] == target_geom_id)
                visible_target_pcd = render["points_exact"][target_hit_mask]
                visible_pixels = len(visible_target_pcd)
                occlusion_ratio = 1 - (visible_pixels / target_pixels)
                if not (min_occlusion_ratio < occlusion_ratio < max_occlusion_ratio):
                    print(f"occlusion ratio out of range: {occlusion_ratio}")
                    continue
                print(f"Accepted pcd occlusion ratio: {occlusion_ratio}")    

                occluders.append(occ)
                scene_meshes.append(occ)
                success = True
                break

            if not success:
                raise RuntimeError(f"Failed to generate occluder {occ_idx}")

        if not synthetic_occlusion: # handles no occlusion
            visible_target_pcd = initial_render["points_exact"]

        ############ Depth Noise ################
        depth_sigma = 0.0005
        t_noisy = apply_depth_noise(t_hit, edge_visible, depth_sigma=depth_sigma, edge_gain=3.0)

        ############ Angular Noise ################
        angular_sigma = 0.00005
        dirs_visible = ray_dirs[visible_hit_idx]
        dirs_noisy = dirs_visible + np.random.normal(0.0, angular_sigma, size=dirs_visible.shape)
        dirs_noisy /= np.linalg.norm(dirs_noisy, axis=1, keepdims=True)

        ############ Dropouts ################
        keep_mask = apply_dropout(edge_visible, p_base=0.05, p_edge=0.4, gamma=2.5)
        origins_kept = ray_origins[visible_hit_idx][keep_mask]
        dirs_kept = dirs_noisy[keep_mask]
        t_kept = t_noisy[keep_mask]
        normals_kept = normals[keep_mask]
        geom_ids_kept = geom_ids[visibility_mask][keep_mask]
        print(f"geom_ids_kept: {len(geom_ids_kept)}")
        points_final = origins_kept + t_kept[:, None] * dirs_kept
        print(f"points_final: {len(points_final)}")

        ############ Edge Outliers ################
        t_min = np.min(t_kept)
        t_max = np.max(t_kept)
        depth_margin = 0.15 * (t_max - t_min)
        depth_low = max(0.0, t_min - depth_margin)
        depth_high = t_max + depth_margin
        edge_outlier_origins = ray_origins[invalid_hit_idx]
        edge_outlier_dirs = ray_dirs[invalid_hit_idx]
        edge_outlier_edge = edge_flat[invalid_hit_idx]
        global_depth_range = (depth_low, depth_high)

        geom_depth_ranges = compute_geom_depth_ranges(t_kept, geom_ids_kept)
        
        geom_ids_img = np.full_like(hit_mask, fill_value=-1, dtype=int)
        geom_ids_img[hit_mask] = geom_ids
        print(f"geom_ids_img: {np.unique(geom_ids_img)}")
        pts_out, geom_out = generate_edge_outliers(
            origins=ray_origins[invalid_hit_idx],
            dirs=ray_dirs[invalid_hit_idx],
            edge_strength=edge_flat[invalid_hit_idx],
            geom_ids=geom_ids_img[invalid_hit_idx],  # FIXED
            geom_depth_ranges=geom_depth_ranges,
            global_depth_range=global_depth_range,
            outlier_prob=0.15,
            geom_id_outlier=-1,
            target_geom_id=target_geom_id
        )

        target_mask = geom_ids_kept == target_geom_id
        points_all = [points_final[target_mask]]
        geom_all = [geom_ids_kept[target_mask]]

        if pts_out is not None:
            points_all.append(pts_out)
            geom_all.append(geom_out)

        ############ Final PCD ################

        points_all = np.vstack(points_all)
        geom_all = np.concatenate(geom_all)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_all)
        # self.visible_target_pcd = add_outliers(self.visible_target_pcd)
        print(f"num point bef ds: {len(pcd.points)}")
        pcd = pcd.voxel_down_sample(0.001)
        print(f"num point aft ds: {len(pcd.points)}")
        pcd.estimate_normals()
        orient_normals_using_cameras(pcd, cam_pos)
        normalize_normals(pcd)
        validate_normals(pcd)
        synthetic_targets.append(pcd)

    # pcd.colors = o3d.utility.Vector3dVector(colors)
    for i,t in enumerate(synthetic_targets):
        o3d.visualization.draw_geometries([t] + synthetic_scene[i], point_show_normal=False)
