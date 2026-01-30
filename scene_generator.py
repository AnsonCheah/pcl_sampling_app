import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
from utilities import *
from pathlib import Path

class SyntheticSceneGenerator:

    def __init__(self,mesh_path,image_width=640,image_height=480,fov=60,max_occlusion_ratio=0.3):
        self.mesh_path = mesh_path
        self.width = image_width
        self.height = image_height
        self.fov = fov
        self.max_occlusion_ratio = max_occlusion_ratio
        self.target_mesh = self._prepare_target_mesh(mesh_path)
        self.view_sphere = 0
        self.view_idx = 0
        self.visible_target_pcd = o3d.geometry.PointCloud()
        self.occluders_pcd = o3d.geometry.PointCloud()

    def _prepare_target_mesh(self, mesh_path):
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if mesh.is_empty():
            raise RuntimeError("Empty mesh")

        bbox = mesh.get_axis_aligned_bounding_box()
        extent_max = bbox.get_extent().max()
        if extent_max > 5 and extent_max < 5000:
            mesh.scale(0.001, center=(0,0,0))  # mm → m

        mesh.compute_vertex_normals()
        mesh.translate(-mesh.get_center())
        return mesh
        
    def random_camera(self, base_distance=1.5, jitter=0.05):
        p = self.view_sphere[self.view_idx]
        cam_pos = p * base_distance
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

    # def aabb_intersect(self, aabb1, aabb2):
    #     min1 = aabb1.get_min_bound()
    #     max1 = aabb1.get_max_bound()
    #     min2 = aabb2.get_min_bound()
    #     max2 = aabb2.get_max_bound()
    #     return np.all(max1 >= min2) and np.all(max2 >= min1)

    def meshes_intersect(self, mesh1, mesh2):
        aabb1 = mesh1.get_axis_aligned_bounding_box()
        aabb2 = mesh2.get_axis_aligned_bounding_box()

        min1 = aabb1.get_min_bound()
        max1 = aabb1.get_max_bound()
        min2 = aabb2.get_min_bound()
        max2 = aabb2.get_max_bound()
        if not np.all(max1 >= min2) and np.all(max2 >= min1):
            return False

        # if not self.aabb_intersect(aabb1, aabb2):
        #     return False

        scene = o3d.t.geometry.RaycastingScene()
        m1 = o3d.t.geometry.TriangleMesh.from_legacy(mesh1)
        scene.add_triangles(m1)
        pts = np.asarray(mesh2.sample_points_uniformly(500).points)
        query = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
        sdf = scene.compute_signed_distance(query).numpy()
        return np.any(sdf < 0)

    # ----------------------------
    # Raycasting
    # ----------------------------

    def _camera_cast_worker(self, meshes, cam_pos, look_at):
        scene = o3d.t.geometry.RaycastingScene()
        for m in meshes:
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))

        rays = scene.create_rays_pinhole(
            fov_deg=self.fov,
            center=look_at,
            eye=cam_pos,
            up=[0,0,1],
            width_px=self.width,
            height_px=self.height
        )

        ans = scene.cast_rays(rays)

        hit = ans['t_hit'].isfinite().numpy().reshape(-1)
        hit_indices = np.where(hit)[0]

        rays_np = rays.numpy().reshape(-1, 6)
        t_hit = ans['t_hit'].numpy().reshape(-1)

        origins = rays_np[hit_indices, :3]
        dirs    = rays_np[hit_indices, 3:]
        points  = origins + dirs * t_hit[hit_indices][:, None]

        geom_ids_all = ans['geometry_ids'].numpy().reshape(-1)
        geom_ids_hit = geom_ids_all[hit_indices]

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        return {
            "pcd": pcd,
            "hit_indices": hit_indices,
            "geom_ids_hit": geom_ids_hit,
            "geom_ids_all": geom_ids_all.reshape(self.height, self.width),
            "hit_mask_img": hit.reshape(self.height, self.width),
            "ray_origins": origins,
            "ray_hits": points
        }
    
    def _scene_cast(self, cam_pos, look_at, num_occluders=3, max_trials=50):
        

        target_center = self.target_mesh.get_center()

        # camera viewing direction
        view_dir = (target_center - cam_pos)
        view_dir = view_dir / np.linalg.norm(view_dir)

        # orthonormal basis around view dir
        right = np.cross(view_dir, [0,0,1])
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(view_dir, [0,1,0])
        right /= np.linalg.norm(right)
        up = np.cross(right, view_dir)

        # --- Target only ---
        res_target = self._camera_cast_worker([self.target_mesh], cam_pos, look_at)
        target_hit_ids = res_target["geom_ids_hit"]

        target_geom_ids = [int(i) for i in np.unique(target_hit_ids) if i != 4294967295]
        if len(target_geom_ids) != 1 or target_geom_ids[0] != 0:
            raise RuntimeError("Foreign id in target generation")

        target_geom_id = target_geom_ids[0]
        target_pixels = len(target_hit_ids)

        # --- Object scale ---
        bbox = self.target_mesh.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_extent())

        self.occluders = []
        scene_meshes = [self.target_mesh]
        view_dir = (target_center - cam_pos)
        dist_ct = np.linalg.norm(view_dir)
        view_dir = view_dir / dist_ct
        target_extent = self.target_mesh.get_axis_aligned_bounding_box().get_extent()
        target_radius = 0.5 * np.linalg.norm(target_extent)

        for occ_idx in range(num_occluders):
            success = False

            for trial in range(max_trials):
                occ = copy.deepcopy(self.target_mesh)
                occ.rotate(R.random().as_matrix(), center=(0,0,0))

                depth_offset = np.random.uniform(1, 3) * target_radius
                base_pos = target_center - view_dir * depth_offset
                lateral = (
                    right * np.random.uniform(-1.5*target_radius, 1.5*target_radius) +
                    up    * np.random.uniform(-1.5*target_radius, 1.5*target_radius)
                )
                occ.translate(base_pos + lateral)

                # --- Intersection checks ---
                if any(self.meshes_intersect(m, occ) for m in scene_meshes):
                    print(f"intersection detected, regenerating")
                    continue

                # --- Test occlusion ---
                test_scene = scene_meshes + [occ]
                res = self._camera_cast_worker(test_scene, cam_pos, look_at)

                self.ray_origins = res["ray_origins"]
                self.ray_hits    = res["ray_hits"]
                geom_ids_hit     = res["geom_ids_hit"]
                scene_pcd        = res["pcd"]

                scene_pcd.estimate_normals()
                orient_normals_using_cameras(scene_pcd, cam_pos)
                scene_pcd = normalize_normals(scene_pcd)
                validate_normals(scene_pcd)

                target_hit_mask = geom_ids_hit == target_geom_id
                occ_hit_mask    = geom_ids_hit != target_geom_id

                self.visible_target_pcd = mask_point_cloud(scene_pcd, target_hit_mask)
                self.occluders_pcd      = mask_point_cloud(scene_pcd, occ_hit_mask)

                visible_pixels = len(self.visible_target_pcd.points)
                occlusion_ratio = 1 - (visible_pixels / target_pixels)

                if occlusion_ratio > self.max_occlusion_ratio:
                    print(f"occlusion ratio exceed threshold: {occlusion_ratio}")
                    continue

                # --- Accept ---
                self.occluders.append(occ)
                scene_meshes.append(occ)
                success = True
                break

            if not success:
                raise RuntimeError(f"Failed to generate occluder {occ_idx}")

        # --- Final scene ---
        res_final = self._camera_cast_worker(scene_meshes, cam_pos, look_at)
        geom_ids_hit = res_final["geom_ids_hit"]
        final_visible = np.sum(geom_ids_hit == target_geom_id)
        final_occlusion_ratio = 1 - (final_visible / target_pixels)

        print(f"Target pcd occlusion ratio: {final_occlusion_ratio}")    

    # ----------------------------
    # Noise
    # ----------------------------

    def add_surface_noise(self, pcd, sigma=0.002):
        pts = np.asarray(self.visible_target_pcd.points)
        nrm = np.asarray(self.visible_target_pcd.normals)
        noise = np.random.normal(0, sigma, (len(pts), 1))
        self.visible_target_pcd.points = o3d.utility.Vector3dVector(pts + nrm * noise)
        return pcd

    def add_outliers(self, pcd):
        bbox = self.target_mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        diag = np.linalg.norm(extent)/2
        pts = np.asarray(self.visible_target_pcd.points)
        center = self.visible_target_pcd.get_center()
        outlier_count = int(len(self.visible_target_pcd.points)/3)
        out = center + np.random.uniform(-diag, diag, (outlier_count,3))
        self.visible_target_pcd.points = o3d.utility.Vector3dVector(np.vstack([pts, out]))
        return pcd

    # ----------------------------
    # Visualization
    # ----------------------------


    # ----------------------------
    # Public API
    # ----------------------------

    def generate_sample(self, path, visualize=True):
        cam_pos, look_at, up = self.random_camera()
        self._scene_cast(cam_pos, look_at)
        self.add_surface_noise()
        self.add_outliers()
        if visualize:
            self.visualize(cam_pos, look_at)
        self.visible_target_pcd.voxel_down_sample(0.001)
        self.visible_target_pcd.estimate_normals()
        orient_normals_using_cameras(self.visible_target_pcd, cam_pos)
        self.visible_target_pcd = normalize_normals(self.visible_target_pcd)
        validate_normals(self.visible_target_pcd)
        pointcloud_to_ply(self.visible_target_pcd, path)

# ----------------------------
# Run
# ----------------------------

if __name__ == "__main__":
    sim = SyntheticSceneGenerator(
        mesh_path="input/25333MB000.STL",
        image_width=1920,
        image_height=1080,
        fov=60,
        max_occlusion_ratio=0.3
    )
    samples = 20
    sim.view_sphere = fibonacci_sphere(samples)  # or 500
    for i in range(20):
        save_path = Path.cwd() / "synthetic_target" / f"sample_{i}.ply"
        sim.view_idx = i
        sample = sim.generate_sample(save_path)
