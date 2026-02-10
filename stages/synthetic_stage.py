import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from enums import Stage
from stages.stage_base import BaseStage
from pathlib import Path
from utilities import normalize_normals, fibonacci_sphere, random_camera, pointcloud_to_ply
from synthetic_pcl_utils import *
import threading

class SyntheticStage(BaseStage):
    def __init__(self, app):
        self.name = Stage.SYNTHETIC.name
        super().__init__(app)
        self.synthetic_occlusion = True
        self.fov_deg = 25
        self.res_width = 1920
        self.res_height = 1200
        self.min_occlusion_ratio = 0.1
        self.max_occlusion_ratio = 0.3
        self.synthetic_targets = []
        self.depth_sigma = 0.0005
        self.angular_sigma = 0.00005

    def build_panel(self):
        v = gui.Vert(4)

        self.num_targets_slider = self.register_widget(gui.Slider(gui.Slider.INT))
        self.num_targets_slider.set_limits(2, 200)
        self.num_targets_slider.int_value = 6
        self.btn_generate = self.register_widget(gui.Button("Generate Synthetic Targets"))
        self.btn_generate.set_on_clicked(self.start)
        self.btn_reset = self.register_widget(gui.Button("Clear Synthetic Targets"), lambda: len(self.synthetic_targets)>0)
        self.btn_reset.set_on_clicked(self.reset)
        self.combobox_targets = self.register_widget(gui.Combobox(), lambda: len(self.synthetic_targets)>0)
        self.combobox_targets.set_on_selection_changed(self.preview_synthetic_target)
        self.btn_export = self.register_widget(gui.Button("Export Synthetic Targets"), lambda: len(self.synthetic_targets)>0)
        self.btn_export.set_on_clicked(self.save_synthetic_targets)

        self.btn_back = self.register_widget(gui.Button("Back: SAVE"))
        self.btn_back.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value - 1)))

        self.btn_restart = self.register_widget(gui.Button("Restart"))
        self.btn_restart.set_on_clicked(lambda: self.app._restart())
        v.add_child(gui.Label("Generate Synthetic Targets"))
        v.add_child(gui.Label(""))
        v.add_child(self.num_targets_slider)
        v.add_child(self.btn_generate)
        v.add_child(self.btn_reset)
        v.add_child(self.combobox_targets)
        v.add_child(self.btn_export)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(self.btn_back)
        v.add_child(self.btn_restart)

        print("loaded synthetic panel")
        return v

    def _refresh_ui(self):
        if self.app.headless:
            return
        self.app.main_thread(lambda: self.app._clear_scene())
        if self.synthetic_targets != []:
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("synthetic_target", self.synthetic_targets[0], self.app.default_point_material))
        elif self.app.target_mesh != None:
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("mesh", self.app.target_mesh, self.app.default_material))
        self.app.scene.force_redraw()
        self.enable_widgets()
    
    def reset(self):
        self.synthetic_targets = []
        self.combobox_targets.clear_items()
        self._refresh_ui()

    def worker(self):
        self.num_targets =  self.num_targets_slider.int_value
        self.view_sphere = fibonacci_sphere(self.num_targets)
        for i in range(self.num_targets):
            cam_pos, look_at, up = random_camera(self.view_sphere[i], 1.5)
            proj_pos = projector_from_camera(cam_pos, look_at, baseline=0.27)
            target_center = self.app.target_mesh.get_center()
            view_dir = (target_center - cam_pos)
            view_dir = view_dir / np.linalg.norm(view_dir)
            scene_meshes = [self.app.target_mesh]

            # orthonormal basis around view dir
            right = np.cross(view_dir, [0,0,1])
            right = np.cross(view_dir, [0,1,0]) if np.linalg.norm(right) < 1e-6 else right
            right /= np.linalg.norm(right)
            up = np.cross(right, view_dir)

            # --- Target only ---
            initial_render = scene_render(scene_meshes, cam_pos, proj_pos, look_at, self.fov_deg, self.res_width, self.res_height)
            t_hit = initial_render["t_hit"]
            geom_ids = initial_render["geom_ids_hit"]
            ray_origins = initial_render["ray_origins"]
            ray_dirs = initial_render["ray_dirs"]
            hit_mask = initial_render["hit_mask"]
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
            target_hit_mask = (geom_ids[visibility_mask] == target_geom_id)
            visible_target_pcd = initial_render["points_exact"][target_hit_mask]
            target_pixels = len(visible_target_pcd)
            print(f"target pixels: {target_pixels}")

            occluders = []
            view_dir = (target_center - cam_pos)
            view_dir = view_dir / np.linalg.norm(view_dir)
            target_extent = self.app.target_mesh.get_axis_aligned_bounding_box().get_extent()
            target_radius = 0.5 * np.linalg.norm(target_extent)
            occlusion_ratio = 0
            self.synthetic_occlusion = i>=(self.num_targets>>1)
            num_occluders = (0 if not self.synthetic_occlusion else 1)
            max_trials = 100
            for occ_idx in range(num_occluders):
                success = False
                for _ in range(max_trials):
                    occ = copy.deepcopy(self.app.target_mesh)
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

                    render = scene_render(test_scene, cam_pos, proj_pos, look_at, self.fov_deg, self.res_width, self.res_height)
                    t_hit = render["t_hit"]
                    geom_ids = render["geom_ids_hit"]
                    ray_origins = render["ray_origins"]
                    ray_dirs = render["ray_dirs"]
                    hit_mask = render["hit_mask"]
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
                    if not (self.min_occlusion_ratio < occlusion_ratio < self.max_occlusion_ratio):
                        print(f"occlusion ratio out of range: {occlusion_ratio}")
                        continue
                    print(f"Accepted pcd occlusion ratio: {occlusion_ratio}")    

                    occluders.append(occ)
                    scene_meshes.append(occ)
                    success = True
                    break

                if not success:
                    raise RuntimeError(f"Failed to generate occluder {occ_idx}")

            ############ Depth Noise ################
            t_noisy = apply_depth_noise(t_hit, edge_visible, depth_sigma=self.depth_sigma, edge_gain=3.0)

            ############ Angular Noise ################
            dirs_visible = ray_dirs[visible_hit_idx]
            dirs_noisy = dirs_visible + np.random.normal(0.0, self.angular_sigma, size=dirs_visible.shape)
            dirs_noisy /= np.linalg.norm(dirs_noisy, axis=1, keepdims=True)

            ############ Dropouts ################
            keep_mask = apply_dropout(edge_visible, p_base=0.05, p_edge=0.4, gamma=2.5)
            origins_kept = ray_origins[visible_hit_idx][keep_mask]
            dirs_kept = dirs_noisy[keep_mask]
            t_kept = t_noisy[keep_mask]
            geom_ids_kept = geom_ids[visibility_mask][keep_mask]
            points_final = origins_kept + t_kept[:, None] * dirs_kept

            ############ Edge Outliers ################
            geom_depth_ranges = compute_geom_depth_ranges(t_kept, geom_ids_kept)
            shadow_plane = compute_shadow_plane(self.app.target_mesh, cam_pos, margin=0.005)

            geom_ids_img = np.full_like(hit_mask, fill_value=-1, dtype=int)
            geom_ids_img[hit_mask] = geom_ids
            pts_out, geom_out = generate_edge_outliers(
                origins=ray_origins[invalid_hit_idx],
                dirs=ray_dirs[invalid_hit_idx],
                edge_strength=edge_flat[invalid_hit_idx],
                geom_ids=geom_ids_img[invalid_hit_idx],
                geom_depth_ranges=geom_depth_ranges,
                outlier_prob=0.5,
                geom_id_outlier=-1,
                target_geom_id=target_geom_id,
                shadow_plane=shadow_plane
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
            # pcd = add_outliers(pcd)
            pcd = pcd.voxel_down_sample(0.001)
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=50))
            orient_normals_using_cameras(pcd, cam_pos)
            normalize_normals(pcd)
            validate_normals(pcd)
            self.synthetic_targets.append(pcd)

            if not self.app.headless:
                self.combobox_targets.add_item(f"synthetic_sample_{len(self.synthetic_targets)}")
                self.app.update_progress((i + 1) / self.num_targets)

    def save_synthetic_targets(self):
        base_path = Path.cwd() / "synthetic_target" / self.app.mesh_basename
        for start, prefix in enumerate(["train", "test"]):
            folder_path = base_path / prefix
            folder_path.mkdir(parents=True, exist_ok=True)
            for i, pcd in enumerate(self.synthetic_targets[start::2]):
                pointcloud_to_ply(pcd, folder_path / f"{prefix}_sample_{i}.ply")

    def preview_synthetic_target(self, selected_text: str, selected_index: int) -> None:
        if self.app.headless:
            return
        if self.synthetic_targets[selected_index] != None:
            self.app.main_thread(lambda: self.app._clear_scene())
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("synthetic_target", self.synthetic_targets[selected_index], self.app.default_point_material))
        self.app.scene.force_redraw()
