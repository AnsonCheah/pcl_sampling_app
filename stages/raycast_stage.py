import numpy as np
import copy
import logging
import open3d as o3d
import open3d.visualization.gui as gui
from enums import Stage
from stages.stage_base import BaseStage
from utilities import fibonacci_sphere, orient_normals_using_cameras, normalize_normals, validate_normals

class RaycastStage(BaseStage):
    def __init__(self, app):
        self.name = Stage.RAYCAST.name
        super().__init__(app)
        self.camera_distance = 1.5
        self.num_views = 20
        self.ray_margin_mm = 10.0
        self.ray_spacing = 0.001

    def build_panel(self):
        v = gui.Vert(4)

        v.add_child(gui.Label("Raycasting"))

        self.camera_distance_slider = gui.Slider(gui.Slider.DOUBLE)
        self.camera_distance_slider.set_limits(0.5, 3.0)
        self.camera_distance_slider.double_value = 1.5

        self.num_views_slider = gui.Slider(gui.Slider.INT)
        self.num_views_slider.set_limits(4, 100)
        self.num_views_slider.int_value = 20

        self.btn_raycast = gui.Button("Raycast")
        self.btn_raycast.set_on_clicked(self.start)

        self.btn_reset = gui.Button("Clear Raycast")
        self.btn_reset.set_on_clicked(self.reset)

        self.btn_next = gui.Button("Next: Crop")
        self.btn_next.set_on_clicked(lambda: self.app.set_stage(Stage.CROP))

        self.btn_back = gui.Button("Back: Import Mesh")
        self.btn_back.set_on_clicked(lambda: self.app.set_stage(Stage.IMPORT_MESH))

        for w in [
            self.camera_distance_slider,
            self.num_views_slider,
            self.btn_raycast,
            self.btn_reset,
            self.btn_next,
            self.btn_back,
        ]:
            self.register_widget(w)

        v.add_child(gui.Label("Camera Distance"))
        v.add_child(self.camera_distance_slider)
        v.add_child(gui.Label("Number of Views"))
        v.add_child(self.num_views_slider)
        v.add_child(self.btn_raycast)
        v.add_child(self.btn_reset)
        v.add_child(self.btn_back)
        v.add_child(self.btn_next)
        print(f"[Raycast] panel loaded")
        # print(f"[Raycast] Childs: {[type(c) for c in v.get_children()]}")
        return v

    # ---------- Stage lifecycle ----------
    def init(self):
        if self.app.headless:
            return
        self.app.main_thread(lambda: self.app._clear_scene())
        if self.app.raw_pcd is not None:
            print(f"[Raycast] adding raw_pcd")
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("raw_pcd", self.app.raw_pcd, self.app.default_point_material))
        elif self.app.target_mesh is not None:
            print(f"[Raycast] adding mesh")
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("mesh", self.app.target_mesh, self.app.default_material))
        self.app.scene.force_redraw()

    def reset(self):
        self.app.raw_pcd = None
        self.app.cropped_pcd = None
        self.app.down_pcd = None
        self.init()

    # ---------- Worker ----------
    def worker(self):
        if self.app.target_mesh is None:
            print("[WARN] No mesh loaded")
            return
        if not self.app.headless:
            self.app.main_thread(lambda: self.disable_widgets())
            # self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.RAYCAST], False))
            # self.main_thread(lambda: self.enable_button(self.next_stage_buttons[Stage.RAYCAST], False))
            self.camera_distance = self.camera_distance_slider.double_value
            self.num_views = self.num_views_slider.int_value
            self.app.show_progress("Raycasting mesh...")

        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.app.target_mesh))
        view_dirs = fibonacci_sphere(self.num_views)
        all_points = []
        all_cam_pos = []
        bbox = self.app.target_mesh.get_axis_aligned_bounding_box()
        bbox_corners = np.asarray(bbox.get_box_points())
        for view_index, view_dir in enumerate(view_dirs):
            cam_pos = view_dir * (self.camera_distance)
            forward = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross([0, 0, 1], forward)
            if np.linalg.norm(right) < 1e-6:
                right = np.cross([0, 1, 0], forward)
            right /= np.linalg.norm(right)
            up = np.cross(forward, right)

            # Project bbox corners onto camera plane with margin
            rel = bbox_corners - cam_pos
            x_proj = rel @ right
            y_proj = rel @ up
            margin = self.ray_margin_mm * 1e-3
            xs = np.arange(x_proj.min() - margin, x_proj.max() + margin, self.ray_spacing)
            ys = np.arange(y_proj.min() - margin, y_proj.max() + margin, self.ray_spacing)

            if len(xs) == 0 or len(ys) == 0:
                continue

            uu, vv = np.meshgrid(xs, ys)
            origins = (cam_pos + uu[..., None] * right + vv[..., None] * up)

            dirs = forward[None, None, :].repeat(origins.shape[0], axis=0)
            dirs = dirs.repeat(origins.shape[1], axis=1)

            rays = np.concatenate([origins.reshape(-1, 3), dirs.reshape(-1, 3)],axis=1)
            rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

            hits = scene.cast_rays(rays)
            hit_mask = hits["t_hit"].isfinite()

            if not hit_mask.any():
                continue

            hit_points = (rays[hit_mask][:, :3] + rays[hit_mask][:, 3:] * hits["t_hit"][hit_mask].reshape((-1, 1))).numpy()
            cam_pos_arr = np.repeat(cam_pos[None, :], len(hit_points), axis=0)
            all_points.append(hit_points)
            all_cam_pos.append(cam_pos_arr)

            if not self.app.headless:
                self.app.update_progress((view_index + 1) / self.num_views)

        if not all_points:
            logging.warning(f"No points generated from mesh")
            return

        # Aggregate
        points = np.vstack(all_points)
        cam_positions = np.vstack(all_cam_pos)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd.remove_non_finite_points()
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = 0.005, max_nn = 60))
        orient_normals_using_cameras(pcd, cam_positions)
        print(f"[INFO] Total points sampled: {len(pcd.points)}")
        initial_voxel = (self.ray_spacing)*0.3
        pcd = pcd.voxel_down_sample(initial_voxel)
        normalize_normals(pcd)
        validate_normals(pcd)
        print(f"[INFO] Initial voxelized points: {len(pcd.points)}")
        self.app.raw_pcd = pcd
        self.app.cropped_pcd = copy.deepcopy(self.app.raw_pcd)

        print("raycast finished")
