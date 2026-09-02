import numpy as np
import copy
import logging
import open3d as o3d
import open3d.visualization.gui as gui
from enums import Stage
from stages.stage_base import BaseStage
from geometry.geom_utils import fibonacci_sphere, orient_normals_using_cameras, normalize_normals, validate_normals, camera_view_matrix, O3DSceneObject
from sensor.scene_render import scene_render

class RaycastStage(BaseStage):
    downstream = {
        "raw_pcd": lambda: None,
        "cropped_pcd": lambda: None,
        "point_count_mean": lambda: None,
        "point_count_range": lambda: None,
        "aspect_ratio_range": lambda: None,
        "area_ratio_range": lambda: None,
        "ref_cam_distance": lambda: None,
    }

    def __init__(self, app):
        self.name = Stage.RAYCAST.name
        super().__init__(app)
        self.camera_distance = 1.5
        self.num_views = 20
        self.ray_spacing = 0.001
        self.point_count_mean = 0
        self.point_count_range = (0,0)
        self.view_sphere = fibonacci_sphere(self.num_views)
        self.fov_deg = 41.11
        self.res_width = 1920
        self.res_height = 1200
        self.point_counts = None
        self.point_count_tolerance = 0.5

    def build_panel(self):
        if self.app.headless:
            return
        v = gui.Vert(4)

        v.add_child(gui.Label("Raycasting"))

        self.camera_distance_slider = self.register_widget(gui.Slider(gui.Slider.DOUBLE))
        self.camera_distance_slider.set_limits(0.5, 3.0)
        self.camera_distance_slider.double_value = 1.5

        self.num_views_slider = self.register_widget(gui.Slider(gui.Slider.INT))
        self.num_views_slider.set_limits(4, 100)
        self.num_views_slider.int_value = 20

        self.btn_raycast = self.register_widget(gui.Button("Raycast"))
        self.btn_raycast.set_on_clicked(self._start_with_current_settings)

        self.btn_reset = self.register_widget(gui.Button("Clear Raycast"), lambda: self.app.raw_pcd is not None)
        self.btn_reset.set_on_clicked(self.reset)

        v.add_child(gui.Label("Camera Distance"))
        v.add_child(self.camera_distance_slider)
        v.add_child(gui.Label("Number of Views"))
        v.add_child(self.num_views_slider)
        v.add_child(self.btn_raycast)
        v.add_child(self.btn_reset)
        print(f"[Raycast] panel loaded")
        return v

    def next_enabled(self) -> bool:
        return self.app.raw_pcd is not None

    # ---------- Stage lifecycle ----------
    def _refresh_ui(self):
        if self.app.headless:
            return
        self.app.main_thread(lambda: self.app._clear_scene())
        if self.app.raw_pcd is not None:
            print(f"[Raycast] adding raw_pcd")
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("raw_pcd", self.app.raw_pcd, self.app.default_point_material))
        elif self.app.target_mesh is not None:
            print(f"[Raycast] adding mesh")
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("mesh", self.app.target_mesh, self.app.default_material))
        self.app.main_thread(self.app._reframe)   # content swap -> reframe (posts a redraw too)
        self.app.main_thread(self.enable_widgets)

    # ---------- Worker ----------
    def _start_with_current_settings(self):
        self.camera_distance = self.camera_distance_slider.double_value
        self.num_views = self.num_views_slider.int_value
        self.start()

    def worker(self):
        if self.app.target_mesh is None:
            print("[WARN] No mesh loaded")
            return
        self.app.clear_state_from(self.stage_key)
        if not self.app.headless:
            self.app.main_thread(lambda: self.disable_widgets())
            self.app.show_progress("Raycasting mesh...")
        print(f"[RAYCAST] Worker started")
        all_points = []
        all_cam_pos = []
        self.point_counts = np.zeros((self.num_views))
        # Per-view 2D silhouette metrics of the isolated part -- used to auto-derive the
        # aspect-ratio / area-ratio candidate filter in RenderStage (mirrors the real
        # Mask2Former post-filter, which gates the network's 2D mask output per part).
        aspects     = np.zeros((self.num_views))
        area_ratios = np.zeros((self.num_views))
        total_px    = float(self.res_width * self.res_height)
        raycast_dict = {"ref_mesh": O3DSceneObject(geom=self.app.target_mesh, T_gt=np.eye(4))}
        mesh_center = np.asarray(self.app.target_mesh.get_center())
        for view_index, view_dir in enumerate(self.view_sphere):
            cam_pos = mesh_center + view_dir * self.camera_distance
            look_at = mesh_center
            T_cam = camera_view_matrix(cam_pos, look_at)
            initial_render = scene_render(raycast_dict, T_cam, look_at, self.fov_deg, self.res_width, self.res_height)
            hit_points = initial_render["points"]
            cam_pos_arr = np.repeat(cam_pos[None, :], len(hit_points), axis=0)
            all_points.append(hit_points)
            all_cam_pos.append(cam_pos_arr)
            self.point_counts[view_index] = len(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hit_points)).voxel_down_sample(self.ray_spacing).points)

            # Single isolated part -> its 2D mask is exactly the set of hit pixels.
            pidx = initial_render["pixel_idx"]
            H, W = initial_render["res"]
            if len(pidx):
                rows = pidx // W
                cols = pidx % W
                h = int(rows.max() - rows.min()) + 1
                w = int(cols.max() - cols.min()) + 1
                aspects[view_index]     = max(h, w) / max(min(h, w), 1)
                area_ratios[view_index] = len(pidx) / total_px

            if not self.app.headless:
                self.app.update_progress((view_index + 1) / self.num_views)
        self.app.point_count_mean = self.point_count_mean = int(np.mean(self.point_counts))
        print(f"[RAYCAST] Point count per view: mean={self.point_count_mean}, min={np.min(self.point_counts)}, max={np.max(self.point_counts)}")
        self.app.point_count_range = self.point_count_range = (int(np.min(self.point_counts * (1-self.point_count_tolerance))),
                                                               int(np.max(self.point_counts * (1+self.point_count_tolerance))))
        print(f"[RAYCAST] Point count range with tolerance {self.point_count_tolerance*100}%: {self.point_count_range}")

        # Auto-derived 2D candidate-filter ranges (tolerance widens [min, max] like the
        # point-count range). aspect_ratio is distance-invariant; area_ratio is recorded
        # at this camera distance and rescaled by (1/d^2) at filter time in RenderStage.
        valid = area_ratios > 0
        if valid.any():
            a = aspects[valid]; ar = area_ratios[valid]
            self.app.aspect_ratio_range = self.aspect_ratio_range = (
                float(a.min() * (1 - self.point_count_tolerance)),
                float(a.max() * (1 + self.point_count_tolerance)))
            self.app.area_ratio_range = self.area_ratio_range = (
                float(ar.min() * (1 - self.point_count_tolerance)),
                float(ar.max() * (1 + self.point_count_tolerance)))
            self.app.ref_cam_distance = self.camera_distance
            print(f"[RAYCAST] Aspect ratio range (tol {self.point_count_tolerance*100}%): {self.app.aspect_ratio_range}")
            print(f"[RAYCAST] Area ratio range  (tol {self.point_count_tolerance*100}%): {self.app.area_ratio_range}  @ cam_distance={self.camera_distance}")
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
        # print(f"[INFO] Initial voxelized points: {len(pcd.points)}")
        self.app.raw_pcd = pcd
        self.app.cropped_pcd = copy.deepcopy(self.app.raw_pcd)

        print("raycast finished")
