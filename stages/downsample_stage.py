import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from enums import Stage
from stages.stage_base import BaseStage
from geometry.geom_utils import mask_point_cloud, normalize_normals, pcd_geocenter, extract_edge_points
from geometry.math_utils import find_cdf_knee
import copy

class DownsampleStage(BaseStage):
    def __init__(self, app):
        self.name = Stage.DOWNSAMPLE.name
        self.use_adaptive = False
        self.coarse_factor = 2.0
        self.curvature_k_neighbors = 5
        self.cloud_radio_idx = 0  # 0=surface, 1=edge
        super().__init__(app)

    def build_panel(self):
        if self.app.headless:
            return
        v = gui.Vert(4)

        self.radio_adaptive = self.register_widget(gui.RadioButton(gui.RadioButton.HORIZ))
        self.radio_adaptive.set_items(["Uniform", "Adaptive"])
        self.radio_adaptive.selected_index = 1 if self.use_adaptive else 0
        self.btn_downsample = self.register_widget(gui.Button("Downsample"))
        self.btn_downsample.set_on_clicked(self.start)
        self.btn_recenter = self.register_widget(gui.Button("Recenter Point Cloud"), lambda: self.app.down_pcd is not None)
        self.btn_recenter.set_on_clicked(self.recenter_mesh_pcd)

        self.btn_reset = self.register_widget(gui.Button("Restart Downsample"), lambda: self.app.down_pcd is not None)
        self.btn_reset.set_on_clicked(self.reset)
        self.btn_next = self.register_widget(gui.Button("Next: Save"), lambda: self.app.down_pcd is not None)
        self.btn_next.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value + 1)))
        self.btn_back = self.register_widget(gui.Button("Back: Crop"))
        self.btn_back.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value - 1)))

        self.radio_cloud = self.register_widget(
            gui.RadioButton(gui.RadioButton.HORIZ),
            lambda: self.app.down_pcd_surface is not None
        )
        self.radio_cloud.set_items(["Surface", "Edge"])
        self.radio_cloud.selected_index = 0
        self.radio_cloud.set_on_selection_changed(self._on_cloud_radio_changed)

        v.add_child(gui.Label("Downsampling"))
        v.add_child(gui.Label(""))
        v.add_child(self.radio_adaptive)
        v.add_child(self.btn_downsample)
        v.add_child(self.btn_recenter)
        v.add_child(self.btn_reset)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label("Show Cloud:"))
        v.add_child(self.radio_cloud)
        v.add_child(gui.Label(""))
        v.add_child(self.btn_back)
        v.add_child(self.btn_next)
        print("loaded downsample panel")

        return v

    def _on_cloud_radio_changed(self, idx):
        self.cloud_radio_idx = idx
        self.app._clear_scene()
        if idx == 0 and self.app.down_pcd_surface is not None:
            self.app.scene.scene.add_geometry("surface_pcd", self.app.down_pcd_surface, self.app.default_point_material)
        elif idx == 1 and self.app.down_pcd_edge is not None:
            self.app.scene.scene.add_geometry("edge_pcd", self.app.down_pcd_edge, self.app.default_point_material)
        self.app.scene.force_redraw()

    def _refresh_ui(self):
        if self.app.headless:
            return

        if self.app.down_pcd_surface is not None:
            self.app.main_thread(lambda: self.app._clear_scene())
            if self.cloud_radio_idx == 0:
                self.app.main_thread(lambda: self.app.scene.scene.add_geometry("surface_pcd", self.app.down_pcd_surface, self.app.default_point_material))
            elif self.cloud_radio_idx == 1 and self.app.down_pcd_edge is not None:
                self.app.main_thread(lambda: self.app.scene.scene.add_geometry("edge_pcd", self.app.down_pcd_edge, self.app.default_point_material))
        elif self.app.cropped_pcd is not None:
            self.app.main_thread(lambda: self.app._clear_scene())
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("cropped_pcd", self.app.cropped_pcd, self.app.default_point_material))
        self.app.main_thread(lambda: self.app.scene.force_redraw())
        self.app.main_thread(self.enable_widgets)

    def reset(self):
        self.app.down_pcd = None
        self.app.down_pcd_surface = None
        self.app.down_pcd_edge = None
        self.app.output_pcd_path = None
        self._refresh_ui()

    def worker(self):
        if not self.app.headless:
            self.use_adaptive = self.radio_adaptive.selected_index == 1
            self.app.show_progress("Downsampling point cloud...")
        if self.app.cropped_pcd is None:
            print("cropped pcd is none")
            return

        bbox = self.app.cropped_pcd.get_minimal_oriented_bounding_box()
        self.voxel_size = np.round(np.clip((np.asarray(bbox.volume()) / 3), 0.001, 0.005), 4)
        self.adaptive_voxel_downsample() if self.use_adaptive else self.uniform_voxel_downsample()

        # Edge extraction on the downsampled surface cloud
        if not self.app.headless:
            self.app.show_progress("Extracting edge points...")
        edge_mask = extract_edge_points(self.app.down_pcd_surface, self.voxel_size)
        pcd_edge_raw = mask_point_cloud(self.app.down_pcd_surface, edge_mask)
        self.app.down_pcd_edge = normalize_normals(pcd_edge_raw.voxel_down_sample(self.voxel_size))

        self.app.geocenter = np.round(pcd_geocenter(self.app.down_pcd), decimals=5)
        print(f"[INFO] Downsampled {self.voxel_size * 1000:.2f}mm from {len(self.app.cropped_pcd.points)} to {len(self.app.down_pcd.points)} points")
        print(f"[INFO] Edge cloud: {len(self.app.down_pcd_edge.points)} points ({edge_mask.sum()} before voxel pass)")
        self._refresh_ui()

    def uniform_voxel_downsample(self):
        self.app.down_pcd = normalize_normals(self.app.cropped_pcd.voxel_down_sample(self.voxel_size))
        self.app.down_pcd_surface = self.app.down_pcd

    def adaptive_voxel_downsample(self):
        variation = self.compute_curvature(self.app.cropped_pcd)
        threshold, percentile, _ = find_cdf_knee(variation)
        feature_mask = variation >= threshold

        pcd_feature = mask_point_cloud(self.app.cropped_pcd, feature_mask)
        pcd_flat = mask_point_cloud(self.app.cropped_pcd, ~feature_mask)
        pcd_feature = pcd_feature.voxel_down_sample(self.voxel_size)
        pcd_flat = pcd_flat.voxel_down_sample(self.voxel_size * self.coarse_factor)
        self.app.feature_pcd = pcd_feature
        self.app.pcd_flat = pcd_flat
        self.app.down_pcd = pcd_feature + pcd_flat
        normalize_normals(self.app.down_pcd)
        self.app.down_pcd_surface = copy.deepcopy(self.app.down_pcd)

    def compute_curvature(self, pcd):
        pts = np.asarray(pcd.points)
        n_points = len(pts)
        tree = o3d.geometry.KDTreeFlann(pcd)
        curv = np.zeros(n_points, dtype=np.float64)
        batch_size = max(1, n_points // 100)  # Update progress every 1%

        for i in range(n_points):
            _, idx, _ = tree.search_knn_vector_3d(pts[i], self.curvature_k_neighbors)
            nbrs = np.asarray(pts[idx])
            centered = nbrs - nbrs.mean(axis=0)
            C = (centered.T @ centered) / (len(nbrs) - 1)
            eigvals = np.linalg.eigvalsh(np.asarray(C))
            eigval_sum = eigvals.sum()
            curv[i] = eigvals[0] / eigval_sum if eigval_sum > 1e-12 else 0.0
            if not self.app.headless and (i + 1) % batch_size == 0:
                self.app.update_progress((i + 1) / n_points)

        if not self.app.headless:
            self.app.update_progress(1.0)

        return curv

    def recenter_mesh_pcd(self):
        if self.app.cropped_pcd is None:
            return
        T = pcd_geocenter(self.app.down_pcd_surface)

        self.app.down_pcd_surface.transform(T)
        # In uniform mode down_pcd is the same object as down_pcd_surface — skip to avoid double-transform
        if self.app.down_pcd is not self.app.down_pcd_surface:
            self.app.down_pcd.transform(T)
        if self.app.down_pcd_edge is not None:
            self.app.down_pcd_edge.transform(T)
        if self.app.feature_pcd is not None:
            self.app.feature_pcd.transform(T)
        if self.app.pcd_flat is not None:
            self.app.pcd_flat.transform(T)
        self.app.raw_pcd.transform(T)
        self.app.cropped_pcd.transform(T)
        self.app.target_mesh.transform(T)
        for mesh in self.app.convex_meshes:
            mesh.transform(T)
        self.app.geocenter = np.round(pcd_geocenter(self.app.down_pcd_surface), decimals=5)
        print("recentered mesh and pointcloud")
        self.app.main_thread(self.app._reframe)
        self._refresh_ui()
