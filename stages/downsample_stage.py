import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from enums import Stage
from stages.stage_base import BaseStage
from utilities import mask_point_cloud, find_cdf_knee, normalize_normals, pcd_geocenter

class DownsampleStage(BaseStage):
    def __init__(self, app):
        self.name = Stage.DOWNSAMPLE.name
        self.use_adaptive = True
        self.coarse_factor = 2.0
        self.curvature_k_neighbors = 5
        super().__init__(app)

    def build_panel(self):
        v = gui.Vert(4)
        
        self.adaptive_checkbox = gui.Checkbox("Adaptive sampling")
        self.adaptive_checkbox.checked = self.use_adaptive
        self.btn_downsample = gui.Button("Downsample")
        self.btn_downsample.set_on_clicked(self.start)
        self.btn_recenter = gui.Button("Recenter Point Cloud")
        self.btn_recenter.set_on_clicked(self.recenter_mesh_pcd)

        self.btn_reset = gui.Button("Restart Downsample")
        self.btn_reset.set_on_clicked(self.reset)
        self.btn_next = gui.Button("Next: Save")
        self.btn_next.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value + 1)))
        # self.worker_buttons[Stage.DOWNSAMPLE] = self.btn_downsample
        # self.next_stage_buttons[Stage.DOWNSAMPLE] = self.btn_next
        self.btn_back = gui.Button("Back: Crop")
        self.btn_back.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value - 1)))

        for w in [
            self.adaptive_checkbox,
            self.btn_downsample,
            self.btn_reset,
            self.btn_next,
            self.btn_back,
        ]:
            self.register_widget(w)

        v.add_child(gui.Label("Downsampling"))
        v.add_child(gui.Label(""))
        v.add_child(self.adaptive_checkbox)
        v.add_child(self.btn_downsample)
        v.add_child(self.btn_recenter)
        v.add_child(self.btn_reset)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(self.btn_back)
        v.add_child(self.btn_next)
        print("loaded downsample panel")

        return v
        
    def init(self): 
        if self.app.headless:
            return       

        if self.app.down_pcd != None:
            self.app.main_thread(lambda: self.app._clear_scene())
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("down_pcd", self.app.down_pcd, self.app.default_point_material))
        elif self.app.cropped_pcd != None:
            self.app.main_thread(lambda: self.app._clear_scene())
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("cropped_pcd", self.app.cropped_pcd, self.app.default_point_material))
        self.app.scene.force_redraw()
        # self.app.main_thread(lambda: self.app.enable_button(self.worker_buttons[Stage.DOWNSAMPLE], True))
        # self.app.main_thread(lambda: self.app.enable_button(self.next_stage_buttons[Stage.DOWNSAMPLE], (self.app.down_pcd != None)))
        # self.app.main_thread(lambda: self.app.enable_button(self.btn_recenter, True))

    def reset(self):
        self.app.down_pcd = None 
        self.init()

    def worker(self):
        if not self.app.headless:
            self.use_adaptive = self.adaptive_checkbox.checked
            # self.app.main_thread(lambda: self.app.enable_button(self.worker_buttons[Stage.DOWNSAMPLE], False))
            self.app.show_progress("Downsampling point cloud...")
        if self.app.cropped_pcd is None:
            print("cropped pcd is none")
            return
        
        bbox = self.app.cropped_pcd.get_minimal_oriented_bounding_box()
        self.voxel_size = np.round(np.clip((bbox.volume() / 3), 0.001, 0.005), 4)
        self.adaptive_voxel_downsample() if self.use_adaptive else self.uniform_voxel_downsample()
        print(f"[INFO] Downsampled {self.voxel_size * 1000}mm from {len(self.app.cropped_pcd.points)} to {len(self.app.down_pcd.points)} points")
        self.init()

    def uniform_voxel_downsample(self):
        self.app.down_pcd = normalize_normals(self.app.cropped_pcd.voxel_down_sample(self.voxel_size))

    def adaptive_voxel_downsample(self):
        variation = self.compute_curvature(self.app.cropped_pcd)
        threshold, percentile, _ = find_cdf_knee(variation)
        feature_mask = variation >= threshold

        pcd_feature = mask_point_cloud(self.app.cropped_pcd, feature_mask)
        pcd_flat = mask_point_cloud(self.app.cropped_pcd, ~feature_mask)
        pcd_feature = pcd_feature.voxel_down_sample(self.voxel_size)
        pcd_flat = pcd_flat.voxel_down_sample(self.voxel_size * self.coarse_factor)
        self.app.down_pcd = pcd_feature + pcd_flat
        normalize_normals(self.app.down_pcd)

    def compute_curvature(self, pcd):
        pts = np.asarray(pcd.points)
        n_points = len(pts)
        tree = o3d.geometry.KDTreeFlann(pcd)
        curv = np.zeros(n_points, dtype=np.float64)
        batch_size = max(1, n_points // 100)  # Update progress every 1%
        
        for i in range(n_points):
            _, idx, _ = tree.search_knn_vector_3d(pts[i], self.curvature_k_neighbors)
            nbrs = pts[idx]
            centered = nbrs - nbrs.mean(axis=0)
            C = (centered.T @ centered) / (len(nbrs) - 1)
            eigvals = np.linalg.eigvalsh(C)
            eigval_sum = eigvals.sum()
            curv[i] = eigvals[0] / eigval_sum if eigval_sum > 1e-12 else 0.0
            if not self.app.headless and (i + 1) % batch_size == 0:
                self.app.update_progress((i + 1) / n_points)
        
        if not self.app.headless:
            self.app.update_progress(1.0)
        
        return curv

    def recenter_mesh_pcd(self):
        if self.app.cropped_pcd == None:
            return
        # if not self.app.headless:
            # self.app.main_thread(lambda: self.app.enable_button(self.btn_recenter, False))
        T = pcd_geocenter(self.app.cropped_pcd)
        if self.app.down_pcd != None:
            self.app.down_pcd.transform(T)
        self.app.target_mesh.transform(T)
        self.app.raw_pcd.transform(T)
        self.app.cropped_pcd.transform(T)
        self.init()
        print("recentered pointcloud")