import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
from tkinter import Tk, filedialog
from pathlib import Path
from stages.stage_base import BaseStage
from enums import Stage

class ImportMeshStage(BaseStage):
    def __init__(self, app):
        self.name = Stage.IMPORT_MESH.name
        super().__init__(app)

    def build_panel(self):
        v = gui.Vert(4)
        title = gui.Label("Import Mesh")
        
        self.btn_load = gui.Button("Load STL")
        self.btn_load.set_on_clicked(lambda: self.start(run_on_main=True))

        self.btn_reset = gui.Button("Clear Mesh")
        self.btn_reset.set_on_clicked(self.reset)

        # self.btn_express = gui.Button("Express Sampling")
        # self.btn_express.set_on_clicked(self.app.start_express_sampling)

        # self.btn_batch = gui.Button("Batch Sampling")
        # self.btn_batch.set_on_clicked(self.app.start_batch_sampling)

        self.btn_next = gui.Button("Next: Raycast")
        self.btn_next.set_on_clicked(lambda: self.app.set_stage(Stage.RAYCAST))

        # Register widgets for auto enable/disable
        for w in [
            self.btn_load,
            self.btn_reset,
            # self.btn_express,
            # self.btn_batch,
            self.btn_next,
        ]:
            self.register_widget(w)

        v.add_child(title)
        v.add_child(gui.Label(""))
        v.add_child(self.btn_load)
        v.add_child(self.btn_reset)
        v.add_child(gui.Label(""))
        # v.add_child(self.btn_express)
        # v.add_child(self.btn_batch)
        # v.add_child(gui.Label(""))
        # v.add_child(gui.Label(""))
        v.add_child(self.btn_next)

        print(f"[Import Mesh] panel loaded")
        return v

    def init(self):
        """Refresh scene and button states"""
        if self.app.headless:
            print("headless, skipped")
            return

        self.app.main_thread(self.app._clear_scene)

        if self.app.target_mesh is not None:
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("mesh", self.app.target_mesh, self.app.default_material))
            self.app.main_thread(self.app._reframe)

        # Button states
        # has_mesh = self.app.target_mesh is not None
        # self.btn_next.enabled = has_mesh
        # self.btn_express.enabled = has_mesh
        # self.btn_batch.enabled = has_mesh

    def reset(self):
        """Clear mesh-related data"""
        self.app.target_mesh = None
        self.app.raw_pcd = None
        self.app.cropped_pcd = None
        self.app.down_pcd = None
        self.app.bbox_corners = None

        self.init()

    # ===============================
    # Worker
    # ===============================
    def _on_worker_start(self):
        self.file_path = self._open_stl_dialog()
        if not self.file_path:
            self.enable_widgets()
            return

    def worker(self):
        """Load and preprocess mesh"""
        if not self.file_path:
            return

        mesh = o3d.io.read_triangle_mesh(str(self.file_path))
        if mesh.is_empty():
            print("[WARN] Empty mesh")
            return

        # Reset downstream data
        self.app.raw_pcd = None
        self.app.cropped_pcd = None
        self.app.down_pcd = None

        # -------------------------
        # Unit conversion
        # -------------------------
        bbox = mesh.get_axis_aligned_bounding_box()
        extent_max = bbox.get_extent().max()

        if 5.0 < extent_max < 5000.0:
            print(f"[INFO] Converting units mm → m: {self.file_path.name}")
            mesh.scale(0.001, center=(0, 0, 0))

        # -------------------------
        # Normalize mesh
        # -------------------------
        mesh.compute_vertex_normals()
        mesh.translate(-mesh.get_center())

        self.app.target_mesh = mesh
        self.app.mesh_basename = self.file_path.stem

        # -------------------------
        # Bounding box + sampling params
        # -------------------------
        bbox = mesh.get_axis_aligned_bounding_box()
        self.app.bbox_corners = np.asarray(bbox.get_box_points())
        extent_min = bbox.get_extent().min()

        self.app.ray_spacing = np.round(np.clip(extent_min / 100.0, 0.0005, 0.003), 4)
        self.app.voxel_size = np.round(np.clip(extent_min / 50.0, 0.001, 0.005), 4)
        print(f"[INFO] Ray spacing: {self.app.ray_spacing*1000:.2f} mm")
        print(f"[INFO] Voxel size:  {self.app.voxel_size*1000:.2f} mm")
        print(f"mesh loaded")

    def _open_stl_dialog(self):
        Tk().withdraw()
        path = filedialog.askopenfilename(initialdir=Path.cwd(), filetypes=[("STL files", "*.stl *.STL")])
        if not path:
            return None
        path = Path(path)
        self.mesh_basename = path.stem
        return path
