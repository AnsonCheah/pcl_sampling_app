import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import numpy as np
from tkinter import Tk, filedialog
from pathlib import Path
from stages.stage_base import BaseStage
from enums import Stage

class ImportMeshStage(BaseStage):
    def __init__(self, app):
        self.name = Stage.IMPORT_MESH.name
        self.file_path = None  # may be pre-set by caller to skip the dialog
        super().__init__(app)

    def build_panel(self):
        if self.app.headless:
            return
        v = gui.Vert(4)
        title = gui.Label("Import Mesh")
        
        self.btn_load = self.register_widget(gui.Button("Load STL"))
        self.btn_load.set_on_clicked(lambda: self.start(run_on_main=True))
        self.btn_reset = self.register_widget(gui.Button("Clear Mesh"), enabled_if=lambda: self.app.target_mesh is not None)
        self.btn_reset.set_on_clicked(self.reset)

        self.btn_express = self.register_widget(gui.Button("Express Sampling"), enabled_if=lambda: self.app.target_mesh is not None and not self.app.express_sampling_busy)
        self.btn_express.set_on_clicked(self.app.start_express_sampling)

        self.btn_batch = self.register_widget(gui.Button("Batch Sampling"))
        self.btn_batch.set_on_clicked(self.app.start_batch_sampling)

        self.btn_next = self.register_widget(gui.Button("Next: Raycast"), enabled_if=lambda: self.app.target_mesh is not None)
        self.btn_next.set_on_clicked(lambda: self.app.set_stage(Stage.RAYCAST))

        v.add_child(title)
        v.add_child(gui.Label(""))
        v.add_child(self.btn_load)
        v.add_child(self.btn_reset)
        v.add_child(gui.Label(""))
        v.add_child(self.btn_express)
        v.add_child(self.btn_batch)
        # v.add_child(gui.Label(""))
        # v.add_child(gui.Label(""))
        v.add_child(self.btn_next)

        print(f"[Import Mesh] panel loaded")
        return v

    def _refresh_ui(self):
        """Refresh scene and button states"""
        if self.app.headless:
            print("headless, skipped")
            return

        # self.app.hide_geoms_in_scene()
        # if self.app.has_geom("mesh"): 
        #     print(f"scene has mesh geom")
        #     self.app.show_geom_in_scene(["mesh"])
        self.app.main_thread(self.app._clear_scene)
        
        if self.app.target_mesh is not None:
            # print(f"adding mesh")
            # self.app.add_geom_in_scene("mesh", self.app.target_mesh)
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("mesh", self.app.target_mesh, self.app.default_material))
            self.app.main_thread(self.app._reframe)
        self.app.scene.force_redraw()
        self.enable_widgets()

    def reset(self):
        """Clear mesh-related data"""
        # self.app.main_thread(self.app._clear_scene)
        self.app.target_mesh = None
        self.app.raw_pcd = None
        self.app.cropped_pcd = None
        self.app.down_pcd = None
        self.app.output_pcd_path = None
        self.app.mesh_basename = None
        self.app.convex_meshes = []

        self._refresh_ui()

    # ===============================
    # Worker
    # ===============================
    def _on_worker_start(self):
        if not self.file_path:
            if self.app.headless:
                raw = input("Enter path to STL file: ").strip().strip('"').strip("'")
                self.file_path = Path(raw) if raw else None
            else:
                self.file_path = self._open_stl_dialog()
        if not self.file_path:
            self.enable_widgets()
            return

    def worker(self, mesh=None):
        """Load and preprocess mesh"""
        if not self.file_path:
            return
        if not mesh:
            mesh = o3d.io.read_triangle_mesh(str(self.file_path))
        if mesh.is_empty():
            print("[WARN] Empty mesh")
            return

        bbox = mesh.get_axis_aligned_bounding_box()
        extent_max = bbox.get_extent().max()
        if 5.0 < extent_max < 5000.0:
            print(f"[INFO] Converting units mm → m: {self.file_path.name}")
            mesh.scale(0.001, center=(0, 0, 0))

        mesh.compute_vertex_normals()
        mesh.translate(-mesh.get_center())
        self.app.target_mesh = mesh
        self.app.mesh_basename = self.file_path.stem

        # self.app.main_thread(self.app._clear_scene)

        self.app.raw_pcd = None
        self.app.cropped_pcd = None
        self.app.down_pcd = None
        # Convex decomposition now runs in DecomposeStage (between SAVE and SYNTHETIC),
        # so convex_meshes is rebuilt fresh there. Clear any stale hulls here.
        self.app.convex_meshes = []
        print(f"mesh loaded")

    def _open_stl_dialog(self):
        Tk().withdraw()
        path = filedialog.askopenfilename(initialdir=Path.cwd(), filetypes=[("STL files", "*.stl *.STL")])
        if not path:
            return None
        path = Path(path)
        self.mesh_basename = path.stem
        return path
