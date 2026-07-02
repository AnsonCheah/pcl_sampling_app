import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import numpy as np
from tkinter import Tk, filedialog
from pathlib import Path
from stages.stage_base import BaseStage
from enums import Stage

class ImportMeshStage(BaseStage):
    downstream = {
        "target_mesh": lambda: None,
        "mesh_basename": lambda: None,
    }

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
        self.btn_load.set_on_clicked(self._interactive_load)
        self.btn_reset = self.register_widget(gui.Button("Clear Mesh"), enabled_if=lambda: self.app.target_mesh is not None)
        self.btn_reset.set_on_clicked(self.reset)
        self.btn_center = self.register_widget(gui.Button("Center Mesh"), enabled_if=lambda: self.app.target_mesh is not None)
        self.btn_center.set_on_clicked(self.center_mesh)

        self.btn_express = self.register_widget(gui.Button("Express Sampling"), enabled_if=lambda: self.app.target_mesh is not None and not self.app.express_sampling_busy)
        self.btn_express.set_on_clicked(self.app.start_express_sampling)

        self.btn_batch = self.register_widget(gui.Button("Batch Sampling"))
        self.btn_batch.set_on_clicked(self.app.start_batch_sampling)

        v.add_child(title)
        v.add_child(gui.Label(""))
        v.add_child(self.btn_load)
        v.add_child(self.btn_reset)
        v.add_child(self.btn_center)
        v.add_child(gui.Label(""))
        v.add_child(self.btn_express)
        v.add_child(self.btn_batch)

        print(f"[Import Mesh] panel loaded")
        return v

    def next_enabled(self) -> bool:
        return self.app.target_mesh is not None

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

    # ===============================
    # Worker
    # ===============================
    def _interactive_load(self):
        self.file_path = None
        self.start(run_on_main=True)

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

        self.app.clear_state_from(self.stage_key)  # new mesh is valid; wipe all stale downstream state

        bbox = mesh.get_axis_aligned_bounding_box()
        extent_max = bbox.get_extent().max()
        if 5.0 < extent_max < 5000.0:
            print(f"[INFO] Converting units mm → m: {self.file_path.name}")
            mesh.scale(0.001, center=(0, 0, 0))

        mesh.compute_vertex_normals()
        self.app.target_mesh = mesh
        self.app.mesh_basename = self.file_path.stem
        print(f"mesh loaded")

    def center_mesh(self):
        """Translate mesh (and all downstream clouds) so mesh centroid is at world origin."""
        if self.app.target_mesh is None:
            return
        center = np.asarray(self.app.target_mesh.get_center())
        if np.linalg.norm(center) < 1e-9:
            return
        offset = -center
        self.app.target_mesh.translate(offset)
        seen = set()
        for attr in ("raw_pcd", "cropped_pcd", "down_pcd", "down_pcd_surface",
                     "down_pcd_edge", "feature_pcd", "pcd_flat"):
            obj = getattr(self.app, attr, None)
            if obj is not None and id(obj) not in seen:
                obj.translate(offset)
                seen.add(id(obj))
        for mesh in self.app.convex_meshes:
            mesh.translate(offset)
        self._refresh_ui()

    def _open_stl_dialog(self):
        Tk().withdraw()
        path = filedialog.askopenfilename(initialdir=Path.cwd(), filetypes=[("STL files", "*.stl *.STL")])
        if not path:
            return None
        path = Path(path)
        self.mesh_basename = path.stem
        return path
