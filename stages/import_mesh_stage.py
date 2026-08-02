import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import numpy as np
from tkinter import Tk, filedialog
from pathlib import Path
from stages.stage_base import BaseStage
from geometry.mesh_repair import analyze_mesh
from physics.mujoco_bin_scene import MAX_BIN_DIM
from enums import Stage


class ImportMeshStage(BaseStage):
    downstream = {
        "target_mesh": lambda: None,
        "mesh_basename": lambda: None,
    }

    def __init__(self, app):
        self.name = Stage.IMPORT_MESH.name
        self.file_path = None  # may be pre-set by caller to skip the dialog
        # (cleaned_mesh, report) awaiting the operator's keep/remove decision; see _resolve_pending.
        self._pending = None
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

        def redraw():
            self.app._clear_scene()
            if self.app.target_mesh is not None:
                self.app.scene.scene.add_geometry("mesh", self.app.target_mesh,
                                                  self.app.default_material)
            self.app._reframe()

        self.app.main_thread(redraw)
        self.enable_widgets()
        # Any import findings are surfaced once the mesh is on screen, so the operator can see
        # what the dialog is talking about.
        if self._pending is not None:
            self.app.main_thread(self._resolve_pending)

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
        self._pending = None

        # Validate before anything reads the bounding box. Export debris skews the AABB, which
        # otherwise drives the unit heuristic below, the camera framing, and — through the convex
        # hulls — the partition/tray cell sizing in MujocoBinScene.
        cleaned, report = analyze_mesh(mesh, bin_limit=MAX_BIN_DIM[:3])
        print(f"[MESH] {self.file_path.name}\n{report.summary()}")
        if report.errors:
            print(f"[WARN] refusing to import {self.file_path.name}: {'; '.join(report.errors)}")
            return

        # The unit scale is decided on the debris-free extent and applied to BOTH variants, so
        # the operator's keep/remove choice can never change the resulting scale.
        if report.unit_scale != 1.0:
            print(f"[INFO] Converting units mm -> m: {self.file_path.name}")
            mesh.scale(report.unit_scale, center=(0, 0, 0))
            cleaned.scale(report.unit_scale, center=(0, 0, 0))

        mesh.compute_vertex_normals()
        cleaned.compute_vertex_normals()
        self.app.target_mesh = mesh
        self.app.mesh_basename = self.file_path.stem
        if report.has_findings():
            self._pending = (cleaned, report)
            # Headless has no _refresh_ui to hang the prompt off, and choice_dialog resolves to
            # its first option without a dialog — so batch runs auto-remove and log here.
            if self.app.headless:
                self._resolve_pending()
        print(f"mesh loaded")

    # ===============================
    # Import findings
    # ===============================
    def _resolve_pending(self):
        """Surface the import report. Debris is a choice (GUI) — `app.choice_dialog` runs the
        first option with no dialog when headless, so batch runs auto-remove and log.

        `_pending` is cleared up front, so the prompt fires once per import: `_refresh_ui` runs
        again on every worker completion and on stage entry, and dismissing the dialog (Cancel)
        must not re-open it. Cancel therefore lands on the same outcome as "Keep as-is".
        """
        if self._pending is None:
            return
        cleaned, report = self._pending
        self._pending = None
        if report.has_debris():
            self.app.choice_dialog(
                report.summary(),
                [("Remove debris", lambda: self._apply_cleaned(cleaned, report)),
                 ("Keep as-is", self._keep_original)],
                title="Mesh Import Check")
        else:
            # Warnings only (oversize, or debris removal refused) — informational.
            self.app.confirm_dialog(report.summary(), on_ok=lambda: None,
                                    title="Mesh Import Check")

    def _apply_cleaned(self, cleaned, report):
        self.app.target_mesh = cleaned
        print(f"[MESH] debris removed: {report.debris_triangles} triangle(s) in "
              f"{len(report.debris)} component(s); bounding box diagonal "
              f"{report.diag_shrink_frac:.1%} smaller")
        self._refresh_ui()

    def _keep_original(self):
        print("[MESH] debris kept at operator's request; bounding box left inflated")

    def on_clear(self):
        self._pending = None

    def center_mesh(self):
        """Translate mesh (and all downstream clouds) so the mesh AABB centre is at the world
        origin. Anchored on the bounding-box midpoint rather than get_center() (the vertex mean),
        which is biased by tessellation density — a densely meshed fillet drags it off-centre."""
        if self.app.target_mesh is None:
            return
        center = np.asarray(self.app.target_mesh.get_axis_aligned_bounding_box().get_center())
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
