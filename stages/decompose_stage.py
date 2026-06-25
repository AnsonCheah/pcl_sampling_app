import colorsys
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from concurrent.futures import ProcessPoolExecutor
from enums import Stage
from stages.stage_base import BaseStage
from geometry.geom_utils import o3d_to_trimesh
from geometry.convex_decomp import vhacd_decompose


class DecomposeStage(BaseStage):
    """Convex-decompose the imported mesh into a set of convex hulls.

    Runs synchronously in its own worker (no daemon thread): on completion
    ``app.convex_meshes`` is guaranteed populated before SYNTHETIC consumes it.
    """

    def __init__(self, app):
        self.name = Stage.DECOMPOSE.name
        super().__init__(app)

    def build_panel(self):
        if self.app.headless:
            return
        v = gui.Vert(4)

        self.btn_decompose = self.register_widget(gui.Button("Decompose Mesh"),
                                                  enabled_if=lambda: self.app.target_mesh is not None)
        self.btn_decompose.set_on_clicked(self.start)

        self.btn_back = self.register_widget(gui.Button("Back: Save"))
        self.btn_back.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value - 1)))
        # Only advance once decomposition has produced convex hulls.
        self.btn_next = self.register_widget(gui.Button("Next: Scene"),
                                            enabled_if=lambda: len(self.app.convex_meshes) > 0)
        self.btn_next.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value + 1)))

        self.btn_restart = self.register_widget(gui.Button("Restart"))
        self.btn_restart.set_on_clicked(lambda: self.app._restart())

        v.add_child(gui.Label("Convex Decomposition"))
        v.add_child(gui.Label(""))
        v.add_child(self.btn_decompose)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(self.btn_back)
        v.add_child(self.btn_next)
        v.add_child(self.btn_restart)

        print("loaded decompose panel")
        return v

    def _display_convex_meshes(self):
        """GUI helper: show each convex hull in a distinct HSV colour."""
        saturation, value = 0.6, 0.9
        n = len(self.app.convex_meshes)
        for i, mesh in enumerate(self.app.convex_meshes):
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            rgb = colorsys.hsv_to_rgb(i / max(n, 1), saturation, value)
            material.base_color = list(rgb) + [1.0]
            geom = o3d.geometry.TriangleMesh(mesh)
            geom.compute_vertex_normals()
            self.app.scene.scene.add_geometry(f"convex_{i}", geom, material)

    def _refresh_ui(self):
        if self.app.headless:
            return
        self.app.main_thread(lambda: self.app._clear_scene())
        if len(self.app.convex_meshes) > 0:
            # After decomposition: show the convex hulls.
            self.app.main_thread(self._display_convex_meshes)
        elif self.app.target_mesh is not None:
            # On stage entry, before decomposition: show the original mesh.
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry(
                "mesh", self.app.target_mesh, self.app.default_material))
        # self.app.main_thread(self.app._reframe)
        self.app.scene.force_redraw()
        self.enable_widgets()

    def worker(self):
        if self.app.target_mesh is None:
            print("[WARN] No mesh to decompose")
            return
        self.app.update_progress(0.1, "Preparing mesh for decomposition...")
        self.app.convex_meshes = []
        part_mesh = o3d_to_trimesh(self.app.target_mesh)
        verts = np.asarray(part_mesh.vertices)
        faces = np.asarray(part_mesh.faces)

        # vhacdx.compute_vhacd holds the GIL for its full runtime, which would starve the
        # Open3D GUI thread and freeze the progress bar. Run it in a child process and block
        # on the result here; the wait releases the GIL so queued progress callbacks render.
        self.app.update_progress(0.3, "Running convex decomposition...")
        with ProcessPoolExecutor(max_workers=1) as ex:
            decomposed_convex_list = ex.submit(vhacd_decompose, verts, faces).result()

        n = len(decomposed_convex_list)
        for i, (v, f) in enumerate(decomposed_convex_list):
            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(v),
                triangles=o3d.utility.Vector3iVector(f))
            self.app.convex_meshes.append(mesh)
            self.app.update_progress(0.6 + 0.4 * (i + 1) / max(n, 1),
                                     f"Building convex hulls ({i + 1}/{n})...")
        print(f"[DECOMPOSE] decomposed mesh into {len(self.app.convex_meshes)} convex hulls")

    def reset(self):
        self.app.convex_meshes = []
        self._refresh_ui()
