import colorsys
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import trimesh
from enums import Stage
from stages.stage_base import BaseStage
from geometry.geom_utils import o3d_to_trimesh


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
        self.btn_next = self.register_widget(gui.Button("Next: Synthetic Target"),
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
        self.app.main_thread(self.app._reframe)
        self.app.scene.force_redraw()
        self.enable_widgets()

    def worker(self):
        if self.app.target_mesh is None:
            print("[WARN] No mesh to decompose")
            return
        self.app.convex_meshes = []
        part_mesh = o3d_to_trimesh(self.app.target_mesh)
        decomposed_convex_list = trimesh.decomposition.convex_decomposition(part_mesh)
        for h in decomposed_convex_list:
            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(h["vertices"]),
                triangles=o3d.utility.Vector3iVector(h["faces"]))
            self.app.convex_meshes.append(mesh)
        print(f"[DECOMPOSE] decomposed mesh into {len(self.app.convex_meshes)} convex hulls")

    def reset(self):
        self.app.convex_meshes = []
        self._refresh_ui()
