from enums import *
import open3d.core as o3c
from open3d.geometry import Geometry3D
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
from stages.import_mesh_stage import ImportMeshStage
from stages.raycast_stage import RaycastStage
from stages.crop_stage import CropStage
from stages.downsample_stage import DownsampleStage
from stages.save_stage import SaveStage
from stages.synthetic_stage import SyntheticStage
import threading
from pathlib import Path
from file_utils import pointcloud_to_ply, open_source_folder_dialog
from geom_utils import O3DSceneObject, camera_view_matrix
# import cupy as cp
# print(cp.cuda.runtime.getDeviceCount())

class MeshSamplingApp:

    def __init__(self, headless=False):
        self.headless = headless
        
        self.stages = {
            Stage.IMPORT_MESH: ImportMeshStage(self),
            Stage.RAYCAST: RaycastStage(self),
            Stage.CROP: CropStage(self),
            Stage.DOWNSAMPLE: DownsampleStage(self),
            Stage.SAVE: SaveStage(self),
            Stage.SYNTHETIC: SyntheticStage(self),
        }
        if not self.headless:
            # === Scene widget ===
            self.window_width = 1440
            self.window_height = 900
            self.window = gui.Application.instance.create_window("Mesh Sampling Wizard", self.window_width, self.window_height)
            self.scene = gui.SceneWidget()
            self.scene.scene = rendering.Open3DScene(self.window.renderer)
            self.scene.scene.set_background([0.2, 0.2, 0.2, 1.0])
            self.scene_geoms = {}
            self.window.set_on_layout(self._on_layout)
            self.window.set_on_key(self._on_key)
            self.scene.set_on_mouse(self._on_mouse_event)
            self.window.add_child(self.scene)

            # === Materials ===
            self.default_material = rendering.MaterialRecord()
            self.default_material.shader = "defaultLit"
            self.default_point_material = rendering.MaterialRecord()
            self.default_point_material.point_size = 1.5
            self.default_point_material.base_color = [1.0, 1.0, 1.0, 1.0]
            self.overlay_material = rendering.MaterialRecord()
            self.overlay_material.point_size = 1.5
            self.overlay_material.base_color = [1.0, 0.5, 0.3, 1.0]

            # ===============================
            # Build Control Panel
            # ===============================
            em = self.window.theme.font_size
            self.panel = gui.Vert(0.25 * em, gui.Margins(em, em, em, em))
            self.window.add_child(self.panel)

            for stage_class in self.stages.values():
                stage_class.panel.visible = False
                self.panel.add_child(stage_class.panel)

            # === Progress Bar widget ===
            self.progress_panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
            self.progress_panel.visible = False
            self.progress_label = gui.Label("Processing...")
            self.progress_bar = gui.ProgressBar()
            self.progress_bar.value = 0.0  # range [0, 1]
            self.progress_panel.add_child(self.progress_label)
            self.progress_panel.add_child(self.progress_bar)
            self.pb_panel_size = (400, 50)
            x=(self.window_width - self.pb_panel_size[0])>>1
            y=(self.window_height - self.pb_panel_size[1])>>1
            self.progress_panel.frame = gui.Rect(x, y, self.pb_panel_size[0], self.pb_panel_size[1])
            self.window.add_child(self.progress_panel)

        self._restart()

    def _restart(self):
        self.target_mesh = None
        self.raw_pcd = None
        self.down_pcd = None
        self.visible_target_pcd = None
        self.occluders_pcd = None
        self.mesh_basename = None
        self.convex_meshes = []
        self.synthetic_targets = {}
        self.synthetic_scenes = {}
        self.feature_pcd = None
        self.flat_pcd = None
        self.output_pcd_path = None
        self.geocenter = np.eye(4)
        self.point_count_mean = None
        self.point_count_range = None
        self.stage = Stage.IMPORT_MESH
        if not self.headless and Stage.SYNTHETIC in self.stages:
            self.stages[Stage.SYNTHETIC].combobox_targets.clear_items()
        self.set_stage(Stage.IMPORT_MESH)

    def set_stage(self, stage: Stage):
        self.stage = stage
        if self.headless:
            return
        for s in self.stages.values():
            s.panel.visible = (s is self.stages[stage])
            for w in s.widgets:
                if hasattr(w.widget, "toggleable") and w.widget.toggleable:
                    w.widget.is_on = False
        self.window.set_needs_layout()
        self.stages[stage]._refresh_ui()
        self.stages[stage].enable_widgets()
        self._update_title()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 300
        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

    # ===============================
    # UI helpers
    # ===============================
    def _update_title(self):
        self.window.title = f"Mesh Sampling Wizard | Stage: {self.stage.name}"

    def _reframe(self, fov_deg=41.1, margin=1.0):
        """
        Dynamically frame object based on its bounding box size.
        """
        if self.headless:
            return
        if self.target_mesh is None:
            return
        else:
            if self.stage == Stage.SYNTHETIC:
                mj_scene = self.stages[Stage.SYNTHETIC].mj_scene
                look_at, cam_pos, up = mj_scene._get_camera_lookat()
                bbox = mj_scene.bin_mesh.get_axis_aligned_bounding_box()
            else:
                bbox = self.target_mesh.get_axis_aligned_bounding_box()
                look_at = bbox.get_center()
                distance = 1.0 * np.linalg.norm(bbox.get_extent())
                if distance < 1e-6:
                    return
                # distance = (radius / np.tan(np.deg2rad(fov_deg) / 2.0)) * margin
                cam_pos = look_at + np.array([distance, distance, distance])
                T_cam = camera_view_matrix(cam_pos, look_at)
                up = T_cam[:3, 1]

        self.scene.center_of_rotation = look_at
        self.scene.setup_camera(fov_deg, bbox, look_at)
        self.scene.scene.camera.look_at(look_at, cam_pos, up)
        self.scene.force_redraw()
        print("reframed")

    def show_progress(self, text="Processing..."):
        if self.headless:
            return
        def _show():
            self.progress_label.text = text
            self.progress_bar.value = 0.0
            self.progress_panel.visible = True
        gui.Application.instance.post_to_main_thread(self.window, _show)

    def update_progress(self, value, text="Processing..."):
        if self.headless:
            return
        value = max(0.0, min(1.0, value))
        def _update():
            self.progress_label.text = text
            self.progress_bar.value = value
        gui.Application.instance.post_to_main_thread(self.window, _update)

    def hide_progress(self):
        if self.headless:
            return
        def _hide():
            self.progress_panel.visible = False
        gui.Application.instance.post_to_main_thread(self.window, _hide)
    
    def _clear_scene(self):
        self.scene_geoms = {}
        self.scene.scene.clear_geometry()

    def add_geom_in_scene(self, name:str, geom: Geometry3D, color=[1.0, 1.0, 1.0], alpha=1.0, point_size=1.5):
        material = rendering.MaterialRecord()
        material.point_size = point_size
        material.base_color = color + [alpha]
        material.shader = "defaultLit"
        self.scene_geoms[name] = O3DSceneObject(geom, material)
        self.main_thread(lambda: self.scene.scene.add_geometry(name, geom, material))

    def remove_geom_in_scene(self, name:str):
        self.scene_geoms.pop(name)
        self.main_thread(lambda: self.scene.scene.remove_geometry(name))
    
    def hide_geoms_in_scene(self, geoms=[]):
        def hide_geoms():
            if not geoms: 
                print("no specified geom, hiding all")
                for name in self.scene_geoms.keys():
                    print(f"hiding {name}")
                    self.scene.scene.show_geometry(name, show=False)
            else: 
                print(f"geoms = {geoms}")
                for name in geoms:
                    print(f"hiding {name}")
                    self.scene.scene.show_geometry(name, show=False)
        self.main_thread(hide_geoms)

    def show_geoms_in_scene(self, geoms:list=[]):
        def show_geoms():
            if not geoms: print("no specified geom, showing all")
            print(geoms)

            for name in (geoms if geoms else self.scene_geoms.keys()):
                print(f"showing {name}")
                self.scene.scene.show_geometry(name, show=True)
        self.main_thread(show_geoms)

    def has_geom(self, name:str):
        self.main_thread(lambda: self.scene.scene.has_geometry(name))

    def main_thread(self, fn):
        if self.headless:
            return
        gui.Application.instance.post_to_main_thread(self.window, fn)

    # ===============================
    # Keybindings
    # ===============================
    def _on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return False

        key = event.key

        # --- Global ---
        if key == gui.KeyName.R:
            self._reframe()
            return True

        # --- Stage specific ---
        if self.stage in self.stages:
            self.stages[self.stage]._on_key(event)

        return True
    
    def _on_mouse_event(self, event):
        if self.stage in self.stages:
            self.stages[self.stage]._on_mouse_event(event)
        return gui.Widget.EventCallbackResult.IGNORED

    # ===============================
    # Express Handler
    # ===============================
    def start_express_sampling(self):
        self._express_sampling_thread = threading.Thread(target=self._express_sampling_worker)
        self._express_sampling_thread.start()

    def _express_sampling_worker(self):
        self.stages[Stage.RAYCAST].worker()
        self.down_pcd=self.raw_pcd
        self.stages[Stage.DOWNSAMPLE].worker()
        self.stages[Stage.DOWNSAMPLE].recenter_mesh_pcd()
        self.set_stage(Stage.SAVE)
        self.hide_progress()

    def start_batch_sampling(self):
        self.src_dir = open_source_folder_dialog()
        if self.src_dir is None:
            print("No source path selected")
            return
        
        self._batch_sampling_thread = threading.Thread(target=self._batch_sampling_worker)
        self._batch_sampling_thread.start()

    def _batch_sampling_worker(self):
        dst_dir = Path.cwd() / "reference_pcd"
        if dst_dir is None:
            print("No destination path selected")
            return

        stl_files = list(self.src_dir.glob("*.stl"))
        print(f"[INFO] Found {len(stl_files)} STL files")
        
        for stl_path in stl_files:
            try:
                print(f"[INFO] Processing {stl_path.name}")
                self.stages[Stage.IMPORT_MESH].file_path = stl_path
                self.stages[Stage.IMPORT_MESH].worker()
                self._express_sampling_worker()
                pointcloud_to_ply(self.down_pcd, str(dst_dir / (stl_path.stem + ".ply")))

                if self.headless:
                    continue
                self.main_thread(lambda: self._clear_scene())
                self.main_thread(lambda: self.scene.scene.add_geometry("down_pcd", self.down_pcd, self.default_point_material))
                # self.hide_geoms_in_scene()
                # self.add_geom_in_scene("down_pcd", self.down_pcd)
                self.scene.force_redraw()
                self._reframe()
            except Exception as e:
                print(f"[ERROR] Failed to process {stl_path.name}: {e}")
                continue

if __name__ == "__main__":
    # gui.Application.instance.initialize()
    # app = MeshSamplingApp()
    # gui.Application.instance.run()
    try: 
        gui.Application.instance.initialize()
        app = MeshSamplingApp()
        gui.Application.instance.run()
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}")
        exit()
