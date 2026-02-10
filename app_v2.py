from enums import *
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
from stages.import_mesh_stage import ImportMeshStage
from stages.raycast_stage import RaycastStage
from stages.crop_stage import CropStage
from stages.downsample_stage import DownsampleStage
from stages.save_stage import SaveStage
import threading
from pathlib import Path
from utilities import pointcloud_to_ply, open_source_folder_dialog

class MeshSamplingApp:

    def __init__(self, headless=False):
        self.headless = headless

        # Shared state
        self.target_mesh = None
        self.raw_pcd = None
        self.down_pcd = None
        self.stage = Stage.IMPORT_MESH
        self.visible_target_pcd = None
        self.occluders_pcd = None

        # === State parameters ===
        self.mesh_basename = None
        self.synthetic_occlusion = True
        self.fov_deg = 25
        self.res_width = 1920
        self.res_height = 1200
        self.min_occlusion_ratio = 0.1
        self.max_occlusion_ratio = 0.3
        self.synthetic_targets = []
        self.depth_sigma = 0.0005
        self.angular_sigma = 0.00005

        # -------------------------
        # Stages
        # -------------------------

        if not self.headless:
            # === Scene widget ===
            self.window_width = 1440
            self.window_height = 900
            self.window = gui.Application.instance.create_window("Mesh Sampling Wizard", self.window_width, self.window_height)
            self.scene = gui.SceneWidget()
            self.scene.scene = rendering.Open3DScene(self.window.renderer)
            self.scene.scene.set_background([0.2, 0.2, 0.2, 1.0])
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

            self.stages = {
                Stage.IMPORT_MESH: ImportMeshStage(self),
                Stage.RAYCAST: RaycastStage(self),
                Stage.CROP: CropStage(self),
                Stage.DOWNSAMPLE: DownsampleStage(self),
                Stage.SAVE: SaveStage(self),
                # Stage.SYNTHETIC: SyntheticStage(self),
            }
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
            self.pb_panel_size = (300, 50)
            x=(self.window_width - self.pb_panel_size[0])>>1
            y=(self.window_height - self.pb_panel_size[1])>>1
            self.progress_panel.frame = gui.Rect(x, y, self.pb_panel_size[0], self.pb_panel_size[1])
            self.window.add_child(self.progress_panel)

        self.set_stage(Stage.IMPORT_MESH)

    def set_stage(self, stage: Stage):
        self.stage = stage
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
        r = self.window.content_rect
        self.scene.frame = r

    # ===============================
    # UI helpers
    # ===============================
    def _update_title(self):
        self.window.title = f"Mesh Sampling Wizard | Stage: {self.stage.name}"

    def _clear_scene(self):
        self.scene.scene.clear_geometry()

    def _reframe(self, fov_deg=25.0, margin=1.0):
        """
        Dynamically frame object based on its bounding box size.
        """
        if self.headless:
            return
        if self.target_mesh is None:
            return
        else:
            bbox = self.target_mesh.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = bbox.get_extent()
            radius = 0.2 * np.linalg.norm(extent)
            if radius < 1e-6:
                return

            distance = (radius / np.tan(np.deg2rad(fov_deg) / 2.0)) * margin
            eye = center + np.array([distance, distance, 0])
            up = np.array([0, 1, 1])

        self.scene.scene.camera.look_at(center, eye, up)
        self.scene.force_redraw()
        print("reframed")

    def show_progress(self, text="Processing..."):
        def _show():
            self.progress_label.text = text
            self.progress_bar.value = 0.0
            self.progress_panel.visible = True
        gui.Application.instance.post_to_main_thread(self.window, _show)

    def update_progress(self, value):
        value = max(0.0, min(1.0, value))
        def _update():
            self.progress_bar.value = value
        gui.Application.instance.post_to_main_thread(self.window, _update)

    def hide_progress(self):
        def _hide():
            self.progress_panel.visible = False
        gui.Application.instance.post_to_main_thread(self.window, _hide)
        
    def main_thread(self, fn):
        gui.Application.instance.post_to_main_thread(self.window, fn)

    def enable_button(self, button, enabled:bool):
        button.enabled = enabled
        
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
        return o3d.visualization.gui.Widget.EventCallbackResult.IGNORED

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
                self.scene.force_redraw()
                self._reframe()
            except Exception as e:
                print(f"[ERROR] Failed to process {stl_path.name}: {e}")
                continue

if __name__ == "__main__":
    gui.Application.instance.initialize()
    app = MeshSamplingApp()
    gui.Application.instance.run()
    # try: 
    #     gui.Application.instance.initialize()
    #     app = MeshSamplingApp()
    #     gui.Application.instance.run()
    # except Exception as e:
    #     print(f"[FATAL] Unhandled exception: {e}")
    #     exit()
