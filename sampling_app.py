import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import numpy as np
from enum import Enum
from pathlib import Path
from tkinter import filedialog, Tk
from utilities import *
import time
import logging
import copy
import threading

# ===============================
# Stage definition
# ===============================
class Stage(Enum):
    IMPORT_MESH = 0
    RAYCAST = 1
    CROP = 2
    DOWNSAMPLE = 3
    SAVE = 4

class ToolMode(Enum):
    NONE = 0
    BOX_SELECT = 1
    POLY_SELECT = 2

# ===============================
# Main Application
# ===============================
class MeshSamplingApp:

    def __init__(self, headless=False):
        self.headless = headless
        self.raycasting_thread = None
        self.downsampling_thread = None

        self.stage = Stage.IMPORT_MESH

        # === Data variables ===
        self.target_mesh = None
        self.raw_pcd = None
        self.cropped_pcd = None
        self.down_pcd = None

        self.stage_init = {}        
        self.stage_init[Stage.IMPORT_MESH] = self.load_mesh_stage_init
        self.stage_init[Stage.RAYCAST] = self.raycast_stage_init
        self.stage_init[Stage.CROP] = self.crop_stage_init
        self.stage_init[Stage.DOWNSAMPLE] = self.downsample_stage_init
        self.stage_init[Stage.SAVE] = self.save_stage_init

        # === State parameters ===
        self.camera_distance = 1.5
        self.num_views = 20
        self.ray_margin_mm = 10.0
        self.ray_spacing = 0.001
        self.voxel_size = 0.001
        self.use_adaptive = True
        self.coarse_factor = 2
        self.curvature_k_neighbors = 5
        self.bbox_corners = None

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
            self.tool_mode = ToolMode.NONE
            self.is_dragging = False
            self.drag_start = None
            self.drag_end = None
            self.selected_indices = []

            # === Materials ===
            self.default_material = rendering.MaterialRecord()
            self.default_material.shader = "defaultLit"
            self.default_point_material = rendering.MaterialRecord()
            self.default_point_material.shader = "defaultUnlit"
            self.default_point_material.point_size = 1.5
            self.default_point_material.base_color = [1.0, 1.0, 1.0, 1.0]
            self.overlay_material = rendering.MaterialRecord()
            self.overlay_material.shader = "defaultUnlit"
            self.overlay_material.point_size = 1.5
            self.overlay_material.base_color = [1.0, 0.5, 0.3, 1.0]

            # ===============================
            # Build Control Panel
            # ===============================
            em = self.window.theme.font_size
            self.panel = gui.Vert(0.25 * em, gui.Margins(em, em, em, em))
            self.window.add_child(self.panel)

            self.next_stage_buttons = {}
            self.worker_buttons = {}
            self.stage_panels = {}
            self.stage_panels[Stage.IMPORT_MESH] = self._load_mesh_panel()
            self.stage_panels[Stage.RAYCAST] = self._raycast_panel()
            self.stage_panels[Stage.CROP] = self._crop_panel()
            self.stage_panels[Stage.DOWNSAMPLE] = self._downsample_panel()
            self.stage_panels[Stage.SAVE] = self._save_panel()

            for p in self.stage_panels.values():
                p.visible = False
                self.panel.add_child(p)

            self.pb_layout()

            self.set_stage(Stage.IMPORT_MESH)

    # ===============================
    # Control Panel
    # ===============================

    def set_stage(self, stage: Stage):
        print(f"Transitioning from {self.stage.name} to {stage.name}")
        self.stage = stage
        if self.headless:
            return
        for s, panel in self.stage_panels.items():
            panel.visible = (s == stage) # toggle on off stage panels

        if stage != Stage.CROP:
            self.btn_box_select.is_on = False
        self.window.set_needs_layout()
        self.stage_init[stage]()
        self._update_title()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 300
        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
        r = self.window.content_rect
        self.scene.frame = r

    def pb_layout(self):
        self.progress_panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        self.progress_panel.visible = False
        self.progress_label = gui.Label("Processing...")
        self.progress_bar = gui.ProgressBar()
        self.progress_bar.value = 0.0  # range [0, 1]
        self.progress_panel.add_child(self.progress_label)
        self.progress_panel.add_child(self.progress_bar)
        panel_width = 300
        panel_height = 50
        self.progress_panel.frame = gui.Rect(
            int((self.window_width - panel_width)/2),
            int((self.window_height - panel_height)/ 2),
            panel_width,
            panel_height
        )
        self.window.add_child(self.progress_panel)

    # ===============================
    # UI helpers
    # ===============================
    def _update_title(self):
        self.window.title = f"Mesh Sampling Wizard | Stage: {self.stage.name}"

    def _clear_scene(self):
        self.scene.scene.clear_geometry()

    def _reframe(self, fov_deg=60.0, margin=1.0):
        """
        Dynamically frame object based on its bounding box size.
        """
        print("reframing")
        if self.target_mesh is None:
            center = np.array([0.0, 0.0, 0.0])
            distance = 0.5
            eye = np.array([1.0,1.0,1.0])
            up = np.array([1, 1, 1])

        else:
            bbox = self.target_mesh.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = bbox.get_extent()
            radius = 0.5 * np.linalg.norm(extent)
            if radius < 1e-6:
                return

            distance = (radius / np.tan(np.deg2rad(fov_deg) / 2.0)) * margin
            eye = center + np.array([distance, distance, 0])
            up = np.array([0, 1, 1])

        cam = self.scene.scene.camera
        cam.look_at(center, eye, up)
        # cam.set_projection(
        #     60.0,                    # FOV
        #     self.scene.frame.width / self.scene.frame.height,
        #     0.01,
        #     1000.0,
        #     o3d.visualization.rendering.Camera.FovType.Vertical
        # )

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
        
    def safe_scene_update(self, fn):
        gui.Application.instance.post_to_main_thread(self.window, fn)

    # ===============================
    # File dialogs
    # ===============================
    def _open_stl_dialog(self):
        Tk().withdraw()
        path = filedialog.askopenfilename(filetypes=[("STL files", "*.stl *.STL")])
        if not path:
            return None
        path = Path(path)
        self.stl_basename = path.stem  # filename without extension
        return path

    def _save_ply_dialog(self):
        Tk().withdraw()
        default_name = "output.ply"
        if hasattr(self, "stl_basename") and self.stl_basename:
            default_name = f"{self.stl_basename}.ply"

        path = filedialog.asksaveasfilename(
            defaultextension=".ply",
            initialfile=default_name,
            filetypes=[("PLY files", "*.ply")]
        )

        return Path(path) if path else None
    
    def _open_source_folder_dialog(self):
        Tk().withdraw()
        path = filedialog.askdirectory(title="Select source folder (STL files)")
        return Path(path) if path else None

    def _open_dest_folder_dialog(self):
        Tk().withdraw()
        path = filedialog.askdirectory(title="Select destination folder (PLY output)")
        return Path(path) if path else None
    
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

        if key == gui.KeyName.N:
            if self.stage.value >=4:
                return False
            self.set_stage(Stage(self.stage.value + 1))
            return True
        
        if key == gui.KeyName.B:
            if self.stage.value <=0:
                return False
            self.set_stage(Stage(self.stage.value - 1))
            return True

        # --- Stage specific ---
        if self.stage == Stage.IMPORT_MESH:
            if key == gui.KeyName.O:
                self.import_mesh()
        elif self.stage == Stage.RAYCAST:
            if key == gui.KeyName.S:
                self.start_raycasting()
        elif self.stage == Stage.CROP:
            if key == gui.KeyName.S:
                self.save_pcd()
            elif key == gui.KeyName.D:
                self.delete_selected_points()

        elif self.stage == Stage.DOWNSAMPLE:
            if key == gui.KeyName.S:
                self.start_downsampling()
            elif key == gui.KeyName.T:
                self.use_adaptive = not self.use_adaptive
                self.adaptive_checkbox.checked = not self.adaptive_checkbox.checked

        elif self.stage == Stage.SAVE:
            if key == gui.KeyName.S:
                self.save_pcd()
            elif key == gui.KeyName.R:
                self.target_mesh = None
                self.raw_pcd = None
                self.down_pcd = None
                self.stage = Stage.IMPORT_MESH
                self._clear_scene()
                self._update_title()
                
        return True

    # ===============================
    # Import stage functions
    # ===============================  

    def _load_mesh_panel(self):
        print("loading mesh panel")
        v = gui.Vert(4)
        btn_load = gui.Button("Load STL")
        btn_load.set_on_clicked(self.import_mesh)

        btn_reset = gui.Button("Clear Mesh")
        btn_reset.set_on_clicked(self.reset_mesh_stage)
        self.btn_express = gui.Button("Express Sampling")
        self.btn_express.set_on_clicked(self._express_sampling)
        btn_batch =  gui.Button("Batch Sampling")
        btn_batch.set_on_clicked(self._batch_sampling)

        btn_next = gui.Button("Next: Raycast")
        btn_next.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value + 1)))
        self.worker_buttons[Stage.IMPORT_MESH] = btn_load
        self.next_stage_buttons[Stage.IMPORT_MESH] = btn_next
        v.add_child(gui.Label("Import Mesh"))
        v.add_child(gui.Label(""))
        v.add_child(self.worker_buttons[Stage.IMPORT_MESH])
        v.add_child(btn_reset)
        v.add_child(gui.Label(""))
        v.add_child(self.btn_express)
        v.add_child(btn_batch)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(self.next_stage_buttons[Stage.IMPORT_MESH])
        print("loaded mesh panel")
        return v
    
    def load_mesh_stage_init(self):
        if self.headless:
            return
        self.worker_buttons[Stage.IMPORT_MESH].enabled = True
        self.next_stage_buttons[Stage.IMPORT_MESH].enabled = (self.target_mesh != None)
        self.btn_express.enabled = (self.target_mesh != None)
        if self.target_mesh is not None:
            self.safe_scene_update(lambda: self._clear_scene())
            self.safe_scene_update(lambda: self.scene.scene.add_geometry("mesh", self.target_mesh, self.default_material))

    def reset_mesh_stage(self):
        self.target_mesh = None
        self.raw_pcd = None
        self.cropped_pcd = None
        self.down_pcd = None
        self.worker_buttons[Stage.IMPORT_MESH].enabled = True
        self.next_stage_buttons[Stage.IMPORT_MESH].enabled = False
        self.btn_express.enabled = False
        self.safe_scene_update(lambda: self._clear_scene())

    def import_mesh(self):
        self.worker_buttons[Stage.IMPORT_MESH].enabled = False
        self.file_path = self._open_stl_dialog()
        if not self.file_path:
            self.worker_buttons[Stage.IMPORT_MESH].enabled = True
            return
        self._load_mesh_worker()
        self.next_stage_buttons[Stage.IMPORT_MESH].enabled = True
        if self.headless:
            return
        self.btn_express.enabled = True
        self.safe_scene_update(lambda: self._clear_scene())
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("mesh", self.target_mesh, self.default_material))
        self._reframe()

    def _load_mesh_worker(self):
        mesh = o3d.io.read_triangle_mesh(str(self.file_path))
        if mesh.is_empty():
            print("[WARN] Empty mesh")
            return
        
        self.raw_pcd = None
        self.cropped_pcd = None
        self.down_pcd = None
        
        bbox = mesh.get_axis_aligned_bounding_box()
        extent_max = bbox.get_extent().max()
        unit_conversion = 1.0
        if extent_max > 5 and extent_max < 5000.0:
            # Likely in millimeters -> convert to meters
            print(f"[INFO] Converting units from mm to m for: {self.file_path.name}")
            unit_conversion = 0.001
            mesh.scale(unit_conversion, center=(0, 0, 0))

        mesh.compute_vertex_normals()
        mesh.translate(-mesh.get_center())
        self.target_mesh = mesh

        bbox = mesh.get_axis_aligned_bounding_box()
        self.bbox_corners = np.asarray(bbox.get_box_points())
        extent_min = bbox.get_extent().min()   # (dx, dy, dz) in world units
        self.ray_spacing = np.round(np.clip((extent_min / 100), 0.0005, 0.003), 4)
        self.voxel_size = np.round(np.clip((extent_min / 50), 0.001, 0.005), 4)
        print(f"calculated ray spacing: {self.ray_spacing*1000}mm")
        print(f"calculated voxel size needed: {self.voxel_size * 1000}mm")

    # ===============================
    # Raycast stage functions
    # ===============================

    def _raycast_panel(self):
        v = gui.Vert(4)

        self.camera_distance_slider = gui.Slider(gui.Slider.DOUBLE)
        self.camera_distance_slider.set_limits(0.5, 3.0)
        self.camera_distance_slider.double_value = 1.5
        self.num_views_slider = gui.Slider(gui.Slider.INT)
        self.num_views_slider.set_limits(4, 100)
        self.num_views_slider.int_value = 20
        btn_raycast = gui.Button("Raycast")
        btn_raycast.set_on_clicked(self.start_raycasting)

        btn_reset = gui.Button("Clear Raycast")
        btn_reset.set_on_clicked(self.reset_raycast_stage)
        btn_next = gui.Button("Next: Crop")
        btn_next.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value + 1)))
        btn_back = gui.Button("Back: Import Mesh")
        btn_back.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value - 1)))

        self.worker_buttons[Stage.RAYCAST] = btn_raycast
        self.next_stage_buttons[Stage.RAYCAST] = btn_next
        v.add_child(gui.Label("Raycasting"))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label("Camera Distance"))
        v.add_child(self.camera_distance_slider)
        v.add_child(gui.Label("Number of Views"))
        v.add_child(self.num_views_slider)
        v.add_child(self.worker_buttons[Stage.RAYCAST])
        v.add_child(btn_reset)
        v.add_child(gui.Label(""))
        v.add_child(btn_back)
        v.add_child(self.next_stage_buttons[Stage.RAYCAST])
        return v

    def raycast_stage_init(self):
        if self.headless:
            return
        self.worker_buttons[Stage.RAYCAST].enabled = True
        self.next_stage_buttons[Stage.RAYCAST].enabled = (self.raw_pcd != None)
        if self.raw_pcd is not None:
            self.safe_scene_update(lambda: self._clear_scene())
            self.safe_scene_update(lambda: self.scene.scene.add_geometry("raw_pcd", self.raw_pcd, self.default_material))

    def reset_raycast_stage(self):
        self.raw_pcd = None
        self.cropped_pcd = None
        self.down_pcd = None
        if self.headless: 
            return
        self.worker_buttons[Stage.RAYCAST].enabled = True
        self.next_stage_buttons[Stage.RAYCAST].enabled = (self.raw_pcd != None)
        self.safe_scene_update(lambda: self._clear_scene())
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("mesh", self.target_mesh, self.default_material))

    def start_raycasting(self):
        self.set_stage(Stage.RAYCAST)
        self.raycasting_thread = threading.Thread(target=self._raycasting_worker)
        self.raycasting_thread.start()

    def _raycasting_worker(self):
        if self.target_mesh is None:
            print("[WARN] No mesh loaded")
            return
        if not self.headless:
            self.worker_buttons[Stage.RAYCAST].enabled = False
            self.next_stage_buttons[Stage.RAYCAST].enabled = False
            self.camera_distance = self.camera_distance_slider.double_value
            self.num_views = self.num_views_slider.int_value
            self.show_progress("Raycasting mesh...")

        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.target_mesh))
        view_dirs = fibonacci_sphere(self.num_views)
        all_points = []
        all_cam_pos = []
        for view_index, view_dir in enumerate(view_dirs):
            # Camera position
            cam_pos = view_dir * (self.camera_distance)
            forward = -cam_pos / np.linalg.norm(cam_pos)
            # Stable camera basis
            right = np.cross([0, 0, 1], forward)
            if np.linalg.norm(right) < 1e-6:
                right = np.cross([0, 1, 0], forward)
            right /= np.linalg.norm(right)
            up = np.cross(forward, right)

            # Project bbox corners onto camera plane
            rel = self.bbox_corners - cam_pos
            x_proj = rel @ right
            y_proj = rel @ up
            x_min, x_max = x_proj.min(), x_proj.max()
            y_min, y_max = y_proj.min(), y_proj.max()

            # Add margin (convert mm → meters)
            margin = self.ray_margin_mm * 1e-3
            x_min -= margin
            x_max += margin
            y_min -= margin
            y_max += margin

            # Generate ray grid in METERS
            xs = np.arange(x_min, x_max, self.ray_spacing)
            ys = np.arange(y_min, y_max, self.ray_spacing)

            if len(xs) == 0 or len(ys) == 0:
                continue

            uu, vv = np.meshgrid(xs, ys)
            # Ray origins and directions
            origins = (cam_pos + uu[..., None] * right + vv[..., None] * up)

            dirs = forward[None, None, :].repeat(origins.shape[0], axis=0)
            dirs = dirs.repeat(origins.shape[1], axis=1)

            rays = np.concatenate([origins.reshape(-1, 3), dirs.reshape(-1, 3)],axis=1)
            rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

            hits = scene.cast_rays(rays)
            hit_mask = hits["t_hit"].isfinite()

            if not hit_mask.any():
                continue

            hit_points = (rays[hit_mask][:, :3] + rays[hit_mask][:, 3:] * hits["t_hit"][hit_mask].reshape((-1, 1))).numpy()
            cam_pos_arr = np.repeat(cam_pos[None, :], len(hit_points), axis=0)
            all_points.append(hit_points)
            all_cam_pos.append(cam_pos_arr)

            if not self.headless:
                progress = (view_index + 1) / self.num_views
                self.update_progress(progress)

        if not all_points:
            logging.warning(f"No points generated from mesh")
            return

        # Aggregate
        points = np.vstack(all_points)
        cam_positions = np.vstack(all_cam_pos)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd.remove_non_finite_points()
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = 0.005, max_nn = 60))
        pcd=normalize_normals(pcd)
        orient_normals_using_cameras(pcd, cam_positions)
        
        pcd=normalize_normals(pcd)
        validate_normals(pcd)
        print(f"[INFO] Total points sampled: {len(pcd.points)}")
        initial_voxel = (self.ray_spacing)*0.3
        pcd = normalize_normals(pcd.voxel_down_sample(initial_voxel)) # downsample to 3x resolution 
        print(f"[INFO] Initial voxelized points: {len(pcd.points)}")
        self.raw_pcd = pcd
        self.cropped_pcd = copy.deepcopy(self.raw_pcd) # for crop stage

        if not self.headless:
            self.safe_scene_update(lambda: self._clear_scene())
            self.safe_scene_update(lambda: self.scene.scene.add_geometry("raw_pcd", self.raw_pcd, self.default_point_material))
            self.next_stage_buttons[Stage.RAYCAST].enabled = True
            self.worker_buttons[Stage.RAYCAST].enabled = True
            self.hide_progress()
        print("raycast finished")

    # ==============================
    # Crop Stage
    # ==============================
    
    def _crop_panel(self):
        v = gui.Vert(4)
        
        self.btn_box_select = gui.Button("Box Select")
        self.btn_box_select.toggleable = True
        self.btn_box_select.set_on_clicked(self._enable_box_selection)
        delete_btn = gui.Button("Delete Selected Points")
        delete_btn.set_on_clicked(self.delete_selected_points)
        btn_reset = gui.Button("Reset Crop")
        btn_reset.set_on_clicked(self.reset_crop_stage)

        btn_next = gui.Button("Next: Downsample")
        btn_next.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value + 1)))
        btn_back = gui.Button("Back: Raycast")
        btn_back.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value - 1)))

        self.worker_buttons[Stage.CROP] = delete_btn
        self.next_stage_buttons[Stage.CROP] = btn_next

        v.add_child(gui.Label("Crop Point Cloud"))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label("Controls"))
        v.add_child(self.btn_box_select)
        v.add_child(self.worker_buttons[Stage.CROP])
        v.add_child(btn_reset)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(btn_back)
        v.add_child(btn_next)
        print("loaded crop panel")

        return v

    def _on_mouse_event(self, event):
        if self.stage != Stage.CROP:
            return o3d.visualization.gui.Widget.EventCallbackResult.IGNORED

        if self.tool_mode != ToolMode.BOX_SELECT:
            return o3d.visualization.gui.Widget.EventCallbackResult.IGNORED

        if event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_DOWN:
            if event.buttons == 1:
                self.is_dragging = True
                self.drag_start = (event.x, event.y)
                self.drag_end = self.drag_start
                return o3d.visualization.gui.Widget.EventCallbackResult.HANDLED

        elif event.type == o3d.visualization.gui.MouseEvent.Type.DRAG:
            if self.is_dragging:
                self.drag_end = (event.x, event.y)
                self.scene.set_view_controls(o3d.visualization.gui.SceneWidget.Controls.NONE)
                self.drag_end = (event.x, event.y)
                return o3d.visualization.gui.Widget.EventCallbackResult.HANDLED

        elif event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_UP:
            if self.is_dragging and event.buttons == 1:
                self.is_dragging = False
                self.drag_end = (event.x, event.y)
                print(f"[INFO] Selection box from {self.drag_start} to {self.drag_end}")
                self._select_points_screen_space()
                self.drag_start = None
                self.drag_end = None

                return o3d.visualization.gui.Widget.EventCallbackResult.HANDLED
        return o3d.visualization.gui.Widget.EventCallbackResult.IGNORED

    def crop_stage_init(self):
        self.selected_indices = []
        self.cropped_pcd = copy.deepcopy(self.raw_pcd) if self.cropped_pcd == None else self.cropped_pcd
        if self.headless:
            return
        self.worker_buttons[Stage.CROP].enabled = (len(self.selected_indices) != 0)
        self.safe_scene_update(lambda: self._clear_scene())
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("crop_pcd", self.cropped_pcd, self.default_point_material))

    def reset_crop_stage(self):
        self.down_pcd = None
        self.selected_indices = []
        self.cropped_pcd = copy.deepcopy(self.raw_pcd)
        if self.headless:
            return
        self.worker_buttons[Stage.CROP].enabled = (len(self.selected_indices) != 0)
        self.safe_scene_update(lambda: self._clear_scene())
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("cropped", self.cropped_pcd, self.default_point_material))

    def _select_points_screen_space(self):
        valid_idx, screen_pts = self.project_world_to_screen(np.asarray(self.cropped_pcd.points))

        selected = []        
        for i, (x, y) in zip(valid_idx, screen_pts):
            selected.append(i) if self._inside_rect(x, y) else None
        print(f"[INFO] Selected {len(selected)} points")
        self.selected_indices = selected

        selection_mask = np.ones(len(self.cropped_pcd.points), dtype=bool)
        selection_mask[selected] = False #
        selected_pcd = mask_point_cloud(self.cropped_pcd, ~selection_mask)
        non_selected_pcd = mask_point_cloud(self.cropped_pcd, selection_mask)

        self.safe_scene_update(lambda: self._clear_scene())
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("selected", selected_pcd, self.overlay_material))
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("non selected", non_selected_pcd, self.default_point_material))
        self.worker_buttons[Stage.CROP].enabled = (len(self.selected_indices) != 0)
        self.next_stage_buttons[Stage.CROP].enabled = True

    def delete_selected_points(self):
        mask = np.ones(len(self.cropped_pcd.points), dtype=bool)
        mask[self.selected_indices] = False
        self.cropped_pcd = mask_point_cloud(self.cropped_pcd, mask)
        print(f"[INFO] Deleted selected {len(self.selected_indices)} points. Remaining points: {len(self.cropped_pcd.points)}")
        self.selected_indices = []
        self.safe_scene_update(lambda: self._clear_scene())
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("cropped", self.cropped_pcd, self.default_point_material))
        self.worker_buttons[Stage.CROP].enabled = (len(self.selected_indices) != 0)

    def _enable_box_selection(self):
        if self.btn_box_select.is_on:
            print("[INFO] Box selection enabled for cropping.")
            self.tool_mode = ToolMode.BOX_SELECT
        else:
            print("[INFO] Box selection disabled.")
            self.tool_mode = ToolMode.NONE

    def _inside_rect(self, x, y):
        xmin, xmax = sorted([self.drag_start[0], self.drag_end[0]])
        ymin, ymax = sorted([self.drag_start[1], self.drag_end[1]])
        return xmin <= x <= xmax and ymin <= y <= ymax
    
    def project_world_to_screen(self, points):
        cam = self.scene.scene.camera
        view = np.asarray(cam.get_view_matrix())
        proj = np.asarray(cam.get_projection_matrix())

        # World → clip space
        pts_h = np.hstack([points, np.ones((len(points), 1))])
        clip = (proj @ view @ pts_h.T).T

        # Perspective divide
        ndc = clip[:, :3] / clip[:, 3:4]

        # Cull points behind camera
        valid = clip[:, 3] > 0
        ndc = ndc[valid]
        valid_indices = np.where(valid)[0]

        # NDC → screen
        x = (ndc[:, 0] * 0.5 + 0.5) * self.scene.frame.width
        y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self.scene.frame.height

        return valid_indices, np.column_stack([x, y])
    
    # ===============================
    # Downsampling stage functions
    # ===============================
    def _downsample_panel(self):
        v = gui.Vert(4)
        
        self.adaptive_checkbox = gui.Checkbox("Adaptive sampling")
        self.adaptive_checkbox.checked = self.use_adaptive
        btn_downsample = gui.Button("Downsample")
        btn_downsample.set_on_clicked(self.start_downsampling)
        btn_recenter = gui.Button("Recenter Point Cloud")
        btn_recenter.set_on_clicked(self.recenter_down_pcd)

        btn_reset = gui.Button("Restart Downsample")
        btn_reset.set_on_clicked(self.reset_downsample_stage)
        btn_next = gui.Button("Next: Save")
        btn_next.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value + 1)))
        self.worker_buttons[Stage.DOWNSAMPLE] = btn_downsample
        self.next_stage_buttons[Stage.DOWNSAMPLE] = btn_next
        btn_back = gui.Button("Back: Crop")
        btn_back.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value - 1)))

        v.add_child(gui.Label("Downsampling"))
        v.add_child(gui.Label(""))
        v.add_child(self.adaptive_checkbox)
        v.add_child(self.worker_buttons[Stage.DOWNSAMPLE])
        v.add_child(btn_recenter)
        v.add_child(btn_reset)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(btn_back)
        v.add_child(self.next_stage_buttons[Stage.DOWNSAMPLE])
        print("loaded downsample panel")

        return v
        
    def downsample_stage_init(self): 
        if self.headless:
            return       
        self.worker_buttons[Stage.DOWNSAMPLE].enabled = True
        self.next_stage_buttons[Stage.DOWNSAMPLE].enabled = (self.down_pcd != None)
        if self.down_pcd != None:
            self.safe_scene_update(lambda: self._clear_scene())
            self.safe_scene_update(lambda: self.scene.scene.add_geometry("down_pcd", self.down_pcd, self.default_point_material))
        elif self.cropped_pcd != None:
            self.safe_scene_update(lambda: self._clear_scene())
            self.safe_scene_update(lambda: self.scene.scene.add_geometry("cropped_pcd", self.cropped_pcd, self.default_point_material))

    def reset_downsample_stage(self):
        self.down_pcd = None
        self.downsample_stage_init()
        # if self.headless:
        #     return
        # self.worker_buttons[Stage.DOWNSAMPLE].enabled = True
        # self.next_stage_buttons[Stage.DOWNSAMPLE].enabled = (self.down_pcd != None)
        # self.safe_scene_update(lambda: self._clear_scene())
        # self.safe_scene_update(lambda: self.scene.scene.add_geometry("cropped_pcd", self.cropped_pcd, self.default_point_material))

    def start_downsampling(self):
        self.set_stage(Stage.DOWNSAMPLE)
        self.downsampling_thread = threading.Thread(target=self._downsample_worker)
        self.downsampling_thread.start()

    def _downsample_worker(self):
        if not self.headless:
            self.use_adaptive = self.adaptive_checkbox.checked
            self.worker_buttons[Stage.DOWNSAMPLE].enabled = False
        if self.cropped_pcd is None:
            print("cropped pcd is none")
            return
        
        bbox = self.cropped_pcd.get_minimal_oriented_bounding_box()
        self.voxel_size = np.round(np.clip((bbox.volume() / 3), 0.001, 0.005), 4)
        self.adaptive_voxel_downsample() if self.use_adaptive else self.uniform_voxel_downsample()
        print(f"[INFO] Downsampled {self.voxel_size * 1000}mm from {len(self.cropped_pcd.points)} to {len(self.down_pcd.points)} points")

        if self.headless:
            return
        self.worker_buttons[Stage.DOWNSAMPLE].enabled = True
        self.next_stage_buttons[Stage.DOWNSAMPLE].enabled = (self.down_pcd != None)
        self.safe_scene_update(lambda: self._clear_scene())
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("down_pcd", self.down_pcd, self.default_point_material))
 
    def uniform_voxel_downsample(self):
        self.down_pcd = normalize_normals(self.cropped_pcd.voxel_down_sample(self.voxel_size))

    def adaptive_voxel_downsample(self):
        if not self.headless:
            self.show_progress("Downsampling point cloud...")
        variation = self.compute_curvature(self.cropped_pcd)
        threshold, percentile, _ = find_cdf_knee(variation)
        feature_mask = variation >= threshold

        pcd_feature = mask_point_cloud(self.cropped_pcd, feature_mask)
        pcd_flat = mask_point_cloud(self.cropped_pcd, ~feature_mask)
        pcd_feature = pcd_feature.voxel_down_sample(self.voxel_size)
        pcd_flat = pcd_flat.voxel_down_sample(self.voxel_size * self.coarse_factor)
        self.down_pcd = normalize_normals(pcd_feature + pcd_flat)
        if not self.headless:
            self.update_progress(1.0)
            self.hide_progress()

    def compute_curvature(self, pcd):
        pts = np.asarray(pcd.points)
        tree = o3d.geometry.KDTreeFlann(pcd)
        curv = np.zeros(len(pts))

        for i in range(len(pts)):
            _, idx, _ = tree.search_knn_vector_3d(pts[i], self.curvature_k_neighbors)
            nbrs = pts[idx]
            C = np.cov(nbrs.T)
            eigvals = np.linalg.eigvalsh(C)
            curv[i] = eigvals[0] / eigvals.sum()   # smallest eigenvalue ratio
            if not self.headless:
                progress = (i + 1) / len(pts)
                self.update_progress(progress)
        return curv

    def recenter_down_pcd(self):
        self.down_pcd, T = center_pointcloud_to_geometric_center(self.down_pcd)
        self.target_mesh.transform(T)
        self.raw_pcd.transform(T)
        self.cropped_pcd.transform(T)
        self.safe_scene_update(lambda: self._clear_scene())
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("down_pcd", self.down_pcd, self.default_point_material))
        print("recentered pointcloud")

    # ===============================
    # Save PLY Stage
    # ===============================

    def _save_panel(self):
        v = gui.Vert(4)

        btn_save = gui.Button("Export Point Cloud")
        btn_save.set_on_clicked(self.save_pcd)

        btn_back = gui.Button("Back: Downsample")
        btn_back.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value - 1)))

        btn_restart = gui.Button("Restart")
        btn_restart.set_on_clicked(lambda: self._restart())
        self.worker_buttons[Stage.SAVE] = btn_save
        v.add_child(gui.Label("Export PCL"))
        v.add_child(gui.Label(""))
        v.add_child(btn_save)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(btn_back)
        v.add_child(btn_restart)

        print("loaded save panel")
        return v
        
    def save_stage_init(self):
        if self.headless:
            return
        self.worker_buttons[Stage.SAVE].enabled = True
        if self.down_pcd != None:
            self.safe_scene_update(lambda: self._clear_scene())
            self.safe_scene_update(lambda: self.scene.scene.add_geometry("down_pcd", self.down_pcd, self.default_point_material))

    def save_pcd(self):
        if self.down_pcd is None:
            self.worker_buttons[Stage.SAVE].enabled = True
            print("[WARN] No pointcloud to save")
            return
        self.worker_buttons[Stage.SAVE].enabled = False
        path = self._save_ply_dialog()
        self.worker_buttons[Stage.SAVE].enabled = True
        if not path:
            return
        try:
            pointcloud_to_ply(self.down_pcd, str(path))
            print(f"[INFO] Point cloud saved to {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save PLY: {e}")
            return

    def _restart(self):
        print("restart wizard")
        self.target_mesh = None
        self.raw_pcd = None
        self.cropped_pcd = None
        self.down_pcd = None
        self.set_stage(Stage.IMPORT_MESH)
        self.safe_scene_update(lambda: self._clear_scene())

    def _express_sampling(self):
        self._raycasting_worker()
        self.down_pcd=app.raw_pcd
        self._downsample_worker()
        self.set_stage(Stage.SAVE)

    def _batch_sampling(self):
        src_dir = self._open_source_folder_dialog()
        if src_dir is None:
            return

        dst_dir = self._open_dest_folder_dialog()
        if dst_dir is None:
            return

        stl_files = list(src_dir.glob("*.stl"))
        print(f"[INFO] Found {len(stl_files)} STL files")

        for stl_path in stl_files:
            print(f"[INFO] Processing {stl_path.name}")
            self.file_path = stl_path
            self._load_mesh_worker()
            self._express_sampling()
            pointcloud_to_ply(self.down_pcd, str(dst_dir / (stl_path.stem + ".ply")))

# ===============================
# Entry point
# ===============================
if __name__ == "__main__":
    try: 
        gui.Application.instance.initialize()
        app = MeshSamplingApp()
        gui.Application.instance.run()
    except Exception as e:
        exit()
