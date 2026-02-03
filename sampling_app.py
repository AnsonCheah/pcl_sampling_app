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
from scipy.spatial.transform import Rotation as R

# ===============================
# Stage definition
# ===============================
class Stage(Enum):
    IMPORT_MESH = 0
    RAYCAST = 1
    CROP = 2
    DOWNSAMPLE = 3
    SAVE = 4
    SYNTHETIC = 5

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
        self.visible_target_pcd = None
        self.occluders_pcd = None

        self.stage_init = {}        
        self.stage_init[Stage.IMPORT_MESH] = self.load_mesh_stage_init
        self.stage_init[Stage.RAYCAST] = self.raycast_stage_init
        self.stage_init[Stage.CROP] = self.crop_stage_init
        self.stage_init[Stage.DOWNSAMPLE] = self.downsample_stage_init
        self.stage_init[Stage.SAVE] = self.save_stage_init
        self.stage_init[Stage.SYNTHETIC] = self.synthetic_stage_init

        # === State parameters ===
        self.mesh_basename = None
        self.camera_distance = 1.5
        self.num_views = 20
        self.ray_margin_mm = 10.0
        self.ray_spacing = 0.001
        self.voxel_size = 0.001
        self.use_adaptive = False
        self.coarse_factor = 2
        self.curvature_k_neighbors = 5
        self.bbox_corners = None
        self.synthetic_occlusion = True
        self.fov_deg = 25
        self.res_width = 1920
        self.res_height = 1200
        self.min_occlusion_ratio = 0.1
        self.max_occlusion_ratio = 0.3
        self.synthetic_targets = []

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
            # self.default_point_material.shader = "Unlit"
            self.default_point_material.point_size = 1.5
            self.default_point_material.base_color = [1.0, 1.0, 1.0, 1.0]
            self.overlay_material = rendering.MaterialRecord()
            # self.overlay_material.shader = "Unlit"
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
            self.stage_panels[Stage.SYNTHETIC] = self._synthetic_panel()

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
        self.pb_panel_size = (300, 50)
        x=(self.window_width - self.pb_panel_size[0])>>1
        y=(self.window_height - self.pb_panel_size[1])>>1
        self.progress_panel.frame = gui.Rect(x, y, self.pb_panel_size[0], self.pb_panel_size[1])
        self.window.add_child(self.progress_panel)

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
        print("reframing")
        if self.target_mesh is None:
            center = np.array([0.0, 0.0, 0.0])
            distance = 0.5
            eye = np.array([-1.0,-1.0,1.0])
            up = np.array([0, 0, 1])

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
    # File dialogs
    # ===============================
    def _open_stl_dialog(self):
        Tk().withdraw()
        path = filedialog.askopenfilename(initialdir=Path.cwd(), filetypes=[("STL files", "*.stl *.STL")])
        if not path:
            return None
        path = Path(path)
        self.mesh_basename = path.stem
        return path

    def _save_ply_dialog(self):
        Tk().withdraw()
        default_name = "output.ply"
        if hasattr(self, "mesh_basename") and self.mesh_basename:
            default_name = f"{self.mesh_basename}.ply"

        path = filedialog.asksaveasfilename(
            defaultextension=".ply",
            initialdir=Path.cwd(), 
            initialfile=default_name,
            filetypes=[("PLY files", "*.ply")]
        )

        return Path(path) if path else None
    
    def _open_source_folder_dialog(self):
        Tk().withdraw()
        path = filedialog.askdirectory(initialdir=Path.cwd(), title="Select source folder (STL files)")
        return Path(path) if path else None

    def _open_dest_folder_dialog(self):
        Tk().withdraw()
        path = filedialog.askdirectory(initialdir=Path.cwd(), title="Select destination folder (PLY output)")
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
            if self.stage.value >=5:
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
        # btn_load.set_on_clicked(self.start_import_worker)

        btn_reset = gui.Button("Clear Mesh")
        btn_reset.set_on_clicked(self.reset_mesh_stage)
        self.btn_express = gui.Button("Express Sampling")
        self.btn_express.set_on_clicked(self.start_express_sampling)
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
        self.main_thread(lambda: self._clear_scene())
        if self.target_mesh is not None:
            self.main_thread(lambda: self.scene.scene.add_geometry("mesh", self.target_mesh, self.default_material))
        self.scene.force_redraw()
        self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.IMPORT_MESH], True))
        self.main_thread(lambda: self.enable_button(self.next_stage_buttons[Stage.IMPORT_MESH], (self.target_mesh != None)))
        self.main_thread(lambda: self.enable_button(self.btn_express, (self.target_mesh != None)))

    def reset_mesh_stage(self):
        self.target_mesh = None
        self.raw_pcd = None
        self.cropped_pcd = None
        self.down_pcd = None
        self.load_mesh_stage_init()

    def import_mesh(self):
        if not self.headless:
            self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.IMPORT_MESH], False))
        self.file_path = self._open_stl_dialog()
        if not self.file_path:
            if not self.headless:
                self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.IMPORT_MESH], True))
            return
        self._load_mesh_worker()
        self.load_mesh_stage_init()
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
        if self.raw_pcd is not None:
            self.main_thread(lambda: self._clear_scene())
            self.main_thread(lambda: self.scene.scene.add_geometry("raw_pcd", self.raw_pcd, self.default_material))
        elif self.target_mesh is not None:
            self.main_thread(lambda: self._clear_scene())
            self.main_thread(lambda: self.scene.scene.add_geometry("mesh", self.target_mesh, self.default_material))
        self.scene.force_redraw()
        self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.RAYCAST], True))
        self.main_thread(lambda: self.enable_button(self.next_stage_buttons[Stage.RAYCAST], (self.raw_pcd != None)))

    def reset_raycast_stage(self):
        self.raw_pcd = None
        self.cropped_pcd = None
        self.down_pcd = None
        self.raycast_stage_init()

    def start_raycasting(self):
        self.set_stage(Stage.RAYCAST)
        self.raycasting_thread = threading.Thread(target=self._raycasting_worker)
        self.raycasting_thread.start()

    def _raycasting_worker(self):
        if self.target_mesh is None:
            print("[WARN] No mesh loaded")
            return
        if not self.headless:
            self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.RAYCAST], False))
            self.main_thread(lambda: self.enable_button(self.next_stage_buttons[Stage.RAYCAST], False))
            self.camera_distance = self.camera_distance_slider.double_value
            self.num_views = self.num_views_slider.int_value
            self.show_progress("Raycasting mesh...")

        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.target_mesh))
        view_dirs = fibonacci_sphere(self.num_views)
        all_points = []
        all_cam_pos = []
        for view_index, view_dir in enumerate(view_dirs):
            cam_pos = view_dir * (self.camera_distance)
            forward = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross([0, 0, 1], forward)
            if np.linalg.norm(right) < 1e-6:
                right = np.cross([0, 1, 0], forward)
            right /= np.linalg.norm(right)
            up = np.cross(forward, right)

            # Project bbox corners onto camera plane with margin
            rel = self.bbox_corners - cam_pos
            x_proj = rel @ right
            y_proj = rel @ up
            margin = self.ray_margin_mm * 1e-3
            xs = np.arange(x_proj.min() - margin, x_proj.max() + margin, self.ray_spacing)
            ys = np.arange(y_proj.min() - margin, y_proj.max() + margin, self.ray_spacing)

            if len(xs) == 0 or len(ys) == 0:
                continue

            uu, vv = np.meshgrid(xs, ys)
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
                self.update_progress((view_index + 1) / self.num_views)

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
        orient_normals_using_cameras(pcd, cam_positions)
        print(f"[INFO] Total points sampled: {len(pcd.points)}")
        initial_voxel = (self.ray_spacing)*0.3
        pcd = pcd.voxel_down_sample(initial_voxel)
        normalize_normals(pcd)
        validate_normals(pcd)
        print(f"[INFO] Initial voxelized points: {len(pcd.points)}")
        self.raw_pcd = pcd
        self.cropped_pcd = copy.deepcopy(self.raw_pcd)

        print("raycast finished")
        self.raycast_stage_init()
        self.hide_progress()

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
                self._draw_selection_rectangle()
                self.scene.set_view_controls(o3d.visualization.gui.SceneWidget.Controls.PICK_POINTS)
                return o3d.visualization.gui.Widget.EventCallbackResult.HANDLED

        elif event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_UP:
            if self.is_dragging and event.buttons == 1:
                self.is_dragging = False
                self.drag_end = (event.x, event.y)
                self._clear_selection_rectangle()
                self._select_points_screen_space()
                self.drag_start = None
                self.drag_end = None
                self.scene.set_view_controls(o3d.visualization.gui.SceneWidget.Controls.ROTATE_CAMERA)
                return o3d.visualization.gui.Widget.EventCallbackResult.HANDLED
        return o3d.visualization.gui.Widget.EventCallbackResult.IGNORED

    def crop_stage_init(self):
        self.selected_indices = []
        if self.headless:
            return
        self.main_thread(lambda: self._clear_scene())
        self.main_thread(lambda: self.scene.scene.add_geometry("crop_pcd", self.cropped_pcd, self.default_point_material))
        self.scene.force_redraw()
        self.main_thread(lambda: self.enable_button(self.next_stage_buttons[Stage.CROP], True))
        self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.CROP], (len(self.selected_indices) != 0)))

    def reset_crop_stage(self):
        self.down_pcd = None
        self.cropped_pcd = copy.deepcopy(self.raw_pcd)
        self._clear_selection_rectangle()
        self.crop_stage_init()

    def _select_points_screen_space(self):
        valid_idx, screen_pts = self.project_world_to_screen(np.asarray(self.cropped_pcd.points))
        selected = []
        for i, (x, y) in zip(valid_idx, screen_pts):
            selected.append(i) if self._inside_rect(x, y) else None
        print(f"[INFO] Selected {len(selected)} points")
        self.selected_indices = selected

        selection_mask = np.ones(len(self.cropped_pcd.points), dtype=bool)
        selection_mask[selected] = False
        selected_pcd = mask_point_cloud(self.cropped_pcd, ~selection_mask)
        non_selected_pcd = mask_point_cloud(self.cropped_pcd, selection_mask)

        self.main_thread(lambda: self.enable_button(self.next_stage_buttons[Stage.CROP], True))
        self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.CROP], (len(self.selected_indices) != 0)))
        self.main_thread(lambda: self._clear_scene())
        self.main_thread(lambda: self.scene.scene.add_geometry("selected", selected_pcd, self.overlay_material))
        self.main_thread(lambda: self.scene.scene.add_geometry("non selected", non_selected_pcd, self.default_point_material))
        self.scene.force_redraw()

    def delete_selected_points(self):
        mask = np.ones(len(self.cropped_pcd.points), dtype=bool)
        mask[self.selected_indices] = False
        self.cropped_pcd = mask_point_cloud(self.cropped_pcd, mask)
        print(f"[INFO] Deleted selected {len(self.selected_indices)} points. Remaining points: {len(self.cropped_pcd.points)}")
        self._clear_selection_rectangle()
        self.crop_stage_init()

    def _enable_box_selection(self):
        if self.btn_box_select.is_on:
            print("[INFO] Box selection enabled for cropping.")
            self.tool_mode = ToolMode.BOX_SELECT
        else:
            print("[INFO] Box selection disabled.")
            self.tool_mode = ToolMode.NONE
            self._clear_selection_rectangle()

    def _inside_rect(self, x, y):
        """Check if a point is inside the selection rectangle"""
        xmin, xmax = sorted([self.drag_start[0], self.drag_end[0]])
        ymin, ymax = sorted([self.drag_start[1], self.drag_end[1]])
        return xmin <= x <= xmax and ymin <= y <= ymax
    
    def project_world_to_screen(self, points):
        """Project 3D world points to 2D screen coordinates"""
        cam = self.scene.scene.camera
        view = np.asarray(cam.get_view_matrix())
        proj = np.asarray(cam.get_projection_matrix())

        # World → clip space
        pts_h = np.hstack([points, np.ones((len(points), 1))])
        clip = (proj @ view @ pts_h.T).T

        # Cull points behind camera BEFORE perspective divide (more efficient)
        valid = clip[:, 3] > 0
        
        if not np.any(valid):
            return np.array([], dtype=int), np.empty((0, 2))
        
        clip = clip[valid]
        valid_indices = np.where(valid)[0]

        # Perspective divide
        ndc = clip[:, :3] / clip[:, 3:4]

        # NDC → screen
        x = (ndc[:, 0] * 0.5 + 0.5) * self.scene.frame.width
        y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self.scene.frame.height

        return valid_indices, np.column_stack([x, y])
    
    def _get_selection_frustum_corners(self):
        """Get the 3D corners of the selection frustum in world space"""
        if self.drag_start is None or self.drag_end is None:
            return None
        
        x1, y1 = self.drag_start
        x2, y2 = self.drag_end
        
        # Check minimum selection size (at least 5 pixels)
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return None
        
        cam = self.scene.scene.camera
        view_matrix = np.asarray(cam.get_view_matrix())
        proj_matrix = np.asarray(cam.get_projection_matrix())
        
        # Pre-compute inverse matrices (only once)
        inv_view = np.linalg.inv(view_matrix)
        inv_proj_view = np.linalg.inv(proj_matrix @ view_matrix)
        cam_pos = inv_view[:3, 3]  # Camera position
        
        # Calculate target depth
        target_depth = self._calculate_selection_depth(x1, y1, x2, y2, cam_pos)
        
        # Screen corners
        corners_screen = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        corners_world = []
        
        # Pre-calculate NDC-to-screen conversion factors
        width = self.scene.frame.width
        height = self.scene.frame.height
        
        for sx, sy in corners_screen:
            # Convert to Normalized Device Coordinates
            ndc_x = (2.0 * sx / width) - 1.0
            ndc_y = 1.0 - (2.0 * sy / height)
            # Use only the NEAR plane (z=-1) which has valid w-component
            near_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0])
            # Transform to world space
            near_world_h = inv_proj_view @ near_ndc
            # Perspective divide with safety check
            w = near_world_h[3]
            if abs(w) < 1e-6:
                return None
            
            near_world = near_world_h[:3] / w
            
            # Calculate ray direction from camera to near point
            ray_dir = near_world - cam_pos
            ray_length = np.linalg.norm(ray_dir)
            
            if ray_length < 1e-6:
                return None
            
            ray_dir = ray_dir / ray_length
            
            # Place point at target depth along the ray
            point = cam_pos + ray_dir * target_depth
            corners_world.append(point)
        
        # Verify corners form a non-degenerate rectangle
        if not self._validate_rectangle_corners(corners_world):
            return None
        
        return corners_world
    
    def _calculate_selection_depth(self, x1, y1, x2, y2, cam_pos):
        """Calculate the optimal depth for the selection rectangle"""
        # Default depth
        default_depth = 0.5
        
        if not hasattr(self, 'cropped_pcd') or self.cropped_pcd is None:
            return default_depth
        
        points = np.asarray(self.cropped_pcd.points)
        if len(points) == 0:
            return default_depth
        
        # Project all points to screen (this is cached-friendly)
        valid_idx, screen_pts = self.project_world_to_screen(points)
        
        if len(valid_idx) == 0:
            return default_depth
        
        # Find points within the selection rectangle (vectorized)
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        in_rect = (
            (screen_pts[:, 0] >= xmin) & 
            (screen_pts[:, 0] <= xmax) & 
            (screen_pts[:, 1] >= ymin) & 
            (screen_pts[:, 1] <= ymax)
        )
        
        if np.any(in_rect):
            # Get the closest point within the selection (vectorized)
            rect_points = points[valid_idx[in_rect]]
            distances_to_cam = np.linalg.norm(rect_points - cam_pos, axis=1)
            min_depth = np.min(distances_to_cam)
            
            # Place rectangle significantly in front (50% closer for better visibility)
            return min_depth * 0.50
        else:
            # No points in selection, use center-based approach
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            # Vectorized distance calculation
            distances = np.linalg.norm(
                screen_pts - np.array([center_x, center_y]), 
                axis=1
            )
            closest_idx = valid_idx[np.argmin(distances)]
            closest_point = points[closest_idx]
            
            # Use 90% depth for fallback (less aggressive than selected points)
            return np.linalg.norm(closest_point - cam_pos) * 0.90
    
    def _validate_rectangle_corners(self, corners_world):
        """Validate that the rectangle corners form a non-degenerate shape"""
        if corners_world is None or len(corners_world) != 4:
            return False
        
        # Check all corners are finite
        corners_array = np.array(corners_world)
        if not np.all(np.isfinite(corners_array)):
            return False
        
        # Check bounding box has non-zero volume
        bbox_min = corners_array.min(axis=0)
        bbox_max = corners_array.max(axis=0)
        bbox_size = bbox_max - bbox_min
        
        # All dimensions should be larger than epsilon
        return np.all(bbox_size >= 1e-6)
    
    def _select_points_screen_space(self):
        """Select points within the screen-space rectangle"""
        valid_idx, screen_pts = self.project_world_to_screen(np.asarray(self.cropped_pcd.points))
        
        # Vectorized selection (much faster than loop)
        xmin, xmax = sorted([self.drag_start[0], self.drag_end[0]])
        ymin, ymax = sorted([self.drag_start[1], self.drag_end[1]])
        
        in_rect = (
            (screen_pts[:, 0] >= xmin) & 
            (screen_pts[:, 0] <= xmax) & 
            (screen_pts[:, 1] >= ymin) & 
            (screen_pts[:, 1] <= ymax)
        )
        
        selected = valid_idx[in_rect].tolist()
        
        print(f"[INFO] Selected {len(selected)} points")
        self.selected_indices = selected

        # Create masks (vectorized)
        selection_mask = np.ones(len(self.cropped_pcd.points), dtype=bool)
        selection_mask[selected] = False
        
        selected_pcd = mask_point_cloud(self.cropped_pcd, ~selection_mask)
        non_selected_pcd = mask_point_cloud(self.cropped_pcd, selection_mask)

        # Update UI
        self.main_thread(lambda: self.enable_button(self.next_stage_buttons[Stage.CROP], True))
        self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.CROP], (len(self.selected_indices) != 0)))
        self.main_thread(lambda: self._clear_scene())
        self.main_thread(lambda: self.scene.scene.add_geometry("selected", selected_pcd, self.overlay_material))
        self.main_thread(lambda: self.scene.scene.add_geometry("non selected", non_selected_pcd, self.default_point_material))
        self.scene.force_redraw()

    def _draw_selection_rectangle(self):
        """Draw a live rectangle overlay during box selection"""
        if self.drag_start is None or self.drag_end is None:
            return
        if self.scene.scene.has_geometry("selection_rect"):
            self.scene.scene.remove_geometry("selection_rect")

        corners_world = self._get_selection_frustum_corners()
        if corners_world is None or len(corners_world) != 4:
            return
        
        for corner in corners_world:
            if not np.all(np.isfinite(corner)):
                return
        
        lines = [[0, 1],[1, 2],[2, 3],[3, 0]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0] for _ in range(len(lines))])
        
        rect_material = rendering.MaterialRecord()
        rect_material.shader = "unlitLine"
        rect_material.line_width = 3.0
        rect_material.base_color = [1.0, 1.0, 0.0, 1.0]
        
        try:
            self.scene.scene.add_geometry("selection_rect", line_set, rect_material)
            self.scene.force_redraw()
        except Exception as e:
            print(f"[WARNING] Could not draw selection rectangle: {e}")

    def _clear_selection_rectangle(self):
        """Remove the selection rectangle from the scene"""
        if self.scene.scene.has_geometry("selection_rect"):
            self.scene.scene.remove_geometry("selection_rect")
            self.scene.force_redraw()

    # ===============================
    # Downsampling stage functions
    # ===============================
    def _downsample_panel(self):
        v = gui.Vert(4)
        
        self.adaptive_checkbox = gui.Checkbox("Adaptive sampling")
        self.adaptive_checkbox.checked = self.use_adaptive
        btn_downsample = gui.Button("Downsample")
        btn_downsample.set_on_clicked(self.start_downsampling)
        self.btn_recenter = gui.Button("Recenter Point Cloud")
        self.btn_recenter.set_on_clicked(self.recenter_mesh_pcd)

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
        v.add_child(self.btn_recenter)
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

        if self.down_pcd != None:
            self.main_thread(lambda: self._clear_scene())
            self.main_thread(lambda: self.scene.scene.add_geometry("down_pcd", self.down_pcd, self.default_point_material))
        elif self.cropped_pcd != None:
            self.main_thread(lambda: self._clear_scene())
            self.main_thread(lambda: self.scene.scene.add_geometry("cropped_pcd", self.cropped_pcd, self.default_point_material))
        self.scene.force_redraw()
        self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.DOWNSAMPLE], True))
        self.main_thread(lambda: self.enable_button(self.next_stage_buttons[Stage.DOWNSAMPLE], (self.down_pcd != None)))
        self.main_thread(lambda: self.enable_button(self.btn_recenter, True))

    def reset_downsample_stage(self):
        self.down_pcd = None
        self.downsample_stage_init()

    def start_downsampling(self):
        self.set_stage(Stage.DOWNSAMPLE)
        self.downsampling_thread = threading.Thread(target=self._downsample_worker)
        self.downsampling_thread.start()

    def _downsample_worker(self):
        if not self.headless:
            self.use_adaptive = self.adaptive_checkbox.checked
            self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.DOWNSAMPLE], False))
            self.show_progress("Downsampling point cloud...")
        if self.cropped_pcd is None:
            print("cropped pcd is none")
            return
        
        bbox = self.cropped_pcd.get_minimal_oriented_bounding_box()
        self.voxel_size = np.round(np.clip((bbox.volume() / 3), 0.001, 0.005), 4)
        self.adaptive_voxel_downsample() if self.use_adaptive else self.uniform_voxel_downsample()
        print(f"[INFO] Downsampled {self.voxel_size * 1000}mm from {len(self.cropped_pcd.points)} to {len(self.down_pcd.points)} points")
        self.downsample_stage_init()
        if not self.headless:
            self.update_progress(1.0)
            self.hide_progress()

    def uniform_voxel_downsample(self):
        self.down_pcd = normalize_normals(self.cropped_pcd.voxel_down_sample(self.voxel_size))

    def adaptive_voxel_downsample(self):
        variation = self.compute_curvature(self.cropped_pcd)
        threshold, percentile, _ = find_cdf_knee(variation)
        feature_mask = variation >= threshold

        pcd_feature = mask_point_cloud(self.cropped_pcd, feature_mask)
        pcd_flat = mask_point_cloud(self.cropped_pcd, ~feature_mask)
        pcd_feature = pcd_feature.voxel_down_sample(self.voxel_size)
        pcd_flat = pcd_flat.voxel_down_sample(self.voxel_size * self.coarse_factor)
        self.down_pcd = pcd_feature + pcd_flat
        normalize_normals(self.down_pcd)

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
            if not self.headless and (i + 1) % batch_size == 0:
                self.update_progress((i + 1) / n_points)
        
        if not self.headless:
            self.update_progress(1.0)
        
        return curv

    def recenter_mesh_pcd(self):
        if self.cropped_pcd == None:
            return
        if not self.headless:
            self.main_thread(lambda: self.enable_button(self.btn_recenter, False))
        T = pcd_geocenter(self.cropped_pcd)
        if self.down_pcd != None:
            self.down_pcd.transform(T)
        self.target_mesh.transform(T)
        self.raw_pcd.transform(T)
        self.cropped_pcd.transform(T)
        self.downsample_stage_init()
        print("recentered pointcloud")

    # ===============================
    # Save PLY Stage
    # ===============================

    def _save_panel(self):
        v = gui.Vert(4)

        btn_save = gui.Button("Export Point Cloud")
        btn_save.set_on_clicked(self.save_pcd)

        btn_next = gui.Button("Next: Synthetic Target")
        btn_next.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value + 1)))
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
        v.add_child(btn_next)
        v.add_child(btn_restart)

        print("loaded save panel")
        return v
        
    def save_stage_init(self):
        if self.headless:
            return
        if self.down_pcd != None:
            self.main_thread(lambda: self._clear_scene())
            self.main_thread(lambda: self.scene.scene.add_geometry("down_pcd", self.down_pcd, self.default_point_material))
            self.scene.force_redraw()
            self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.SAVE], True))

    def save_pcd(self):
        if not self.headless:
            if self.down_pcd is None:
                self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.SAVE], True))
                print("[WARN] No pointcloud to save")
                return
            self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.SAVE], False))
        # path = self._save_ply_dialog()
        # path = ""
        # if not path:
        #     return
        try:
            folder_path = Path.cwd() / "reference_pcd"
            folder_path.mkdir(parents=True, exist_ok=True)
            pcd_path = folder_path / (self.mesh_basename + ".ply")
            pointcloud_to_ply(self.down_pcd, str(pcd_path))
            print(f"[INFO] Point cloud saved to {pcd_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save PLY: {e}")
            return
        finally:
            self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.SAVE], True))

    # ===============================
    # Synthetic Target Stage
    # ===============================

    def _synthetic_panel(self):
        v = gui.Vert(4)

        self.num_targets_slider = gui.Slider(gui.Slider.INT)
        self.num_targets_slider.set_limits(2, 200)
        self.num_targets_slider.int_value = 6
        btn_generate = gui.Button("Generate Synthetic Targets")
        btn_generate.set_on_clicked(self.start_synthetic)
        self.clear_synthetic_btn = gui.Button("Clear Synthetic Targets")
        self.clear_synthetic_btn.set_on_clicked(self.clear_synthetic)
        self.combobox_targets = gui.Combobox()
        self.combobox_targets.set_on_selection_changed(self.preview_synthetic_target)
        btn_export = gui.Button("Export Synthetic Targets")
        btn_export.set_on_clicked(self.save_synthetic_targets)

        btn_back = gui.Button("Back: SAVE")
        btn_back.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value - 1)))

        btn_restart = gui.Button("Restart")
        btn_restart.set_on_clicked(lambda: self._restart())
        self.worker_buttons[Stage.SYNTHETIC] = btn_generate
        v.add_child(gui.Label("Generate Synthetic Targets"))
        v.add_child(gui.Label(""))
        v.add_child(self.num_targets_slider)
        v.add_child(btn_generate)
        v.add_child(self.clear_synthetic_btn)
        v.add_child(self.combobox_targets)
        v.add_child(btn_export)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(btn_back)
        v.add_child(btn_restart)

        print("loaded synthetic panel")
        return v
        
    def start_synthetic(self):
        self.synthetic_thread = threading.Thread(target=self._synthetic_target_worker)
        self.synthetic_thread.start()
        if not self.headless:
            self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.SYNTHETIC], False))
            self.show_progress("Generating synthetic targets")

    def synthetic_stage_init(self):
        if self.headless:
            return
        self.main_thread(lambda: self._clear_scene())
        if self.synthetic_targets != []:
            self.main_thread(lambda: self.scene.scene.add_geometry("synthetic_target", self.synthetic_targets[0], self.default_point_material))
        elif self.target_mesh != None:
            self.main_thread(lambda: self.scene.scene.add_geometry("mesh", self.target_mesh, self.default_material))
        self.scene.force_redraw()
        self.main_thread(lambda: self.enable_button(self.worker_buttons[Stage.SYNTHETIC], True))
        self.main_thread(lambda: self.enable_button(self.clear_synthetic_btn, (len(self.synthetic_targets)>0)))
    
    def clear_synthetic(self):
        self.synthetic_targets = []
        self.combobox_targets.clear_items()
        self.synthetic_stage_init()

    def _synthetic_target_worker(self):
        self.num_targets =  self.num_targets_slider.int_value
        self.view_sphere = fibonacci_sphere(self.num_targets)
        self.visible_target_pcd = o3d.geometry.PointCloud()
        self.occluders_pcd = o3d.geometry.PointCloud()
        dropout = 0.1
        for i in range(self.num_targets):
            cam_pos, look_at, up = random_camera(self.view_sphere[i], self.camera_distance)
            proj_pos = projector_from_camera(cam_pos, look_at, baseline=0.3)
            print(cam_pos, proj_pos)
            target_center = self.target_mesh.get_center()
            view_dir = (target_center - cam_pos)
            view_dir = view_dir / np.linalg.norm(view_dir)
            scene_meshes = [self.target_mesh]

            # orthonormal basis around view dir
            right = np.cross(view_dir, [0,0,1])
            right = np.cross(view_dir, [0,1,0]) if np.linalg.norm(right) < 1e-6 else right
            right /= np.linalg.norm(right)
            up = np.cross(right, view_dir)

            # --- Target only ---
            initial_res = scene_render(scene_meshes, cam_pos, proj_pos, look_at, self.fov_deg, self.res_width, self.res_height, dropout_prob=dropout)

            target_hit_ids = initial_res["geom_ids_hit"]
            target_geom_ids = [int(i) for i in np.unique(target_hit_ids) if i != 4294967295 and i != -1]
            if len(target_geom_ids) != 1 or target_geom_ids[0] != 0:
                raise RuntimeError("Foreign id in target generation")
            target_geom_id = target_geom_ids[0]
            target_pixels = len(target_hit_ids)
            print(f"target pixels: {target_pixels}")

            self.occluders = []
            view_dir = (target_center - cam_pos)
            view_dir = view_dir / np.linalg.norm(view_dir)
            target_extent = self.target_mesh.get_axis_aligned_bounding_box().get_extent()
            target_radius = 0.5 * np.linalg.norm(target_extent)
            occlusion_ratio = 0
            self.synthetic_occlusion = i>=(self.num_targets>>1)
            num_occluders = (0 if not self.synthetic_occlusion else 1)
            max_trials = 1000
            for occ_idx in range(num_occluders):
                success = False

                for trial in range(max_trials):
                    occ = copy.deepcopy(self.target_mesh)
                    occ.rotate(random_rotation_matrix(), center=(0,0,0))
                    right_offset = right * np.random.uniform(-1.5*target_radius, 1.5*target_radius)
                    up_offset = up * np.random.uniform(-1.5*target_radius, 1.5*target_radius)
                    depth_offset = np.random.uniform(1, 3) * target_radius
                    base_pos = target_center - view_dir * depth_offset
                    occ.translate(base_pos + right_offset + up_offset)

                    if any(meshes_intersect(m, occ) for m in scene_meshes):
                        print(f"intersection detected, regenerating")
                        continue

                    test_scene = scene_meshes + [occ]
                    res = scene_render(test_scene, cam_pos, proj_pos, look_at, self.fov_deg, self.res_width, self.res_height, dropout_prob=dropout)
                    # self.ray_origins = res["ray_origins"]
                    # self.ray_hits    = res["ray_hits"]
                    geom_ids_hit     = res["geom_ids_hit"]
                    scene_pcd        = res["pcd"]

                    scene_pcd.estimate_normals()
                    target_hit_mask = geom_ids_hit == target_geom_id
                    self.visible_target_pcd = mask_point_cloud(scene_pcd, target_hit_mask)
                    self.occluders_pcd      = mask_point_cloud(scene_pcd, ~target_hit_mask)
                    visible_pixels = len(self.visible_target_pcd.points)
                    occlusion_ratio = 1 - (visible_pixels / target_pixels)
                    if not (self.min_occlusion_ratio < occlusion_ratio < self.max_occlusion_ratio):
                        print(f"occlusion ratio out of range: {occlusion_ratio}")
                        continue
                    print(f"Accepted pcd occlusion ratio: {occlusion_ratio}")    

                    self.occluders.append(occ)
                    scene_meshes.append(occ)
                    success = True
                    break

                if not success:
                    raise RuntimeError(f"Failed to generate occluder {occ_idx}")

            if not self.synthetic_occlusion: # handles no occlusion
                self.visible_target_pcd = initial_res["pcd"]
                self.visible_target_pcd.estimate_normals()
            # orient_normals_using_cameras(self.visible_target_pcd, cam_pos)
            # normalize_normals(self.visible_target_pcd)
            # validate_normals(self.visible_target_pcd)

            # self.visible_target_pcd = add_outliers(self.visible_target_pcd)
            print(f"num point bef ds: {len(self.visible_target_pcd.points)}")
            self.visible_target_pcd = self.visible_target_pcd.voxel_down_sample(0.001)
            print(f"num point aft ds: {len(self.visible_target_pcd.points)}")
            self.visible_target_pcd.estimate_normals()
            orient_normals_using_cameras(self.visible_target_pcd, cam_pos)
            normalize_normals(self.visible_target_pcd)
            validate_normals(self.visible_target_pcd)
            self.synthetic_targets.append(self.visible_target_pcd)

            if not self.headless:
                self.combobox_targets.add_item(f"synthetic_sample_{len(self.synthetic_targets)}")
                self.update_progress((i + 1) / self.num_targets)

        if not self.headless:
            self.hide_progress()
            self.synthetic_stage_init()

    def save_synthetic_targets(self):
        base_path = Path.cwd() / "synthetic_target" / self.mesh_basename
        for start, prefix in enumerate(["train", "test"]):
            folder_path = base_path / prefix
            folder_path.mkdir(parents=True, exist_ok=True)
            for i, pcd in enumerate(self.synthetic_targets[start::2]):
                pointcloud_to_ply(pcd, folder_path / f"{prefix}_sample_{i}.ply")

    def preview_synthetic_target(self, selected_text: str, selected_index: int) -> None:
        if self.headless:
            return
        if self.synthetic_targets[selected_index] != None:
            self.main_thread(lambda: self._clear_scene())
            self.main_thread(lambda: self.scene.scene.add_geometry("synthetic_target", self.synthetic_targets[selected_index], self.default_point_material))
        self.scene.force_redraw()

    def _restart(self):
        print("restart wizard")
        self.target_mesh = None
        self.raw_pcd = None
        self.cropped_pcd = None
        self.down_pcd = None
        self.synthetic_targets = []
        self.combobox_targets.clear_items()
        self.set_stage(Stage.IMPORT_MESH)
        if not self.headless:
            self.main_thread(lambda: self._clear_scene())

    # ===============================
    # Express Handler
    # ===============================
    def start_express_sampling(self):
        self._express_sampling_thread = threading.Thread(target=self._express_sampling_worker)
        self._express_sampling_thread.start()

    def _express_sampling_worker(self):
        self._raycasting_worker()
        self.down_pcd=self.raw_pcd
        self._downsample_worker()
        self.recenter_mesh_pcd()
        self.set_stage(Stage.SAVE)

    def _batch_sampling(self):
        src_dir = self._open_source_folder_dialog()
        if src_dir is None:
            return

        # dst_dir = self._open_dest_folder_dialog()
        dst_dir = Path.cwd() / "reference_pcd"
        if dst_dir is None:
            return

        stl_files = list(src_dir.glob("*.stl"))
        print(f"[INFO] Found {len(stl_files)} STL files")

        for stl_path in stl_files:
            print(f"[INFO] Processing {stl_path.name}")
            self.file_path = stl_path
            self._load_mesh_worker()
            self.start_express_sampling()
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
