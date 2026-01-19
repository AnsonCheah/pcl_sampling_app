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

    def __init__(self):
        self.stage = Stage.IMPORT_MESH
        self.window = gui.Application.instance.create_window("Mesh Sampling Tool", 1440, 900)

        # === Data variables ===
        self.mesh = None
        self.raw_pcd = None
        self.cropped_pcd = None
        self.down_pcd = None

        self.tool_mode = ToolMode.NONE
        self.is_dragging = False
        self.drag_start = None
        self.drag_end = None
        self.selected_indices = []
        self.default_material = rendering.MaterialRecord()
        self.default_material.shader = "defaultLit"
        self.overlay_material = rendering.MaterialRecord()
        self.overlay_material.shader = "defaultLit"
        self.overlay_material.base_color = [1.0, 0.5, 0.3, 1.0]

        self.stage_init = {}        
        self.stage_init[Stage.IMPORT_MESH] = self.load_mesh_stage_init
        self.stage_init[Stage.RAYCAST] = self.raycast_stage_init
        self.stage_init[Stage.CROP] = self.crop_stage_init
        self.stage_init[Stage.DOWNSAMPLE] = self.downsample_stage_init
        self.stage_init[Stage.SAVE] = self.save_stage_init

        # === State parameters ===
        self.camera_distance = 1.5
        self.num_views = 30
        self.image_res = 1000
        self.voxel_size = 0.001
        self.use_adaptive = True
        self.feature_ratio = 0.1
        self.coarse_factor = 3.0
        self.curvature_k_neighbors = 5

        # === Scene widget ===
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.2, 0.2, 0.2, 1.0])
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)
        self.scene.set_on_mouse(self._on_mouse_event)
        self.window.add_child(self.scene)

        # ===============================
        # Build Control Panel
        # ===============================

        em = self.window.theme.font_size
        self.panel = gui.Vert(0.25 * em, gui.Margins(em, em, em, em))
        self.window.add_child(self.panel)

        self.next_stage_buttons = {}
        self.stage_panels = {}
        self.stage_panels[Stage.IMPORT_MESH] = self._load_mesh_panel()
        self.stage_panels[Stage.RAYCAST] = self._raycast_panel()
        self.stage_panels[Stage.CROP] = self._crop_panel()
        self.stage_panels[Stage.DOWNSAMPLE] = self._downsample_panel()
        self.stage_panels[Stage.SAVE] = self._save_panel()


        for p in self.stage_panels.values():
            p.visible = False
            self.panel.add_child(p)

        self.set_stage(Stage.IMPORT_MESH)

    # ===============================
    # Control Panel
    # ===============================

    def set_stage(self, stage: Stage):
        print(f"Transitioning from {self.stage.name} to {stage.name}")
        self.stage = stage
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


    # ===============================
    # UI helpers
    # ===============================
    def _update_title(self):
        self.window.title = f"Mesh Sampling Pipeline | Stage: {self.stage.name}"

    def _clear_scene(self):
        self.scene.scene.clear_geometry()

    def _reframe(self):
        if self.mesh == None:
            return
        bbox = self.mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        self.scene.setup_camera(20.0, bbox, center)

    def safe_scene_update(self, fn):
        gui.Application.instance.post_to_main_thread(self.window, fn)

    # ===============================
    # File dialogs
    # ===============================
    def _open_stl_dialog(self):
        Tk().withdraw()
        path = filedialog.askopenfilename(filetypes=[("STL files", "*.stl *.STL")])
        return Path(path) if path else None

    def _save_ply_dialog(self):
        Tk().withdraw()
        path = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=[("PLY files", "*.ply")])
        return Path(path) if path else None

    # ===============================
    # Keybindings
    # ===============================
    def _on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return False

        key = event.key

        # --- Global ---
        if key == gui.KeyName.Q:
            gui.Application.instance.quit()
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
                self.raycast_mesh()
        elif self.stage == Stage.CROP:
            if key == gui.KeyName.S:
                self.save_pcd()
            elif key == gui.KeyName.D:
                self.delete_selected_points()

        elif self.stage == Stage.DOWNSAMPLE:
            if key == gui.KeyName.S:
                self.downsample()
            elif key == gui.KeyName.T:
                self.use_adaptive = not self.use_adaptive
                self.adaptive_checkbox.checked = not self.adaptive_checkbox.checked

        elif self.stage == Stage.SAVE:
            if key == gui.KeyName.S:
                self.save_pcd()
            elif key == gui.KeyName.R:
                self.mesh = None
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

        btn_next = gui.Button("Next: Raycast")
        btn_next.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value + 1)))
        self.next_stage_buttons[Stage.IMPORT_MESH] = btn_next
        v.add_child(gui.Label("Import Mesh"))
        v.add_child(btn_load)
        v.add_child(btn_reset)
        v.add_child(self.next_stage_buttons[Stage.IMPORT_MESH])
        print("loaded mesh panel")
        return v
    
    def load_mesh_stage_init(self):
        self.next_stage_buttons[Stage.IMPORT_MESH].enabled = (self.mesh != None)

    def reset_mesh_stage(self):
        self.mesh = None
        self.raw_pcd = None
        self.cropped_pcd = None
        self.down_pcd = None
        self.next_stage_buttons[Stage.IMPORT_MESH].enabled = False
        self.safe_scene_update(lambda: self._clear_scene())

    def import_mesh(self):
        path = self._open_stl_dialog()
        if not path:
            return

        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty():
            print("[WARN] Empty mesh")
            return

        bbox = mesh.get_axis_aligned_bounding_box()
        extent_max = bbox.get_extent().max()
        unit_conversion = 1.0
        if extent_max > 5 and extent_max < 5000.0:
            # Likely in millimeters -> convert to meters
            print(f"[INFO] Converting units from mm to m for: {path.name}")
            unit_conversion = 0.001
            mesh.scale(unit_conversion, center=(0, 0, 0))

        mesh.compute_vertex_normals()
        mesh.translate(-mesh.get_center())
        self.mesh = mesh
        self.next_stage_buttons[Stage.IMPORT_MESH].enabled = True
        self.safe_scene_update(lambda: self._clear_scene())
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("mesh", self.mesh, self.default_material))
        self._reframe()

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
        self.image_res_slider = gui.Slider(gui.Slider.INT)
        self.image_res_slider.set_limits(100, 2000)
        self.image_res_slider.int_value = 1000
        btn_raycast = gui.Button("Raycast")
        btn_raycast.set_on_clicked(self.raycast_mesh)

        btn_reset = gui.Button("Clear Raycast")
        btn_reset.set_on_clicked(self.reset_raycast_stage)
        btn_next = gui.Button("Next: Crop")
        btn_next.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value + 1)))
        self.next_stage_buttons[Stage.RAYCAST] = btn_next
        btn_back = gui.Button("Back: Import Mesh")
        btn_back.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value - 1)))

        v.add_child(gui.Label("Raycasting"))
        v.add_child(gui.Label("Raycast Settings"))
        v.add_child(gui.Label("Camera Distance"))
        v.add_child(self.camera_distance_slider)
        v.add_child(gui.Label("Number of Views"))
        v.add_child(self.num_views_slider)
        v.add_child(gui.Label("Resolution"))
        v.add_child(self.image_res_slider)
        v.add_child(btn_raycast)
        v.add_child(btn_reset)
        v.add_child(btn_next)
        v.add_child(btn_back)
        return v

    def raycast_stage_init(self):
        self.next_stage_buttons[Stage.RAYCAST].enabled = (self.raw_pcd != None)

    def reset_raycast_stage(self):
        self.safe_scene_update(lambda: self._clear_scene())
        self.raw_pcd = None
        self.cropped_pcd = None
        self.down_pcd = None
        self.next_stage_buttons[Stage.RAYCAST].enabled = (self.raw_pcd != None)
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("mesh", self.mesh, self.default_material))

    def raycast_mesh(self):
        if self.mesh is None:
            print("[WARN] No mesh loaded")
            return
        self.camera_distance = self.camera_distance_slider.double_value
        self.num_views = self.num_views_slider.int_value
        self.image_res = self.image_res_slider.int_value

        # Use the largest extent as the characteristic size of the model.
        print("[INFO] Performing view-based ray casting...")
        print("[INFO] Parameters: camera_distance=", self.camera_distance, " num_views=", self.num_views, " image_res=", self.image_res)
        bbox = self.mesh.get_axis_aligned_bounding_box()
        effective_scale = bbox.get_extent().max() if bbox.get_extent().max() > 0 else 1.0

        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        view_dirs = fibonacci_sphere(self.num_views)
        all_points = []
        all_cam_pos = []

        # Raycast from each view
        for v in view_dirs:
            # Camera distance is specified relative to object size; multiply by effective_scale
            cam_pos = v * (self.camera_distance * effective_scale)

            u = np.linspace(-1,1,self.image_res)
            v = np.linspace(-1,1,self.image_res)
            du = (2/self.image_res)
            dv = (2/self.image_res)
            uu, vv = np.meshgrid(u,v)
            uu += np.random.uniform(-du/2, du/2, uu.shape)
            vv += np.random.uniform(-dv/2, dv/2, vv.shape)

            # Camera basis
            forward = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross([0, 1, 0], forward)
            if np.linalg.norm(right) < 1e-6:
                right = np.cross([1, 0, 0], forward)
            right /= np.linalg.norm(right)
            up = np.cross(forward, right)

            # Ray directions
            dirs = forward + uu[..., None] * right + vv[..., None] * up
            dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
            origins = np.repeat(np.repeat(cam_pos[None, None, :], self.image_res, axis=0), self.image_res, axis=1)
            rays = o3d.core.Tensor(np.concatenate([origins.reshape(-1, 3), dirs.reshape(-1, 3)], axis=1), dtype=o3d.core.Dtype.Float32)
            hits = scene.cast_rays(rays)
            hit_mask = hits["t_hit"].isfinite()
            if not hit_mask.any():
                continue

            valid_rays = rays[hit_mask]
            hit_points = (valid_rays[:, :3] + valid_rays[:, 3:] * hits["t_hit"][hit_mask].reshape(o3d.core.SizeVector([-1, 1]))).numpy()
            cam_pos_arr = np.repeat(cam_pos[None, :], len(hit_points), axis=0)
            all_points.append(hit_points)
            all_cam_pos.append(cam_pos_arr)

        if not all_points:
            logging.warning(f"No points generated from mesh")
            return

        # Aggregate
        points = np.vstack(all_points)
        cam_positions = np.vstack(all_cam_pos)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd.remove_non_finite_points()
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius = 0.005,
                max_nn = 60
            )
        )
        pcd=normalize_normals(pcd)

        orient_normals_using_cameras(pcd, cam_positions)
        pcd = center_pointcloud_to_geometric_center(pcd)
        pcd=normalize_normals(pcd)
        validate_normals(pcd)
        print(f"[INFO] Total points sampled: {len(pcd.points)}")
        self.raw_pcd = pcd
        self.cropped_pcd = copy.deepcopy(self.raw_pcd) # for crop stage
        self.safe_scene_update(lambda: self._clear_scene())
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("raw_pcd", self.raw_pcd, self.default_material))
        self.next_stage_buttons[Stage.RAYCAST].enabled = True

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
        self.next_stage_buttons[Stage.CROP] = btn_next
        btn_back = gui.Button("Back: Raycast")
        btn_back.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value - 1)))

        v.add_child(gui.Label("Crop Point Cloud"))
        v.add_child(gui.Label("Controls"))
        v.add_child(self.btn_box_select)
        v.add_child(delete_btn)
        v.add_child(btn_reset)

        v.add_child(btn_next)
        v.add_child(btn_back)
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
        self.cropped_pcd = copy.deepcopy(self.raw_pcd) if self.cropped_pcd == None else self.cropped_pcd
        print("[INFO] Entering crop stage. Use box selection to select points to delete.")
        self.safe_scene_update(lambda: self._clear_scene())
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("crop_pcd", self.cropped_pcd, self.default_material))

    def reset_crop_stage(self):
        self.safe_scene_update(lambda: self._clear_scene())
        self.cropped_pcd = copy.deepcopy(self.raw_pcd)
        self.down_pcd = None
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("cropped", self.cropped_pcd, self.default_material))

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
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("non selected", non_selected_pcd, self.default_material))
        self.next_stage_buttons[Stage.CROP].enabled = True

    def delete_selected_points(self):
        mask = np.ones(len(self.cropped_pcd.points), dtype=bool)
        mask[self.selected_indices] = False
        self.cropped_pcd = mask_point_cloud(self.cropped_pcd, mask)
        print(f"[INFO] Deleted selected {len(self.selected_indices)} points. Remaining points: {len(self.cropped_pcd.points)}")
        self.selected_indices = []
        self.safe_scene_update(lambda: self._clear_scene())
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("cropped", self.cropped_pcd, self.default_material))

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
        self.voxel_slider = gui.Slider(gui.Slider.DOUBLE)
        self.voxel_slider.set_limits(0.01, 10)
        self.voxel_slider.double_value = self.voxel_size * 1000.0  # in mm
        self.feature_ratio_slider = gui.Slider(gui.Slider.DOUBLE)
        self.feature_ratio_slider.set_limits(0.01, 1.0)
        self.feature_ratio_slider.double_value = self.feature_ratio
        self.coarse_factor_slider = gui.Slider(gui.Slider.DOUBLE)
        self.coarse_factor_slider.set_limits(1.0, 10.0)
        self.coarse_factor_slider.double_value = self.coarse_factor
        self.curvature_k_neighbors_slider = gui.Slider(gui.Slider.INT)
        self.curvature_k_neighbors_slider.set_limits(5, 100)
        self.curvature_k_neighbors_slider.int_value = self.curvature_k_neighbors

        btn_downsample = gui.Button("Downsample")
        btn_downsample.set_on_clicked(self.downsample)

        btn_reset = gui.Button("Restart Downsample")
        btn_reset.set_on_clicked(self.reset_downsample_stage)
        btn_next = gui.Button("Next: Save")
        btn_next.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value + 1)))
        self.next_stage_buttons[Stage.DOWNSAMPLE] = btn_next
        btn_back = gui.Button("Back: Crop")
        btn_back.set_on_clicked(lambda: self.set_stage(Stage(self.stage.value - 1)))
        v.add_child(gui.Label("Downsampling"))
        v.add_child(gui.Label("Downsample Settings"))
        v.add_child(self.adaptive_checkbox)

        v.add_child(gui.Label("Voxel Size"))
        v.add_child(self.voxel_slider)
        v.add_child(gui.Label("Feature Ratio"))
        v.add_child(self.feature_ratio_slider)
        v.add_child(gui.Label("Curvature K Neighbours"))
        v.add_child(self.curvature_k_neighbors_slider)

        v.add_child(btn_downsample)
        v.add_child(btn_reset)
        v.add_child(btn_next)
        v.add_child(btn_back)
        print("loaded downsample panel")

        return v
        
    def downsample_stage_init(self):        
        if self.down_pcd != None:
            self.safe_scene_update(lambda: self._clear_scene())
            self.safe_scene_update(lambda: self.scene.scene.add_geometry("down_pcd", self.down_pcd, self.default_material))
        self.next_stage_buttons[Stage.DOWNSAMPLE].enabled = (self.down_pcd != None)

    def reset_downsample_stage(self):
        self.next_stage_buttons[Stage.DOWNSAMPLE].enabled = (self.down_pcd != None)
        self.safe_scene_update(lambda: self._clear_scene())
        self.down_pcd = None
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("cropped_pcd", self.cropped_pcd, self.default_material))

    def downsample(self):
        self.use_adaptive = self.adaptive_checkbox.checked
        self.voxel_size = self.voxel_slider.double_value / 1000.0  # convert mm to m
        self.feature_ratio = self.feature_ratio_slider.double_value
        self.coarse_factor = self.coarse_factor_slider.double_value
        self.curvature_k_neighbors = self.curvature_k_neighbors_slider.int_value 

        if self.cropped_pcd is None:
            return

        if self.use_adaptive:
            self.adaptive_voxel_downsample()
        else:
            self.uniform_voxel_downsample()
        print(f"[INFO] Downsampled from to {len(self.cropped_pcd.points)} {len(self.down_pcd.points)} points")
        self.next_stage_buttons[Stage.DOWNSAMPLE].enabled = (self.down_pcd != None)
        
        self.safe_scene_update(lambda: self._clear_scene())
        self.safe_scene_update(lambda: self.scene.scene.add_geometry("down_pcd", self.down_pcd, self.default_material))
    
    def adaptive_voxel_downsample(self):
        print(f"[INFO] Performing adaptive voxel downsampling on raycasted point cloud {len(self.cropped_pcd.points)} points.")
        print(f"[INFO] Parameters: voxel_size={self.voxel_size}, curvature_k_neighbors={self.curvature_k_neighbors}, feature_ratio={self.feature_ratio}, coarse_factor={self.coarse_factor}")
        
        variation = compute_curvature(self.cropped_pcd, self.curvature_k_neighbors)
        threshold = np.percentile(variation, 100 * (1 - self.feature_ratio))
        feature_mask = variation >= threshold

        pcd_feature = mask_point_cloud(self.cropped_pcd, feature_mask)
        pcd_flat = mask_point_cloud(self.cropped_pcd, ~feature_mask)
        pcd_feature = pcd_feature.voxel_down_sample(self.voxel_size)
        pcd_flat = pcd_flat.voxel_down_sample(self.voxel_size * self.coarse_factor)

        self.down_pcd = normalize_normals(pcd_feature + pcd_flat)

    def uniform_voxel_downsample(self):
        print("[INFO] Performing uniform voxel downsampling...")
        print(f"[INFO] Parameters: voxel_size={self.voxel_size}")
        self.down_pcd = normalize_normals(self.cropped_pcd.voxel_down_sample(self.voxel_size))


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
        btn_restart.set_on_clicked(lambda: self._restart)
        
        v.add_child(gui.Label("Export PCL"))
        v.add_child(btn_save)
        v.add_child(btn_back)
        v.add_child(btn_restart)
        return v
        
    def save_stage_init(self):
        print("save stage init")
        pass

    def save_pcd(self):
        if self.down_pcd is None:
            print("[WARN] No pointcloud to save")
            return
        path = self._save_ply_dialog()
        if not path:
            return
        try:
            write_mechmind_ply(self.down_pcd, str(path))
            print(f"[INFO] Point cloud saved to {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save PLY: {e}")
            return

    def _restart(self):
        self.mesh = None
        self.raw_pcd = None
        self.cropped_pcd = None
        self.down_pcd = None
        self.set_stage(Stage.IMPORT_MESH)

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
