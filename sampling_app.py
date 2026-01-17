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
        self.crop_selection_overlay = None
        # self.overlay_material = rendering.Material()
        # self.overlay_material.point_size = 5.0
        # self.overlay_material.shader = "defaultUnlit"

        # === State variables ===
        self.camera_distance = 1.5
        self.num_views = 30
        self.image_res = 1000
        self.voxel_size = 0.001
        self.use_adaptive = True
        self.feature_ratio = 0.1
        self.coarse_factor = 3.0
        self.curvature_k_neighbors = 5

        self.tool_mode = ToolMode.NONE
        self.is_dragging = False
        self.drag_start = None
        self.drag_end = None
        self.selected_indices = []

        # === Scene widget ===
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.1, 0.1, 0.1, 1.0])
        self.scene.set_on_mouse(self._on_mouse_event)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)
        self.window.add_child(self.scene)
        self._build_parameter_panel()
        self._setup_keybindings()
        self._update_title()

    # ===============================
    # Control Panel
    # ===============================

    def _build_parameter_panel(self):
        em = self.window.theme.font_size
        self.panel = gui.Vert(0.25 * em, gui.Margins(em, em, em, em))

        self._load_mesh_panel()
        self._raycast_panel()
        self._crop_panel()
        self._downsample_panel()
        self._save_panel()
        self.window.add_child(self.panel)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 300
        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

    def _load_mesh_panel(self):
        self.panel.add_child(gui.Label("Import Mesh"))
        import_btn = gui.Button("Import")
        import_btn.set_on_clicked(self.import_mesh)
        self.panel.add_child(import_btn)

    def _raycast_panel(self):
        self.panel.add_child(gui.Label("Raycasting"))

        self.camera_distance_slider = gui.Slider(gui.Slider.DOUBLE)
        self.camera_distance_slider.set_limits(0.5, 3.0)
        self.num_views_slider = gui.Slider(gui.Slider.INT)
        self.num_views_slider.set_limits(4, 100)
        self.num_views_slider.int_value = 20
        self.panel.add_child(gui.Label("Number of Views"))
        self.panel.add_child(self.num_views_slider)

        self.image_res_slider = gui.Slider(gui.Slider.INT)
        self.image_res_slider.set_limits(100, 2000)
        self.image_res_slider.int_value = 1000
        self.panel.add_child(gui.Label("Image Resolution"))
        self.panel.add_child(self.image_res_slider)

        apply_btn = gui.Button("Apply")
        apply_btn.set_on_clicked(self._apply_raycast_parameters)
        self.panel.add_child(apply_btn)
        raycast_btn = gui.Button("Raycast")
        raycast_btn.set_on_clicked(self.raycast_mesh)
        self.panel.add_child(raycast_btn)

    def _crop_panel(self):
        self.panel.add_child(gui.Label("Crop Point Cloud"))

        # self.btn_box_select = gui.ToggleSwitch("Box Select")
        # self.btn_box_select.set_on_clicked(self._enable_box_selection)
        # self.panel.add_child(self.btn_box_select)
        self.btn_box_select = gui.Checkbox("Box Select")
        self.btn_box_select.checked = False
        self.btn_box_select.set_on_checked(self._enable_box_selection)
        self.panel.add_child(self.btn_box_select)

        delete_btn = gui.Button("Delete Selected Points")
        delete_btn.set_on_clicked(self.delete_selected_points)
        self.panel.add_child(delete_btn)

        reset_btn = gui.Button("Reset Crop")
        reset_btn.set_on_clicked(self.reset_crop)
        self.panel.add_child(reset_btn)

    def _downsample_panel(self):
        self.panel.add_child(gui.Label("Downsampling"))

        self.adaptive_checkbox = gui.Checkbox("Adaptive sampling")
        self.adaptive_checkbox.checked = self.use_adaptive
        self.panel.add_child(self.adaptive_checkbox)

        self.voxel_slider = gui.Slider(gui.Slider.DOUBLE)
        self.voxel_slider.set_limits(0.01, 10)
        self.voxel_slider.double_value = self.voxel_size * 1000.0  # in mm
        self.panel.add_child(gui.Label("Voxel Size (mm)"))
        self.panel.add_child(self.voxel_slider)

        self.feature_ratio_slider = gui.Slider(gui.Slider.DOUBLE)
        self.feature_ratio_slider.set_limits(0.01, 1.0)
        self.feature_ratio_slider.double_value = self.feature_ratio
        self.panel.add_child(gui.Label("Feature Ratio"))
        self.panel.add_child(self.feature_ratio_slider)

        self.coarse_factor_slider = gui.Slider(gui.Slider.DOUBLE)
        self.coarse_factor_slider.set_limits(1.0, 10.0)
        self.coarse_factor_slider.double_value = self.coarse_factor
        self.panel.add_child(gui.Label("Coarse Factor"))
        self.panel.add_child(self.coarse_factor_slider)

        self.curvature_k_neighbors_slider = gui.Slider(gui.Slider.INT)
        self.curvature_k_neighbors_slider.set_limits(5, 100)
        self.curvature_k_neighbors_slider.int_value = self.curvature_k_neighbors
        self.panel.add_child(gui.Label("Curvature K Neighbors"))
        self.panel.add_child(self.curvature_k_neighbors_slider)

        apply_btn = gui.Button("Apply")
        apply_btn.set_on_clicked(self._apply_downsample_parameters)
        self.panel.add_child(apply_btn)
        downsample_btn = gui.Button("Downsample")
        downsample_btn.set_on_clicked(self.downsample)
        self.panel.add_child(downsample_btn)

    def _save_panel(self):
        self.panel.add_child(gui.Label("Save"))
        save_btn = gui.Button("Export Point Cloud")
        save_btn.set_on_clicked(self.save_pcd)
        self.panel.add_child(save_btn)

    # ===============================
    # Button callbacks
    # ===============================

    def _apply_raycast_parameters(self):
        self.num_views = self.num_views_slider.int_value
        self.image_res = self.image_res_slider.int_value

        print("[INFO] Updated raycast parameters:")
        print(f"[INFO] num_views: {self.num_views}, image_res: {self.image_res}")

    def _apply_downsample_parameters(self):
        self.use_adaptive = self.adaptive_checkbox.checked
        self.voxel_size = self.voxel_slider.double_value / 1000.0  # convert mm to m
        self.feature_ratio = self.feature_ratio_slider.double_value
        self.coarse_factor = self.coarse_factor_slider.double_value
        self.curvature_k_neighbors = self.curvature_k_neighbors_slider.int_value 

        print("[INFO] Updated downsample parameters:")
        print(f"[INFO] Parameters: adaptive={self.use_adaptive}, voxel_size={self.voxel_size}, feature_ratio={self.feature_ratio}, coarse_factor={self.coarse_factor}, curvature_k_neighbors={self.curvature_k_neighbors}")

    def _next_stage(self):
        if self.stage == Stage.IMPORT_MESH:
            self.stage = Stage.RAYCAST
        elif self.stage == Stage.RAYCAST:
            self.stage = Stage.CROP
        elif self.stage == Stage.CROP:
            self.stage = Stage.DOWNSAMPLE
        elif self.stage == Stage.DOWNSAMPLE:
            self.stage = Stage.SAVE
        elif self.stage == Stage.SAVE:
            self.stage = Stage.IMPORT_MESH

        self._update_title()

    def _previous_stage(self):
        if self.stage == Stage.SAVE:
            self.stage = Stage.DOWNSAMPLE
        elif self.stage == Stage.DOWNSAMPLE:
            self.stage = Stage.CROP
        elif self.stage == Stage.CROP:
            self.stage = Stage.RAYCAST
        elif self.stage == Stage.RAYCAST:
            self.stage = Stage.IMPORT_MESH
        elif self.stage == Stage.IMPORT_MESH:
            self.stage = Stage.SAVE

        self._update_title()
    
    def _reset(self):
        self.mesh = None
        self.raw_pcd = None
        self.down_pcd = None
        self.stage = Stage.IMPORT_MESH
        self._clear_scene()
        self._update_title()

    # ===============================
    # UI helpers
    # ===============================
    def _update_title(self):
        self.window.title = f"Mesh Sampling Pipeline | Stage: {self.stage.name}"

    def _clear_scene(self):
        self.scene.scene.clear_geometry()

    def _show_geometry(self, geom, name="geom"):
        self._clear_scene()
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self.scene.scene.add_geometry(name, geom, mat)
        # self.scene.setup_camera(60, geom.get_axis_aligned_bounding_box(), geom.get_center())
        bbox = geom.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        # extent = np.linalg.norm(bbox.get_extent())

        self.scene.setup_camera(20.0, bbox, center)

        # Manually expand clipping planes
    def safe_scene_update(self, fn):
        gui.Application.instance.post_to_main_thread(self.window, fn)

    def _enable_box_selection(self):
        print("[INFO] Box selection enabled for cropping.")
        self.tool_mode = ToolMode.BOX_SELECT
        if self.btn_box_select.checked:
            print("[INFO] Box selection enabled for cropping.")
            self.tool_mode = ToolMode.BOX_SELECT
        else:
            print("[INFO] Box selection disabled.")
            self.tool_mode = ToolMode.NONE
            self._clear_crop_overlay()

    def exit_crop_stage(self):
        self.tool_mode = ToolMode.NONE
        self._clear_crop_overlay()

    def _show_overlay(self, overlay):
        self._clear_crop_overlay()
        self.crop_selection_overlay = overlay
        self.scene.scene.add_geometry("crop_overlay", overlay, self.overlay_material)

    def _clear_crop_overlay(self):
        if self.crop_selection_overlay:
            self.scene.scene.remove_geometry("crop_overlay")
            self.crop_selection_overlay = None

    # ===============================
    # File dialogs
    # ===============================
    def _open_stl_dialog(self):
        Tk().withdraw()
        path = filedialog.askopenfilename(
            filetypes=[("STL files", "*.stl *.STL")]
        )
        return Path(path) if path else None

    def _save_ply_dialog(self):
        Tk().withdraw()
        path = filedialog.asksaveasfilename(
            defaultextension=".ply",
            filetypes=[("PLY files", "*.ply")]
        )
        return Path(path) if path else None

    # ===============================
    # Stage logic
    # ===============================
    def import_mesh(self):
        self.stage = Stage.IMPORT_MESH
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

        self.safe_scene_update(lambda: self._show_geometry(self.mesh, "mesh"))
        self.safe_scene_update(lambda: self._update_title())

    def raycast_mesh(self):
        if self.mesh is None:
            print("[WARN] No mesh loaded")
            return
        self.stage = Stage.RAYCAST
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
        self.safe_scene_update(lambda: self._show_geometry(self.raw_pcd, "raw_pcd"))
        self.safe_scene_update(lambda: self._update_title())
        
    def crop_stage(self):
        self.stage = Stage.CROP
        print("[INFO] Entering crop stage. Use box selection to select points to delete.")
        self._update_title()
        self._show_geometry(self.raycast_pcd_working, "crop_pcd")
        self._enable_box_selection()

    def downsample(self):
        if self.raw_pcd is None:
            return

        if self.use_adaptive:
            # === your adaptive downsample function ===
            self.adaptive_voxel_downsample()
            # self.down_pcd = self.raw_pcd.voxel_down_sample(self.voxel_size)
        else:
            # self.down_pcd = self.raw_pcd.voxel_down_sample(self.voxel_size)
            # self.safe_scene_update( self.uniform_voxel_downsample())
            self.uniform_voxel_downsample()
        print(f"[INFO] Downsampled from to {len(self.raw_pcd.points)} {len(self.down_pcd.points)} points")
        self.stage = Stage.DOWNSAMPLE
        self.safe_scene_update(lambda: self._show_geometry(self.down_pcd, "down_pcd"))
        self.safe_scene_update(lambda: self._update_title())

    def save_pcd(self):
        self.stage = Stage.SAVE
        if self.down_pcd is None:
            print("[WARN] No pointcloud to save")
            return
        path = self._save_ply_dialog()
        if not path:
            return
        try:
            write_mechmind_ply(self.down_pcd, str(path))
            print(f"[INFO] MechMind PLY saved to {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save PLY: {e}")
            return

        self.safe_scene_update(self._update_title)

    # ===============================
    # Keybindings
    # ===============================
    def _setup_keybindings(self):
        self.window.set_on_key(self._on_key)

    def _on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return False

        key = event.key

        # --- Global ---
        if key == gui.KeyName.Q:
            gui.Application.instance.quit()
            return True

        # --- Stage specific ---
        if self.stage == Stage.IMPORT_MESH:
            if key == gui.KeyName.O:
                self.import_mesh()
            elif key == gui.KeyName.N:
                self.stage = Stage.RAYCAST
                self._update_title()

        elif self.stage == Stage.RAYCAST:
            if key == gui.KeyName.S:
                self.raycast_mesh()
            elif key == gui.KeyName.B:
                self.stage = Stage.IMPORT_MESH
                self._update_title()
            elif key == gui.KeyName.N:
                self.stage = Stage.CROP
                self.crop_stage()
                self._update_title()

        elif self.stage == Stage.CROP:
            if key == gui.KeyName.S:
                self.save_pcd()
            elif key == gui.KeyName.B:
                self.stage = Stage.RAYCAST
                self._update_title()
            elif key == gui.KeyName.N:
                self.stage = Stage.DOWNSAMPLE
                self._update_title()
            elif key == gui.KeyName.D:
                self.delete_selected_points()

        elif self.stage == Stage.DOWNSAMPLE:
            if key == gui.KeyName.S:
                self.downsample()
            elif key == gui.KeyName.T:
                self.use_adaptive = not self.use_adaptive
                print(f"Adaptive downsampling: {self.use_adaptive}")
            elif key == gui.KeyName.B:
                self.stage = Stage.RAYCAST
                self._update_title()
            elif key == gui.KeyName.N:
                self.stage = Stage.SAVE
                self._update_title()

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

    def _on_mouse_event(self, event):
        if self.stage != Stage.CROP:
            return o3d.visualization.gui.Widget.EventCallbackResult.IGNORED

        if self.tool_mode != ToolMode.BOX_SELECT:
            return o3d.visualization.gui.Widget.EventCallbackResult.IGNORED

        print("[INFO] Box selection mouse event")
        if event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_DOWN:
            if event.is_left_button:
                self.is_dragging = True
                self.drag_start = (event.x, event.y)
                self.drag_end = self.drag_start
                return o3d.visualization.gui.Widget.EventCallbackResult.HANDLED

        elif event.type == o3d.visualization.gui.MouseEvent.Type.DRAG:
            if self.is_dragging:
                self.drag_end = (event.x, event.y)
                self._update_selection_preview()
                return o3d.visualization.gui.Widget.EventCallbackResult.HANDLED

        elif event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_UP:
            if self.is_dragging:
                self.is_dragging = False
                self._finalize_selection()
                return o3d.visualization.gui.Widget.EventCallbackResult.HANDLED

        return o3d.visualization.gui.Widget.EventCallbackResult.IGNORED

    def _select_points_screen_space(self):
        cam = self.scene_widget.scene.camera
        pts = np.asarray(self.raycast_pcd_working.points)

        selected = []
        for i, p in enumerate(pts):
            screen = cam.project(p)
            if screen is None:
                continue

            x, y = screen[0], screen[1]
            if self._inside_rect(x, y):
                selected.append(i)

        self.selected_indices = selected


    def adaptive_voxel_downsample(self):
        print(f"[INFO] Performing adaptive voxel downsampling on raycasted point cloud {len(self.raw_pcd.points)} points.")
        print(f"[INFO] Parameters: voxel_size={self.voxel_size}, curvature_k_neighbors={self.curvature_k_neighbors}, feature_ratio={self.feature_ratio}, coarse_factor={self.coarse_factor}")
        
        variation = compute_curvature(self.raw_pcd, self.curvature_k_neighbors)
        threshold = np.percentile(variation, 100 * (1 - self.feature_ratio))
        feature_mask = variation >= threshold
        flat_mask = ~feature_mask

        pts = np.asarray(self.raw_pcd.points)
        nrm = np.asarray(self.raw_pcd.normals)

        pcd_feature = o3d.geometry.PointCloud()
        pcd_feature.points = o3d.utility.Vector3dVector(pts[feature_mask])
        pcd_feature.normals = o3d.utility.Vector3dVector(nrm[feature_mask])
        pcd_flat = o3d.geometry.PointCloud()
        pcd_flat.points = o3d.utility.Vector3dVector(pts[flat_mask])
        pcd_flat.normals = o3d.utility.Vector3dVector(nrm[flat_mask])

        pcd_feature = pcd_feature.voxel_down_sample(self.voxel_size)
        pcd_feature = normalize_normals(pcd_feature)
        pcd_flat = pcd_flat.voxel_down_sample(self.voxel_size * self.coarse_factor)
        pcd_flat = normalize_normals(pcd_flat)

        self.down_pcd = pcd_feature + pcd_flat

    def uniform_voxel_downsample(self):
        print("[INFO] Performing uniform voxel downsampling...")
        print(f"[INFO] Parameters: voxel_size={self.voxel_size}")
        self.down_pcd = normalize_normals(self.raw_pcd.voxel_down_sample(self.voxel_size))

    def delete_selected_points(self):
        mask = np.ones(len(self.raycast_pcd_working.points), dtype=bool)
        mask[self.selected_indices] = False

        pts = np.asarray(self.raycast_pcd_working.points)[mask]
        self.raycast_pcd_working.points = o3d.utility.Vector3dVector(pts)

        self.selected_indices = []
        self._clear_crop_overlay()

        self.safe_scene_update(lambda: self._show_geometry(self.raycast_pcd_working, "crop_pcd"))

    def _on_crop_selection(self, selection):
        if self.stage != Stage.CROP:
            return

        self.selected_indices = selection.indices

        if not self.selected_indices:
            self._clear_crop_overlay()
            return

        pts = np.asarray(self.raycast_pcd_working.points)[self.selected_indices]

        overlay = o3d.geometry.PointCloud()
        overlay.points = o3d.utility.Vector3dVector(pts)

        colors = np.zeros((len(pts), 3))
        colors[:, 0] = 1.0  # red
        overlay.colors = o3d.utility.Vector3dVector(colors)

        self._show_overlay(overlay)

    def reset_crop(self):
        self.raycast_pcd_working = self.raw_pcd.clone()
        self.safe_scene_update(lambda: self._show_geometry(self.raycast_pcd_working, "crop_pcd"))
        

# ===============================
# Entry point
# ===============================
if __name__ == "__main__":
    gui.Application.instance.initialize()
    app = MeshSamplingApp()
    gui.Application.instance.run()
