import numpy as np
import copy
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from enums import Stage, ToolMode
from stages.stage_base import BaseStage
from geometry.geom_utils import mask_point_cloud

class CropStage(BaseStage):
    def __init__(self, app):
        self.name = Stage.CROP.name
        super().__init__(app)
        self.tool_mode = ToolMode.NONE
        self.is_dragging = False
        self.drag_start = None
        self.drag_end = None
        self.selected_pcd = None
        self.non_selected_pcd = None
        self.selected_indices = []
        self.lines = [[0, 1],[1, 2],[2, 3],[3, 0]]
        self.line_set = o3d.geometry.LineSet()
        self.rect_material = rendering.MaterialRecord()
        self.rect_material.shader = "unlitLine"
        self.rect_material.line_width = 3.0
        self.rect_material.base_color = [1.0, 1.0, 0.0, 1.0]
        
    def build_panel(self):
        if self.app.headless:
            return
        v = gui.Vert(4)
        
        self.btn_box_select = self.register_widget(gui.Button("Box Select"))
        self.btn_box_select.toggleable = True
        self.btn_box_select.set_on_clicked(self._enable_box_selection)
        self.delete_btn = self.register_widget(gui.Button("Delete Selected Points"), lambda: len(self.selected_indices)>0)
        self.delete_btn.set_on_clicked(self.start)
        self.btn_reset = self.register_widget(gui.Button("Reset Crop"), lambda: len(self.app.cropped_pcd.points)<len(self.app.raw_pcd.points))
        self.btn_reset.set_on_clicked(self.reset)

        self.btn_next = self.register_widget(gui.Button("Next: Downsample"), lambda: self.app.cropped_pcd is not None)
        self.btn_next.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value + 1)))
        self.btn_back = self.register_widget(gui.Button("Back: Raycast"))
        self.btn_back.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value - 1)))

        v.add_child(gui.Label("Crop Point Cloud"))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label("Controls"))
        v.add_child(self.btn_box_select)
        v.add_child(self.delete_btn)
        v.add_child(self.btn_reset)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(self.btn_back)
        v.add_child(self.btn_next)
        print("loaded crop panel")

        return v

    def _refresh_ui(self):
        if self.app.headless:
            return
        self._clear_selection_rectangle()
        self.app.main_thread(lambda: self.app._clear_scene())
        if self.selected_pcd is None:
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("crop_pcd", self.app.cropped_pcd, self.app.default_point_material))
        else:
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("selected", self.selected_pcd, self.app.overlay_material))
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("non selected", self.non_selected_pcd, self.app.default_point_material))
        self.app.scene.force_redraw()
        self.enable_widgets()

    def reset(self):
        self.app.down_pcd = None
        self.app.cropped_pcd = copy.deepcopy(self.app.raw_pcd)
        self.app.output_pcd_path = None

        self._refresh_ui()

    def worker(self):
        """Deletes selected points from pointcloud"""
        mask = np.ones(len(self.app.cropped_pcd.points), dtype=bool)
        mask[self.selected_indices] = False
        self.app.cropped_pcd = mask_point_cloud(self.app.cropped_pcd, mask)
        self.selected_indices = []
        self.selected_pcd = None
        self.non_selected_pcd = None
        print(f"[INFO] Deleted selected {len(self.selected_indices)} points. Remaining points: {len(self.app.cropped_pcd.points)}")

    def _enable_box_selection(self):
        if self.btn_box_select.is_on:
            print("[INFO] Box selection enabled for cropping.")
            self.tool_mode = ToolMode.BOX_SELECT
        else:
            print("[INFO] Box selection disabled.")
            self.tool_mode = ToolMode.NONE
            self._clear_selection_rectangle()
    
    def project_world_to_screen(self, points):
        """Project 3D world points to 2D screen coordinates"""
        cam = self.app.scene.scene.camera
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
        x = (ndc[:, 0] * 0.5 + 0.5) * self.app.scene.frame.width
        y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self.app.scene.frame.height

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
        
        cam = self.app.scene.scene.camera
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
        width = self.app.scene.frame.width
        height = self.app.scene.frame.height
        
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
        
        if not hasattr(self, 'cropped_pcd') or self.app.cropped_pcd is None:
            return default_depth
        
        points = np.asarray(self.app.cropped_pcd.points)
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
        valid_idx, screen_pts = self.project_world_to_screen(np.asarray(self.app.cropped_pcd.points))
        
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
        selection_mask = np.ones(len(self.app.cropped_pcd.points), dtype=bool)
        selection_mask[selected] = False
        
        self.selected_pcd = mask_point_cloud(self.app.cropped_pcd, ~selection_mask)
        self.non_selected_pcd = mask_point_cloud(self.app.cropped_pcd, selection_mask)
        self._refresh_ui()

    def _draw_selection_rectangle(self):
        """Draw a live rectangle overlay during box selection"""
        if self.drag_start is None or self.drag_end is None:
            return
        if self.app.scene.scene.has_geometry("selection_rect"):
            self.app.scene.scene.remove_geometry("selection_rect")

        self.corners_world = self._get_selection_frustum_corners()
        if self.corners_world is None or len(self.corners_world) != 4:
            return
        self.line_set.points = o3d.utility.Vector3dVector(self.corners_world)
        self.line_set.lines = o3d.utility.Vector2iVector(self.lines)
        self.line_set.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0] for _ in range(len(self.lines))])
        for corner in self.corners_world:
            if not np.all(np.isfinite(corner)):
                return
        
        try:
            self.app.scene.scene.add_geometry("selection_rect", self.line_set, self.rect_material)
            self.app.scene.force_redraw()
        except Exception as e:
            print(f"[WARNING] Could not draw selection rectangle: {e}")

    def _clear_selection_rectangle(self):
        """Remove the selection rectangle from the scene"""
        if self.app.scene.scene.has_geometry("selection_rect"):
            self.app.scene.scene.remove_geometry("selection_rect")
            self.app.scene.force_redraw()
    
    def _on_mouse_event(self, event):
        if self.tool_mode != ToolMode.BOX_SELECT:
            return gui.Widget.EventCallbackResult.IGNORED
        
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            if event.buttons == 1:
                self.is_dragging = True
                self.drag_start = (event.x, event.y)
                self.drag_end = self.drag_start
                return gui.Widget.EventCallbackResult.HANDLED

        elif event.type == gui.MouseEvent.Type.DRAG:
            if self.is_dragging:
                self.drag_end = (event.x, event.y)
                self._draw_selection_rectangle()
                self.app.scene.set_view_controls(gui.SceneWidget.Controls.PICK_POINTS)
                return gui.Widget.EventCallbackResult.HANDLED

        elif event.type == gui.MouseEvent.Type.BUTTON_UP:
            if self.is_dragging and event.buttons == 1:
                self.is_dragging = False
                self.drag_end = (event.x, event.y)
                self._clear_selection_rectangle()
                self._select_points_screen_space()
                self.drag_start = None
                self.drag_end = None
                self.app.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED