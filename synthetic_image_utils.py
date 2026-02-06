import open3d as o3d
import numpy as np
import os
from utilities import fibonacci_sphere

def look_at(camera_pos, target=np.array([0, 0, 0]), up=np.array([0, 0, 1])):
    forward = (target - camera_pos)
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    true_up = np.cross(forward, right)

    R = np.vstack([right, true_up, forward]).T
    t = -R @ camera_pos

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return extrinsic

def project_extent(points, cam_pos, cam_dir, up):
    right = np.cross(cam_dir, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, cam_dir)

    rel = points - cam_pos
    x = rel @ right
    y = rel @ up
    return np.max(np.abs(x)), np.max(np.abs(y))

def fit_mesh_in_view_obb(mesh, cam_pos, ctr, margin=1.2):
    obb = mesh.get_oriented_bounding_box()
    corners = np.asarray(obb.get_box_points())

    cam_dir = (np.array([0, 0, 0]) - cam_pos)
    cam_dir /= np.linalg.norm(cam_dir)

    up = np.array([0, 0, 1])
    if abs(np.dot(cam_dir, up)) > 0.95:
        up = np.array([0, 1, 0])

    max_x, max_y = project_extent(corners, cam_pos, cam_dir, up)

    dist = np.linalg.norm(cam_pos)

    # Empirical zoom conversion (legacy Open3D quirk)
    zoom = margin * max(max_x, max_y) / dist
    # zoom = np.clip(zoom, 0.1, 0.7)

    ctr.set_zoom(zoom)

if __name__=="__main__":
    file_path = "400_97703GI400.stl"
    mesh = o3d.io.read_triangle_mesh(file_path)
    if mesh.is_empty():
        print("[WARN] Empty mesh")
        exit()

    bbox = mesh.get_minimal_oriented_bounding_box()
    bbox_corners = np.asarray(bbox.get_box_points())
    extent = bbox.extent  # Get the extent (dimensions) of the OBB
    extent_min = extent.min()
    extent_max = extent.max()
    unit_conversion = 1.0
    if extent_max > 5 and extent_max < 5000.0:
        # Likely in millimeters -> convert to meters
        print(f"[INFO] Converting units from mm to m for: {file_path}")
        unit_conversion = 0.001
        mesh.scale(unit_conversion, center=(0, 0, 0))

    mesh.compute_vertex_normals()
    mesh.translate(-bbox.get_center())
    mesh.paint_uniform_color([1,1,1])

    # Extract base name from file path
    file_base_name = os.path.splitext(os.path.basename(file_path))[0]
    num_targets = 100
    view_sphere = fibonacci_sphere(num_targets)
    out_dir = os.path.join("snapshots", file_base_name)
    os.makedirs(out_dir, exist_ok=True)

    # Renderer
    width, height = 512, 512

    # Calculate appropriate radius based on minimal oriented bounding box
    # Get the diagonal of the bounding box
    bbox_diagonal = np.linalg.norm(extent)

    # Set radius to ensure object fits in view
    # The factor depends on the FOV - typically 60 degrees for Open3D
    # Adjust this multiplier if needed (larger = camera farther away)
    radius = bbox_diagonal * 2.0

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)

    ctr = vis.get_view_control()
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.mesh_show_back_face = True
    # opt.light_on = False   # disables lighting for pure color contrast

    for i, v in enumerate(view_sphere):
        cam_pos = np.asarray(v) * radius

        ctr.set_lookat([0, 0, 0])
        ctr.set_front(-cam_pos / np.linalg.norm(cam_pos))
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.7)  # Slightly zoom out for some padding (< 1.0 zooms out)

        vis.poll_events()
        vis.update_renderer()

        img = vis.capture_screen_float_buffer(False)
        img = (np.asarray(img) * 255).astype(np.uint8)

        o3d.io.write_image(os.path.join(out_dir, f"view_{i:03d}.png"), o3d.geometry.Image(img))

    vis.destroy_window()