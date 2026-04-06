import numpy as np
import open3d as o3d

def visualize_rays(origins, hits, color=(0.6, 0.8, 1.0)):
    points = []
    lines = []
    colors = []

    for i, (o, h) in enumerate(zip(origins, hits)):
        idx = len(points)
        points.extend([o, h])
        lines.append([idx, idx+1])
        colors.append(color)

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls
def visualize_projector_rays(surface_pts, proj_pos, color=(0.2, 1.0, 0.2)):
    points = []
    lines = []
    colors = []

    for p in surface_pts:
        idx = len(points)
        points.extend([p, proj_pos])
        lines.append([idx, idx+1])
        colors.append(color)

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def create_frustum_lines(pos, look_at, fov_deg, aspect=1.0, scale=0.2, color=(1,0,0)):
    forward = look_at - pos
    forward /= np.linalg.norm(forward)

    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    fov = np.deg2rad(fov_deg)
    h = np.tan(fov / 2) * scale
    w = h * aspect

    center = pos + forward * scale
    corners = [
        center + right*w + up*h,
        center - right*w + up*h,
        center - right*w - up*h,
        center + right*w - up*h
    ]

    points = [pos] + corners
    lines = [ [0,1],[0,2],[0,3],[0,4], [1,2],[2,3],[3,4],[4,1]]

    colors = [color] * len(lines)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def visualize_scene(meshes,cam_pos,look_at,proj_pos,cam_origins,cam_hits,proj_visible_pts,fov=60):
    geometries = []

    # Meshes
    for m in meshes:
        geometries.append(m)

    # Camera frame + frustum
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    cam_frame.translate(cam_pos)
    geometries.append(cam_frame)

    cam_frustum = create_frustum_lines(cam_pos, look_at, fov, color=(1,0,0))
    geometries.append(cam_frustum)

    # Projector frame + frustum
    proj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    proj_frame.translate(proj_pos)
    geometries.append(proj_frame)

    proj_frustum = create_frustum_lines(proj_pos, look_at, fov, color=(0,1,0))
    geometries.append(proj_frustum)

    # Camera rays
    cam_ray_lines = visualize_rays(cam_origins, cam_hits)
    geometries.append(cam_ray_lines)

    # Projector rays (only visible points)
    proj_ray_lines = visualize_projector_rays(proj_visible_pts, proj_pos)
    geometries.append(proj_ray_lines)

    # Hit points
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cam_hits))
    pcd.paint_uniform_color([0.2, 0.2, 1.0])
    geometries.append(pcd)

    o3d.visualization.draw_geometries(geometries)

def make_camera_frustum(cam_pos, look_at, fov_deg=60, aspect=1.0, depth=0.5):
    forward = look_at - cam_pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, [0,0,1])
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, [0,1,0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    h = np.tan(np.deg2rad(fov_deg/2)) * depth
    w = h * aspect

    center = cam_pos + forward * depth
    corners = [
        center + up*h + right*w,
        center + up*h - right*w,
        center - up*h - right*w,
        center - up*h + right*w,
    ]

    points = [cam_pos] + corners
    lines = [
        [0,1],[0,2],[0,3],[0,4],
        [1,2],[2,3],[3,4],[4,1]
    ]

    frustum = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector2iVector(lines)
    )
    frustum.paint_uniform_color([1,0,0])
    return frustum

def make_grid(center, normal, size=1.0, step=0.1):
    normal = normal / np.linalg.norm(normal)

    # find two orthogonal axes on plane
    tmp = np.array([1,0,0]) if abs(normal[0]) < 0.9 else np.array([0,1,0])
    axis1 = np.cross(normal, tmp)
    axis1 /= np.linalg.norm(axis1)
    axis2 = np.cross(normal, axis1)

    lines = []
    points = []
    n = int(size / step)

    for i in range(-n, n+1):
        p1 = center + axis1 * i * step + axis2 * size
        p2 = center + axis1 * i * step - axis2 * size
        p3 = center + axis2 * i * step + axis1 * size
        p4 = center + axis2 * i * step - axis1 * size

        points.append(p1); points.append(p2)
        lines.append([len(points)-2, len(points)-1])

        points.append(p3); points.append(p4)
        lines.append([len(points)-2, len(points)-1])

    grid = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector2iVector(lines)
    )
    grid.paint_uniform_color([0.3,0.3,0.3])
    return grid
