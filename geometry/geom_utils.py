import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import trimesh
import colorsys
from dataclasses import dataclass
from typing import Optional
import open3d.visualization.rendering as rendering
from geometry.math_utils import find_cdf_knee
# import cupy as cp

@dataclass
class O3DSceneObject:
    geom: o3d.geometry.Geometry3D
    ref_geom: Optional[o3d.geometry.Geometry3D]= None
    material: Optional[rendering.MaterialRecord] = None
    id: Optional[int] = None
    T_gt: Optional[np.ndarray] = None
    xyz0: Optional[np.ndarray] = None
    xyz1: Optional[np.ndarray] = None
    overlap: Optional[float] = None

def init_open3d():
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.destroy_window()

def random_rotation_matrix():
    return R.random().as_matrix()

def random_quaternion(scalar_first=False):
    return R.random().as_quat(scalar_first=scalar_first)

def rotation_aligning_vector_to_axis(src, dst=(0., 0., 1.)) -> np.ndarray:
    """Return a 3x3 rotation matrix R such that ``R @ src`` is parallel to ``dst``.

    Used to turn a user-picked face normal ("this face points up") into the stable-pose
    rotation the structured-scene builder expects. Mirrors the face-normal alignment in
    ``MujocoBinScene.get_stable_poses`` but aligns to an arbitrary axis (default world +Z).
    Handles the parallel / antiparallel degeneracies; for the antiparallel case any
    perpendicular rotation axis is valid, so we pick one deterministically.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)

    cos_a = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if cos_a > 1.0 - 1e-6:
        return np.eye(3)
    if cos_a < -1.0 + 1e-6:
        # Antiparallel: rotate 180 deg about any axis perpendicular to src.
        perp = np.cross(src, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(src, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return R.from_rotvec(perp * np.pi).as_matrix()

    axis = np.cross(src, dst)
    axis /= np.linalg.norm(axis)
    return R.from_rotvec(axis * np.arccos(cos_a)).as_matrix()

def face_facet_map(tri_mesh):
    """Group coplanar-adjacent faces into facets (trimesh.facets) and return a face->facet lookup.

    Returns (face_to_facet, facets, facets_normal):
      face_to_facet : (F,) int array; face_to_facet[f] is the facet index of face f, or -1 when f is
                      a singleton (trimesh only lists groups of 2+ coplanar-adjacent faces).
      facets        : list of int arrays, each the face indices of one coplanar-adjacent group.
      facets_normal : (len(facets), 3) float array, one normal per facet.

    Merge coincident vertices first (``tri_mesh.merge_vertices()``) so face adjacency is detectable
    on split-vertex STL meshes — otherwise every face comes back as a singleton.
    """
    facets = list(tri_mesh.facets)
    facets_normal = np.asarray(tri_mesh.facets_normal, dtype=float).reshape(-1, 3)
    face_to_facet = np.full(len(tri_mesh.faces), -1, dtype=np.int64)
    for fi, faces in enumerate(facets):
        face_to_facet[np.asarray(faces, dtype=np.int64)] = fi
    return face_to_facet, facets, facets_normal

def o3d_display(geometries:list, width:int=1280, height:int=720, dynamic_color:bool=False):
    """
    Display a list of Open3D geometries with a black background.

    Args:
        geometries (list): List of open3d.geometry objects, OR a list of such
            lists (groups) — e.g. two hull sets being compared side by side.
        width (int): Window width.
        height (int): Window height.
        dynamic_color (bool): Paint every geometry a distinct HSV colour. Each
            group gets its own full [0, 1) hue wheel, so a group painted in a
            single dynamic_color=True call as part of a bigger concatenated
            list would otherwise only get a contiguous *slice* of the wheel
            (proportional to its size) rather than the full rainbow — and a
            slice landing in the perceptually-compressed blue/magenta region
            reads as "all the same colour" even though every item did get a
            unique hue. Passing groups keeps every group fully distinct
            regardless of its size.
    """
    grouped = bool(geometries) and isinstance(geometries[0], (list, tuple))
    groups = geometries if grouped else [geometries]

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    vis.add_geometry(frame)
    for group in groups:
        for g in group:
            vis.add_geometry(g)

    render_opt = vis.get_render_option()
    render_opt.background_color = [0, 0, 0]  # black background
    render_opt.light_on = True
    # Optional visual tweaks
    render_opt.mesh_show_back_face = True
    render_opt.point_size = 3.0

    if dynamic_color:
        saturation, value = 0.4, 0.9
        for group in groups:
            hues = np.linspace(0, 1, len(group), endpoint=False)
            for g, h in zip(group, hues):
                g.paint_uniform_color(colorsys.hsv_to_rgb(h, saturation, value))

    return vis


def o3d_to_trimesh(o3d_mesh):
    return trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), faces=np.asarray(o3d_mesh.triangles), process=False)

def trimesh_to_o3d(tri_mesh:trimesh.Trimesh):
    o3d_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(tri_mesh.vertices), 
                                         triangles=o3d.utility.Vector3iVector(tri_mesh.faces))
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

def _fix_axis_signs(rotation_matrix):
    """Deterministic sign convention: each column's dominant component is positive.

    Without this the eigen-decomposition's arbitrary sign choice makes the frame differ
    run to run, which matters because the frame is baked into every exported cloud.
    """
    for i in range(3):
        axis = rotation_matrix[:, i]
        max_idx = np.argmax(np.abs(axis))
        if axis[max_idx] < 0:
            rotation_matrix[:, i] *= -1
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1
    return rotation_matrix


def pcd_geocenter(pcd, axis=None, axis_align_tol_deg=5.0, axis_point_tol=None):
    """
    Returns transformation matrix with consistent orientation.

    With ``axis=None`` this is the historical PCA-canonical frame: axes are the
    covariance eigenvectors through the cloud mean.

    With an ``AmbiguityAxis`` (see ``geometry.ambiguity``) the frame is built so the
    ambiguity axis IS the frame's **Z** and passes through the frame origin. That is what
    makes the axis addressable by MechVision's ``rotationStrategy``, which can only rotate
    about a geocenter frame axis: a PCA frame is derived from mass distribution and has no
    reason to line up with an ambiguity axis, so without this the symmetry search rotates
    about the wrong line no matter what ``angleStep`` is used.

    If the PCA frame already agrees with the axis (direction within
    ``axis_align_tol_deg``, and the axis passes within ``axis_point_tol`` of the PCA
    origin) the PCA frame is returned unchanged, so parts that were already correct need
    no bundle or scene regeneration.

    Returns
    -------
    (tf, frame_changed) when ``axis`` is given, else ``tf`` alone — the historical
    single-value contract is preserved for existing callers.

    Note the return convention is the *inverse* transform; see geometry/CLAUDE.md.
    """
    points = np.asarray(pcd.points)
    center = points.mean(axis=0)

    centered_points = points - center
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    rotation_matrix = eigenvectors[:, idx]
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1

    rotation_matrix = _fix_axis_signs(rotation_matrix)

    def _pack(rot, origin):
        tf = np.eye(4)
        tf[:3, 3] = origin
        tf[:3, :3] = np.round(rot, decimals=6)
        return np.linalg.inv(tf)

    if axis is None:
        return _pack(rotation_matrix, center)

    d = np.asarray(axis.direction, dtype=float)
    d = d / np.linalg.norm(d)
    axis_pt = np.asarray(axis.point, dtype=float)
    if axis_point_tol is None:
        extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        axis_point_tol = 0.01 * float(extent)

    # Does the existing PCA frame already do the job?
    for i in range(3):
        if abs(float(rotation_matrix[:, i] @ d)) < np.cos(np.deg2rad(axis_align_tol_deg)):
            continue
        offset = center - axis_pt
        if np.linalg.norm(offset - float(offset @ d) * d) <= axis_point_tol:
            return _pack(rotation_matrix, center), False

    # Frame Z is the ambiguity axis. Two DOF remain: where the origin sits along the axis,
    # and the in-plane rotation. Fix the first by projecting the cloud mean onto the axis
    # and the second from the PCA of the cloud projected into the perpendicular plane, so
    # the frame is fully determined and reproducible.
    origin = axis_pt + float((center - axis_pt) @ d) * d

    rel = points - origin
    planar = rel - np.outer(rel @ d, d)
    cov2 = np.cov(planar.T)
    vals2, vecs2 = np.linalg.eig(cov2)
    order = np.argsort(vals2.real)[::-1]
    x_axis = vecs2[:, order[0]].real
    x_axis = x_axis - float(x_axis @ d) * d
    norm = np.linalg.norm(x_axis)
    if norm < 1e-9:                      # perfectly isotropic in-plane; any X will do
        x_axis = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
        norm = np.linalg.norm(x_axis)
    x_axis /= norm
    y_axis = np.cross(d, x_axis)

    rot = _fix_axis_signs(np.column_stack([x_axis, y_axis, d]))
    # Re-orthonormalise: the sign fixing may have flipped Z, and Z must stay the axis
    # (direction only -- either sense of the axis is the same line).
    rot[:, 1] = np.cross(rot[:, 2], rot[:, 0])
    rot[:, 1] /= np.linalg.norm(rot[:, 1])
    rot[:, 0] = np.cross(rot[:, 1], rot[:, 2])
    rot[:, 0] /= np.linalg.norm(rot[:, 0])

    return _pack(rot, origin), True

def estimate_normals(points:np.ndarray, view_pos:np.ndarray, radius=0.005, max_nn=50):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_towards_camera_location(view_pos)
    return np.asarray(pcd.normals)

def normalize_normals(pcd):
    """
    Normalize all normals in a point cloud to unit length.
    Removes any points with zero-magnitude normals.
    
    Args:
        pcd: Open3D PointCloud object
    
    Returns:
        The same point cloud object (modified in-place)
    
    Raises:
        ValueError: If point cloud has no normals
    """
    if not pcd.has_normals():
        raise ValueError("Point cloud has no normals")
    
    normals = np.asarray(pcd.normals, dtype=np.float64)
    points = np.asarray(pcd.points, dtype=np.float64)
    
    # Vectorized magnitude calculation
    magnitudes = np.linalg.norm(normals, axis=1)
    valid_mask = magnitudes > 1e-10
    num_removed = np.sum(~valid_mask)
    
    if num_removed > 0:
        print(f"Removing {num_removed} points with zero-magnitude normals")
        points = points[valid_mask]
        normals = normals[valid_mask]
        magnitudes = magnitudes[valid_mask]
    
    # Vectorized normalization - no need for np.newaxis, broadcasting handles it
    normalized = normals / magnitudes[:, None]
    
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normalized)
    
    return pcd

def validate_normals(pcd, tolerance=1e-5):
    """
    Check if all normals in the point cloud are normalized (magnitude = 1.0).
    Raises ValueError if any normal is not normalized beyond the tolerance.
    
    Args:
        pcd: Open3D PointCloud object
        tolerance: Allowed deviation from magnitude 1.0 (default: 1e-5)
    
    Raises:
        ValueError: If normals are missing or not normalized
    """
    if not pcd.has_normals():
        raise ValueError("Point cloud has no normals")
    
    normals = np.asarray(pcd.normals, dtype=np.float64)
    if normals.shape[0] == 0:
        raise ValueError("Normals array is empty")
    
    magnitudes = np.linalg.norm(normals, axis=1)
    deviations = np.abs(magnitudes - 1.0)
    
    if np.any(deviations > tolerance):
        invalid_count = np.sum(deviations > tolerance)
        worst_idx = np.argmax(deviations)
        raise ValueError(
            f"Found {invalid_count} unnormalized normals. "
            f"Max deviation: {deviations[worst_idx]:.2e} at index {worst_idx} "
            f"(magnitude: {magnitudes[worst_idx]:.8f}). "
            f"Min/Max magnitudes: {np.min(magnitudes):.8f}/{np.max(magnitudes):.8f}"
        )
    print(f"All {len(magnitudes)} normals passed validation.")

def fibonacci_sphere(samples):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def mask_point_cloud(pcd, mask):
    masked_pcd = o3d.geometry.PointCloud()
    masked_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask])
    masked_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
    return masked_pcd

def extract_edge_points(pcd, voxel_size):
    """
    Classify each point as edge or non-edge using angular gap analysis in the
    local tangent plane.  For each point, neighbor vectors are projected onto
    the plane perpendicular to the point's normal; the largest gap between
    consecutive azimuthal angles is the edge score.  The threshold is chosen
    automatically via CDF knee detection — no per-part tuning required.

    Returns a boolean mask (True = edge) over the input point cloud.
    """
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    n_points = len(pts)

    tree = o3d.geometry.KDTreeFlann(pcd)
    radius = voxel_size * 5.0

    max_gaps = np.zeros(n_points, dtype=np.float64)

    for i in range(n_points):
        _, idx, _ = tree.search_radius_vector_3d(pts[i], radius)
        idx = np.asarray(idx)
        if len(idx) < 4:  # need self + at least 3 neighbours
            continue

        n = nrm[i]
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm

        neighbors = pts[idx[1:]] - pts[i]  # vectors to neighbours (exclude self)

        # Project onto tangent plane
        projected = neighbors - (neighbors @ n)[:, None] * n

        # Orthonormal basis in tangent plane
        arb = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, arb)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

        cu = projected @ u
        cv = projected @ v
        valid = np.sqrt(cu ** 2 + cv ** 2) > 1e-10
        if valid.sum() < 2:
            continue

        angles = np.sort(np.arctan2(cv[valid], cu[valid]))
        gaps = np.diff(angles)
        wrap_gap = (angles[0] + 2.0 * np.pi) - angles[-1]
        max_gaps[i] = np.max(np.append(gaps, wrap_gap))

    threshold, _, _ = find_cdf_knee(max_gaps)
    return max_gaps >= threshold


def orient_normals_using_cameras(pcd, cam_positions):
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    
    view_vecs = pts - cam_positions   # from camera → point
    view_vecs /= np.linalg.norm(view_vecs, axis=1, keepdims=True)

    dots = np.sum(nrm * view_vecs, axis=1)

    # If dot > 0, normal points *away* from camera → flip it
    flip = dots > 0
    nrm[flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(nrm)

def camera_view_matrix(cam_pos, look_at, up=np.array([0.0, 0.0, 1.0])):
    forward = look_at - cam_pos
    forward /= np.linalg.norm(forward)

    # If forward is parallel to up, choose a fallback up vector
    right = np.cross(up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross([0,1,0], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    T = np.eye(4)
    T[:3, :3] = np.stack([right, up, forward], axis=1)
    T[:3, 3] = cam_pos
    return T

def compute_overlap(xyz0: np.ndarray,
                    xyz1: np.ndarray,
                    threshold: float) -> float:
    """
    Fraction of xyz0 points (already transformed into xyz1 frame)
    that have a neighbour in xyz1 within threshold.
    """
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(xyz1)
    tree = o3d.geometry.KDTreeFlann(pcd1)
    hits = sum(1 for p in xyz0 if tree.search_radius_vector_3d(p, threshold)[0] > 0)
    return hits / max(len(xyz0), 1)

if __name__=="__main__":
    import time
    start = time.time()
    mesh = o3d.t.io.read_triangle_mesh("mesh_raw/37150MB000.STL")
    print(f"legacy read time: {time.time() - start}")
    start = time.time()
    mesh = o3d.t.io.read_triangle_mesh("mesh_raw/37150MB000.STL")
    print(f"tensor read time: {time.time() - start}")

    bbox = mesh.get_axis_aligned_bounding_box()
    extent_max = bbox.get_extent().max()
    if 5.0 < extent_max < 5000.0:
        print(f"[INFO] Converting units mm → m")
        mesh.scale(0.001, center=(0, 0, 0))
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())

    # mesh.paint_uniform_color([0.5,0.5,0.5])
    start = time.time()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fibonacci_sphere(200)*0.5))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=50))
    print(f"legacy estimate time: {time.time() - start}")
    # start = time.time()
    # pcd1 = o3d.geometry.PointCloud(o3c.Tensor(fibonacci_sphere(200)*0.5, o3c.float32, device))
    # pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=50))
    # print(f"tensor estimate time: {time.time() - start}")
    # o3d_display([pcd1])
    # o3d.visualization.draw_geometries([mesh, pcd1], width=1080, height=720, zoom=1.0)

