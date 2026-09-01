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

# --- Resolution-based mesh decimation ----------------------------------------------------
# A dense STL (a scanned part, or CAD exported at high chord tolerance) costs time in the
# Open3D viewport, in VHACD, and in the raycast — with no downstream benefit, because every
# consumer works at a coarser resolution than the mesh carries.
#
# The target is a RESOLUTION (an edge/voxel length), not a triangle budget: a fraction of the
# part's OBB diagonal so it is scale-invariant, clamped by absolute metric bounds so a tiny
# part is not decimated into a tetrahedron and a huge one is not left coarser than the render
# voxel. 0.4% of a 0.15 m-diagonal part is 0.6 mm, which matches VHACD's own effective voxel
# at resolution=1e6 and sits below the 1 mm render voxel — finer detail is not observable.
DECIMATE_DIAG_FRAC      = 0.004    # target voxel = 0.4% of the part's OBB diagonal
DECIMATE_MIN_VOXEL_M    = 1.0e-4   # 0.1 mm absolute floor (tiny parts)
DECIMATE_MAX_VOXEL_M    = 2.0e-3   # 2 mm absolute ceiling (huge parts)
DECIMATE_MIN_TRIANGLES  = 2000     # below this a mesh is already cheap everywhere
DECIMATE_MAX_VOLUME_ERR = 0.02     # 2% hull-volume drift -> revert to the original mesh
DECIMATE_EDGE_SAMPLES   = 5000     # edges sampled to estimate the mesh's current resolution


def decimate_mesh_to_resolution(mesh,
                                diag_frac: float = DECIMATE_DIAG_FRAC,
                                min_voxel_m: float = DECIMATE_MIN_VOXEL_M,
                                max_voxel_m: float = DECIMATE_MAX_VOXEL_M,
                                min_triangles: int = DECIMATE_MIN_TRIANGLES,
                                max_volume_err: float = DECIMATE_MAX_VOLUME_ERR):
    """Decimate `mesh` to a target spatial resolution. Returns (mesh_out, stats).

    stats: {voxel_size, tri_before, tri_after, volume_err, skipped, reason}

    On skip or revert the ORIGINAL object is returned (identity, not a copy), so a caller can
    test `out is mesh` to tell that nothing happened.

    Uses vertex clustering rather than quadric decimation, deliberately: clustering is natively
    resolution-based (`voxel_size` IS the target edge), whereas driving quadric decimation needs
    an edge->face-count conversion that assumes near-uniform tessellation — and CAD-exported
    STLs are the opposite, pairing huge planar triangles with dense fillet strips. Clustering is
    also a single O(n) pass and cannot fail on the non-watertight, self-intersecting meshes that
    arrive here, where quadric decimation can emit flipped triangles or throw. `Quadric`
    contraction recovers most of the feature retention.

    Input units must be metres — call AFTER any mm->m conversion, or the absolute clamps are
    meaningless.
    """
    stats = {"voxel_size": 0.0, "tri_before": 0, "tri_after": 0,
             "volume_err": 0.0, "skipped": True, "reason": ""}

    tri_before = len(mesh.triangles)
    stats["tri_before"] = stats["tri_after"] = tri_before

    if mesh.is_empty() or tri_before < min_triangles:
        stats["reason"] = "already coarse"
        return mesh, stats

    tri_in = o3d_to_trimesh(mesh)
    try:
        # `.primitive.extents` is the OBB's own side lengths; the inherited `.extents` is the
        # axis-aligned bounds of the rotated box mesh, which inflates toward its diagonal and
        # would silently coarsen the target voxel.
        diag = float(np.linalg.norm(tri_in.bounding_box_oriented.primitive.extents))
    except Exception:
        diag = float(np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent()))
    voxel = float(np.clip(diag_frac * diag, min_voxel_m, max_voxel_m))
    stats["voxel_size"] = voxel

    # Second skip test: if the mesh is already at or below the target resolution, clustering
    # would only add risk. Median edge length is a more honest measure than a raw face count.
    edges = tri_in.vertices[tri_in.edges_unique]
    lengths = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1)
    if lengths.size > DECIMATE_EDGE_SAMPLES:
        idx = np.random.default_rng(0).choice(lengths.size, DECIMATE_EDGE_SAMPLES, replace=False)
        lengths = lengths[idx]
    if float(np.median(lengths)) >= voxel:
        stats["reason"] = "median edge >= voxel"
        return mesh, stats

    out = mesh.simplify_vertex_clustering(
        voxel_size=voxel,
        contraction=o3d.geometry.SimplificationContraction.Quadric)

    # VHACD hygiene: decompose_stage's fillMode="flood" wants a clean watertight mesh, and
    # clustering can leave duplicate/degenerate faces behind.
    out.remove_degenerate_triangles()
    out.remove_duplicated_triangles()
    out.remove_duplicated_vertices()
    out.remove_unreferenced_vertices()
    out.remove_non_manifold_edges()
    out.compute_vertex_normals()

    # Never let decimation destroy a part: revert on any sign of collapse.
    if out.is_empty() or len(out.triangles) < 4:
        stats["reason"] = "degenerate result, reverted"
        return mesh, stats
    try:
        vol_before = float(tri_in.convex_hull.volume)
        vol_after = float(o3d_to_trimesh(out).convex_hull.volume)
        volume_err = abs(vol_after - vol_before) / max(vol_before, 1e-12)
    except Exception:
        volume_err = 0.0
    if volume_err > max_volume_err:
        stats["reason"] = f"volume drift {volume_err:.3f} > {max_volume_err}, reverted"
        return mesh, stats

    stats.update(tri_after=len(out.triangles), volume_err=volume_err,
                 skipped=False, reason="decimated")
    return out, stats


def pcd_geocenter(pcd):
    """
    Returns transformation matrix with consistent orientation.
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

    for i in range(3):
        axis = rotation_matrix[:, i]
        # Find the component with largest absolute value
        max_idx = np.argmax(np.abs(axis))
        # If that component is negative, flip the entire axis
        if axis[max_idx] < 0:
            rotation_matrix[:, i] *= -1
    
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1
    
    rotation_matrix = np.round(rotation_matrix, decimals=6)
    tf = np.eye(4)
    tf[:3, 3] = center
    tf[:3, :3] = rotation_matrix
    tf = np.linalg.inv(tf)
    
    return tf

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

