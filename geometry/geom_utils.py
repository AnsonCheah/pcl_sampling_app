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
                                max_volume_err: float = DECIMATE_MAX_VOLUME_ERR,
                                voxel_m: float = None):
    """Decimate `mesh` to a target spatial resolution. Returns (mesh_out, stats).

    stats: {voxel_size, tri_before, tri_after, volume_err, skipped, reason}

    `voxel_m` overrides the diagonal-derived target. Pass it when two variants of the same part
    must be decimated identically — e.g. ImportMeshStage decimates both the raw mesh and the
    debris-free `cleaned` mesh, and export debris would otherwise inflate the raw mesh's OBB
    diagonal and coarsen its voxel relative to the cleaned one.

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
    if voxel_m is not None:
        voxel = float(voxel_m)
    else:
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


def pcd_geocenter(pcd, axis=None):
    """Model-frame transform for a cloud, with a deterministic axis convention.

    pcd  : point cloud.
    axis : ``None`` for the PCA frame (covariance eigenvectors through the cloud mean), or
           an ``AmbiguityAxis`` (see ``geometry.ambiguity``) to build the frame around it.

    Returns
    -------
    tf : (4, 4) transform to APPLY to the cloud to put it in this frame. It is the *inverse*
         frame transform, so the translation sits in column 3; see geometry/CLAUDE.md.

    With an ``AmbiguityAxis`` the axis becomes frame **Z** through the origin, always —
    MechVision's ``rotationStrategy`` can only rotate about a geocenter axis, and a PCA frame
    has no reason to line up with an ambiguity axis. Consequently **the part is not centred
    on the origin, and that is correct**: an ambiguity axis generally misses the centroid, so
    both cannot sit at the origin, and centring the part would aim the symmetry search at the
    wrong line. The origin is placed at the projection of the cloud mean onto the axis, which
    zeroes the along-axis offset without disturbing that.

    Pure and deterministic: same cloud in, bitwise-identical matrix out. Which ambiguity axis
    wins is *not* stable across re-analyses — see ``geometry.ambiguity.analyse_ambiguity``.
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

    return _pack(rot, origin)

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

def _as_points(obj) -> np.ndarray:
    """Accept a PointCloud, a TriangleMesh, or an (N,3) array and return the points."""
    if isinstance(obj, np.ndarray):
        return np.asarray(obj, dtype=float).reshape(-1, 3)
    if hasattr(obj, "points"):
        return np.asarray(obj.points, dtype=float)
    if hasattr(obj, "vertices"):
        return np.asarray(obj.vertices, dtype=float)
    raise TypeError(f"cannot extract points from {type(obj)}")


def model_diameter(obj, max_hull_points: int = 3000) -> float:
    """Largest distance between any two points, searched over the convex hull.

    Exact while the hull has at most ``max_hull_points`` vertices; above that the hull is
    strided down and the result is a tight lower bound. The subsample keeps the six
    axis-extreme points, so it can never come back shorter than the longest AABB edge. The
    cap is not theoretical: a sphere sampled at 20 000 points has a 19 172-vertex hull, and
    an unguarded all-pairs distance matrix over that would ask for 8.8 GB.

    This is the single definition of "diameter". The longest minimal-OBB extent and the AABB
    diagonal disagree by 1.55x on the bunny (94.4 mm vs 146.5 mm), so "5% of diameter" means
    two different things unless every caller uses this one. The exception is
    ``geometry.ambiguity``, which deliberately keeps its own AABB-diagonal measure because
    its tolerances are calibrated against that number.
    """
    from scipy.spatial.distance import pdist

    pts = _as_points(obj)
    if len(pts) < 2:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        # Degenerate (coplanar/collinear) clouds have no 3D hull; the answer is still the
        # max pairwise distance, just over every point.
        hull = pts
    if len(hull) > max_hull_points:
        extremes = np.concatenate([hull.argmin(axis=0), hull.argmax(axis=0)])
        stride = np.linspace(0, len(hull) - 1, max_hull_points).astype(np.int64)
        hull = hull[np.unique(np.concatenate([stride, extremes]))]
    return float(pdist(hull).max())


def median_spacing(obj) -> float:
    """Median nearest-neighbour distance — the cloud's own resolution.

    This is the floor on any geometric tolerance: agreement asserted below the sampling
    pitch is measuring the sampling, not the geometry.
    """
    from scipy.spatial import cKDTree
    pts = _as_points(obj)
    if len(pts) < 2:
        return 0.0
    return float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1]))


def pairing_error(mesh, cloud):
    """Why this mesh and this cloud are not the same part in the same frame, or None.

    Sibling of :func:`reference_frames_agree`: that one compares two *clouds*, this one
    compares a cloud against the *mesh* an analysis will raycast.

    The mesh is not a decoration to `ambiguity.analyse_ambiguity` -- ``_visibility_masks``
    raycasts it and snaps each hit to the nearest cloud point within ~``2*radius/res``, so a
    cloud that does not sit ON that mesh loses almost every point from ``visible``.  Nothing
    errors: the sweep just reports a thin sliver as "the visible patch", the per-view survival
    test is then applied to a few hundred accidental points, and the ranking is decided by
    whichever transform happens to explain that sliver.

    Measured on a stale export paired with the current STL, 13% of points were ever visible
    (311 per view) against 99% (6400 per view) for a matched pair, and the run reported 12
    axes with a bogus continuous one where the matched pair reports 3.  A silently wrong
    answer, so it is checked rather than assumed -- the check costs milliseconds.
    """
    from scipy.spatial import cKDTree

    pts = _as_points(cloud)
    if len(pts) < 2 or len(mesh.triangles) == 0:
        return None
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    dist = scene.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy()

    # Judged against the cloud's own resolution, not an absolute distance, so this scales
    # from a 20 mm part to a 500 mm one exactly as the analysis tolerances do.
    spacing = float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1]))
    offset = float(np.percentile(dist, 95))
    tol = max(2.0 * spacing, 0.0005)
    if offset <= tol:
        return None
    return (f"95% of the cloud's points lie within {offset * 1000:.2f} mm of the mesh "
            f"surface, against a {tol * 1000:.2f} mm tolerance "
            f"(2x the cloud's own {spacing * 1000:.2f} mm spacing)")


def reference_frames_agree(a, b, tol_factor: float = 0.5, sample: int = 4000):
    """Are these two reference clouds the same cloud in the same model frame?

    Returns ``None`` when they agree, else a human-readable reason.

    A scene's ``reference_cloud.ply`` and the exported bundle's ``<part>_surface.ply`` are
    written from ``down_pcd`` and ``down_pcd_surface``, which hold identical points in both
    uniform and adaptive modes — so on a matching pair this is an exact comparison, not an
    approximate one, and the tolerance only absorbs PLY's float32 round trip.

    This exists because the model frame is **not reproducible across re-exports**. It is
    bitwise deterministic for a fixed mesh and fixed settings, but change the voxel size,
    the view count or the ambiguity config and the winning ambiguity axis can change:
    measured on 25333MB000, three sample densities gave folds C4/C1/C2 with the third
    landing on a different axis 45 degrees away, moving the model frame by 60.6 degrees and
    28.3 mm. Nothing downstream can detect that from the files alone — a stale scene still
    loads, still has a T_gt, and simply scores every pose against the wrong frame.

    ``bench/generate_scenes.py`` is resumable and tops up parts that already hold scenes, so
    a single part can accumulate scenes from two sessions in two different frames without
    anything noticing. That is the case this is here to catch.
    """
    from scipy.spatial import cKDTree

    pa, pb = _as_points(a), _as_points(b)
    if len(pa) == 0 or len(pb) == 0:
        return f"empty cloud ({len(pa)} vs {len(pb)} points)"
    if len(pa) != len(pb):
        return (f"point counts differ ({len(pa)} vs {len(pb)}) -- these are not the same "
                f"cloud, so they cannot be the same export")

    spacing = median_spacing(pb)
    if spacing <= 0.0:
        return None                              # degenerate cloud; nothing to compare against
    tol = tol_factor * spacing

    idx = np.arange(len(pa)) if len(pa) <= sample else \
        np.linspace(0, len(pa) - 1, sample).astype(np.int64)
    dist = cKDTree(pb).query(pa[idx], k=1)[0]
    # p99 rather than max: a single PLY float32 round-trip outlier should not condemn an
    # otherwise identical cloud, but a frame change moves essentially every point.
    worst = float(np.percentile(dist, 99))
    if worst <= tol:
        return None
    return (f"clouds are in different model frames: p99 nearest-neighbour distance "
            f"{worst * 1000:.3f} mm exceeds {tol * 1000:.3f} mm "
            f"(0.5x median spacing {spacing * 1000:.3f} mm)\n"
            f"        A aabb (mm): {np.round(pa.min(axis=0) * 1000, 2)} .. "
            f"{np.round(pa.max(axis=0) * 1000, 2)}\n"
            f"        B aabb (mm): {np.round(pb.min(axis=0) * 1000, 2)} .. "
            f"{np.round(pb.max(axis=0) * 1000, 2)}")


def estimate_surface_area(obj, k: int = 8) -> float:
    """Surface area of a point cloud, from a k-nearest-neighbour density estimate.

    For a locally 2D point set of areal density ``lambda``, the k-th nearest neighbour sits
    at ``d_k`` with ``lambda ~ k / (pi * d_k^2)``; the area is then ``N / lambda``.

    This replaces ``N * s^2`` (with ``s`` the *first*-neighbour distance), which is wrong by
    a large constant for randomly sampled clouds: the median nearest-neighbour distance of a
    Poisson process is ``0.4697/sqrt(lambda)``, not ``1/sqrt(lambda)``, so that formula
    understates area by ~4.5x.  Measured on a box of known area it returned 2 389 mm^2
    against a true 11 200 mm^2.  Using a larger ``k`` averages over the local arrangement
    and lands within a few tens of percent for both randomly sampled and voxel-gridded
    clouds, which matters because this pipeline produces both.

    Also deterministic, unlike the version it replaces: that one sampled 500 points with an
    unseeded ``np.random.choice``, so repeated calls disagreed and every parameter derived
    from it moved with them.
    """
    from scipy.spatial import cKDTree
    pts = _as_points(obj)
    if len(pts) < 10:
        return float(np.pi * (model_diameter(pts) / 2.0) ** 2)     # sphere fallback
    kk = min(k, len(pts) - 1)
    d_k = cKDTree(pts).query(pts, k=kk + 1)[0][:, kk]              # skip self at column 0
    d_k = d_k[d_k > 1e-12]
    if len(d_k) == 0:
        return 0.0
    density = kk / (np.pi * np.median(d_k) ** 2)                   # points per unit area
    return float(len(pts) / max(density, 1e-12))


def golden_hue_color(i: int, saturation: float, value: float):
    """RGB for item `i`, hue advanced by the golden ratio so adjacent indices contrast.

    Unlike an even split over n, this needs no total up front and stays well separated
    however many items appear. Callers pick the tone: overlays match whatever they sit on.
    """
    return colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, saturation, value)


def make_transform(rotation=None, translation=None) -> np.ndarray:
    """4x4 homogeneous transform from a 3x3 rotation and/or a 3-vector translation."""
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    if translation is not None:
        T[:3, 3] = translation
    return T


def pose_to_matrix(pose) -> np.ndarray:
    """MuJoCo/MechVision pose [x, y, z, qw, qx, qy, qz] (scalar-first) -> 4x4."""
    x, y, z, qw, qx, qy, qz = pose
    return make_transform(R.from_quat([qw, qx, qy, qz], scalar_first=True).as_matrix(),
                          (x, y, z))


def mat_to_wxyz(T) -> np.ndarray:
    """Rotation part of a 3x3/4x4 matrix as a scalar-first (w, x, y, z) quaternion."""
    return R.from_matrix(np.asarray(T, dtype=float)[:3, :3]).as_quat(scalar_first=True)


def project_to_so3(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix to ``R``. Accepts ``(3,3)`` or a batch ``(..., 3, 3)``.

    Needed wherever rotations are averaged — pose clustering averages the rotations of the
    hypotheses in a cluster, and the mean of several rotation matrices is not itself one.
    Feeding an unprojected mean downstream produces a transform that quietly scales and
    shears the model, which shows up as a plausible-looking pose that fails verification.

    (This is also the failure OpenCV's ``ppf_match_3d`` ships with — opencv_contrib #3223,
    an unnormalised quaternion in ``clusterPoses`` — so the same guard is needed whether the
    clustering is ours or theirs.)

    Delegates to ``scipy.spatial.transform.Rotation.from_matrix``, which orthogonalises a
    non-proper input via Markley's quaternion method rather than raising. That agrees with a
    hand-written SVD projection to 1.3e-15 over 200 perturbed rotations, so there is nothing
    to gain from keeping our own — and scipy handles batches and reflections for free.
    """
    from scipy.spatial.transform import Rotation

    arr = np.asarray(R, dtype=float)
    return Rotation.from_matrix(arr).as_matrix().reshape(arr.shape)


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
