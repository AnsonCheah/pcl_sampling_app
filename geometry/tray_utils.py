"""Tray geometry: build an injection-molded tray whose pockets trace a part's footprint.

Base-layer module (geometry/): imports nothing from sensor/physics/registration. Reuses the in-process
VHACD wrapper in ``geometry.convex_decomp`` to decompose the (non-convex) pocket frame into convex pieces
for MuJoCo collision (MuJoCo collides meshes as their convex hull, so concave geometry must be split).

Pipeline per stable pose:
  1. footprint_polygon(): project a part (in its aligned stable pose) onto XY -> shapely polygon, fill
     interior holes, then buffer() by the clearance (outward offset + corner smoothing). The COLLISION
     pocket is traced from the part's CONVEX COLLISION (union of its VHACD pieces), not its raw mesh --
     that is what MuJoCo actually collides; the mesh silhouette is too tight and ejects the part. The
     VISUAL pocket uses the true concave mesh silhouette for the molded look.
  2. build_tray_collision_frame(): a cell block minus a through-pocket (NO base) -> VHACD convex pieces.
     The caller adds a separate floor slab; keeping the base out of the decomposed frame stops hull
     pieces from bridging the base into the cavity. High vhacd_resolution keeps wall intrusion < clearance.
  3. tray_visual_tile(): a cell block minus the pocket -- the smooth molded mesh for rendering only.

Requires ``shapely`` (silhouette union + buffer) and ``mapbox_earcut`` (polygon triangulation behind
``trimesh.creation.extrude_polygon``).
"""
import numpy as np
import trimesh
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

from geometry.convex_decomp import vhacd_decompose

# Tiny vertical overshoot so a pocket prism pokes through the slab top/bottom -- avoids coplanar faces
# that make the manifold boolean unstable at the cut surfaces.
_CUT_EPS = 1e-4


def footprint_polygon(part_mesh, R_aligned, clearance: float, fill_holes: bool = True,
                      corner_segs: int = 2, simplify_tol: float = 0.0) -> Polygon:
    """Return a 2D footprint (shapely Polygon) in the aligned-pose mesh frame.

    part_mesh   : trimesh.Trimesh -- pass the part mesh for the VISUAL pocket, or the concatenated convex
                  collision pieces for the COLLISION pocket (what MuJoCo collides).
    R_aligned   : 3x3 rotation placing the part in its tray-facing stable pose.
    clearance   : outward offset (m). buffer() rounds convex corners, giving the smoothed outline.
    fill_holes  : drop interior holes so the pocket is a solid cavity the part always seats into.
    corner_segs : buffer arc segments per quarter-circle (low keeps the outline cheap).
    """
    m = part_mesh.copy()
    T = np.eye(4)
    T[:3, :3] = np.asarray(R_aligned, dtype=float)
    m.apply_transform(T)

    try:
        path = m.projected(normal=[0.0, 0.0, 1.0])   # trimesh Path2D (shapely-backed silhouette)
        poly = unary_union(list(path.polygons_full))
    except Exception as e:                            # degenerate/non-watertight -> convex fallback
        print(f"[tray] projection failed ({e}); falling back to convex hull of projected vertices")
        pts2d = np.asarray(m.vertices)[:, :2]         # m is ALREADY rotated by R -- do not re-apply it
        poly = shapely.MultiPoint(pts2d).convex_hull

    poly = _coalesce(poly)                             # merge hairline-split lobes (don't drop them)
    if fill_holes:
        poly = Polygon(poly.exterior)
    if clearance and clearance > 0:
        poly = _coalesce(poly.buffer(clearance, quad_segs=corner_segs, join_style=1))  # offset + smooth
    if simplify_tol and simplify_tol > 0:
        poly = poly.simplify(simplify_tol)
    return poly


def _coalesce(poly: Polygon) -> Polygon:
    """Reduce a (possibly MultiPolygon) footprint to a single Polygon WITHOUT discarding area: close
    hairline gaps between near-touching lobes with a tiny buffer; only if still disconnected fall back
    to the largest lobe. Taking max() directly (the old behaviour) silently dropped real footprint
    regions, shrinking the pocket below the part and ejecting it."""
    if poly.geom_type != "MultiPolygon":
        return poly
    merged = poly.buffer(2e-4).buffer(-2e-4)
    if merged.geom_type == "Polygon":
        return merged
    return max(poly.geoms, key=lambda g: g.area)


def convex_footprint_polygon(convex_pieces, R_aligned, clearance: float, fill_holes: bool = True,
                             corner_segs: int = 2) -> Polygon:
    """Robust COLLISION footprint: the union of each convex piece's 2D convex hull in the aligned pose.

    This is exactly what MuJoCo collides -- each part geom collides as its convex hull -- and unioning
    clean convex polygons avoids trimesh's ``mesh.projected()``, which on real meshes can return an
    INCOMPLETE silhouette (fractured into lobes, of which only the largest was kept). An incomplete
    collision footprint shrinks the pocket below the part, so the part overlaps the walls and is ejected.

    convex_pieces : iterable of trimesh.Trimesh (the part's VHACD pieces).
    """
    R = np.asarray(R_aligned, dtype=float)
    hulls = []
    for piece in convex_pieces:
        v2 = (R @ np.asarray(piece.vertices).T).T[:, :2]
        h = shapely.MultiPoint(v2).convex_hull
        if h.geom_type == "Polygon" and h.area > 0:
            hulls.append(h)
    poly = _coalesce(unary_union(hulls))
    if fill_holes:
        poly = Polygon(poly.exterior)
    if clearance and clearance > 0:
        poly = _coalesce(poly.buffer(clearance, quad_segs=corner_segs, join_style=1))
    return poly


def build_tray_collision_frame(footprint_poly: Polygon, pitch_xy, pocket_depth: float, base_z: float,
                               max_convex_hulls: int = 24, vhacd_resolution: int = 800000):
    """Decompose ONE pocket's wall frame into convex pieces for MuJoCo collision.

    The frame is a pitch-sized cell with the pocket cut all the way THROUGH it (no base): the caller adds
    a separate floor slab the part rests on. Keeping the base out of the frame is what stops VHACD hulls
    from bridging the solid base across the open cavity (the artifact that ejected seated parts). High
    ``vhacd_resolution`` keeps any residual wall-corner intrusion below the running clearance.

    Centred on the footprint bbox so a pure (x, y) translation by a body position places it under the
    part; the pieces already sit at z in [base_z, base_z + pocket_depth].

    Returns (frame_trimesh, pieces) with pieces a list of (vertices, faces) convex hulls.
    """
    px, py = float(pitch_xy[0]), float(pitch_xy[1])
    minx, miny, maxx, maxy = footprint_poly.bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0

    cell = trimesh.creation.box(extents=[px, py, pocket_depth])
    cell.apply_translation([cx, cy, base_z + pocket_depth / 2.0])

    pocket = trimesh.creation.extrude_polygon(footprint_poly, height=pocket_depth + 2 * _CUT_EPS)
    pocket.apply_translation([0.0, 0.0, base_z - _CUT_EPS])           # cut clean through the cell

    frame = trimesh.boolean.difference([cell, pocket], engine="manifold", check_volume=False)
    frame.process(validate=True)

    pieces = vhacd_decompose(np.asarray(frame.vertices), np.asarray(frame.faces),
                             maxConvexHulls=max_convex_hulls, resolution=vhacd_resolution)
    return frame, pieces


def _heightfield_mesh(gx, gy, Z):
    """Triangulated surface mesh of a height field Z(gy, gx) (the tray pocket top, for rendering)."""
    nrow, ncol = Z.shape
    GX, GY = np.meshgrid(gx, gy)
    verts = np.stack([GX.ravel(), GY.ravel(), Z.ravel()], axis=1)
    faces = []
    for i in range(nrow - 1):
        for j in range(ncol - 1):
            a = i * ncol + j; b = a + 1; c = a + ncol; d = c + 1
            faces.append([a, c, b]); faces.append([b, c, d])
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=False)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_tray_conforming_hfield(convex_pieces, R_aligned, pitch_xy, pocket_depth: float,
                                 base_thickness: float, clearance: float,
                                 grid_res: float = 0.0015, max_grid: int = 128,
                                 visual_grid_res: float = 0.00075, visual_max_grid: int = 192,
                                 visual_wall_radius: float = None):
    """Conforming tray pocket as a MuJoCo height field that cradles the part's bottom surface -- like a
    3D-print support bed. The pocket FLOOR follows the part's bottom depth map z_bottom(x,y) of its
    CONVEX COLLISION (what MuJoCo collides), and the WALLS rise to the cell top outside the
    clearance-buffered footprint. The part is seated `clearance` above the floor, so it settles into a
    uniform-clearance cradle across its whole underside.

    Built from a 2D orthographic depth map (Open3D raycast from below) -- fast, no VHACD. Returns a dict:
      elevation  : (nrow, ncol) float32 in [0, 1]  -> MjSpec hfield userdata
      size       : [radius_x, radius_y, z_range, base]  -> MjSpec hfield size
      z_offset   : geom-local z of the minimum hfield elevation
      center_xy  : (cx, cy) footprint centre in the part frame; the hfield geom goes at body_xy + center
      seat_dz    : part-body z so its lowest point rests `clearance` above the floor (z0 = 0)
      visual     : visual-only surface mesh; contact floor matches the hfield, walls/rim are smoothed
    """
    import scipy.ndimage as ndi
    import open3d as o3d
    from geometry.geom_utils import trimesh_to_o3d

    px, py = float(pitch_xy[0]), float(pitch_xy[1])
    R4 = np.eye(4); R4[:3, :3] = np.asarray(R_aligned, dtype=float)
    M = trimesh.util.concatenate([p.copy().apply_transform(R4) for p in convex_pieces])
    bmin, bmax = M.bounds
    cx, cy = float((bmin[0] + bmax[0]) / 2), float((bmin[1] + bmax[1]) / 2)
    min_z = float(bmin[2])
    M.apply_translation([-cx, -cy, 0.0])                       # centre the footprint at the origin

    ncol = int(np.clip(round(px / grid_res), 8, max_grid))
    nrow = int(np.clip(round(py / grid_res), 8, max_grid))
    gx = np.linspace(-px / 2, px / 2, ncol)
    gy = np.linspace(-py / 2, py / 2, nrow)
    GX, GY = np.meshgrid(gx, gy)                               # (nrow, ncol): rows<->y, cols<->x

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(trimesh_to_o3d(M)))

    def raycast_bottom(GX_, GY_):
        zlo = min_z - 0.01
        origins = np.stack([GX_.ravel(), GY_.ravel(), np.full(GX_.size, zlo)], axis=1).astype(np.float32)
        dirs = np.tile(np.array([0, 0, 1], np.float32), (GX_.size, 1))
        rays = o3d.core.Tensor(np.concatenate([origins, dirs], axis=1))
        t_hit_ = scene.cast_rays(rays)["t_hit"].numpy()
        return (zlo + t_hit_).reshape(GX_.shape)               # +inf where the ray misses (outside part)

    z_bottom = raycast_bottom(GX, GY)
    hit = np.isfinite(z_bottom)
    if not hit.any():
        raise ValueError("conforming hfield: no raycast hits (empty footprint)")

    # Nearest-fill z_bottom into the lateral-clearance band, build the conforming floor, wall the rest.
    nearest = ndi.distance_transform_edt(~hit, return_distances=False, return_indices=True)
    contour = np.clip(z_bottom[tuple(nearest)] - min_z, 0.0, None)   # height above the lowest point (>=0)
    cell_top = base_thickness + pocket_depth
    floor = base_thickness + contour
    rad_px = max(1, int(round(clearance / (px / max(ncol - 1, 1)))))
    floor_mask = ndi.binary_dilation(hit, iterations=rad_px)

    # COLLISION surface = a conservative lower envelope: the part's bottom curves below the linear
    # interpolation through the grid samples, so a floor *through* the samples rises above the part and
    # the solver ejects it. The min-filter pulls each sample down to its neighbourhood minimum so the
    # interpolated hfield stays under the true bottom and the part cradles. The visual contact floor
    # samples this same surface so settled MuJoCo poses do not appear vertically offset in the rerender.
    coll_floor = ndi.minimum_filter(floor, size=3, mode="nearest")
    surface_coll = np.where(floor_mask, np.minimum(coll_floor, cell_top), cell_top)

    # MuJoCo normalizes compile-time hfield elevation data to [0, 1]. Keep that normalization explicit
    # and place the geom at z_offset so MuJoCo collision and the Open3D visual share one coordinate frame.
    hfield_min = float(surface_coll.min())
    hfield_range = max(float(surface_coll.max() - hfield_min), 1e-9)
    elevation = np.clip((surface_coll - hfield_min) / hfield_range, 0.0, 1.0).astype(np.float32)
    hfield_surface = hfield_min + elevation.astype(float) * hfield_range

    # VISUAL surface = the same contact floor, but only the walls and top rim are rounded. A finer
    # visual grid reduces the stair-step outline without increasing MuJoCo's hfield resolution.
    if visual_grid_res is None or visual_grid_res <= 0:
        visual_grid_res = grid_res
    visual_max_grid = max(8, int(visual_max_grid or max_grid))
    ncol_vis = int(np.clip(round(px / float(visual_grid_res)), 8, visual_max_grid))
    nrow_vis = int(np.clip(round(py / float(visual_grid_res)), 8, visual_max_grid))
    gx_vis = np.linspace(-px / 2, px / 2, ncol_vis)
    gy_vis = np.linspace(-py / 2, py / 2, nrow_vis)
    GXv, GYv = np.meshgrid(gx_vis, gy_vis)

    z_bottom_vis = raycast_bottom(GXv, GYv)
    hit_vis = np.isfinite(z_bottom_vis)
    if not hit_vis.any():
        hit_vis = ndi.zoom(hit.astype(float), (nrow_vis / nrow, ncol_vis / ncol), order=0) > 0.5

    row = ((GYv - gy[0]) / max(gy[-1] - gy[0], 1e-12) * (nrow - 1)).ravel()
    col = ((GXv - gx[0]) / max(gx[-1] - gx[0], 1e-12) * (ncol - 1)).ravel()
    visual_floor = ndi.map_coordinates(
        np.minimum(coll_floor, cell_top), [row, col], order=1, mode="nearest"
    ).reshape(nrow_vis, ncol_vis)

    dx_vis = px / max(ncol_vis - 1, 1)
    dy_vis = py / max(nrow_vis - 1, 1)
    outside_dist = ndi.distance_transform_edt(~hit_vis, sampling=(dy_vis, dx_vis))
    wall_radius = visual_wall_radius
    if wall_radius is None:
        wall_radius = max(float(clearance), 2.0 * max(dx_vis, dy_vis))
    wall_radius = max(float(wall_radius), 1e-9)
    floor_margin = max(float(clearance), 0.0)
    blend = _smoothstep((outside_dist - floor_margin) / wall_radius)
    surface_vis = visual_floor * (1.0 - blend) + cell_top * blend
    surface_vis[hit_vis] = visual_floor[hit_vis]
    surface_vis[outside_dist >= floor_margin + wall_radius] = cell_top
    surface_vis = np.clip(surface_vis, hfield_min, cell_top)

    return dict(
        elevation=elevation,
        size=[px / 2.0, py / 2.0, hfield_range, float(base_thickness)],
        z_offset=hfield_min,
        center_xy=(cx, cy),
        seat_dz=base_thickness - min_z + clearance,
        pocket_depth=pocket_depth,
        collision_surface=hfield_surface,
        visual=_heightfield_mesh(gx_vis, gy_vis, surface_vis),
        visual_shape=surface_vis.shape,
        visual_contact_mask=hit_vis,
        visual_floor=visual_floor,
    )


def tray_visual_tile(footprint_poly: Polygon, pitch_xy, pocket_depth: float, base_thickness: float):
    """Smooth molded tray cell (a pitch-sized block minus the pocket) for rendering only.

    Centred on the footprint bbox so a pure (x, y) translation by a body position places it under the
    part. Concave -- used for the visual/raycast mesh, never for collision.
    """
    px, py = float(pitch_xy[0]), float(pitch_xy[1])
    minx, miny, maxx, maxy = footprint_poly.bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    tray_top = base_thickness + pocket_depth

    cell = trimesh.creation.box(extents=[px, py, tray_top])
    cell.apply_translation([cx, cy, tray_top / 2.0])

    prism = trimesh.creation.extrude_polygon(footprint_poly, height=pocket_depth + _CUT_EPS)
    prism.apply_translation([0.0, 0.0, base_thickness])       # cut the top pocket_depth of the cell

    tile = trimesh.boolean.difference([cell, prism], engine="manifold", check_volume=False)
    tile.process(validate=True)
    return tile
