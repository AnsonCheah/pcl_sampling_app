"""Tray geometry: build an injection-molded tray whose pockets trace a part's footprint.

Base-layer module (geometry/): imports nothing from sensor/physics/registration. Reuses the in-process
VHACD wrapper in ``geometry.convex_decomp`` to decompose the (non-convex) pocket frame into convex pieces
for MuJoCo collision (MuJoCo collides meshes as their convex hull, so concave geometry must be split).

Pipeline per stable pose:
  1. footprint_polygon(): project a part (in its aligned stable pose) onto XY → shapely polygon, fill
     interior holes, then buffer() by the clearance (outward offset + corner smoothing). The COLLISION
     pocket is traced from the part's CONVEX COLLISION (union of its VHACD pieces), not its raw mesh —
     that is what MuJoCo actually collides; the mesh silhouette is too tight and ejects the part. The
     VISUAL pocket uses the true concave mesh silhouette for the molded look.
  2. build_tray_collision_frame(): a cell block minus a through-pocket (NO base) → VHACD convex pieces.
     The caller adds a separate floor slab; keeping the base out of the decomposed frame stops hull
     pieces from bridging the base into the cavity. High vhacd_resolution keeps wall intrusion < clearance.
  3. tray_visual_tile(): a cell block minus the pocket — the smooth molded mesh for rendering only.

Requires ``shapely`` (silhouette union + buffer) and ``mapbox_earcut`` (polygon triangulation behind
``trimesh.creation.extrude_polygon``).
"""
import numpy as np
import trimesh
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

from geometry.convex_decomp import vhacd_decompose

# Tiny vertical overshoot so a pocket prism pokes through the slab top/bottom — avoids coplanar faces
# that make the manifold boolean unstable at the cut surfaces.
_CUT_EPS = 1e-4


def footprint_polygon(part_mesh, R_aligned, clearance: float, fill_holes: bool = True,
                      corner_segs: int = 2, simplify_tol: float = 0.0) -> Polygon:
    """Return a 2D footprint (shapely Polygon) in the aligned-pose mesh frame.

    part_mesh   : trimesh.Trimesh — pass the part mesh for the VISUAL pocket, or the concatenated convex
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
    except Exception as e:                            # degenerate/non-watertight → convex fallback
        print(f"[tray] projection failed ({e}); falling back to convex hull of projected vertices")
        pts2d = np.asarray(m.vertices)[:, :2]         # m is ALREADY rotated by R — do not re-apply it
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

    This is exactly what MuJoCo collides — each part geom collides as its convex hull — and unioning
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


def tray_visual_tile(footprint_poly: Polygon, pitch_xy, pocket_depth: float, base_thickness: float):
    """Smooth molded tray cell (a pitch-sized block minus the pocket) for rendering only.

    Centred on the footprint bbox so a pure (x, y) translation by a body position places it under the
    part. Concave — used for the visual/raycast mesh, never for collision.
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
