"""Import-time mesh validation and repair.

Conventional hygiene checks applied to every imported CAD/STL mesh before it reaches VHACD,
MuJoCo, or the raycaster. Pure geometry -- no GUI, no local imports beyond numpy/open3d, so it
stays in the base layer of the dependency graph.

The motivating failure: a CAD export can carry a stray 2-triangle sliver a metre away from the
part. It contributes nothing visually but inflates the axis-aligned bounding box, which then
drives the mm->m unit heuristic, the camera framing, and -- via the convex hulls -- the partition
/ tray cell sizing in `physics.mujoco_bin_scene._compute_structured_grid`. One speck is enough
to generate a fixture larger than the bin.

Debris removal is deliberately conservative. See `analyze_mesh` for the three-part rule and why
each part is load-bearing.
"""

import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from typing import Optional, Sequence

# --- Debris thresholds -------------------------------------------------------------------
# A cluster is dropped only if negligible in area AND detached AND inflating the bounding
# box. All three are needed -- see geometry/README.md for the per-mesh measurements.
DEBRIS_MAX_AREA_FRAC = 1e-3    # cluster surface area below 0.1% of the total is "negligible"
DEBRIS_MIN_SHRINK    = 0.02    # only cut if the AABB diagonal shrinks by >= 2%
DEBRIS_MAX_TRI_FRAC  = 0.05    # candidates above 5% of triangles => assembly file, refuse to cut

# --- Unit heuristic ----------------------------------------------------------------------
# Applied to the debris-free extent. A part whose largest dimension reads between these bounds
# is assumed to be authored in millimetres.
MM_SCALE_MIN, MM_SCALE_MAX = 5.0, 5000.0
MM_TO_M = 0.001


@dataclass
class DebrisCluster:
    """One connected component judged to be export debris."""
    index: int
    n_triangles: int
    area_frac: float
    gap: float                  # distance it sits outside the main component's AABB
    center: np.ndarray


@dataclass
class MeshReport:
    """What `analyze_mesh` found and what the cleaned mesh differs by."""
    n_vertices_before: int = 0
    n_triangles_before: int = 0
    n_vertices_after: int = 0
    n_triangles_after: int = 0

    dup_vertices_removed: int = 0
    degenerate_removed: int = 0
    dup_triangles_removed: int = 0
    unreferenced_removed: int = 0

    n_clusters: int = 0
    debris: list = field(default_factory=list)      # list[DebrisCluster]
    debris_triangles: int = 0
    debris_area_frac: float = 0.0
    refused_debris_removal: bool = False            # assembly-file guard tripped

    extent_before: Optional[np.ndarray] = None
    extent_after: Optional[np.ndarray] = None
    diag_shrink_frac: float = 0.0

    is_watertight: bool = False
    is_edge_manifold: bool = False
    is_orientable: bool = False

    unit_scale: float = 1.0
    fits_in_bin: bool = True
    bin_limit: Optional[np.ndarray] = None

    errors: list = field(default_factory=list)      # fatal problems (empty mesh, NaN verts)

    @property
    def topology_cleaned(self) -> int:
        return (self.dup_vertices_removed + self.degenerate_removed
                + self.dup_triangles_removed + self.unreferenced_removed)

    def has_debris(self) -> bool:
        return bool(self.debris) and not self.refused_debris_removal

    def has_findings(self) -> bool:
        """True when something warrants interrupting the operator.

        Deliberately excludes non-watertight geometry and the mm->m conversion: both are
        routine (most CAD exports are not watertight) and a modal on every import would train
        the operator to dismiss it. They still appear in `summary()`, which is always logged.
        """
        return bool(self.errors) or self.has_debris() or self.refused_debris_removal \
            or not self.fits_in_bin

    def summary(self) -> str:
        """Multi-line report, used for both the console log and the GUI dialog."""
        L = []
        for e in self.errors:
            L.append(f"ERROR: {e}")
        L.append(f"Triangles: {self.n_triangles_before} -> {self.n_triangles_after}"
                 f"   Vertices: {self.n_vertices_before} -> {self.n_vertices_after}")
        if self.topology_cleaned:
            L.append(f"Topology cleanup: {self.dup_vertices_removed} duplicate vertices, "
                     f"{self.degenerate_removed} degenerate triangles, "
                     f"{self.dup_triangles_removed} duplicate triangles, "
                     f"{self.unreferenced_removed} unreferenced vertices removed.")
        L.append(f"Connected components: {self.n_clusters}")

        if self.unit_scale != 1.0:
            L.append(f"Units: detected millimetres, scaling by {self.unit_scale} (mm -> m).")

        if self.has_debris():
            L.append("")
            L.append(f"DEBRIS DETECTED: {len(self.debris)} stray component(s), "
                     f"{self.debris_triangles} triangle(s), "
                     f"{self.debris_area_frac:.2e} of surface area.")
            for d in self.debris[:5]:
                L.append(f"  - {d.n_triangles} tri, area frac {d.area_frac:.2e}, "
                         f"{d.gap:.4f} m outside the part, at "
                         f"[{d.center[0]:.4f} {d.center[1]:.4f} {d.center[2]:.4f}]")
            if len(self.debris) > 5:
                L.append(f"  - ... and {len(self.debris) - 5} more")
            if self.extent_before is not None and self.extent_after is not None:
                L.append(f"  Bounding box {np.round(self.extent_before, 4).tolist()} -> "
                         f"{np.round(self.extent_after, 4).tolist()} "
                         f"({self.diag_shrink_frac:.1%} smaller diagonal)")
            L.append("Removing it fixes bounding-box-driven camera framing and "
                     "partition/tray cell sizing.")
        elif self.refused_debris_removal:
            L.append("")
            L.append(f"WARNING: {len(self.debris)} detached component(s) totalling "
                     f"{self.debris_triangles} triangles look like separate bodies, not debris "
                     f"(they do not inflate the bounding box, or exceed "
                     f"{DEBRIS_MAX_TRI_FRAC:.0%} of the mesh). Left untouched - this file may "
                     f"be an assembly rather than a single part.")

        warn = []
        if not self.is_watertight:
            warn.append("not watertight (convex decomposition uses flood fill, which assumes "
                        "a closed surface - hulls may be degraded)")
        if not self.is_edge_manifold:
            warn.append("not edge-manifold")
        if not self.fits_in_bin and self.bin_limit is not None:
            warn.append(f"larger than the bin {np.round(self.bin_limit, 3).tolist()} m - "
                        f"structured scenes cannot place it")
        if warn:
            L.append("")
            L.append("Warnings: " + "; ".join(warn) + ".")
        return "\n".join(L)


def _cluster_bounds(vertices, triangles, mask):
    """AABB (min, max) of the vertices referenced by the masked triangles."""
    used = np.unique(triangles[mask])
    v = vertices[used]
    return v.min(axis=0), v.max(axis=0)


def analyze_mesh(mesh, bin_limit: Optional[Sequence[float]] = None):
    """Validate and repair an imported mesh.

    Non-destructive: `mesh` is not modified. Returns `(cleaned_mesh, report)` where
    `cleaned_mesh` is a new TriangleMesh with topology hygiene applied and any debris removed.
    Neither the returned mesh nor the original has `unit_scale` applied -- the caller applies
    `report.unit_scale` to whichever variant it decides to keep, so the operator's debris
    choice can never change the resulting scale.

    `bin_limit` is the (w, l, h) the part must fit inside, passed in by the caller so this
    module does not have to import `physics` (which would invert the dependency graph).

    Checks, in order:
      1. non-empty, finite vertex coordinates
      2. topology hygiene: duplicate vertices, degenerate triangles, duplicate triangles,
         unreferenced vertices
      3. connected-component analysis -> debris detection
      4. watertight / edge-manifold / orientable (reported, never auto-repaired)
      5. mm -> m unit heuristic, computed on the DEBRIS-FREE extent
      6. bin-fit check
    """
    report = MeshReport()

    cleaned = o3d.geometry.TriangleMesh(mesh)
    V = np.asarray(cleaned.vertices)
    T = np.asarray(cleaned.triangles)
    report.n_vertices_before = len(V)
    report.n_triangles_before = len(T)

    if len(V) == 0 or len(T) == 0:
        report.errors.append("mesh is empty (no vertices or no triangles)")
        return cleaned, report
    if not np.isfinite(V).all():
        n_bad = int((~np.isfinite(V)).any(axis=1).sum())
        report.errors.append(f"{n_bad} vertex coordinate(s) are NaN or infinite")
        return cleaned, report

    if report.extent_before is None:
        report.extent_before = V.max(axis=0) - V.min(axis=0)

    # --- 2. topology hygiene -------------------------------------------------------------
    # STL is a vertex soup (~5.8x duplication measured on real exports). Merging restores the
    # face adjacency that facet grouping and every topology predicate below depend on.
    n_v, n_t = len(cleaned.vertices), len(cleaned.triangles)
    cleaned.remove_duplicated_vertices()
    report.dup_vertices_removed = n_v - len(cleaned.vertices)

    n_t = len(cleaned.triangles)
    cleaned.remove_degenerate_triangles()
    report.degenerate_removed = n_t - len(cleaned.triangles)

    n_t = len(cleaned.triangles)
    cleaned.remove_duplicated_triangles()
    report.dup_triangles_removed = n_t - len(cleaned.triangles)

    n_v = len(cleaned.vertices)
    cleaned.remove_unreferenced_vertices()
    report.unreferenced_removed = n_v - len(cleaned.vertices)

    # --- 3. connected components / debris ------------------------------------------------
    V = np.asarray(cleaned.vertices)
    T = np.asarray(cleaned.triangles)
    if len(T) == 0:
        report.errors.append("no triangles left after removing degenerate geometry")
        report.n_vertices_after = len(V)
        report.n_triangles_after = 0
        return cleaned, report

    idx, _n_per, area_per = cleaned.cluster_connected_triangles()
    idx = np.asarray(idx)
    area_per = np.asarray(area_per)
    report.n_clusters = len(area_per)
    total_area = float(area_per.sum())

    candidates = []
    if report.n_clusters > 1 and total_area > 0:
        main = int(np.argmax(area_per))
        main_lo, main_hi = _cluster_bounds(V, T, idx == main)
        for c in range(report.n_clusters):
            if c == main:
                continue
            area_frac = float(area_per[c] / total_area)
            if area_frac >= DEBRIS_MAX_AREA_FRAC:
                continue                      # not negligible
            lo, hi = _cluster_bounds(V, T, idx == c)
            # positive only if the cluster lies wholly outside the main component's AABB
            gap = float(np.maximum(main_lo - hi, np.maximum(lo - main_hi, 0.0)).max())
            if gap <= 0.0:
                continue                      # nested / touching shell, legitimate geometry
            n_tri = int((idx == c).sum())
            candidates.append(DebrisCluster(index=c, n_triangles=n_tri, area_frac=area_frac,
                                            gap=gap, center=(lo + hi) / 2.0))

    if candidates:
        keep = np.ones(len(T), dtype=bool)
        for d in candidates:
            keep &= (idx != d.index)

        ext_before = V.max(axis=0) - V.min(axis=0)
        kept_v = V[np.unique(T[keep])]
        ext_after = kept_v.max(axis=0) - kept_v.min(axis=0)
        diag_before = float(np.linalg.norm(ext_before))
        diag_after = float(np.linalg.norm(ext_after))
        shrink = 0.0 if diag_before < 1e-12 else (diag_before - diag_after) / diag_before

        report.debris = candidates
        report.debris_triangles = sum(d.n_triangles for d in candidates)
        report.debris_area_frac = sum(d.area_frac for d in candidates)
        report.diag_shrink_frac = shrink
        report.extent_before = ext_before

        tri_frac = report.debris_triangles / len(T)
        if shrink >= DEBRIS_MIN_SHRINK and tri_frac < DEBRIS_MAX_TRI_FRAC:
            cleaned.remove_triangles_by_index(np.flatnonzero(~keep).tolist())
            cleaned.remove_unreferenced_vertices()
            report.extent_after = ext_after
        else:
            # Detached but harmless (does not inflate the bbox), or so large that this is an
            # assembly file rather than a part with debris. Report, do not cut.
            report.refused_debris_removal = True
            report.extent_after = ext_before
            report.diag_shrink_frac = shrink

    V = np.asarray(cleaned.vertices)
    T = np.asarray(cleaned.triangles)
    report.n_vertices_after = len(V)
    report.n_triangles_after = len(T)
    if report.extent_after is None:
        report.extent_after = V.max(axis=0) - V.min(axis=0)

    # --- 4. manifold / watertight (reported, never auto-repaired) ------------------------
    report.is_watertight = bool(cleaned.is_watertight())
    report.is_edge_manifold = bool(cleaned.is_edge_manifold())
    report.is_orientable = bool(cleaned.is_orientable())

    # --- 5. unit heuristic on the debris-free extent -------------------------------------
    extent_max = float(np.asarray(report.extent_after).max())
    if MM_SCALE_MIN < extent_max < MM_SCALE_MAX:
        report.unit_scale = MM_TO_M

    # --- 6. bin fit ----------------------------------------------------------------------
    if bin_limit is not None:
        limit = np.asarray(bin_limit, dtype=float).ravel()[:3]
        report.bin_limit = limit
        scaled_extent = np.asarray(report.extent_after, dtype=float) * report.unit_scale
        # Compare the part's sorted extents against the sorted bin extents: the part may be
        # placed in any axis-aligned orientation, so only the shape ordering matters.
        report.fits_in_bin = bool(np.all(np.sort(scaled_extent) <= np.sort(limit)))

    return cleaned, report
