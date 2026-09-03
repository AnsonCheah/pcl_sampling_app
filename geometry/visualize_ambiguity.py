"""Inspect the ambiguity analysis for a part: heat-mapped cloud plus ranked axes.

Run:
    python geometry/visualize_ambiguity.py 25333MB000
    python geometry/visualize_ambiguity.py 25333MB000.STL --views 200
    python geometry/visualize_ambiguity.py path/to/part_surface.ply --mesh path/to/part.stl
    python geometry/visualize_ambiguity.py --primitive hex_prism
    python geometry/visualize_ambiguity.py 25333MB000 --save out.png --no-show

What you are looking at
    Point cloud   heat ramp over the per-point discriminative score.  HOT points are
                  explained by no ambiguity transform, so they are what actually pins the
                  pose down; COOL points are interchangeable with somewhere else on the
                  model and contribute nothing.  A part that is mostly cool is a part the
                  matcher can flip.
    Rods          the ranked ambiguity axes, hottest = rank 0 = most viewpoints affected.
    Knobs         each rod's closest point to the part centre -- i.e. where the axis
                  actually sits.
    White sphere  the cloud centroid.        Blue sphere  the AABB centre.

If a knob does not sit on the white sphere, the axis misses the centroid, and no analysis
that rotates about centroid or AABB axes could have found it.  That is the case this whole
module exists for, so it is worth looking at first.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d

from geometry.ambiguity import (
    AmbiguityConfig,
    ambiguity_geometries,
    analyse_ambiguity,
)
from geometry.geom_utils import pairing_error

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_ROOT = os.path.join(REPO, "output", "reference_pcd")
MM_ROOT = os.path.join(REPO, "MM_Optimizer", "CAD_Match", "resource", "3d_matching")


def _primitive(name: str):
    """Known-answer shapes, handy for sanity-checking the viewer itself."""
    import trimesh

    from geometry.geom_utils import trimesh_to_o3d

    def prism(sides, radius=0.025, height=0.050):
        ring = [(radius * np.cos(a), radius * np.sin(a))
                for a in np.linspace(0.0, 2.0 * np.pi, sides + 1)[:-1]]
        poly = trimesh.path.polygons.Polygon(ring)
        return trimesh_to_o3d(trimesh.creation.extrude_polygon(poly, height))

    def offset_axis():
        """Cylinder on +Z through the origin with a block dragging the centroid away."""
        cyl = trimesh.creation.cylinder(radius=0.015, height=0.060)
        blk = trimesh.creation.box((0.040, 0.012, 0.012))
        blk.apply_translation((0.033, 0.0, 0.018))
        return trimesh_to_o3d(trimesh.util.concatenate([cyl, blk]))

    table = {
        "cylinder": lambda: o3d.geometry.TriangleMesh.create_cylinder(0.020, 0.080, resolution=64),
        "sphere": lambda: o3d.geometry.TriangleMesh.create_sphere(0.030, resolution=30),
        "cone": lambda: o3d.geometry.TriangleMesh.create_cone(0.025, 0.060, resolution=60),
        "box": lambda: o3d.geometry.TriangleMesh.create_box(0.100, 0.030, 0.020),
        "torus": lambda: o3d.geometry.TriangleMesh.create_torus(0.030, 0.010, 60, 30),
        "tri_prism": lambda: prism(3, 0.030),
        "square_prism": lambda: prism(4),
        "hex_prism": lambda: prism(6),
        "offset_axis": offset_axis,
    }
    if name not in table:
        sys.exit(f"Unknown primitive '{name}'. Choose from: {', '.join(sorted(table))}")
    mesh = table[name]()
    mesh.translate(-mesh.get_axis_aligned_bounding_box().get_center())
    mesh.compute_vertex_normals()
    return mesh, None


def _mesh_candidates(ply_path: str, stem: str):
    """Meshes that might belong to ``ply_path``, nearest-provenance first.

    A reference bundle is written by ``SaveStage`` as ``<part>/<part>.stl`` plus
    ``<part>/<part>_<type>/<part>_<type>.ply``, both transformed into the *same* model frame
    in the same run.  So the bundle's own STL is the only mesh guaranteed to match its
    cloud; everything after it is a guess and has to survive ``pairing_error``.
    """
    d = os.path.dirname(os.path.abspath(ply_path))
    return [os.path.join(os.path.dirname(d), f"{stem}.stl"),   # bundle root -- exported together
            os.path.join(d, f"{stem}.stl"),
            os.path.join(REF_ROOT, stem, f"{stem}.stl"),
            os.path.join(REPO, f"{stem}.STL"),
            os.path.join(REPO, f"{stem}.stl")]


def _resolve(target: str):
    """Accept a part name, an STL, or a reference PLY. Returns (mesh, pcd_or_None).

    ``REF_ROOT`` is searched before ``MM_ROOT``: the reference bundle is the source of
    truth and carries its own STL, whereas the MechVision resource folder is a deployed
    *copy* with no mesh beside it, so its cloud can only ever be paired by guessing -- and
    it goes stale the moment a part is re-exported into a new model frame.  It is still
    searched last, but the pairing check is what actually decides.
    """
    if os.path.isfile(target):
        path = target
    else:
        stem = os.path.splitext(os.path.basename(target))[0]
        for cand in (os.path.join(REF_ROOT, stem, f"{stem}_surface", f"{stem}_surface.ply"),
                     os.path.join(REF_ROOT, stem, f"{stem}.stl"),
                     os.path.join(REPO, f"{stem}.STL"),
                     os.path.join(REPO, f"{stem}.stl"),
                     # The deployed MechVision model, written by MM_Optimizer/model_sync.py.
                     os.path.join(MM_ROOT, stem, f"{stem}.ply")):
            if os.path.isfile(cand):
                path = cand
                break
        else:
            sys.exit(f"Could not find a mesh or reference cloud for '{target}'")

    print(f"Source: {path}")
    if path.lower().endswith(".ply"):
        pcd = o3d.io.read_point_cloud(path)
        stem = os.path.basename(path).replace("_surface.ply", "").replace(".ply", "")
        rejected = []
        for cand in _mesh_candidates(path, stem):
            if not os.path.isfile(cand):
                continue
            mesh = o3d.io.read_triangle_mesh(cand)
            mesh.compute_vertex_normals()
            why = pairing_error(mesh, pcd)
            if why is None:
                print(f"Mesh:   {cand}")
                return mesh, pcd
            rejected.append(f"  {cand}\n    {why}")
        if rejected:
            sys.exit("None of the meshes found for '{}' match this cloud:\n{}\n\n"
                     "The cloud and the mesh must be in the same model frame. A bundle under\n"
                     "output/reference_pcd/ always is; a copy elsewhere (e.g. the MechVision\n"
                     "3d_matching resource) goes stale as soon as the part is re-exported into\n"
                     "an ambiguity-aligned frame. Re-export the part, or pass --mesh explicitly."
                     .format(stem, "\n".join(rejected)))
        sys.exit(f"Found the cloud but no mesh for '{stem}'; pass --mesh explicitly "
                 f"(the raycast visibility sweep needs the surface)")

    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    return mesh, None


def _prepare(mesh, pcd, n_points):
    """Centre the part and, when only a mesh was given, sample a cloud from it."""
    if pcd is None:
        o3d.utility.random.seed(0)
        pcd = mesh.sample_points_uniformly(n_points, use_triangle_normal=False)
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(15)

    # The analysis is translation-equivariant, but centring keeps the reported axis
    # offsets readable against the centroid/AABB markers.
    shift = -mesh.get_axis_aligned_bounding_box().get_center()
    mesh.translate(shift)
    pcd.translate(shift)
    return mesh, pcd


def _report(profile, legend):
    print(f"\n  epsilon        {profile.epsilon_m * 1000:.2f} mm"
          f"   (diameter {profile.diameter_m * 1000:.1f} mm)")
    print(f"  axes found     {len(profile.axes)}  ({profile.n_significant_axes} significant)")
    print(f"  discriminative {profile.discriminative_fraction:.3f}"
          f"   -> {(1 - profile.discriminative_fraction) * 100:.0f}% of the visible surface "
          f"is explained by an ambiguity transform")
    n_amb = sum(1 for v in profile.per_view if v.n_transforms > 0)
    print(f"  ambiguous views {n_amb}/{len(profile.per_view)}")
    if profile.ppf_degeneracy:
        print(f"  ppf entropy    {profile.ppf_degeneracy.get('entropy', float('nan')):.3f}"
              f"   (diagnostic only; low = flat patches PPF cannot separate)")
    if not legend:
        print("\n  No ambiguity axes found -- pose should be uniquely determined.")
        return
    print(f"\n  {'rank':>4} {'colour':<18} {'fold':>10} {'step':>7} {'glob':>5} "
          f"{'views':>6} {'patch':>6} {'off-centroid':>13} {'off-aabb':>9}")
    for row in legend:
        rgb = "".join(f"{int(c * 255):>4}" for c in row["colour"])
        step = "-" if row["angle_step_deg"] == 0 else f"{row['angle_step_deg']:.0f}deg"
        print(f"  {row['rank']:>4} rgb({rgb.strip():<12}) {row['fold']:>10} {step:>7} "
              f"{str(row['is_global']):>5} {row['view_fraction']:>6.2f} "
              f"{row['patch_fraction']:>6.2f} "
              f"{row['offset_from_centroid_mm']:>10.2f} mm {row['offset_from_aabb_centre_mm']:>6.2f} mm")
    worst = max(legend, key=lambda r: r["offset_from_centroid_mm"])
    if worst["offset_from_centroid_mm"] > 1.0:
        print(f"\n  Axis {worst['rank']} sits {worst['offset_from_centroid_mm']:.2f} mm off the "
              f"centroid -- a centroid/OBB-based analysis cannot represent it.")


def _rank_sweep(profile, exponents=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)):
    """Show how the exponent reorders the axes, without re-running the analysis.

    Ranking is a pure function of stored per-axis metrics, so sweeping it is free. Use
    this to see whether a part's ordering is stable or sits on a knife edge.
    """
    ident = {id(ax): i for i, ax in enumerate(profile.axes)}
    print(f"\n  per-axis metrics (label = rank at the exponent the profile was built with)")
    print(f"  {'axis':>4} {'fold':>10} {'views':>6} {'area':>6} {'patch':>6}")
    for i, ax in enumerate(profile.axes):
        fold = "continuous" if ax.fold == 0 else (f"C{ax.fold}" if ax.fold > 1 else "none")
        print(f"  {i:>4} {fold:>10} {ax.view_fraction:>6.2f} "
              f"{ax.area_fraction:>6.2f} {ax.patch_fraction:>6.2f}")

    print(f"\n  {'exponent':>8} | order of axes by score | dominant")
    for k in exponents:
        scored = sorted(profile.axes,
                        key=lambda a: -(a.view_fraction * a.area_fraction ** k))
        order = [ident[id(a)] for a in scored]
        top = scored[0]
        fold = "continuous" if top.fold == 0 else (f"C{top.fold}" if top.fold > 1 else "none")
        step = "-" if top.fold == 0 else f"{top.angle_step_deg():.0f}deg"
        print(f"  {k:>8.1f} | {str(order):<22} | axis {ident[id(top)]} "
              f"({fold}, step {step})")
    print("\n  k=0 ranks by frequency only (equivalent to assuming verification ignores")
    print("  the unseen model); k>1 favours axes whose wrong pose still overlaps the")
    print("  model enough to survive verification.")


def _show(geoms, title, save, show, lookat):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1400, height=900, visible=show)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.10, 0.10, 0.13])
    opt.point_size = 4.0
    opt.mesh_show_back_face = True
    opt.light_on = True

    # A three-quarter view; the default camera looks straight down an axis, which is the
    # one direction that hides whether an axis is offset.
    vc = vis.get_view_control()
    vc.set_lookat(lookat)
    vc.set_front([0.62, -0.66, 0.42])
    vc.set_up([0.0, 0.0, 1.0])
    vc.set_zoom(0.72)

    vis.poll_events()
    vis.update_renderer()
    if save:
        vis.capture_screen_image(save, do_render=True)
        print(f"\n  saved {save}")
    if show:
        vis.run()
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", help="part name, .STL path, or reference .ply path")
    ap.add_argument("--primitive", help="use a known-answer shape instead of a file")
    ap.add_argument("--mesh", help="explicit mesh path (needed if the cloud has no sibling STL)")
    ap.add_argument("--views", type=int, default=200, help="viewpoints in the sweep")
    ap.add_argument("--res", type=int, default=256, help="raycast resolution per view")
    ap.add_argument("--points", type=int, default=12000,
                    help="points sampled when the source is a mesh")
    ap.add_argument("--max-axes", type=int, default=6, help="axes to draw")
    ap.add_argument("--f-tau", type=float, default=None,
                    help="override the survival threshold (default 0.40)")
    ap.add_argument("--rank-exp", type=float, default=None,
                    help="area exponent in the ranking score (default 2.0); "
                         "0 = rank by frequency only, >1 = weight surviving verification")
    ap.add_argument("--rank-sweep", action="store_true",
                    help="print how the ranking changes across exponents, then exit")
    ap.add_argument("--no-markers", action="store_true",
                    help="hide the axis knobs and centroid/AABB reference spheres")
    ap.add_argument("--save", help="write a PNG here")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    args = ap.parse_args()

    if not args.target and not args.primitive:
        ap.error("give a part/path, or --primitive")

    if args.primitive:
        mesh, pcd = _primitive(args.primitive)
        label = args.primitive
    else:
        mesh, pcd = _resolve(args.target)
        label = os.path.splitext(os.path.basename(args.target))[0]
    if args.mesh:
        mesh = o3d.io.read_triangle_mesh(args.mesh)
        mesh.compute_vertex_normals()

    # Re-checked here, not only inside _resolve, because --mesh overrides whatever _resolve
    # validated and is exactly the path an operator reaches for after hitting the error.
    if pcd is not None:
        why = pairing_error(mesh, pcd)
        if why is not None:
            sys.exit(f"Mesh and cloud are not the same part in the same frame: {why}\n"
                     f"The visibility sweep raycasts the mesh and snaps hits onto the cloud, "
                     f"so this would report ambiguity axes fitted to a sliver of points.")

    mesh, pcd = _prepare(mesh, pcd, args.points)
    print(f"Part:   {label}   {len(pcd.points)} points, {len(mesh.triangles)} triangles")

    kwargs = dict(n_views=args.views, res=args.res)
    if args.f_tau is not None:
        kwargs["f_tau"] = args.f_tau
    if args.rank_exp is not None:
        kwargs["rank_area_exponent"] = args.rank_exp
    print(f"Analysing with {args.views} viewpoints at {args.res}x{args.res} ...")
    profile = analyse_ambiguity(mesh, pcd, AmbiguityConfig(**kwargs))

    if args.rank_sweep:
        _rank_sweep(profile)
        return

    geoms, legend = ambiguity_geometries(pcd, profile, max_axes=args.max_axes,
                                         show_markers=not args.no_markers)
    _report(profile, legend)

    lookat = np.asarray(pcd.points).mean(axis=0)
    _show(geoms, f"ambiguity - {label}", args.save, not args.no_show, lookat)


if __name__ == "__main__":
    main()
