"""End-to-end PPF demo: CAD mesh in, per-instance poses out, overlay shown in 3D.

    python registration/demo/demo_match.py

Opens an interactive 3D window at the end: drag to orbit, scroll to zoom, Q to close.

The five steps below are the whole API. Everything else in this file is loading and colouring.

    1. load the CAD mesh, in metres
    2. sample it into a reference cloud          <- NOT the mesh vertices; see below
    3. PPFConfig.derive(cloud)                   <- no per-part tuning
    4. PPFModel.train(cloud)                     <- once per part
    5. match each segmented instance             <- once per instance

Two things this demo exists to get right, because both are silent when wrong:

**Derive from a SAMPLED cloud, never from mesh vertices.** `diameter` does not care (it is a
convex-hull property: 152.50 mm from vertices vs 152.43 mm from the cloud), but `tau` cares a
lot -- 4.82 mm from vertices against 10.41 mm from the cloud, a 2.2x error. Mesh vertices
cluster where the tessellator needed detail, so they are not a uniform sample of the surface,
and `PPFConfig.derive` bisects `tau` on the voxel-downsampled count of whatever it is given.
Model points go as `tau^-2` and work as their square, so 2.2x on tau is ~23x on runtime.

**Units.** T-LESS ships millimetres; these scenes are in metres. Unscaled, the derived
diameter comes back as 152.5 *metres* and every sensor-derived floor is meaningless. There is
no unit tag in a PLY or an STL, so it has to be stated -- hence `--mesh-scale`.

Upstream segmentation is assumed done: `scene_00000/sample_*.ply` are the segmented instance
clouds, one file per object. This demo never looks at `sample_*.npz` (the ground truth). It
reports each pose's *verification score* -- the fraction of that instance's points the placed
model explains -- which needs no ground truth and is what you would have in production.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from registration.ppf import PPFConfig, PPFModel, downsample, match_many

HERE = os.path.dirname(os.path.abspath(__file__))

# Distinct colours for the overlaid model, cycled per instance. The scene stays grey so the
# eye reads "grey = measured, colour = where PPF thinks the part is".
PALETTE = np.array([
    [0.90, 0.25, 0.21], [0.16, 0.65, 0.95], [0.95, 0.70, 0.13], [0.30, 0.76, 0.35],
    [0.72, 0.38, 0.85], [0.95, 0.45, 0.70], [0.20, 0.80, 0.75], [0.98, 0.55, 0.25],
])


def load_reference(mesh_path: str, mesh_scale: float, n_points: int):
    """Steps 1-2: CAD mesh -> a uniformly sampled reference cloud with normals."""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.triangles) == 0:
        raise SystemExit(f"{mesh_path} has no triangles - is it a point cloud?")
    mesh.scale(mesh_scale, center=(0.0, 0.0, 0.0))
    # Sampling needs vertex normals to interpolate from; a raw CAD PLY often has none.
    mesh.compute_vertex_normals()

    o3d.utility.random.seed(0)                      # reproducible demo
    pcd = mesh.sample_points_uniformly(n_points, use_triangle_normal=False)
    return mesh, pcd


def load_instances(scene_dir: str, normal_radius: float, camera):
    """Load the segmented clusters produced upstream, and estimate their normals.

    Normals are estimated from the points rather than read from the file on purpose: a real
    sensor delivers depth, and normals get estimated from noisy points. The stored normals in
    these PLYs came from the mesh via raycast and are exact, which would flatter the matcher.

    Orientation matters as much as direction -- PPF's features use the signed angle between a
    normal and the pair direction, so a flipped normal is a *different* feature, not a near
    one. Everything was seen from the camera, so that is what they are oriented towards.
    """
    paths = sorted(glob.glob(os.path.join(scene_dir, "sample_*.ply")),
                   key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))
    out = []
    for p in paths:
        pcd = o3d.io.read_point_cloud(p)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius,
                                                                  max_nn=30))
        pcd.orient_normals_towards_camera_location(camera)
        out.append((os.path.basename(p),
                    np.asarray(pcd.points), np.asarray(pcd.normals)))
    return out


def build_overlay(instances, results, ref_pts):
    """Grey measured points + the reference cloud placed at each estimated pose.

    ``ref_pts`` is the full sampled reference cloud (``ref_cloud.ply``), not the ~500-point
    model the matcher trains on -- the model is decimated to ``tau`` for speed and is far too
    sparse to read as a shape.

    A correct pose puts the coloured cloud *on* the grey one, so the two interleave and the
    part reads as a speckled mix; colour standing alone is model surface the sensor could not
    see (occluded or self-occluded), and grey standing alone is measured surface the pose
    fails to explain. That second case is the one to look for.
    """
    geoms = []
    for i, ((name, pts, _), res) in enumerate(zip(instances, results)):
        cluster = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        cluster.paint_uniform_color([0.78, 0.78, 0.82])
        geoms.append(cluster)

        if res.best is None:
            continue
        T = res.best.T
        placed = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(ref_pts @ T[:3, :3].T + T[:3, 3]))
        placed.paint_uniform_color(PALETTE[i % len(PALETTE)])
        geoms.append(placed)
    return geoms


def show_3d(geoms):
    """Open an interactive 3D window. Drag to orbit, scroll to zoom, Q to quit.

    Uses the low-level Visualizer rather than ``draw_geometries`` only to set point size and
    background -- at the default size the points are fat enough to hide each other, and how
    the two clouds interleave is the one thing this window exists to show.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1000,
                      window_name="PPF overlay - grey = measured scene, colour = placed model")
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 1.6
    opt.background_color = np.array([0.09, 0.09, 0.11])
    vis.run()                 # blocks until the window is closed
    vis.destroy_window()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", default=os.path.join(HERE, "ref_mesh.ply"))
    ap.add_argument("--scene", default=os.path.join(HERE, "scene_00000"))
    ap.add_argument("--mesh-scale", type=float, default=1e-3,
                    help="mesh units -> metres. T-LESS ships mm, so 0.001 (the default)")
    ap.add_argument("--sample-points", type=int, default=50_000,
                    help="points sampled from the mesh; only needs to be dense enough that "
                         "tau is limited by the compute budget rather than by sampling")
    ap.add_argument("--model-points", type=int, default=500,
                    help="target size of the trained model - the one compute knob")
    ap.add_argument("--camera-z", type=float, default=1.5, help="camera height, for normals")
    ap.add_argument("--no-show", action="store_true",
                    help="skip the 3D window (for running this without a display)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    # --- 1 + 2: mesh -> reference cloud -------------------------------------------------
    mesh, ref = load_reference(args.mesh, args.mesh_scale, args.sample_points)
    ref_path = os.path.join(HERE, "ref_cloud.ply")
    o3d.io.write_point_cloud(ref_path, ref)
    ref_pts = np.asarray(ref.points)
    ref_nrm = np.asarray(ref.normals)
    print(f"mesh      {os.path.basename(args.mesh)}: {len(mesh.triangles)} triangles, "
          f"scaled by {args.mesh_scale:g}")
    print(f"reference {len(ref_pts)} points sampled -> {os.path.basename(ref_path)}")

    # --- 3: derive every parameter from the cloud + a sensor calibration -----------------
    cfg = PPFConfig.derive(ref_pts, model_target_points=args.model_points)
    print("\nderived parameters (no per-part tuning):")
    print(cfg.describe())

    # --- 4: train once for the part ------------------------------------------------------
    m_pts, m_nrm = downsample(ref_pts, ref_nrm, cfg.tau)
    model = PPFModel.train(m_pts, m_nrm, cfg)
    print(f"\nmodel     {model.n_points} points, {len(model.keys):,} table entries")

    # --- 5: match every segmented instance ----------------------------------------------
    # The normal-estimation radius is tied to the matcher's own quantisation (2 * tau). It is
    # not cosmetic: it moved recall by ~15 points in the benchmark when changed.
    instances = load_instances(args.scene, 2.0 * cfg.tau,
                               np.array([0.0, 0.0, args.camera_z]))
    if not instances:
        raise SystemExit(f"no sample_*.ply under {args.scene}")
    print(f"scene     {len(instances)} segmented instances from "
          f"{os.path.basename(args.scene)}\n")

    results = match_many(model, [(p, n) for _, p, n in instances], workers=args.workers)

    print(f"{'instance':<14}{'points':>8}{'score':>8}{'margin':>8}   position (mm)")
    for (name, pts, _), res in zip(instances, results):
        if res.best is None:
            print(f"{name:<14}{len(pts):>8}{'-':>8}{'-':>8}   NO POSE")
            continue
        t = res.best.T[:3, 3] * 1e3
        print(f"{name:<14}{len(pts):>8}{res.best.score:>8.3f}{res.best.peak_margin:>8.3f}"
              f"   [{t[0]:7.1f} {t[1]:7.1f} {t[2]:7.1f}]")

    found = [r for r in results if r.best is not None]
    if found:
        mean_score = float(np.mean([r.best.score for r in found]))
        print(f"\n{len(found)}/{len(results)} instances matched, mean score {mean_score:.3f}")
    else:
        print("\nno instances matched")
    print("score  = fraction of the instance's points the placed model explains "
          "(needs no ground truth)")
    print("margin = 1 - runner_up/peak; low means the accumulator nearly picked "
          "a different pose")

    if args.no_show:
        return
    show_3d(build_overlay(instances, results, ref_pts))


if __name__ == "__main__":
    main()
