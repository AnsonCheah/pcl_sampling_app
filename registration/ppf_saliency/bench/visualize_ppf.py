"""Overlay PPF-estimated poses on a synthetic scene, one panel per weighting arm.

Run::

    python -m registration.ppf_saliency.bench.visualize_ppf                          # bunny, scene 0, both arms
    python -m registration.ppf_saliency.bench.visualize_ppf --part X --scene 1
    python -m registration.ppf_saliency.bench.visualize_ppf --arms uniform           # single panel
    python -m registration.ppf_saliency.bench.visualize_ppf --save out.png --no-show

What you are looking at
    Grey cloud      the raw synthetic scene, exactly as the sensor simulation produced it.
    Coloured clouds the reference model placed at each ESTIMATED pose.  If the pose is right
                    the colour sits flush on the grey; if it is wrong the model visibly
                    floats, sinks, or spins away from the points it claims to explain.  No
                    ground-truth rendering is needed to see that — the scene points *are*
                    the evidence.
    Colour          green within 2 mm / 5 deg, amber within 5 mm / 10 deg, red beyond.
                    (``--color-by instance`` reverts to one hue per instance if you want to
                    trace individual parts instead of judging accuracy.)
    Triads          dim = ground truth, bright = estimate, joined by a yellow error line.
                    A long yellow line with a well-aligned model usually means a symmetry
                    flip rather than a miss.
    Panels          one per arm, laid out along +X in the order given by ``--arms``.

Caveat on symmetric parts
    Error classes here are the plain pose error against a single ground-truth
    representative; they are NOT quotiented by the part's symmetry group.  On a symmetric
    part a physically correct pose can therefore be coloured red.  Judge those by whether
    the coloured cloud sits on the grey one, not by the colour.  The symmetry-aware scoring
    lives in the benchmark metrics, not in this viewer.
"""

from __future__ import annotations

import argparse
import colorsys
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

# Repo root, four levels up: registration/ppf_saliency/bench/visualize_ppf.py
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))

from registration.ppf_saliency.bench import SYNTH_ROOT
from registration.ppf.bench.dataset import (list_parts, list_scenes, load_reference,
                                            load_scene)
from registration.ppf_saliency import PPFConfig, PPFModel, downsample, match  # noqa: E402
from registration.ppf_saliency.saliency import ppf_saliency, transfer_weights

# Repo-wide accuracy gates (MM_Optimizer/search_config.py).
TIGHT = (0.002, 5.0)
LOOSE = (0.005, 10.0)

GREEN = (0.20, 0.85, 0.35)
AMBER = (0.98, 0.70, 0.15)
RED = (0.92, 0.22, 0.22)
GREY = (0.55, 0.55, 0.55)


# ── arms ──────────────────────────────────────────────────────────────────────

def build_arms(model: PPFModel, names: List[str], scene) -> Dict[str, PPFModel]:
    """Map arm name -> a model carrying that arm's vote weights."""
    out: Dict[str, PPFModel] = {}
    for name in names:
        if name == "uniform":
            out[name] = model
        elif name == "ppf":
            out[name] = model.with_weights(ppf_saliency(model))
        elif name == "ambiguity":
            w = _ambiguity_weights(model, scene)
            if w is None:
                print("[skip] arm 'ambiguity': no per-point heat map available. It is never "
                      "persisted, so it has to be recomputed from the mesh — pass "
                      "--mesh <part.stl>.")
                continue
            out[name] = model.with_weights(w)
        else:
            raise ValueError(f"unknown arm {name!r}")
    return out


def _ambiguity_weights(model: PPFModel, scene) -> Optional[np.ndarray]:
    mesh_path = getattr(scene, "_mesh_path", None)
    if not mesh_path or not os.path.exists(mesh_path):
        return None
    from geometry.ambiguity import AmbiguityConfig, analyse_ambiguity

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene.ref_points)
    pcd.normals = o3d.utility.Vector3dVector(scene.ref_normals)
    print("  analysing pose ambiguity (this takes a minute)...")
    profile = analyse_ambiguity(mesh, pcd, AmbiguityConfig())
    disc = profile.per_point_discriminative
    if disc.size != len(scene.ref_points):
        return None
    # The heat map lives on the full reference cloud; the model is voxel-downsampled from
    # it, so the two are not index-aligned however tempting that assumption is.
    return transfer_weights(scene.ref_points, disc, model.points)


# ── geometry helpers (visual language shared with MM_Optimizer/visualize_match.py) ──

def _instance_color(i: int) -> Tuple[float, float, float]:
    return colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.75, 0.98)


def _triads(transforms: List[np.ndarray], length: float, dim: bool) -> o3d.geometry.LineSet:
    base_cols = [(1.0, 0.25, 0.25), (0.25, 1.0, 0.25), (0.35, 0.45, 1.0)]
    cols = [tuple(0.35 * c for c in col) for col in base_cols] if dim else base_cols
    pts, lines, colours = [], [], []
    for T in transforms:
        o = T[:3, 3]
        b = len(pts)
        pts.append(o)
        for a in range(3):
            pts.append(o + T[:3, a] * length)
            lines.append([b, b + 1 + a])
            colours.append(cols[a])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts, float).reshape(-1, 3))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, np.int32).reshape(-1, 2))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colours, float).reshape(-1, 3))
    return ls


def _error_lines(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> o3d.geometry.LineSet:
    pts, lines, cols = [], [], []
    for a, b in pairs:
        i = len(pts)
        pts.extend([a, b])
        lines.append([i, i + 1])
        cols.append((1.0, 1.0, 0.0))
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts, float).reshape(-1, 3))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, np.int32).reshape(-1, 2))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(cols, float).reshape(-1, 3))
    return ls


def _pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
    E = np.linalg.inv(T_gt) @ T_est
    ang = np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
    return float(np.linalg.norm(E[:3, 3])), float(ang)


def _class_color(pos: float, ang: float) -> Tuple[float, float, float]:
    if pos < TIGHT[0] and ang < TIGHT[1]:
        return GREEN
    if pos < LOOSE[0] and ang < LOOSE[1]:
        return AMBER
    return RED


# ── per-arm pass ──────────────────────────────────────────────────────────────

def backdrop_points(scene, mode: str, voxel: float) -> np.ndarray:
    """The grey cloud a panel is drawn against.

    Defaults to the union of the *segmented instances* rather than ``scene.ply``. The full
    bin cloud is ~1.25 M points of which the floor and walls are the overwhelming majority,
    so on a 760 mm bin holding 110 mm parts the parts are visually lost in background the
    matcher never even saw — it is handed one cluster at a time.
    """
    if mode == "none":
        return np.empty((0, 3))
    if mode == "full":
        pts = scene.scene_points
    else:
        pts = (np.concatenate([i.points for i in scene.instances])
               if scene.instances else np.empty((0, 3)))
    if voxel > 0 and len(pts):
        # Reuse the matcher's downsampler rather than re-deriving voxel binning here; it is
        # the same operation and it already handles the Open3D details.
        pts = downsample(pts, np.tile([0.0, 0.0, 1.0], (len(pts), 1)), voxel)[0]
    return pts


def run_arm(name: str, model: PPFModel, scene, ref_overlay: np.ndarray,
            offset: np.ndarray, color_by: str, backdrop: np.ndarray):
    """Match every instance with this arm and build its panel's geometry."""
    geoms: List = []
    if len(backdrop):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(backdrop + offset)
        cloud.paint_uniform_color(GREY)
        geoms.append(cloud)

    axis_len = 0.35 * model.cfg.diameter
    gt_T, est_T, err_pairs, rows = [], [], [], []
    elapsed = 0.0

    for k, inst in enumerate(scene.instances):
        t0 = time.perf_counter()
        best = match(model, inst.points, inst.normals).best
        elapsed += time.perf_counter() - t0

        if best is None:
            rows.append((inst.index, inst.overlap, None, None, 0.0, 0.0))
            continue
        pos, ang = _pose_error(best.T, inst.T_gt)
        rows.append((inst.index, inst.overlap, pos, ang, best.score, best.peak_margin))

        col = _class_color(pos, ang) if color_by == "error" else _instance_color(k)
        placed = o3d.geometry.PointCloud()
        placed.points = o3d.utility.Vector3dVector(
            ref_overlay @ best.T[:3, :3].T + best.T[:3, 3] + offset)
        placed.paint_uniform_color(col)
        geoms.append(placed)

        Tg, Te = inst.T_gt.copy(), best.T.copy()
        Tg[:3, 3] += offset
        Te[:3, 3] += offset
        gt_T.append(Tg)
        est_T.append(Te)
        err_pairs.append((Tg[:3, 3], Te[:3, 3]))

    if gt_T:
        geoms.append(_triads(gt_T, axis_len, dim=True))
        geoms.append(_triads(est_T, axis_len, dim=False))
        if err_pairs:
            geoms.append(_error_lines(err_pairs))
    return geoms, rows, elapsed


def _report(name: str, rows, elapsed: float) -> Dict[str, float]:
    found = [r for r in rows if r[2] is not None]
    e = np.array([[r[2], r[3]] for r in found]) if found else np.zeros((0, 2))
    n = len(rows)
    stat = {
        "found": len(found) / max(n, 1),
        "tight": float(np.mean((e[:, 0] < TIGHT[0]) & (e[:, 1] < TIGHT[1]))) if len(e) else 0.0,
        "loose": float(np.mean((e[:, 0] < LOOSE[0]) & (e[:, 1] < LOOSE[1]))) if len(e) else 0.0,
        "pos_p50": float(np.median(e[:, 0]) * 1e3) if len(e) else float("nan"),
        "rot_p50": float(np.median(e[:, 1])) if len(e) else float("nan"),
        "margin": float(np.mean([r[5] for r in found])) if found else 0.0,
        "sec": elapsed / max(n, 1),
    }
    print(f"\n  {name}")
    print(f"    {'inst':>5} {'vis':>6} {'pos_mm':>8} {'rot_deg':>8} {'score':>7} {'margin':>7}")
    for idx, ov, pos, ang, score, margin in rows:
        if pos is None:
            print(f"    {idx:>5} {ov:>6.2f} {'-- no pose --':>26}")
        else:
            flag = "  " if pos < LOOSE[0] and ang < LOOSE[1] else " *"
            print(f"    {idx:>5} {ov:>6.2f} {pos * 1e3:>8.2f} {ang:>8.2f} "
                  f"{score:>7.3f} {margin:>7.2f}{flag}")
    return stat


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default=None, help="part name under output/synthetic_target/")
    ap.add_argument("--scene", type=int, default=0, help="scene index within the part")
    ap.add_argument("--arms", default="uniform,ppf",
                    help="comma-separated: uniform, ppf, ambiguity")
    ap.add_argument("--mesh", default=None, help="STL, only needed by the 'ambiguity' arm")
    ap.add_argument("--model-points", type=int, default=500, help="PPF model size budget")
    ap.add_argument("--normals", choices=("estimated", "stored"), default="estimated")
    ap.add_argument("--normal-radius", type=float, default=None,
                    help="normal estimation radius in metres; default 2*tau. This one "
                         "genuinely moves the answer (~20 recall points), so it is reported "
                         "in the header rather than left implicit")
    ap.add_argument("--color-by", choices=("error", "instance"), default="error")
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--scene-cloud", choices=("instances", "full", "none"),
                    default="instances",
                    help="grey backdrop: the segmented clusters (default, and what the "
                         "matcher actually sees), the whole bin, or nothing")
    ap.add_argument("--scene-voxel", type=float, default=0.002,
                    help="thin the backdrop for display (m); 0 keeps every point")
    ap.add_argument("--overlay-voxel", type=float, default=0.003,
                    help="thin the overlaid model cloud (m); 0 keeps every point")
    ap.add_argument("--point-size", type=float, default=3.0)
    ap.add_argument("--save", default=None, help="write a PNG here")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    part = args.part
    if part is None:
        parts = list_parts()
        if not parts:
            sys.exit(f"No scenes found under {SYNTH_ROOT}. Generate some first.")
        part = parts[0]
    scenes = list_scenes(part)
    if not scenes:
        sys.exit(f"No scene_* directories for part {part!r} under {SYNTH_ROOT}")
    if not 0 <= args.scene < len(scenes):
        sys.exit(f"--scene {args.scene} out of range; {part} has {len(scenes)}")

    scene_dir = scenes[args.scene]
    print(f"Part:  {part}   scene {args.scene} of {len(scenes)}   ({scene_dir})")

    # Parameters first, instances second: the normal-estimation radius has to come from tau,
    # not the other way round, or the scene's normals are averaged over a different
    # neighbourhood than the one the matcher's angular binning assumes.
    ref_pts, ref_nrm = load_reference(scene_dir)
    cfg = PPFConfig.derive(ref_pts, model_target_points=args.model_points)
    radius = args.normal_radius if args.normal_radius else 2.0 * cfg.tau

    scene = load_scene(scene_dir, normal_radius=radius, normals=args.normals,
                       max_instances=args.max_instances)
    scene._mesh_path = args.mesh
    print(f"Loaded {len(scene.instances)} instances, "
          f"{len(scene.ref_points)} reference points, "
          f"{len(scene.scene_points)} scene points")
    print(f"  normals: {args.normals} over r={radius * 1e3:.2f} mm")

    m_pts, m_nrm = downsample(scene.ref_points, scene.ref_normals, cfg.tau)
    model = PPFModel.train(m_pts, m_nrm, cfg)
    print(cfg.describe())
    print(f"  model points  {model.n_points:8d}     table {len(model.keys):,} entries")

    arms = build_arms(model, [a.strip() for a in args.arms.split(",") if a.strip()], scene)
    if not arms:
        sys.exit("no runnable arms")

    overlay = scene.ref_points
    if args.overlay_voxel > 0:
        overlay = downsample(scene.ref_points, scene.ref_normals, args.overlay_voxel)[0]

    backdrop = backdrop_points(scene, args.scene_cloud, args.scene_voxel)
    # Panels are spaced by what is actually drawn, not by the bin: with the default
    # instances-only backdrop the occupied region is far smaller than the bin, and pitching
    # on bin width would leave the panels stranded far apart on screen.
    extent = backdrop if len(backdrop) else overlay
    pitch = float(extent[:, 0].max() - extent[:, 0].min()) + 1.5 * cfg.diameter

    geoms: List = []
    stats: Dict[str, Dict[str, float]] = {}
    for i, (name, arm_model) in enumerate(arms.items()):
        offset = np.array([i * pitch, 0.0, 0.0])
        g, rows, elapsed = run_arm(name, arm_model, scene, overlay, offset,
                                   args.color_by, backdrop)
        geoms.extend(g)
        stats[name] = _report(name, rows, elapsed)

    print(f"\n  {'arm':<12} {'found':>7} {'@2mm/5d':>9} {'@5mm/10d':>9} "
          f"{'pos_p50':>8} {'rot_p50':>8} {'margin':>7} {'s/inst':>7}")
    for name, s in stats.items():
        print(f"  {name:<12} {s['found']:>7.2f} {s['tight']:>9.2f} {s['loose']:>9.2f} "
              f"{s['pos_p50']:>8.2f} {s['rot_p50']:>8.2f} {s['margin']:>7.2f} {s['sec']:>7.3f}")
    if args.color_by == "error":
        print("\n  green <2mm/5deg   amber <5mm/10deg   red beyond   (* marks amber/red)")
        print("  NOT symmetry-quotiented: on a symmetric part a correct pose can read red.")
    print(f"  panels along +X, {pitch * 1e3:.0f} mm apart, in arm order: "
          f"{' | '.join(arms)}")

    if args.save or not args.no_show:
        _show(geoms, f"ppf overlay — {part} {os.path.basename(scene_dir)}",
              args.save, not args.no_show, args.point_size)


def _show(geoms, title: str, save: Optional[str], show: bool, point_size: float) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1600, height=900, visible=show)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.10, 0.10, 0.13])
    opt.point_size = point_size
    opt.light_on = True

    # Fit to the geometry first, then rotate. Framing by hand from a computed centroid gets
    # it wrong as soon as the panel layout changes, and the failure mode is a screenshot of
    # empty space.
    vis.reset_view_point(True)
    vc = vis.get_view_control()
    # A three-quarter view: looking straight down the bin axis is the one direction that
    # hides a pose error along the viewing ray.
    vc.set_front([0.22, -0.42, 0.88])
    vc.set_up([0.0, 0.0, 1.0])
    # Open3D's zoom is inverted: smaller is closer. reset_view_point fits the bounds for the
    # default view direction, so after rotating the camera the fit is loose and needs taking
    # back in.
    vc.set_zoom(0.42)

    vis.poll_events()
    vis.update_renderer()
    if save:
        vis.capture_screen_image(save, do_render=True)
        print(f"\n  saved {save}")
    if show:
        vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
