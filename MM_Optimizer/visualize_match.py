"""
visualize_match.py — Overlay MechVision matched poses on synthetic scenes
-------------------------------------------------------------------------
Runs MechVision once per scene with a tuned config (default:
optuna_best_config_<part>.json) and produces, for every scene:

  - the raw synthetic scene cloud (grey),
  - the reference/model cloud placed at each MATCHED pose, one colour per
    instance (this is the "matched result" — how the model fits the scene),
  - a coordinate triad for every instance's GROUND-TRUTH pose (dim RGB) and
    MATCHED pose (bright RGB), joined by a yellow error line.

Ground-truth poses are read from each sample_*.ply header. Matched poses come
from a live MechVision run (fine_poses). Matching GT<->returned is nearest
-neighbour within SC.POS_THRESH_MATCH, mirroring mv_evaluator.match_poses_to_gt.

CLI:
  python -m MM_Optimizer.visualize_match --part 25333MB000
  python -m MM_Optimizer.visualize_match --part 25333MB000 --scenes 0 2 --no-show
  python -m MM_Optimizer.visualize_match --config path/to/best_config.json
"""

import argparse
import colorsys
import json
import os
import sys

import numpy as np
import open3d as o3d

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, ".."))
for _p in (_ROOT, _DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mm_adapter.mm_adapter        import MechVisionClient, VisionRunError
from MM_Optimizer.mv_evaluator    import MVEvaluator, _rotation_error_deg, PROJ_NAME
from MM_Optimizer.optimizer_utils import read_gt_pose_from_ply
import MM_Optimizer.search_config as SC
from geometry.file_utils import list_sample_plys
from geometry.geom_utils import pose_to_matrix, golden_hue_color

PART_ROOT_DEFAULT = os.path.join(_ROOT, "output", "synthetic_target")


# ── geometry helpers ──────────────────────────────────────────────────────────

def instance_color(i):
    """High-saturation colour for instance i."""
    return golden_hue_color(i, 0.75, 0.98)


def triad_lineset(transforms, length, axis_colors):
    """One LineSet holding an XYZ triad for every transform in *transforms*."""
    pts, lines, cols = [], [], []
    for T in transforms:
        o = T[:3, 3]
        base = len(pts)
        pts.append(o)
        for a in range(3):                       # columns = axis directions
            pts.append(o + T[:3, a] * length)
            lines.append([base, base + 1 + a])
            cols.append(axis_colors[a])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=float))
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=float))
    return ls


def error_lineset(gt_pos, match_pos, color=(1.0, 1.0, 0.0)):
    """Yellow segments connecting each GT origin to its matched origin."""
    pts, lines, cols = [], [], []
    for g, m in zip(gt_pos, match_pos):
        base = len(pts)
        pts.extend([g, m])
        lines.append([base, base + 1])
        cols.append(color)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=float))
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=float))
    return ls


# ── matching ──────────────────────────────────────────────────────────────────

def match_gt_to_returned(returned, gt, thresh):
    """Nearest-neighbour GT->returned index map within *thresh* (position).

    Mirrors mv_evaluator.match_poses_to_gt but returns the correspondence
    (which returned pose each GT instance was matched to) instead of errors.
    Returns a list over GT: matched returned index, or None if unmatched.
    """
    used, out = set(), [None] * len(gt)
    for gi, g in enumerate(gt):
        gp, best, bd = np.asarray(g[:3], float), None, np.inf
        for ri, r in enumerate(returned):
            if ri in used:
                continue
            d = float(np.linalg.norm(np.asarray(r[:3], float) - gp))
            if d < bd:
                bd, best = d, ri
        if best is not None and bd < thresh:
            used.add(best)
            out[gi] = best
    return out


# ── per-scene pipeline ──────────────────────────────────────────────────────────


def build_scene_geometry(scene_dir, ref_pcd, fine_poses,
                         ref_voxel=0.004, axis_len=0.035):
    """Return (geometries, stats) for one scene."""
    sample_plys = list_sample_plys(scene_dir)
    gt_poses    = [read_gt_pose_from_ply(p, scalar_first=True) for p in sample_plys]
    n_gt        = len(gt_poses)

    match_idx = match_gt_to_returned(fine_poses, gt_poses, SC.POS_THRESH_MATCH)

    # Base scene cloud (grey)
    scene_cloud = o3d.io.read_point_cloud(os.path.join(scene_dir, "scene.ply"))
    scene_cloud.paint_uniform_color((0.55, 0.55, 0.55))
    geoms = [scene_cloud]

    ref_ds = ref_pcd.voxel_down_sample(ref_voxel) if ref_voxel else ref_pcd

    gt_T, match_T, gt_o, match_o = [], [], [], []
    pos_errs, ang_errs = [], []
    n_matched = 0

    for gi, gt in enumerate(gt_poses):
        T_gt = pose_to_matrix(gt)
        gt_T.append(T_gt)
        ri = match_idx[gi]
        if ri is None:
            continue                              # miss: GT frame only
        n_matched += 1
        pred = fine_poses[ri]
        T_m  = pose_to_matrix(pred)
        match_T.append(T_m)
        gt_o.append(T_gt[:3, 3])
        match_o.append(T_m[:3, 3])

        # reference cloud placed at the matched pose, one colour per instance
        inst = o3d.geometry.PointCloud(ref_ds)
        inst.transform(T_m)
        inst.paint_uniform_color(instance_color(gi))
        geoms.append(inst)

        pos_errs.append(float(np.linalg.norm(np.asarray(pred[:3]) - np.asarray(gt[:3]))))
        ang_errs.append(_rotation_error_deg(pred[3:7], gt[3:7]))

    # GT triads for ALL instances (dim), matched triads (bright), error lines
    if gt_T:
        geoms.append(triad_lineset(
            gt_T, axis_len,
            [(0.5, 0.0, 0.0), (0.0, 0.45, 0.0), (0.0, 0.0, 0.55)]))
    if match_T:
        geoms.append(triad_lineset(
            match_T, axis_len,
            [(1.0, 0.25, 0.25), (0.3, 1.0, 0.3), (0.35, 0.5, 1.0)]))
        geoms.append(error_lineset(gt_o, match_o))

    stats = {
        "n_gt":        n_gt,
        "n_returned":  len(fine_poses),
        "n_matched":   n_matched,
        "coverage":    n_matched / max(1, n_gt),
        "mean_pos_mm": float(np.mean(pos_errs) * 1000) if pos_errs else float("nan"),
        "mean_ang_deg": float(np.mean(ang_errs))       if ang_errs else float("nan"),
    }
    return geoms, stats


def show_and_save(geoms, title, png_path, show=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1600, height=1000)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.08, 0.08, 0.10])
    vis.poll_events()
    vis.update_renderer()
    if png_path:
        vis.capture_screen_image(png_path, do_render=True)
        print(f"    saved {png_path}")
    if show:
        vis.run()
    vis.destroy_window()


# ── main ────────────────────────────────────────────────────────────────────

def _resolve_config_path(config, sampler, part):
    """Pick the best_config JSON to visualize.

    The optuna tuner writes ``results/<SAMPLER>_best_config_<part>.json`` (e.g. ``GP_best_config``),
    NOT ``optuna_best_config_<part>.json``. Defaulting to the latter silently overlaid a stale
    config against fresh scenes — the exact failure this diagnosis chased. Resolution order:
      1. explicit --config path,
      2. results/<SAMPLER>_best_config_<part>.json when --sampler given,
      3. the most-recently-modified ``*_best_config_<part>.json`` in results/ (the freshly tuned one).
    """
    import glob
    if config:
        return config
    results_dir = os.path.join(_DIR, "results")
    if sampler:
        p = os.path.join(results_dir, f"{sampler.upper()}_best_config_{part}.json")
        if not os.path.isfile(p):
            sys.exit(f"No config for --sampler {sampler}: {p}")
        return p
    candidates = glob.glob(os.path.join(results_dir, f"*best_config_{part}.json"))
    if not candidates:
        sys.exit(f"No *_best_config_{part}.json found in {results_dir}")
    newest = max(candidates, key=os.path.getmtime)
    print(f"[auto] newest tuned config for {part}: {os.path.basename(newest)}")
    return newest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="25333MB000")
    ap.add_argument("--config", default=None,
                    help="Path to best_config JSON. Overrides --sampler/auto-detect.")
    ap.add_argument("--sampler", default=None,
                    help="Load results/<SAMPLER>_best_config_<part>.json (e.g. GP, NSGAII, TPE). "
                         "If omitted, the newest *_best_config_<part>.json in results/ is used.")
    ap.add_argument("--part-root", default=PART_ROOT_DEFAULT,
                    help="Root holding <part>/scene_NNNNN directories")
    ap.add_argument("--scenes", type=int, nargs="*", default=None,
                    help="Scene indices to process (default: all)")
    ap.add_argument("--no-show", action="store_true",
                    help="Save PNGs only; do not open interactive windows")
    ap.add_argument("--ref-voxel", type=float, default=0.004,
                    help="Voxel size (m) to downsample the reference cloud")
    args = ap.parse_args()

    cfg_path = _resolve_config_path(args.config, args.sampler, args.part)
    with open(cfg_path) as f:
        cfg = json.load(f)
    coarse, fine = cfg["coarse"], cfg["fine"]
    print(f"Config: {cfg_path}")
    print(f"  reported coverage={cfg.get('coverage'):.3f} "
          f"mean_time={cfg.get('mean_time'):.3f}s")

    part_dir = os.path.join(args.part_root, args.part)
    scene_dirs = sorted(
        os.path.join(part_dir, d) for d in os.listdir(part_dir)
        if d.startswith("scene_") and os.path.isdir(os.path.join(part_dir, d)))
    if args.scenes is not None:
        scene_dirs = [d for d in scene_dirs
                      if int(d.split("_")[-1]) in args.scenes]
    if not scene_dirs:
        sys.exit("No matching scenes found.")

    out_dir = os.path.join(part_dir, "_match_viz")
    os.makedirs(out_dir, exist_ok=True)

    client = MechVisionClient()
    projects = client.get_projects()
    project_id = projects[PROJ_NAME]
    ev = MVEvaluator(args.part, client, project_id,
                     scene_groups=[], warm_start=None, cache=None, dry_run=False)

    ref_pcd = o3d.io.read_point_cloud(
        os.path.join(scene_dirs[0], "reference_cloud.ply"))
    print(f"Reference cloud: {len(ref_pcd.points)} pts "
          f"(downsampled @ {args.ref_voxel} m)\n")

    for scene_dir in scene_dirs:
        name = os.path.basename(scene_dir)
        print(f"[{name}] running MechVision ...")
        params = ev._make_params_dict(coarse, fine, scene_dir)
        client.set_params(project_id, params)
        try:
            result     = client.run_vision(project_id)
            fine_poses = result.get("fine_poses", [])
        except VisionRunError as e:
            print(f"    VisionRunError: {e} -> no detections")
            fine_poses = []

        geoms, st = build_scene_geometry(scene_dir, ref_pcd, fine_poses,
                                         ref_voxel=args.ref_voxel)
        print(f"    GT={st['n_gt']}  returned={st['n_returned']}  "
              f"matched={st['n_matched']}  coverage={st['coverage']:.2f}  "
              f"pos={st['mean_pos_mm']:.1f}mm  ang={st['mean_ang_deg']:.1f}deg")

        png = os.path.join(out_dir, f"{name}_match.png")
        show_and_save(geoms, f"{args.part}/{name}  "
                             f"cov={st['coverage']:.2f} "
                             f"pos={st['mean_pos_mm']:.1f}mm",
                      png, show=not args.no_show)

    client.close()
    print(f"\nDone. PNGs in {out_dir}")


if __name__ == "__main__":
    main()
