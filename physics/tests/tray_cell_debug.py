"""Isolated tray-cell diagnostic — sweeps every stable pose of a part.

Runs the geometry/tray_utils.py pocket pipeline for each stable pose and reports where a part would
be popped out of its cell. For every pose it prints a row of quantitative diagnostics and (unless
--no-show) visualizes:

  footprint  — a grid of all poses: part mesh silhouette (blue) vs the CONVEX-COLLISION silhouette
               (red, what MuJoCo actually collides). A mismatch/incomplete silhouette shrinks the pocket.
  negative   — per pose: the pocket prism (the cavity) + floor slab.
  vhacd      — per pose: the wall-frame convex pieces (one colour each) + floor.
  overlay    — per pose: SEATED part collision pieces (red) inside the tray collision (blue frame +
               grey floor). Red poking into a frame piece / below the floor top = initial penetration.

Key numbers per pose: clearance, floor penetration at the seat, and the boolean OVERLAP VOLUME between
the seated part collision and the tray collision (walls / floor) — nonzero overlap ⇒ the part ejects.

With --sim, each pose also DROPS the part into one tray cell and settles it under MuJoCo (same solver
settings as the app): the table gains the settled lateral/vertical drift and a seated/POPPED status, and
the overlay shows the SETTLED part (red) in the cell.

Usage:
  python tray_cell_debug.py --mesh part.stl                 # sweep all poses, all views
  python tray_cell_debug.py --mesh part.stl --no-show       # sweep all poses, numbers only (table)
  python tray_cell_debug.py --mesh part.stl --sim --no-show # + MuJoCo settle drift per pose
  python tray_cell_debug.py --mesh part.stl --sim --view    # watch each settle live in the MuJoCo viewer
  python tray_cell_debug.py --mesh part.stl --pose 1 --sim --view --stage overlay
  python tray_cell_debug.py --shape L --clearance snug --no-show
"""
import argparse
import colorsys
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from physics.mujoco_bin_scene import (MujocoBinScene, load_part, TRAY_WALL, TRAY_BASE, TRAY_SEAT_GAP,
                                      TRAY_VHACD_MAX_HULLS, TRAY_VHACD_RESOLUTION, _CLEARANCE_M)
from geometry.tray_utils import (footprint_polygon, convex_footprint_polygon,
                                  build_tray_collision_frame, build_tray_conforming_hfield)
from geometry.convex_decomp import vhacd_decompose

try:                          # trimesh.mesh.projected() needs rtree; only used for the blue MESH line.
    import rtree  # noqa: F401
    _HAS_RTREE = True
except Exception:
    _HAS_RTREE = False


# ---------------------------------------------------------------------------- part loading

def _synthetic_L():
    """A concave L prism (a box with a corner notch) — a quick stand-in when no STL is given."""
    base = trimesh.creation.box(extents=[0.10, 0.10, 0.03])
    notch = trimesh.creation.box(extents=[0.05, 0.05, 0.06])
    notch.apply_translation([0.025, 0.025, 0.0])
    return trimesh.boolean.difference([base, notch], engine="manifold")


def load_part_and_collision(mesh_path, shape):
    """Return (part_mesh trimesh, convex_pieces list[trimesh]) via VHACD
    (geometry.convex_decomp.vhacd_decompose with defaults). NOTE: the app's DecomposeStage
    now decomposes with CoACD instead; this debug helper stays on VHACD for a fast, deterministic
    tray-cell repro and does not need to match the stage's collision-mesh fidelity."""
    if mesh_path:
        part_mesh, _ = load_part(mesh_path=mesh_path)            # scales mm->m, centres, decomposes
    else:
        if shape.upper() == "L":
            part_mesh = _synthetic_L()
        elif shape.lower() == "cyl":
            part_mesh = trimesh.creation.cylinder(radius=0.05, height=0.03, sections=64)  # round footprint
        else:
            part_mesh = trimesh.creation.box(extents=[0.09, 0.06, 0.03])
        part_mesh.apply_translation(-part_mesh.center_mass)
    pieces = vhacd_decompose(np.asarray(part_mesh.vertices), np.asarray(part_mesh.faces))
    convex = [trimesh.Trimesh(vertices=np.asarray(v), faces=np.asarray(f), process=False) for v, f in pieces]
    return part_mesh, convex


# ---------------------------------------------------------------------------- helpers

def resolve_clearance(mode, footprint_diag):
    c = _CLEARANCE_M.get(mode, 0.0025)
    return float(0.05 * footprint_diag if c is None else c)


def _o3d(mesh_tri, color):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh_tri.vertices)),
                                  o3d.utility.Vector3iVector(np.asarray(mesh_tri.faces)))
    m.compute_vertex_normals()
    m.paint_uniform_color(color)
    return m


def _intersection_volume(a, b):
    try:
        inter = trimesh.boolean.intersection([a, b], engine="manifold", check_volume=False)
        return float(inter.volume) if inter is not None and len(inter.vertices) else 0.0
    except Exception:
        return 0.0


def _plot_poly(ax, poly, style, label):
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    for k, g in enumerate(geoms):
        ax.plot(*g.exterior.xy, style, label=label if k == 0 else None)


# ---------------------------------------------------------------------------- per-pose pipeline

def compute_cell(R_aligned, part_mesh, convex, collision_part, clearance_mode, height_pct, conform=False):
    """Run the tray_utils pocket pipeline for one stable pose; return all stage geoms + diagnostics.

    conform=True builds a CONFORMING height-field pocket (cradles the part's bottom surface) instead of
    the flat extruded VHACD frame."""
    sv = (R_aligned @ np.asarray(part_mesh.vertices).T).T
    fp_x, fp_y = float(np.ptp(sv[:, 0])), float(np.ptp(sv[:, 1]))
    part_h = float(np.ptp(sv[:, 2]))
    clearance = resolve_clearance(clearance_mode, float(np.hypot(fp_x, fp_y)))
    gap = 0.1 * max(fp_x, fp_y) + 2 * TRAY_WALL + 2 * clearance
    pitch = (fp_x + gap, fp_y + gap)
    pocket_depth = (height_pct / 100.0) * part_h

    poly_mesh_raw = footprint_polygon(part_mesh, R_aligned, 0.0)        # mesh silhouette (may fracture)
    poly_col_raw = convex_footprint_polygon(convex, R_aligned, 0.0)     # robust collision silhouette
    poly_col = convex_footprint_polygon(convex, R_aligned, clearance)
    out = dict(fp_x=fp_x, fp_y=fp_y, part_h=part_h, clearance=clearance, pitch=pitch, pocket_depth=pocket_depth,
               poly_mesh_raw=poly_mesh_raw, poly_col_raw=poly_col_raw, poly_col=poly_col)

    if conform:
        hf = build_tray_conforming_hfield(convex, R_aligned, pitch, pocket_depth, TRAY_BASE, clearance)
        cx, cy = hf["center_xy"]
        T_seat = np.eye(4); T_seat[:3, :3] = R_aligned; T_seat[2, 3] = hf["seat_dz"] + TRAY_SEAT_GAP
        out.update(hf=hf, seat_z=hf["seat_dz"] + TRAY_SEAT_GAP, hf_center=(cx, cy),
                   seated_pieces=[p.copy().apply_transform(T_seat) for p in convex],
                   visual=hf["visual"].copy().apply_translation([cx, cy, 0.0]),
                   floor_pen=0.0, wall_overlap=0.0, floor_overlap=0.0, pieces=[])
        return out

    frame, pieces = build_tray_collision_frame(poly_col, pitch, pocket_depth, TRAY_BASE,
                                               TRAY_VHACD_MAX_HULLS, TRAY_VHACD_RESOLUTION)
    min_z_mesh = float(sv[:, 2].min())
    min_z_col = float((R_aligned @ np.asarray(collision_part.vertices).T).T[:, 2].min())
    seat_z = TRAY_BASE - min_z_mesh + TRAY_SEAT_GAP
    T_seat = np.eye(4); T_seat[:3, :3] = R_aligned; T_seat[2, 3] = seat_z
    seated_pieces = [p.copy().apply_transform(T_seat) for p in convex]
    b = poly_col.bounds
    floor = trimesh.creation.box(extents=[b[2] - b[0] + 2 * TRAY_WALL, b[3] - b[1] + 2 * TRAY_WALL, TRAY_BASE])
    floor.apply_translation([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, TRAY_BASE / 2])
    out.update(seat_z=seat_z, frame=frame, pieces=pieces, seated_pieces=seated_pieces, floor=floor,
               floor_pen=TRAY_BASE - (seat_z + min_z_col),
               wall_overlap=sum(_intersection_volume(sp, frame) for sp in seated_pieces),
               floor_overlap=sum(_intersection_volume(sp, floor) for sp in seated_pieces))
    return out


# ---------------------------------------------------------------------------- single-cell MuJoCo settle

def simulate_cell(cell, convex, R_aligned, settle_time=2.0, view=False):
    """Drop ONE part (its convex pieces) into ONE tray cell (frame pieces + floor) and settle under
    gravity. Mirrors MujocoBinScene's solver settings so behaviour matches the full app. Returns the
    lateral/vertical drift from the seat pose, an escaped flag, and the settled body transform.

    view=True opens a live MuJoCo passive viewer: the settle is stepped in real time, then the window
    is held open (close it to continue to the next pose)."""
    import mujoco
    dt = 0.001
    spec = mujoco.MjSpec()
    spec.option.timestep = dt
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.o_margin = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    g0 = spec.default.geom
    g0.condim = 6
    g0.friction = [1.5, 0.005, 0.0001]
    g0.solref = [max(0.002, 2 * dt), 1]
    g0.solimp = [0.95, 0.99, 0.001, 0.5, 2]
    g0.contype = 1
    g0.conaffinity = 1
    world = spec.worldbody

    # Static tray cell: either a conforming height field, or the flat VHACD wall pieces + floor slab.
    tray = world.add_body(); tray.name = "tray"
    if "hf" in cell:
        hf = cell["hf"]; el = hf["elevation"]; cx, cy = hf["center_xy"]
        h = spec.add_hfield(); h.name = "pocket"; h.nrow = el.shape[0]; h.ncol = el.shape[1]
        h.size = hf["size"]; h.userdata = el.flatten().astype(float).tolist()
        gm = tray.add_geom(); gm.type = mujoco.mjtGeom.mjGEOM_HFIELD; gm.hfieldname = "pocket"
        gm.pos = [cx, cy, float(hf.get("z_offset", 0.0))]; gm.mass = 0.0
    else:
        for j, (v, f) in enumerate(cell["pieces"]):
            mesh = spec.add_mesh(); mesh.name = f"tp_{j}"
            mesh.uservert = np.asarray(v, dtype=float).flatten().tolist()
            mesh.userface = np.asarray(f).flatten().tolist()
            gm = tray.add_geom(); gm.type = mujoco.mjtGeom.mjGEOM_MESH; gm.meshname = f"tp_{j}"; gm.mass = 0.0
        fb = cell["floor"].bounds
        fg = tray.add_geom(); fg.type = mujoco.mjtGeom.mjGEOM_BOX
        fg.size = [(fb[1][0] - fb[0][0]) / 2, (fb[1][1] - fb[0][1]) / 2, TRAY_BASE / 2]
        fg.pos = [(fb[0][0] + fb[1][0]) / 2, (fb[0][1] + fb[1][1]) / 2, TRAY_BASE / 2]; fg.mass = 0.0

    # Free part seated at the pocket: convex pieces, body oriented R_aligned at the seat height.
    part = world.add_body(); part.name = "part"
    part.pos = [0.0, 0.0, cell["seat_z"]]
    part.quat = R.from_matrix(R_aligned).as_quat(scalar_first=True).tolist()
    part.add_freejoint()
    for j, pc in enumerate(convex):
        mesh = spec.add_mesh(); mesh.name = f"pp_{j}"
        mesh.uservert = np.asarray(pc.vertices, dtype=float).flatten().tolist()
        mesh.userface = np.asarray(pc.faces).flatten().tolist()
        gm = part.add_geom(); gm.type = mujoco.mjtGeom.mjGEOM_MESH; gm.meshname = f"pp_{j}"; gm.mass = 0.1

    model = spec.compile(); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "part")
    pre = data.xpos[pid].copy()
    nsteps = int(settle_time / dt)
    if view:
        import mujoco.viewer
        import time
        b = cell["poly_col"].bounds
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 4.0 * max(cell["fp_x"], cell["fp_y"])
            viewer.cam.azimuth = 90; viewer.cam.elevation = -25
            viewer.cam.lookat[:] = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, TRAY_BASE + cell["pocket_depth"] / 2]
            for _ in range(nsteps):
                if not viewer.is_running():
                    break
                t0 = time.time()
                mujoco.mj_step(model, data)
                viewer.sync()
                rem = dt - (time.time() - t0)
                if rem > 0:
                    time.sleep(rem)
            post = data.xpos[pid].copy(); pq = data.xquat[pid].copy()
            # print("   [viewer] settled — close the window to continue")
            # time.sleep(1)
            viewer.close()
            # while viewer.is_running():
            #     viewer.sync(); time.sleep(0.02)
    else:
        for _ in range(nsteps):
            mujoco.mj_step(model, data)
        post = data.xpos[pid].copy(); pq = data.xquat[pid].copy()
    drift_xy = float(np.linalg.norm(post[:2] - pre[:2]))
    T = np.eye(4); T[:3, :3] = R.from_quat(pq, scalar_first=True).as_matrix(); T[:3, 3] = post
    escaped = bool(post[2] < -0.005 or drift_xy > 0.5 * max(cell["fp_x"], cell["fp_y"]))
    return dict(drift_xy=drift_xy, drift_z=float(post[2] - pre[2]), escaped=escaped, T_settled=T)


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Tray-cell diagnostic across all stable poses")
    ap.add_argument("--mesh", default=None, help="STL path (else use --shape synthetic)")
    ap.add_argument("--shape", default="L", choices=["L", "box", "cyl"], help="synthetic part when no --mesh")
    ap.add_argument("--clearance", default="medium", choices=["snug", "medium", "loose"])
    ap.add_argument("--height", type=int, default=70, help="pocket depth as %% of part height")
    ap.add_argument("--pose", type=int, default=None, help="restrict to one pose index (default: sweep all)")
    ap.add_argument("--stage", default="all",
                    choices=["footprint", "negative", "vhacd", "overlay", "all"])
    ap.add_argument("--conform", action="store_true",
                    help="conforming height-field pocket (cradles the part bottom) instead of flat VHACD")
    ap.add_argument("--sim", action="store_true", help="drop the part into each cell and settle under MuJoCo")
    ap.add_argument("--settle", type=float, default=2.0, help="settle time in seconds for --sim")
    ap.add_argument("--view", action="store_true",
                    help="open a live MuJoCo viewer during --sim (real-time settle, per pose)")
    ap.add_argument("--no-show", action="store_true", help="print the diagnostic table only, skip viewers")
    args = ap.parse_args()

    part_mesh, convex = load_part_and_collision(args.mesh, args.shape)
    collision_part = trimesh.util.concatenate(convex)
    poses = MujocoBinScene.get_stable_poses(part_mesh)
    sel = range(len(poses)) if args.pose is None else [min(args.pose, len(poses) - 1)]

    # --- sweep + table --------------------------------------------------------------------------
    name = ("mesh " + args.mesh) if args.mesh else ("synthetic " + args.shape)
    print(f"\n===== TRAY DIAGNOSTIC: {name}   convex pieces: {len(convex)}   "
          f"stable poses: {len(poses)}   clearance: {args.clearance} =====")
    if not _HAS_RTREE:
        print(" [note] rtree not installed -> the blue MESH silhouette falls back to its convex hull "
              "(cosmetic only).\n        The collision pocket (red) uses convex hulls directly and is "
              "unaffected. `pip install rtree` for the true mesh outline.")
    if args.conform:
        print(f" [mode] CONFORMING height-field pocket (cradles the part bottom)")
    sim_hdr = f" {'driftXY':>8} {'driftZ':>7} {'status':>7}" if args.sim else ""
    geo_col = "grid" if args.conform else "hulls"
    print(f"{'pose':>4} {'prob':>5} {'footprint(mm)':>14} {'depth':>6} {geo_col:>7} "
          f"{'floorPen':>8} {'wall(mm^3)':>11} {'floor(mm^3)':>11}{sim_hdr}  flags")
    cells = []
    for i in sel:
        c = compute_cell(poses[i][0], part_mesh, convex, collision_part, args.clearance, args.height,
                         conform=args.conform)
        cells.append((i, float(poses[i][1]), c))
        sim_cols = ""
        if args.sim:
            c["sim"] = simulate_cell(c, convex, poses[i][0], args.settle,
                                     view=(args.view and not args.no_show))
            sim_cols = (f" {c['sim']['drift_xy']*1e3:>8.2f} {c['sim']['drift_z']*1e3:>7.2f} "
                        f"{'POPPED' if c['sim']['escaped'] else 'seated':>7}")
        flags = ("  <-- WALL INTRUSION" if c["wall_overlap"] > 1e-10 else "") + \
                ("  <-- FLOOR PEN" if c["floor_pen"] > 1e-4 else "") + \
                ("  <-- POPPED OUT" if args.sim and c["sim"]["escaped"] else "")
        geo = f"{c['hf']['elevation'].shape[0]}x{c['hf']['elevation'].shape[1]}" if args.conform else str(len(c["pieces"]))
        print(f"{i:>4} {poses[i][1]:>5.2f} {c['fp_x']*1e3:>6.1f}x{c['fp_y']*1e3:<7.1f} "
              f"{c['pocket_depth']*1e3:>6.1f} {geo:>7} {c['floor_pen']*1e3:>8.2f} "
              f"{c['wall_overlap']*1e9:>11.1f} {c['floor_overlap']*1e9:>11.1f}{sim_cols}{flags}")
    bad = [i for i, _, c in cells
           if c["wall_overlap"] > 1e-10 or c["floor_pen"] > 1e-4 or (args.sim and c["sim"]["escaped"])]
    print(f"{'='*70}\n {'problem poses: ' + str(bad) if bad else 'all poses clean (no overlap / no floor penetration)' + (' / all settled' if args.sim else '')}\n")

    if args.no_show:
        return

    # --- footprint grid (all poses at once) -----------------------------------------------------
    def show_footprint_grid():
        n = len(cells)
        ncols = min(4, n); nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for k, (i, prob, c) in enumerate(cells):
            ax = axes.flat[k]; ax.set_visible(True); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
            ov = c["wall_overlap"] * 1e9
            ax.set_title(f"pose {i} (p={prob:.2f})  overlap={ov:.0f}mm³",
                         color=("red" if ov > 1e-1 else "black"), fontsize=9)
            _plot_poly(ax, c["poly_mesh_raw"], "b-", "mesh" if _HAS_RTREE else "mesh (hull)")
            _plot_poly(ax, c["poly_col_raw"], "r-", "collision")
            if k == 0:
                ax.legend(fontsize=8)
        fig.suptitle(f"{name}: mesh (blue) vs convex-collision (red) silhouette per pose")
        plt.tight_layout(); plt.show()

    def show_3d(geoms, title):
        print(f"[view] {title} (close window to continue)")
        o3d.visualization.draw_geometries(geoms, window_name=title)

    if args.stage in ("footprint", "all"):
        show_footprint_grid()

    for i, prob, c in cells:
        tag = f"pose {i}/{len(poses)-1} (p={prob:.2f})"
        if args.stage == "negative" and not args.conform:
            prism = trimesh.creation.extrude_polygon(c["poly_col"], height=c["pocket_depth"])
            prism.apply_translation([0, 0, TRAY_BASE])
            show_3d([_o3d(prism, [0.9, 0.5, 0.2]), _o3d(c["floor"], [0.6, 0.6, 0.6])], f"{tag} — 3D negative")
        if args.stage == "vhacd" and not args.conform:
            geoms = [_o3d(trimesh.Trimesh(np.asarray(v), np.asarray(f), process=False),
                          list(colorsys.hsv_to_rgb(j / max(len(c["pieces"]), 1), 0.6, 0.9)))
                     for j, (v, f) in enumerate(c["pieces"])]
            show_3d(geoms + [_o3d(c["floor"], [0.55, 0.55, 0.55])], f"{tag} — VHACD pieces ({len(c['pieces'])})")
        if args.stage in ("overlay", "all"):
            if args.conform:   # conforming hfield surface (cradle) instead of VHACD walls + floor
                tray = [_o3d(c["visual"], [0.3, 0.5, 0.9])]
            else:
                tray = [_o3d(trimesh.Trimesh(np.asarray(v), np.asarray(f), process=False), [0.3, 0.5, 0.9])
                        for v, f in c["pieces"]] + [_o3d(c["floor"], [0.55, 0.55, 0.55])]
            if args.sim:   # show where the part actually SETTLED
                Tf = c["sim"]["T_settled"]
                part = [_o3d(pc.copy().apply_transform(Tf), [0.9, 0.2, 0.2]) for pc in convex]
                extra = f"settled drift={c['sim']['drift_xy']*1e3:.1f}mm  {'POPPED' if c['sim']['escaped'] else 'seated'}"
            else:
                part = [_o3d(sp, [0.9, 0.2, 0.2]) for sp in c["seated_pieces"]]
                extra = "conforming pocket" if args.conform else f"overlap={c['wall_overlap']*1e9:.0f}mm³"
            show_3d(tray + part, f"{tag} — OVERLAY (red part in blue tray)  {extra}")


if __name__ == "__main__":
    main()
