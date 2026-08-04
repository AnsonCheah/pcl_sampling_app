"""
Soft cable drape/drop simulation using MuJoCo.

The primary scene is loaded from cable_model.xml. It fixes one end of the
cable from a higher hook, drops it onto pegs, and lets it settle into a tray. That
scene is more informative than a straight horizontal free fall because gravity
and contact immediately create bending.

Requirements:
    pip install mujoco

Run:
    python physics/cable_drop.py --mode viewer
    python physics/cable_drop.py --mode headless --duration 2
    python physics/cable_drop.py --mode record --output cable_drape.mp4
"""

from __future__ import annotations

import argparse
import pathlib
import re
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np


SCRIPT_DIR = pathlib.Path(__file__).parent
XML_PATH = SCRIPT_DIR / "cable_model.xml"

CAMERA_LOOKAT = np.array([-0.02, 0.0, 0.72])
CAMERA_DISTANCE = 2.4
CAMERA_ELEVATION = -25
CAMERA_AZIMUTH = 135


INLINE_XML = """
<mujoco model="soft_cable_drape">
  <compiler angle="degree" autolimits="true"/>
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"
          solver="Newton" tolerance="1e-8" iterations="80"/>
  <statistic center="0 0 0.72" extent="1.8"/>
  <default>
    <geom condim="3" friction="1.0 0.02 0.001"
          solref="0.01 1" solimp="0.95 0.99 0.0005"/>
    <default class="fixture">
      <geom material="fixture_mat" friction="1.2 0.03 0.001"/>
    </default>
  </default>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="0.24 0.25 0.26" rgb2="0.42 0.43 0.44" width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="6 6" reflectance="0.18"/>
    <material name="tray_mat" rgba="0.28 0.33 0.38 1"/>
    <material name="fixture_mat" rgba="0.18 0.22 0.26 1"/>
    <material name="hook_mat" rgba="0.78 0.72 0.62 1"/>
    <material name="cable_mat" rgba="0.86 0.22 0.10 1"/>
  </asset>
  <visual>
    <global azimuth="135" elevation="-25" offwidth="1280" offheight="720"/>
  </visual>
  <worldbody>
    <light name="key" pos="-1.2 -1.4 3.0" dir="0.4 0.5 -1"
           diffuse="0.9 0.9 0.86" directional="true"/>
    <light name="fill" pos="1.4 1.0 2.0" dir="-0.7 -0.4 -1"
           diffuse="0.35 0.38 0.42" directional="true"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="floor_mat"/>
    <geom name="tray_floor" type="box" pos="0 0 0.015" size="0.72 0.32 0.015" material="tray_mat"/>
    <geom name="tray_front" type="box" pos="0 -0.34 0.085" size="0.72 0.02 0.07" material="tray_mat"/>
    <geom name="tray_back" type="box" pos="0 0.34 0.085" size="0.72 0.02 0.07" material="tray_mat"/>
    <geom name="tray_left" type="box" pos="-0.74 0 0.085" size="0.02 0.34 0.07" material="tray_mat"/>
    <geom name="tray_right" type="box" pos="0.74 0 0.085" size="0.02 0.34 0.07" material="tray_mat"/>
    <geom name="fixed_hook" class="fixture" type="capsule"
          fromto="-0.66 -0.20 1.48 -0.66 0.20 1.48" size="0.025" material="hook_mat"/>
    <geom name="left_peg" class="fixture" type="cylinder" pos="-0.32 0 0.42"
          zaxis="0 1 0" size="0.045 0.34"/>
    <geom name="right_peg" class="fixture" type="cylinder" pos="0.22 0 0.30"
          zaxis="0 1 0" size="0.040 0.34"/>
    <body name="cable_release" pos="0 0 0.70">
      <composite prefix="cable" type="cable" initial="none"
               vertex="
                 -0.62 0 0.78 -0.58 0 0.76 -0.54 0 0.72 -0.50 0 0.66
                 -0.46 0 0.58 -0.42 0 0.51 -0.37 0 0.48 -0.31 0 0.50
                 -0.25 0 0.48 -0.20 0 0.42 -0.16 0 0.34 -0.12 0 0.25
                 -0.06 0 0.18  0.00 0 0.15  0.06 0 0.18  0.12 0 0.25
                  0.17 0 0.33  0.23 0 0.37  0.29 0 0.35  0.35 0 0.28
                  0.40 0 0.20  0.44 0 0.12  0.48 0 0.08  0.54 0 0.07
                  0.60 0 0.075 0.66 0 0.08">
        <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="2e5"/>
        <config key="bend" value="8e4"/>
        <config key="flat" value="true"/>
        <config key="vmax" value="0.2"/>
        </plugin>
        <joint kind="main" damping="0.04" armature="0.0002"/>
        <geom type="capsule" size="0.007" mass="0.003" material="cable_mat"/>
      </composite>
    </body>
  </worldbody>
</mujoco>
"""


def read_xml_source() -> str:
    if XML_PATH.exists():
        print(f"[INFO] Loading model from: {XML_PATH}")
        return XML_PATH.read_text(encoding="utf-8")

    print("[INFO] XML file not found; using inline XML.")
    return INLINE_XML


def patch_stiffness(
    xml_string: str,
    bend: Optional[float] = None,
    twist: Optional[float] = None,
) -> str:
    """Patch only the stiffness values explicitly supplied on the CLI."""
    if bend is not None:
        xml_string = re.sub(
            r'(key="bend"\s+value=")[^"]+(")',
            rf"\g<1>{bend:.6g}\g<2>",
            xml_string,
        )
    if twist is not None:
        xml_string = re.sub(
            r'(key="twist"\s+value=")[^"]+(")',
            rf"\g<1>{twist:.6g}\g<2>",
            xml_string,
        )
    return xml_string


def load_model(args: argparse.Namespace) -> tuple[mujoco.MjModel, mujoco.MjData]:
    xml_src = read_xml_source()

    if args.bend is not None or args.twist is not None:
        parts = []
        if args.bend is not None:
            parts.append(f"bend={args.bend:.3g} Pa")
        if args.twist is not None:
            parts.append(f"twist={args.twist:.3g} Pa")
        print(f"[INFO] Overriding stiffness: {', '.join(parts)}")
        xml_src = patch_stiffness(xml_src, bend=args.bend, twist=args.twist)

    model = mujoco.MjModel.from_xml_string(xml_src)
    data = mujoco.MjData(model)
    return model, data


def apply_initial_perturbation(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """
    Tilt the first free joint if the loaded scene has one.

    The drape scene fixes the first cable point, so there is no root freejoint
    to perturb. Older/free-drop XMLs still get the previous small release tilt.
    """
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) == 0:
        return False

    qadr = int(model.jnt_qposadr[free_joints[0]])
    tilt_angle = np.deg2rad(15.0)
    data.qpos[qadr + 3] = np.cos(tilt_angle / 2.0)
    data.qpos[qadr + 4] = 0.0
    data.qpos[qadr + 5] = np.sin(tilt_angle / 2.0)
    data.qpos[qadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return True


def cable_body_ids(model: mujoco.MjModel) -> list[int]:
    """Return generated cable body ids, excluding fixtures in worldbody."""
    return [
        body_id
        for body_id in range(1, model.nbody)
        if model.body_geomnum[body_id] > 0
    ]


def print_model_info(model: mujoco.MjModel) -> None:
    cable_ids = cable_body_ids(model)
    print("\n" + "=" * 55)
    print("  MODEL STATISTICS")
    print("=" * 55)
    print(f"  Bodies         : {model.nbody}")
    print(f"  Cable bodies   : {len(cable_ids)}")
    print(f"  Joints (DOF)   : {model.njnt}  ({model.nv} DOF)")
    print(f"  Geoms          : {model.ngeom}")
    print(f"  Plugins        : {model.nplugin}")
    print(f"  Timestep       : {model.opt.timestep * 1000:.1f} ms")
    print(f"  Integrator     : {model.opt.integrator}")
    print(f"  Gravity        : {model.opt.gravity}")
    print("=" * 55 + "\n")


def compute_cable_stats(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    body_ids = cable_body_ids(model)
    stats: dict[str, float] = {}

    if not body_ids:
        return stats

    points = data.xpos[np.array(body_ids)]
    z_positions = points[:, 2]
    stats["z_mean"] = float(np.mean(z_positions))
    stats["z_min"] = float(np.min(z_positions))
    stats["z_max"] = float(np.max(z_positions))
    stats["z_spread"] = stats["z_max"] - stats["z_min"]

    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-9
    if np.count_nonzero(valid) >= 2:
        tangents = segments[valid] / lengths[valid, None]
        dots = np.sum(tangents[:-1] * tangents[1:], axis=1)
        bend_angles = np.arccos(np.clip(dots, -1.0, 1.0))
        stats["bend_mean_deg"] = float(np.rad2deg(np.mean(bend_angles)))
        stats["bend_max_deg"] = float(np.rad2deg(np.max(bend_angles)))

    stats["energy_kinetic"] = float(data.energy[0]) if len(data.energy) else 0.0
    return stats


def configure_camera(camera: mujoco.MjvCamera) -> None:
    camera.lookat[:] = CAMERA_LOOKAT
    camera.distance = CAMERA_DISTANCE
    camera.elevation = CAMERA_ELEVATION
    camera.azimuth = CAMERA_AZIMUTH


def run_viewer_mode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sim_duration: float = 10.0,
    log_interval: float = 0.5,
) -> None:
    print("[MODE] Interactive viewer (close window or Ctrl+C to exit)")
    print(f"[INFO] Simulating up to {sim_duration:.1f} s real-time\n")

    next_log_time = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_camera(viewer.cam)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        while viewer.is_running() and data.time < sim_duration:
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time >= next_log_time:
                stats = compute_cable_stats(model, data)
                print(
                    f"  t={data.time:6.3f}s | "
                    f"z_mean={stats.get('z_mean', 0):.4f}m | "
                    f"z_spread={stats.get('z_spread', 0):.4f}m | "
                    f"bend_max={stats.get('bend_max_deg', 0):.1f}deg"
                )
                next_log_time += log_interval

            remaining = model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    print("\n[INFO] Simulation complete.")


def run_headless_mode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sim_duration: float = 5.0,
    log_interval: float = 0.25,
) -> None:
    print("[MODE] Headless (no viewer)")
    print(f"[INFO] Simulating {sim_duration:.1f} s of physics\n")

    n_steps = int(sim_duration / model.opt.timestep)
    log_every = max(1, int(log_interval / model.opt.timestep))
    wall_start = time.perf_counter()

    for step in range(n_steps):
        mujoco.mj_step(model, data)

        if step % log_every == 0:
            stats = compute_cable_stats(model, data)
            print(
                f"  t={data.time:6.3f}s | "
                f"z_min={stats.get('z_min', 0):.4f}m | "
                f"z_mean={stats.get('z_mean', 0):.4f}m | "
                f"z_spread={stats.get('z_spread', 0):.4f}m | "
                f"bend_max={stats.get('bend_max_deg', 0):.1f}deg"
            )

    wall_elapsed = time.perf_counter() - wall_start
    rtf = sim_duration / wall_elapsed
    print(
        f"\n[PERF] Simulated {sim_duration:.1f}s in {wall_elapsed:.2f}s "
        f"({rtf:.1f}x real-time)"
    )


def run_record_mode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sim_duration: float = 5.0,
    fps: int = 60,
    output_path: str = "cable_drape.mp4",
) -> None:
    try:
        import cv2

        has_cv2 = True
    except ImportError:
        has_cv2 = False
        print("[WARN] OpenCV not found; frames will be stored as a numpy array.")

    print(f"[MODE] Record -> {output_path} ({fps} FPS, {sim_duration:.1f}s)")

    width, height = 1280, 720
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    configure_camera(camera)

    frames = []
    render_every = max(1, int((1.0 / fps) / model.opt.timestep))
    n_steps = int(sim_duration / model.opt.timestep)

    for step in range(n_steps):
        mujoco.mj_step(model, data)

        if step % render_every == 0:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

            if step % (render_every * fps) == 0:
                print(f"  Captured frame {len(frames):4d} | t={data.time:.3f}s")

    renderer.close()

    if has_cv2 and frames:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"\n[DONE] Video saved -> {output_path} ({len(frames)} frames)")
    else:
        np_path = output_path.replace(".mp4", "_frames.npy")
        np.save(np_path, np.array(frames))
        print(f"\n[DONE] Frames saved -> {np_path} (shape: {np.array(frames).shape})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soft cable drape/drop simulation")
    parser.add_argument(
        "--mode",
        choices=["viewer", "headless", "record"],
        default="viewer",
        help="Simulation mode (default: viewer)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second for record mode (default: 60)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cable_drape.mp4",
        help="Output file path for record mode",
    )
    parser.add_argument(
        "--no-perturb",
        action="store_true",
        help="Disable the initial tilt used only by free-drop XML scenes",
    )
    parser.add_argument(
        "--bend",
        type=float,
        default=None,
        help="Override XML bend stiffness in Pa",
    )
    parser.add_argument(
        "--twist",
        type=float,
        default=None,
        help="Override XML twist stiffness in Pa",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, data = load_model(args)

    print_model_info(model)

    if args.no_perturb:
        print("[INFO] Initial perturbation disabled.")
    elif apply_initial_perturbation(model, data):
        print("[INFO] Applied 15 degree initial tilt perturbation.")
    else:
        print("[INFO] High-drop fixed-end scene: no root freejoint perturbation needed.")

    if args.mode == "viewer":
        run_viewer_mode(model, data, sim_duration=args.duration)
    elif args.mode == "headless":
        run_headless_mode(model, data, sim_duration=args.duration)
    elif args.mode == "record":
        run_record_mode(
            model,
            data,
            sim_duration=args.duration,
            fps=args.fps,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()



