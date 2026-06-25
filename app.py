"""PCL Sampling App — single entrypoint for GUI and headless use.

    python app.py                 # GUI (default)
    python app.py --headless      # interactive terminal pipeline
    python app.py --headless --mesh path/to/part.stl   # pre-seed mesh, skip the prompt

GUI mode opens the Open3D wizard. Headless mode walks the full synthetic-data
pipeline from the terminal:

    IMPORT_MESH -> RAYCAST -> DOWNSAMPLE -> SAVE -> DECOMPOSE -> SYNTHETIC

CROP is GUI-only (interactive box-select on a live viewport) and is skipped headless.

Headless has two top-level modes:
  * express  - sensible defaults, minimal prompts; runs sampling straight through,
               then pauses for confirmation before the synthetic scene stage.
  * custom   - prompts at every stage; pressing Enter accepts the shown default.

The synthetic stage has its own express/custom choice and works for both
"random" and "structured" arrangements:
  * structured -> one scene per stable resting pose (no fill-rate/count prompt).
  * random     -> N scenes; express sweeps fill rate 20%->100%, custom prompts
                  fill-rate/count per scene.
"""

from enums import *
# import open3d.core as o3c
from open3d.geometry import Geometry3D
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
from stages.import_mesh_stage import ImportMeshStage
from stages.raycast_stage import RaycastStage
from stages.crop_stage import CropStage
from stages.downsample_stage import DownsampleStage
from stages.save_stage import SaveStage
from stages.decompose_stage import DecomposeStage
from stages.scene_stage import SceneStage
from stages.render_stage import RenderStage
import threading
import time
from pathlib import Path
from rich import print as rp
from geometry.file_utils import pointcloud_to_ply, open_source_folder_dialog
from geometry.geom_utils import O3DSceneObject, camera_view_matrix, o3d_to_trimesh
from physics.mujoco_bin_scene import MujocoBinScene
# import cupy as cp
# print(cp.cuda.runtime.getDeviceCount())

class MeshSamplingApp:

    def __init__(self, headless=False, mesh_path=None):
        self.headless = headless
        self.express_sampling_busy = False   # blocks the Express Sampling button while a run is in flight

        self.mesh_path = Path(mesh_path) if mesh_path is not None else None
        self.stages = {
            Stage.IMPORT_MESH: ImportMeshStage(self),
            Stage.RAYCAST: RaycastStage(self),
            Stage.CROP: CropStage(self),
            Stage.DOWNSAMPLE: DownsampleStage(self),
            Stage.SAVE: SaveStage(self),
            Stage.DECOMPOSE: DecomposeStage(self),
            Stage.SCENE: SceneStage(self),
            Stage.RENDER: RenderStage(self),
        }
        # Give each stage a back-reference to its own enum key so reset()/clear_state_from
        # can locate it without a reverse lookup.
        for st, inst in self.stages.items():
            inst.stage_key = st
        if not self.headless:
            # === Scene widget ===
            self.window_width = 1440
            self.window_height = 900
            self.window = gui.Application.instance.create_window("Mesh Sampling Wizard", self.window_width, self.window_height)
            self.scene = gui.SceneWidget()
            self.scene.scene = rendering.Open3DScene(self.window.renderer)
            self.scene.scene.set_background([0.2, 0.2, 0.2, 1.0])
            self.scene_geoms = {}
            self.window.set_on_layout(self._on_layout)
            self.window.set_on_key(self._on_key)
            self.scene.set_on_mouse(self._on_mouse_event)
            self.window.add_child(self.scene)

            # === Materials ===
            self.default_material = rendering.MaterialRecord()
            self.default_material.shader = "defaultLit"
            self.default_point_material = rendering.MaterialRecord()
            self.default_point_material.point_size = 1.5
            self.default_point_material.base_color = [1.0, 1.0, 1.0, 1.0]
            self.overlay_material = rendering.MaterialRecord()
            self.overlay_material.point_size = 1.5
            self.overlay_material.base_color = [1.0, 0.5, 0.3, 1.0]

            # ===============================
            # Build Control Panel
            # ===============================
            em = self.window.theme.font_size
            self.panel = gui.Vert(0.25 * em, gui.Margins(em, em, em, em))
            self.window.add_child(self.panel)

            for stage_class in self.stages.values():
                stage_class.panel.visible = False
                self.panel.add_child(stage_class.panel)

            # === Progress Bar widget ===
            self.progress_panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
            self.progress_panel.visible = False
            self.progress_label = gui.Label("Processing...")
            self.progress_bar = gui.ProgressBar()
            self.progress_bar.value = 0.0  # range [0, 1]
            self.progress_panel.add_child(self.progress_label)
            self.progress_panel.add_child(self.progress_bar)
            self.pb_panel_size = (400, 50)
            x=(self.window_width - self.pb_panel_size[0])>>1
            y=(self.window_height - self.pb_panel_size[1])>>1
            self.progress_panel.frame = gui.Rect(x, y, self.pb_panel_size[0], self.pb_panel_size[1])
            self.window.add_child(self.progress_panel)

        # Pre-seed the mesh path so headless callers skip the file dialog.
        if self.mesh_path is not None:
            self.stages[Stage.IMPORT_MESH].file_path = self.mesh_path

        self._restart()

    def _restart(self):
        # Single source of truth: every app.* pipeline attribute is declared in some
        # stage's `downstream` map (see stages/*.py). reset_all_state() applies them all,
        # so adding/removing state means editing one stage, never this method.
        self.reset_all_state()
        self.stage = Stage.IMPORT_MESH
        self.set_stage(Stage.IMPORT_MESH)

    # ===============================
    # State clearing (driven by per-stage `downstream` declarations)
    # ===============================
    def clear_state_from(self, stage: Stage, inclusive: bool = True):
        """Reset the owned state of `stage` (when inclusive) and every later stage to
        defaults, by strict pipeline order (Stage enum value)."""
        start = stage.value if inclusive else stage.value + 1
        for st, inst in self.stages.items():
            if st.value >= start:
                inst.clear_produced()

    def reset_all_state(self):
        """Reset every stage's owned state to defaults."""
        for inst in self.stages.values():
            inst.clear_produced()

    def set_stage(self, stage: Stage):
        self.stage = stage
        if self.headless:
            return
        for s in self.stages.values():
            s.panel.visible = (s is self.stages[stage])
            for w in s.widgets:
                if hasattr(w.widget, "toggleable") and w.widget.toggleable:
                    w.widget.is_on = False
        self.window.set_needs_layout()
        self.stages[stage]._refresh_ui()
        self.stages[stage].enable_widgets()
        self._update_title()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 300
        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

    # ===============================
    # UI helpers
    # ===============================
    def _update_title(self):
        self.window.title = f"Mesh Sampling Wizard | Stage: {self.stage.name}"

    def _reframe(self, fov_deg=41.1, margin=1.0):
        """
        Dynamically frame object based on its bounding box size.
        """
        if self.headless:
            return
        if self.target_mesh is None:
            return
        else:
            if self.stage in (Stage.SCENE, Stage.RENDER) and self.mj_scene is not None:
                mj_scene = self.mj_scene
                look_at, cam_pos, up = mj_scene._get_camera_lookat()
                bbox = mj_scene.bin_mesh.get_axis_aligned_bounding_box()
            else:
                bbox = self.target_mesh.get_axis_aligned_bounding_box()
                look_at = bbox.get_center()
                distance = 1.0 * np.linalg.norm(bbox.get_extent())
                if distance < 1e-6:
                    return
                cam_pos = look_at + np.array([distance, distance, distance])
                T_cam = camera_view_matrix(cam_pos, look_at)
                up = T_cam[:3, 1]

        self.scene.center_of_rotation = look_at
        self.scene.setup_camera(fov_deg, bbox, look_at)
        self.scene.scene.camera.look_at(look_at, cam_pos, up)
        self.scene.force_redraw()
        print("reframed")

    def show_progress(self, text="Processing..."):
        if self.headless:
            return
        def _show():
            self.progress_label.text = text
            self.progress_bar.value = 0.0
            self.progress_panel.visible = True
        gui.Application.instance.post_to_main_thread(self.window, _show)

    def update_progress(self, value, text="Processing..."):
        if self.headless:
            return
        value = max(0.0, min(1.0, value))
        def _update():
            self.progress_label.text = text
            self.progress_bar.value = value
        gui.Application.instance.post_to_main_thread(self.window, _update)

    def hide_progress(self):
        if self.headless:
            return
        def _hide():
            self.progress_panel.visible = False
        gui.Application.instance.post_to_main_thread(self.window, _hide)

    def _clear_scene(self):
        self.scene_geoms = {}
        self.scene.scene.clear_geometry()

    def add_geom_in_scene(self, name:str, geom: Geometry3D, color=[1.0, 1.0, 1.0], alpha=1.0, point_size=1.5):
        material = rendering.MaterialRecord()
        material.point_size = point_size
        material.base_color = color + [alpha]
        material.shader = "defaultLit"
        self.scene_geoms[name] = O3DSceneObject(geom, material)
        self.main_thread(lambda: self.scene.scene.add_geometry(name, geom, material))

    def remove_geom_in_scene(self, name:str):
        self.scene_geoms.pop(name)
        self.main_thread(lambda: self.scene.scene.remove_geometry(name))

    def hide_geoms_in_scene(self, geoms=[]):
        def hide_geoms():
            if not geoms:
                print("no specified geom, hiding all")
                for name in self.scene_geoms.keys():
                    print(f"hiding {name}")
                    self.scene.scene.show_geometry(name, show=False)
            else:
                print(f"geoms = {geoms}")
                for name in geoms:
                    print(f"hiding {name}")
                    self.scene.scene.show_geometry(name, show=False)
        self.main_thread(hide_geoms)

    def show_geoms_in_scene(self, geoms:list=[]):
        def show_geoms():
            if not geoms: print("no specified geom, showing all")
            print(geoms)

            for name in (geoms if geoms else self.scene_geoms.keys()):
                print(f"showing {name}")
                self.scene.scene.show_geometry(name, show=True)
        self.main_thread(show_geoms)

    def has_geom(self, name:str):
        self.main_thread(lambda: self.scene.scene.has_geometry(name))

    def main_thread(self, fn):
        if self.headless:
            return
        gui.Application.instance.post_to_main_thread(self.window, fn)

    # ===============================
    # Keybindings
    # ===============================
    def _on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return False

        key = event.key

        # --- Global ---
        if key == gui.KeyName.R:
            self._reframe()
            return True

        # --- Stage specific ---
        if self.stage in self.stages:
            self.stages[self.stage]._on_key(event)

        return True

    def _on_mouse_event(self, event):
        if self.stage in self.stages:
            self.stages[self.stage]._on_mouse_event(event)
        return gui.Widget.EventCallbackResult.IGNORED

    # ===============================
    # Express Handler
    # ===============================
    def start_express_sampling(self):
        # Block the button immediately; cleared in the worker's finally (done or failed).
        self.express_sampling_busy = True
        self.stages[Stage.IMPORT_MESH].enable_widgets()
        self.stages[Stage.IMPORT_MESH].center_mesh()
        self._express_sampling_thread = threading.Thread(target=self._express_sampling_worker)
        self._express_sampling_thread.start()

    def _express_sampling_worker(self):
        try:
            self.stages[Stage.RAYCAST].worker()
            self.down_pcd=self.raw_pcd
            self.stages[Stage.DOWNSAMPLE].worker()
            self.stages[Stage.DOWNSAMPLE].recenter_mesh_pcd()
            self.main_thread(lambda: self.set_stage(Stage.SAVE))
            self.hide_progress()
        finally:
            self.express_sampling_busy = False
            self.main_thread(self.stages[Stage.IMPORT_MESH].enable_widgets)

    def start_batch_sampling(self):
        self.src_dir = open_source_folder_dialog()
        if self.src_dir is None:
            print("No source path selected")
            return

        self._batch_sampling_thread = threading.Thread(target=self._batch_sampling_worker)
        self._batch_sampling_thread.start()

    def _batch_sampling_worker(self):
        dst_dir = Path.cwd() / "reference_pcd"
        if dst_dir is None:
            print("No destination path selected")
            return

        stl_files = list(self.src_dir.glob("*.stl"))
        print(f"[INFO] Found {len(stl_files)} STL files")

        for stl_path in stl_files:
            try:
                print(f"[INFO] Processing {stl_path.name}")
                self.stages[Stage.IMPORT_MESH].file_path = stl_path
                self.stages[Stage.IMPORT_MESH].worker()
                self.stages[Stage.IMPORT_MESH].center_mesh()
                self._express_sampling_worker()
                pointcloud_to_ply(self.down_pcd, str(dst_dir / (stl_path.stem + ".ply")))

                if self.headless:
                    continue
                self.main_thread(lambda: self._clear_scene())
                self.main_thread(lambda: self.scene.scene.add_geometry("down_pcd", self.down_pcd, self.default_point_material))
                # self.hide_geoms_in_scene()
                # self.add_geom_in_scene("down_pcd", self.down_pcd)
                self.scene.force_redraw()
                self._reframe()
            except Exception as e:
                print(f"[ERROR] Failed to process {stl_path.name}: {e}")
                continue


# ===========================================================================
# Headless interactive driver
# ===========================================================================
def _hint(text):
    if text:
        rp(f"[dim]  {text}[/dim]")


def ask(prompt, default, cast=str, hint=None):
    """Prompt with a default; empty input returns the default. Retries on bad cast."""
    _hint(hint)
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip().strip('"').strip("'")
        if not raw:
            return default
        try:
            return cast(raw)
        except (ValueError, TypeError):
            rp(f"[red]Invalid value '{raw}', expected {cast.__name__}.[/red]")


def ask_choice(prompt, options, default, hint=None):
    """Single-char menu select. Each option gets a one-char key: its unique initial
    letter, or a 1-based digit if initials collide. Only the first typed char is read."""
    _hint(hint)
    firsts = [o[0].lower() for o in options]
    use_letters = len(set(firsts)) == len(firsts)
    keys = firsts if use_letters else [str(i + 1) for i in range(len(options))]

    menu = "  ".join(f"[{k}] {o}" for k, o in zip(keys, options))
    default_key = keys[options.index(default)]
    while True:
        raw = input(f"{prompt}\n  {menu}\n  select [{default_key}]: ").strip().lower()
        if not raw:
            return default
        ch = raw[0]                      # only one char is honoured
        if ch in keys:
            return options[keys.index(ch)]
        rp(f"[red]Press one of: {', '.join(keys)}[/red]")


def ask_yes_no(prompt, default=True, hint=None):
    """Single-char y/n; only the first typed char is read."""
    _hint(hint)
    d = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{d}]: ").strip().lower()
    if not raw:
        return default
    return raw[0] == "y"


def banner(text):
    rp(f"\n[bold cyan]=== {text} ===[/bold cyan]")


def run_sampling(app, express):
    """Run raycast + downsample (both modes funnel through the express worker,
    which reads stage attributes in headless mode), then export the reference
    bundle and decompose the mesh."""
    raycast = app.stages[Stage.RAYCAST]
    downsample = app.stages[Stage.DOWNSAMPLE]

    if express:
        downsample.use_adaptive = True
    else:
        banner("Raycast settings")
        raycast.camera_distance = ask(
            "Camera distance (m)", raycast.camera_distance, float,
            hint="Distance of the virtual camera from the part on the view sphere; "
                 "larger sees more but at lower resolution.")
        raycast.num_views = ask(
            "Number of views", raycast.num_views, int,
            hint="How many viewpoints around the part to raycast and merge into the "
                 "reference cloud; more = denser coverage but slower.")
        banner("Downsample settings")
        downsample.use_adaptive = ask_yes_no(
            "Use adaptive (curvature-based) downsampling?", default=False,
            hint="Adaptive keeps more points on edges/high-curvature regions; "
                 "uniform samples the surface evenly.")

    rp("[yellow]Note: CROP stage is GUI-only and is skipped in headless mode.[/yellow]")

    t0 = time.time()
    app._express_sampling_worker()   # raycast -> downsample -> recenter -> set_stage(SAVE)
    rp(f"[green]Sampling took {time.time() - t0:.1f}s[/green]")
    rp(f"mean point count = {raycast.point_count_mean}")
    rp(f"point count range = {raycast.point_count_range}")

    # Reference bundle is always exported (app.stage == SAVE after sampling).
    banner("Exporting reference point-cloud bundle")
    app.stages[Stage.SAVE].worker()

    # Convex decomposition as its own synchronous stage (fills app.convex_meshes).
    banner("Decomposing mesh (convex hulls)")
    t0 = time.time()
    app.stages[Stage.DECOMPOSE]._run_worker()
    rp(f"[green]Decomposition took {time.time() - t0:.1f}s "
       f"-> {len(app.convex_meshes)} hulls[/green]")


def generate_one_scene(app, scene_label):
    scene = app.stages[Stage.SCENE]
    render = app.stages[Stage.RENDER]
    rp(f"[bold]Generating {scene_label}...[/bold]")
    t0 = time.time()
    scene._run_worker()      # build + settle the physical scene -> app.o3d_scene / app.mj_scene
    render._run_worker()     # sensor sim + segmentation -> app.synthetic_targets
    rp(f"  {len(app.synthetic_targets)} valid targets in {time.time() - t0:.1f}s")
    render.save_synthetic_targets()


def run_synthetic(app, express):
    scene = app.stages[Stage.SCENE]
    # Required so save_synthetic_targets() -> SaveStage.worker takes the RENDER
    # branch and writes reference_cloud.ply into each scene directory.
    app.set_stage(Stage.RENDER)

    banner("Synthetic scene generation")
    # Arrangement is asked in BOTH express and custom modes.
    arrangement = ask_choice(
        "Arrangement", ["random", "structured"], "random",
        hint="random = parts dropped & physically settled in the bin (clutter/occlusion); "
             "structured = a grid of one stable resting pose, one scene per pose.")

    if arrangement == "structured":
        # One scene per stable resting pose; no fill-rate / count needed.
        part_mesh = o3d_to_trimesh(app.target_mesh)
        poses = MujocoBinScene.get_stable_poses(part_mesh)
        rp(f"[cyan]Found {len(poses)} stable pose(s); generating one scene each.[/cyan]")
        scene.arrangement = "structured"
        structure = ask_choice(
            "Structure", ["none", "partition", "tray"], "none",
            hint="none = bare grid (static); partition = cardboard egg-crate dividers; "
                 "tray = injection-molded pockets tracing the part footprint. partition/tray settle "
                 "the parts under gravity.")
        scene.structure_type = structure
        if structure != "none":
            pct = ask(
                "Structure height (% of part height)", scene.structure_height_pct, int,
                hint="partition = divider height; tray = pocket depth. 50-100% of part height "
                     "(higher = more enclosed / less exposed).")
            scene.structure_height_pct = max(50, min(100, pct))
            scene.clearance_mode = ask_choice(
                "Fit clearance", ["snug", "medium", "loose"], scene.clearance_mode,
                hint="part-to-wall running clearance; snug (~1 mm) holds tighter, "
                     "loose (~5% of footprint) settles more easily.")
        for i, (R_stable, prob) in enumerate(poses):
            scene.stable_pose_R = R_stable
            generate_one_scene(app, f"structured scene {i + 1}/{len(poses)} (p={prob:.2f})")
        scene.stable_pose_R = None
        return

    # arrangement == "random"
    scene.arrangement = "random"
    scene.stable_pose_R = None
    n_scenes = ask(
        "Number of scenes to generate", 1, int,
        hint="Each scene is a full physics sim + sensor-noise render (can take minutes). "
             "In express mode, >1 sweeps fill rate 20%->100% across the scenes.")
    n_scenes = max(1, n_scenes)

    if express:
        if n_scenes == 1:
            scene.generate_mode = "fill_rate"
            scene.fill_rate = 0.6
            generate_one_scene(app, "scene 1/1 (fill 60%)")
        else:
            # Sweep fill rate 20% -> 100% across the scenes.
            for i, fr in enumerate(np.linspace(0.2, 1.0, n_scenes)):
                scene.generate_mode = "fill_rate"
                scene.fill_rate = float(fr)
                generate_one_scene(app, f"scene {i + 1}/{n_scenes} (fill {fr:.0%})")
    else:
        # Custom: prompt per scene.
        for i in range(n_scenes):
            banner(f"Scene {i + 1}/{n_scenes} settings")
            mode = ask_choice(
                "Generate mode", ["fill_rate", "count"], scene.generate_mode,
                hint="fill_rate = auto-size the part count to a target bin fill %; "
                     "count = drop an exact number of parts.")
            scene.generate_mode = mode
            if mode == "fill_rate":
                pct = ask(
                    "Fill rate (%)", int(round(scene.fill_rate * 100)), int,
                    hint="Target volumetric fill of the bin; higher = more parts, "
                         "more clutter and occlusion.")
                scene.fill_rate = max(0.0, min(1.0, pct / 100.0))
            else:
                scene.num_targets = ask(
                    "Part count", scene.num_targets, int,
                    hint="Exact number of parts to drop into the bin.")
            generate_one_scene(app, f"scene {i + 1}/{n_scenes}")


def run_headless(mesh_arg=None):
    banner("PCL Sampling App - headless (beta)")

    # 1. Mesh path (CLI arg pre-seeds and skips the prompt when valid).
    mesh_path = mesh_arg if (mesh_arg and Path(mesh_arg).is_file()) else None
    if mesh_arg and mesh_path is None:
        rp(f"[red]--mesh not found: {mesh_arg}[/red]")
    while mesh_path is None:
        candidate = ask(
            "Path to input mesh (STL)", None,
            hint="The part mesh to sample a reference cloud from and populate scenes with. "
                 "mm meshes are auto-converted to metres.")
        if candidate and Path(candidate).is_file():
            mesh_path = candidate
        else:
            rp(f"[red]File not found: {candidate}[/red]")

    app = MeshSamplingApp(headless=True, mesh_path=mesh_path)
    app.stages[Stage.IMPORT_MESH]._run_worker()
    if app.target_mesh is None:
        rp("[red]Failed to import mesh. Aborting.[/red]")
        return
    rp(f"[green]Imported {app.mesh_basename}[/green]")

    # 1b. Centering
    if ask_yes_no("Center mesh at origin before raycasting?", default=True,
                  hint="Translates the mesh centroid to the world origin. "
                       "Recommended for consistent view-sphere coverage."):
        app.stages[Stage.IMPORT_MESH].center_mesh()

    # 2. Top-level mode
    mode = ask_choice(
        "Processing mode", ["express", "custom"], "express",
        hint="express = sensible defaults, minimal prompts; "
             "custom = configure raycast/downsample/synthetic at each stage.")
    express = (mode == "express")

    # 3-5. Sampling + reference export + decompose
    run_sampling(app, express)

    # 6. Express confirmation gate before the synthetic stage.
    if express:
        if not ask_yes_no(
                "\nSampling complete. Start synthetic scene stage?", default=True,
                hint="Runs the physics sim + sensor-noise pipeline to generate training "
                     "scenes. Answer 'n' to stop now with just the reference bundle."):
            rp("[yellow]Stopping before synthetic stage.[/yellow]")
            return

    # 7. Synthetic stage (own express/custom choice).
    synth_mode = ask_choice(
        "Synthetic generation mode", ["express", "custom"],
        "express" if express else "custom",
        hint="express = auto fill-rate (or 20%->100% sweep for >1 scene); "
             "custom = set fill-rate or exact count per scene.")
    run_synthetic(app, synth_mode == "express")

    # 8. Report output locations.
    banner("Done")
    rp(f"Reference bundle : output/reference_pcd/{app.mesh_basename}/")
    rp(f"Synthetic scenes : output/synthetic_target/{app.mesh_basename}/scene_*/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PCL Sampling App")
    parser.add_argument("--headless", action="store_true",
                        help="Run the interactive terminal pipeline instead of the GUI")
    parser.add_argument("--mesh", type=str, default=None,
                        help="Pre-seed input mesh path (headless); skips the mesh prompt")
    args = parser.parse_args()

    if args.headless or args.mesh:
        run_headless(mesh_arg=args.mesh)
    else:
        try:
            gui.Application.instance.initialize()
            app = MeshSamplingApp()
            gui.Application.instance.run()
        except Exception as e:
            print(f"[FATAL] Unhandled exception: {e}")
            exit()
