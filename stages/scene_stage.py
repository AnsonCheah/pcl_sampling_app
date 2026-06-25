import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import trimesh
from enums import Stage
from stages.stage_base import BaseStage
from geometry.geom_utils import o3d_to_trimesh, trimesh_to_o3d, O3DSceneObject
from physics.mujoco_bin_scene import MujocoBinScene, MAX_BIN_DIM
import copy
import colorsys
from rich import print as rp
np.set_printoptions(precision=6, suppress=True)

SCENE_FILL_RATE     = 0.6    # default fill rate (fraction of safe capacity)
# Shape-aware random-packing efficiency for part OBB volumes. A fixed factor is only valid for
# cube-like parts; random packing fraction falls ~1/aspect_ratio for elongated/flat parts
# (Philipse random-contact scaling; see _auto_part_count). We taper from a cube anchor by the
# OBB edge ratio so big elongated parts no longer over-count.
PACKING_FACTOR_BASE  = 0.62  # cube/blocky parts (sphere random-close-pack band)
PACKING_FACTOR_FLOOR = 0.18  # keep very thin/long parts from collapsing toward 0
BIN_TOP_MARGIN_FRAC = 0.20   # reserve top 20% of bin height as spill headroom
MIN_AUTO_PARTS      = 2      # never fewer than this
MAX_AUTO_PARTS      = 500    # safety cap (tune): tiny parts otherwise explode the sim
FILL_SLIDER_MIN, FILL_SLIDER_MAX   = 20, 100   # fill-rate slider range (percent)
COUNT_SLIDER_MIN, COUNT_SLIDER_MAX = 2, 500    # override-count slider range (parts)
PREVIEW_PARKED_Z = 1.0   # m — live-preview bodies above this z are still parked (PARKING_Z≈10);
#                          hide them until release_batch() drops them onto the pile (bin ~0.25 m tall)


class SceneStage(BaseStage):
    """Build the physical bin scene: arrange parts (random pile or structured grid, optionally with
    partitions/tray), settle under MuJoCo, and hand the resulting meshes + GT poses to RenderStage via
    app attributes (app.o3d_scene, app.mj_scene, app.scene_mesh)."""

    downstream = {
        "o3d_scene": lambda: {},    # SceneStage -> RenderStage handoff: physical scene meshes + GT poses
        "mj_scene": lambda: None,   # SceneStage -> RenderStage handoff: MujocoBinScene (GT/bin export, camera)
        "scene_mesh": lambda: None,  # SceneStage physical-scene preview mesh
    }

    def __init__(self, app):
        self.name = Stage.SCENE.name
        # Set before super().__init__(): it calls build_panel(), which reads these.
        self.num_targets = 6
        self.arrangement = "random"        # headless callers may set before _run_worker()
        self.generate_mode = "fill_rate"   # "fill_rate" (auto-size to part) or "count" (manual override)
        self.fill_rate = SCENE_FILL_RATE   # used when generate_mode == "fill_rate"
        # Structured-scene structure (Phase 2): "none" | "partition" | "tray"
        self.structure_type = "none"
        self.structure_height_pct = 75     # divider height / pocket depth as % of part height
        self.clearance_mode = "medium"     # "snug" | "medium" | "loose"
        super().__init__(app)
        self.rendering_flag = False
        self.verbose = True
        self.o3d_scene = {}
        self.mj_scene = None
        self.stable_pose_R = None   # optional forced stable orientation (structured mode, set per-pose by headless driver)

    def build_panel(self):
        if self.app.headless:
            return
        v = gui.Vert(4)

        self.arrangement_combo = self.register_widget(gui.Combobox())
        self.arrangement_combo.add_item("Random")
        self.arrangement_combo.add_item("Structured")
        self.arrangement_combo.selected_index = 0
        self.arrangement_combo.set_on_selection_changed(self._on_arrangement_changed)
        # One slider, repurposed by the radio: fill-rate % (auto) or override count.
        self.radio_mode = self.register_widget(gui.RadioButton(gui.RadioButton.HORIZ),
                                               enabled_if=lambda: self._is_random())
        self.radio_mode.set_items(["By Fill Rate", "Override Count"])
        self.radio_mode.selected_index = 0
        self.radio_mode.set_on_selection_changed(self._on_generate_mode_changed)
        self.gen_label = gui.Label("Fill Rate (%)")   # plain label, not registered
        self.gen_slider = self.register_widget(gui.Slider(gui.Slider.INT),
                                               enabled_if=lambda: self._is_random())
        self._apply_slider_mode(0)                     # init in fill-rate mode

        # --- Structured-scene structure (partition / tray) ---
        self.structure_radio = self.register_widget(gui.RadioButton(gui.RadioButton.HORIZ),
                                                    enabled_if=lambda: not self._is_random())
        self.structure_radio.set_items(["None", "Partition", "Tray"])
        self.structure_radio.selected_index = 0
        self.structure_radio.set_on_selection_changed(self._on_structure_changed)
        self.structure_slider_label = gui.Label("Structure Height (%)")   # plain label, not registered
        self.structure_slider = self.register_widget(
            gui.Slider(gui.Slider.INT),
            enabled_if=lambda: (not self._is_random()) and self._structure_type() != "none")
        self.structure_slider.set_limits(50, 100)      # limits BEFORE value
        self.structure_slider.int_value = int(self.structure_height_pct)
        self.clearance_label = gui.Label("Fit Clearance")                 # plain label, not registered
        self.clearance_radio = self.register_widget(
            gui.RadioButton(gui.RadioButton.HORIZ),
            enabled_if=lambda: (not self._is_random()) and self._structure_type() != "none")
        self.clearance_radio.set_items(["Snug", "Medium", "Loose"])
        self.clearance_radio.selected_index = 1        # medium

        self.btn_generate = self.register_widget(gui.Button("Generate Scene"))
        self.btn_generate.set_on_clicked(self.start)
        self.btn_reset = self.register_widget(gui.Button("Clear Scene"), lambda: self.app.mj_scene is not None)
        self.btn_reset.set_on_clicked(self.reset)

        self.btn_next = self.register_widget(gui.Button("Next: Render"), lambda: len(self.app.o3d_scene) > 0)
        self.btn_next.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value + 1)))
        self.btn_back = self.register_widget(gui.Button("Back: Decompose"))
        self.btn_back.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value - 1)))

        self.btn_restart = self.register_widget(gui.Button("Restart"))
        self.btn_restart.set_on_clicked(lambda: self.app._restart())
        v.add_child(gui.Label("Generate Physical Scene"))
        v.add_child(gui.Label(""))
        v.add_child(self.arrangement_combo)
        v.add_child(self.radio_mode)
        v.add_child(self.gen_label)
        v.add_child(self.gen_slider)
        v.add_child(gui.Label("Structure"))
        v.add_child(self.structure_radio)
        v.add_child(self.structure_slider_label)
        v.add_child(self.structure_slider)
        v.add_child(self.clearance_label)
        v.add_child(self.clearance_radio)
        v.add_child(self.btn_generate)
        v.add_child(self.btn_reset)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(self.btn_back)
        v.add_child(self.btn_next)
        v.add_child(self.btn_restart)

        print("loaded scene panel")
        return v

    def _is_random(self) -> bool:
        return self.arrangement_combo.selected_text == "Random"

    def _on_arrangement_changed(self, text, idx):
        # Predicates drive enable/disable; just re-evaluate them.
        self.enable_widgets()

    def _structure_type(self) -> str:
        return ["none", "partition", "tray"][self.structure_radio.selected_index]

    def _clearance_mode(self) -> str:
        return ["snug", "medium", "loose"][self.clearance_radio.selected_index]

    def _apply_structure_label(self, idx: int):
        self.structure_slider_label.text = {
            0: "Structure Height (%)", 1: "Partition Height (%)", 2: "Pocket Depth (%)"}[idx]

    def _on_structure_changed(self, idx: int):
        self._apply_structure_label(idx)
        self.enable_widgets()

    def _apply_slider_mode(self, idx: int):
        """Repurpose the single slider: idx 0 = fill-rate %, idx 1 = override count."""
        if idx == 0:
            self.gen_label.text = "Fill Rate (%)"
            self.gen_slider.set_limits(FILL_SLIDER_MIN, FILL_SLIDER_MAX)   # limits BEFORE value
            self.gen_slider.int_value = int(round(self.fill_rate * 100))
        else:
            self.gen_label.text = "Override Count"
            self.gen_slider.set_limits(COUNT_SLIDER_MIN, COUNT_SLIDER_MAX)
            self.gen_slider.int_value = int(self.num_targets)

    def _on_generate_mode_changed(self, idx: int):
        # Persist the value from the mode being left, then repurpose the slider.
        if idx == 0:                                   # leaving count -> fill
            self.num_targets = int(self.gen_slider.int_value)
        else:                                          # leaving fill -> count
            self.fill_rate = self.gen_slider.int_value / 100.0
        self.generate_mode = "fill_rate" if idx == 0 else "count"
        self._apply_slider_mode(idx)
        self.enable_widgets()

    def _auto_part_count(self, part_mesh) -> int:
        """Size the part count to the part so the fixed max-size bin reaches a
        consistent volumetric fill across all parts, reserving BIN_TOP_MARGIN_FRAC
        of the bin height as spill headroom. See module constants.

        The packing factor is shape-aware: it tapers from PACKING_FACTOR_BASE by the OBB
        aspect ratio (longest/shortest edge) so elongated/flat parts — which pack far less
        densely — no longer over-count. We do NOT also multiply by solidity: the count divides
        by OBB volume, so the correct multiplier is the box-packing fraction (folding solidity
        in again would double-count). Anchors: AR 1->0.62, 2->0.44, 4->0.31, >=12->0.18 floor."""
        bw, bl, bh, _ = MAX_BIN_DIM
        usable_h = bh * (1.0 - BIN_TOP_MARGIN_FRAC)   # reserve top headroom to prevent spilling
        bin_vol  = bw * bl * usable_h
        mobb_vol = max(float(part_mesh.bounding_box_oriented.volume), 1e-9)
        ext = np.sort(part_mesh.bounding_box_oriented.extents)[::-1]   # e1 >= e2 >= e3
        aspect_ratio   = float(ext[0] / max(ext[2], 1e-9))
        packing_factor = max(PACKING_FACTOR_FLOOR, PACKING_FACTOR_BASE / np.sqrt(aspect_ratio))
        n = round(self.fill_rate * packing_factor * bin_vol / mobb_vol)
        rp(f"[PACKING] OBB aspect ratio={aspect_ratio:.2f} -> packing_factor={packing_factor:.3f}")
        return int(np.clip(n, MIN_AUTO_PARTS, MAX_AUTO_PARTS))

    def _display_convex_meshes(self):
        """GUI helper: show each convex hull in a distinct HSV colour."""
        n = len(self.app.convex_meshes)
        for i, mesh in enumerate(self.app.convex_meshes):
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            rgb = colorsys.hsv_to_rgb(i / max(n, 1), 0.6, 0.9)
            material.base_color = list(rgb) + [1.0]
            geom = o3d.geometry.TriangleMesh(mesh)
            geom.compute_vertex_normals()
            self.app.scene.scene.add_geometry(f"convex_{i}", geom, material)
        self.app.main_thread(self.app._reframe)

    def _show_scene_mesh(self):
        """GUI helper: show the settled physical scene, framed on the bin. Everything runs on the main
        thread (Open3D GUI is not thread-safe — touching it from the worker silently corrupts the scene,
        which is what blanked the tray view). If the merged preview mesh is missing/empty it falls back
        to the per-object settled meshes so the scene never blanks out."""
        def show():
            self.app._clear_scene()
            sm = self.app.scene_mesh
            if sm is not None and len(sm.vertices) > 0:
                self.app.scene.scene.add_geometry("scene_mesh", sm, self.app.default_material)
            else:
                for name, obj in self.app.o3d_scene.items():
                    g = copy.deepcopy(obj.geom)
                    if obj.T_gt is not None:
                        g.transform(obj.T_gt)
                    self.app.scene.scene.add_geometry(name, g, self.app.default_material)
            self.app._reframe()
            self.app.scene.force_redraw()
        self.app.main_thread(show)

    def _refresh_ui(self):
        if self.app.headless:
            return
        if self.app.scene_mesh is not None:
            # Scene already generated — keep the settled physical scene on screen.
            self._show_scene_mesh()
        else:
            # Stage entry, before generation: show the decomposed convex hulls.
            self.app.main_thread(lambda: self.app._clear_scene())
            if len(self.app.convex_meshes) > 0:
                self.app.main_thread(self._display_convex_meshes)
            elif self.app.target_mesh is not None:
                self.app.main_thread(lambda: self.app.scene.scene.add_geometry(
                    "mesh", self.app.target_mesh, self.app.default_material))
        self.enable_widgets()

    def on_clear(self):
        # Reset stage-local scratch alongside the app-level scene state.
        self.o3d_scene = {}
        self.mj_scene = None
        self.worker_step = 0

    def worker(self):
        TOTAL_STEPS = 4
        self.app.clear_state_from(self.stage_key)

        # The MuJoCo passive viewer is blocking and must never run headless/batch.
        if self.app.headless:
            self.rendering_flag = False
        if not self.app.convex_meshes:
            print("[WARN] No convex meshes available — run DecomposeStage before SCENE; "
                  "simulation collisions will be degraded.")

        if not self.app.headless:
            self.app.show_progress("Simulating physical scene...")
            self.app.main_thread(lambda: self.app._clear_scene())
            self.arrangement = self.arrangement_combo.selected_text.lower()
            self.generate_mode = "fill_rate" if self.radio_mode.selected_index == 0 else "count"
            if self.generate_mode == "fill_rate":
                self.fill_rate = self.gen_slider.int_value / 100.0
            else:
                self.num_targets = int(self.gen_slider.int_value)
            self.structure_type = self._structure_type()
            self.structure_height_pct = int(self.structure_slider.int_value)
            self.clearance_mode = self._clearance_mode()

        def _update_pb(message: str):
            if not self.app.headless:
                print(message)
            self.worker_step += 1
            self.app.update_progress(self.worker_step / TOTAL_STEPS, message)

        # MujocoBinScene requires mesh centered at its own origin: every rotation, spawn-height,
        # stable-pose, and tray-pocket calculation rotates vertices around (0,0,0). Center a
        # local copy without modifying app.target_mesh so the user's centering choice is preserved.
        mesh_center = np.asarray(self.app.target_mesh.get_center())
        needs_centering = np.linalg.norm(mesh_center) > 1e-9
        part_mesh = o3d_to_trimesh(self.app.target_mesh)
        if needs_centering:
            part_mesh.apply_translation(-mesh_center)
        if needs_centering and self.app.convex_meshes:
            physics_convex = []
            for cm in self.app.convex_meshes:
                c = copy.deepcopy(cm)
                c.translate(-mesh_center)
                physics_convex.append(c)
        else:
            physics_convex = self.app.convex_meshes
        # Fill-rate mode auto-sizes count to the part; structured ignores n_parts (grid sets it).
        if self.generate_mode == "fill_rate" and self.arrangement == "random":
            self.num_targets = self._auto_part_count(part_mesh)
            rp(f"[AUTO-COUNT] fill={self.fill_rate:.0%} "
               f"part OBB vol={part_mesh.bounding_box_oriented.volume:.2e} m³ "
               f"-> n_parts={self.num_targets}")
        self.mj_scene = MujocoBinScene(part_mesh, physics_convex, n_parts=self.num_targets,
                                       render=self.rendering_flag, arrangement=self.arrangement,
                                       stable_pose_R=self.stable_pose_R,
                                       structure_type=self.structure_type,
                                       structure_height_frac=self.structure_height_pct / 100.0,
                                       clearance_mode=self.clearance_mode)
        self.app.mj_scene = self.mj_scene   # handoff to RenderStage + app._reframe

        # Live mesh preview: shown whenever the scene actually simulates (random, or structured
        # partition/tray which now settle under gravity). Structured/none is static → no preview.
        # The callback runs inside simulate()'s step loop (this thread) — no MjData race.
        sim_runs = self.arrangement == "random" or (
            self.arrangement == "structured" and self.structure_type in ("partition", "tray"))
        if not self.app.headless and sim_runs:
            self._setup_live_preview()
            self.mj_scene.simulate(on_step=self._live_preview_update)
        else:
            self.mj_scene.simulate()
        self.mj_scene.verify_parts_in_bin()
        scene_state = self.mj_scene.extract_scene_state()
        self.o3d_scene = self.mj_scene.mujoco_scene_to_o3d(scene_state)
        # T_gt from MuJoCo is in the centered body frame. Remap it and the mesh geom back to
        # the original mesh frame so RenderStage and GT export stay consistent with down_pcd.
        # Math: T_gt_adj = T_gt_phys @ [[I, -mesh_center]; [0,1]]
        #   because p_world = T_gt_phys @ p_centered = T_gt_phys @ (p_orig - mesh_center) = T_gt_adj @ p_orig
        if needs_centering:
            T_shift = np.eye(4)
            T_shift[:3, 3] = -mesh_center
            original_mesh_o3d = o3d.geometry.TriangleMesh(self.app.target_mesh)
            original_mesh_o3d.compute_vertex_normals()
            for key, obj in self.o3d_scene.items():
                if key != "bin":
                    obj.T_gt = obj.T_gt @ T_shift
                    obj.geom = copy.deepcopy(original_mesh_o3d)
        self.app.o3d_scene = self.o3d_scene   # handoff to RenderStage
        _update_pb("Compiling physical scene mesh...")

        # Merged physical-scene preview mesh — GUI only (RenderStage consumes app.o3d_scene, not this).
        # The boolean union can degenerate on the non-manifold tray bin mesh, so guard it and fall back
        # to a plain concatenation (which always renders) rather than leaving the scene blank.
        if not self.app.headless:
            mesh_list = []
            for obj in self.o3d_scene.values():
                mesh = copy.deepcopy(obj.geom)
                tri_mesh = o3d_to_trimesh(mesh.transform(obj.T_gt))
                tri_mesh.process(validate=True)
                mesh_list.append(tri_mesh)
            try:
                union = trimesh.boolean.union(mesh_list, engine="manifold", check_volume=False)
                if union is None or len(union.vertices) == 0:
                    raise ValueError("empty union result")
            except Exception as e:
                print(f"[SCENE] preview union failed ({e}); using concatenation instead")
                union = trimesh.util.concatenate(mesh_list)
            self.app.scene_mesh = trimesh_to_o3d(union)
            self._show_scene_mesh()
        _update_pb("Physical scene ready.")

    # ===============================
    # Live mesh preview during settling (GUI + random only)
    # ===============================
    @staticmethod
    def _pose_to_T(pos, quat):
        """Build a fresh 4x4 transform from a MuJoCo position + wxyz quaternion. Returns a new
        array (safe to hand to the GUI thread; does not alias data.xpos/xquat)."""
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
        T[:3, 3] = pos
        return T

    def _setup_live_preview(self):
        """Add one part mesh per body + the bin to the scene once, framed on the bin. Subsequent
        per-step transforms are pushed by _live_preview_update(). Parked (not-yet-released) bodies
        start hidden. Must run before mj_scene.simulate()."""
        import mujoco
        # Populate xpos/xquat for the initial spawn poses (pure kinematics; does not advance the
        # sim — simulate() recomputes everything from qpos/qvel via mj_step).
        mujoco.mj_forward(self.mj_scene.model, self.mj_scene.data)
        state = self.mj_scene.extract_scene_state()
        part_mesh_o3d = trimesh_to_o3d(self.mj_scene.part_mesh)
        bin_mesh = self.mj_scene.bin_mesh

        def add_all():
            self.app._clear_scene()
            for body_name, bd in state.items():
                pos = bd["position"]
                geom = o3d.geometry.TriangleMesh(part_mesh_o3d)   # one copy per body
                self.app.scene.scene.add_geometry(body_name, geom, self.app.default_material)
                parked = float(pos[2]) > PREVIEW_PARKED_Z
                self.app.scene.scene.show_geometry(body_name, not parked)
                if not parked:
                    self.app.scene.scene.set_geometry_transform(
                        body_name, self._pose_to_T(pos, bd["quaternion"]))
            self.app.scene.scene.add_geometry("bin", bin_mesh, self.app.default_material)
        self.app.main_thread(add_all)
        self.app.main_thread(self.app._reframe)   # GUI op → main thread (Open3D not thread-safe)

    def _live_preview_update(self):
        """Called from inside simulate()'s step loop (worker thread). Snapshots body poses into
        fresh transforms and posts a single GUI update; parked bodies are hidden until released."""
        state = self.mj_scene.extract_scene_state()
        updates = {
            name: (self._pose_to_T(bd["position"], bd["quaternion"]),
                   float(bd["position"][2]) > PREVIEW_PARKED_Z)
            for name, bd in state.items()
        }

        def apply():
            for name, (T, parked) in updates.items():
                if not self.app.scene.scene.has_geometry(name):
                    continue
                self.app.scene.scene.show_geometry(name, not parked)
                if not parked:
                    self.app.scene.scene.set_geometry_transform(name, T)
            self.app.scene.force_redraw()
        self.app.main_thread(apply)
