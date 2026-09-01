import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import trimesh
from enums import Stage
from stages.stage_base import BaseStage
from geometry.geom_utils import o3d_to_trimesh, trimesh_to_o3d, O3DSceneObject
from geometry.file_utils import list_scene_dirs
from physics.mujoco_bin_scene import (
    MujocoBinScene, MAX_BIN_DIM, LAYERS_AT_FULL_FILL, MIN_USEFUL_LAYERS,
    PACKING_FACTOR_BASE, PACKING_FACTOR_FLOOR, BIN_TOP_MARGIN_FRAC,
    MIN_AUTO_PARTS, MAX_AUTO_PARTS, obb_packing_factor, solve_bin_dim,
)
import os
import copy
import colorsys
from pathlib import Path
from rich import print as rp
np.set_printoptions(precision=6, suppress=True)


def _box(center, size):
    """An axis-aligned box mesh of `size` (sx,sy,sz) centred at `center` (world)."""
    size = np.asarray(size, dtype=float)
    b = o3d.geometry.TriangleMesh.create_box(size[0], size[1], size[2])
    b.translate(-size / 2.0)                 # create_box has its min corner at origin
    b.translate(np.asarray(center, dtype=float))
    return b


def _reconstruct_bin_mesh(bin_dim, bin_transform):
    """Rebuild an open-top bin box (floor + 4 walls) from `bin_dim`
    (width, length, height, wall_thickness) + `bin_transform`, for the disk-scene mesh
    preview. Note: partition/tray fixtures are NOT saved to scene_state.npz, so structured
    scenes preview with the outer bin only (documented approximation)."""
    w, l, h, t = [float(x) for x in np.asarray(bin_dim).ravel()[:4]]
    hw, hl = w / 2.0, l / 2.0
    pieces = [
        _box((0.0, 0.0, -t / 2.0), (w, l, t)),           # floor (top at z=0)
        _box(( hw - t / 2.0, 0.0, h / 2.0), (t, l, h)),  # +x wall
        _box((-hw + t / 2.0, 0.0, h / 2.0), (t, l, h)),  # -x wall
        _box((0.0,  hl - t / 2.0, h / 2.0), (w, t, h)),  # +y wall
        _box((0.0, -hl + t / 2.0, h / 2.0), (w, t, h)),  # -y wall
    ]
    mesh = pieces[0]
    for p in pieces[1:]:
        mesh += p
    mesh.transform(np.asarray(bin_transform, dtype=float))
    mesh.compute_vertex_normals()
    return mesh

SCENE_FILL_RATE     = 0.2    # default fill rate (fraction of safe capacity)
# PACKING_FACTOR_BASE/FLOOR, BIN_TOP_MARGIN_FRAC and MIN/MAX_AUTO_PARTS now live in
# physics/mujoco_bin_scene.py (imported above) so the packing math has a single home shared
# with solve_bin_dim(); they are re-exported here because this module's public names are
# referenced by tests and headless drivers.
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
        # Derive the bin from the part instead of the part count from a fixed bin. Random +
        # fill-rate only; headless callers may set this False to restore the fixed max bin.
        self.dynamic_bin = True
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
        # Dynamic bin: size the bin to the part instead of the part count to a fixed bin.
        # Only meaningful for random + fill-rate, so grey it out elsewhere.
        self.dynamic_bin_check = self.register_widget(
            gui.Checkbox("Dynamic Bin"),
            enabled_if=lambda: self._is_random() and self.radio_mode.selected_index == 0)
        self.dynamic_bin_check.checked = self.dynamic_bin

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

        # --- Preview an already-saved scene (mesh form) from output/synthetic_target/<part>/ ---
        self.combo_disk_scenes = self.register_widget(
            gui.Combobox(), lambda: self._has_disk_scenes())
        self.combo_disk_scenes.set_on_selection_changed(self._on_disk_scene_selected)
        self.btn_disk_refresh = self.register_widget(gui.Button("Refresh Saved Scenes"))
        self.btn_disk_refresh.set_on_clicked(self._refresh_disk_scenes)

        self.btn_restart = self.register_widget(gui.Button("Restart"))
        self.btn_restart.set_on_clicked(lambda: self.app._restart())
        v.add_child(gui.Label("Generate Physical Scene"))
        v.add_child(gui.Label(""))
        v.add_child(self.arrangement_combo)
        v.add_child(self.radio_mode)
        v.add_child(self.gen_label)
        v.add_child(self.gen_slider)
        v.add_child(self.dynamic_bin_check)
        v.add_child(gui.Label("Structure"))
        v.add_child(self.structure_radio)
        v.add_child(self.structure_slider_label)
        v.add_child(self.structure_slider)
        v.add_child(self.clearance_label)
        v.add_child(self.clearance_radio)
        v.add_child(self.btn_generate)
        v.add_child(self.btn_reset)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label("Preview Saved Scene (mesh)"))
        v.add_child(self.combo_disk_scenes)
        v.add_child(self.btn_disk_refresh)
        v.add_child(gui.Label(""))
        v.add_child(self.btn_restart)

        print("loaded scene panel")
        return v

    def next_enabled(self) -> bool:
        # Advance if a scene was generated this session, OR on-disk scenes already exist
        # for this part (so the operator can skip straight to tuning).
        return len(self.app.o3d_scene) > 0 or self._has_disk_scenes()

    def request_next(self, proceed):
        # Skipping = advancing without a scene generated this session but with scenes on disk.
        if len(self.app.o3d_scene) == 0 and self._has_disk_scenes():
            self.app.confirm_dialog(
                "No scene was generated this session.\nSkip to the next stage using the "
                "existing on-disk scenes for this part?", on_ok=proceed)
        else:
            proceed()

    # ─────────────────────────────────────────────────────────────────────
    # On-disk saved-scene mesh preview
    # ─────────────────────────────────────────────────────────────────────

    def _synth_dir(self):
        part = getattr(self.app, "mesh_basename", None)
        return (Path.cwd() / "output" / "synthetic_target" / part) if part else None

    def _has_disk_scenes(self) -> bool:
        d = self._synth_dir()
        return bool(list_scene_dirs(d)) if d else False

    def _populate_disk_scenes(self):
        if self.app.headless:
            return
        prev = self.combo_disk_scenes.selected_text
        self.combo_disk_scenes.clear_items()
        d = self._synth_dir()
        names = list_scene_dirs(d) if d else []
        for name in names:
            self.combo_disk_scenes.add_item(name)
        if prev and prev in names:
            self.combo_disk_scenes.selected_text = prev

    def _refresh_disk_scenes(self):
        self._populate_disk_scenes()
        self.enable_widgets()

    def _on_disk_scene_selected(self, text, idx):
        self._preview_disk_scene(text)

    def _build_disk_scene_geoms(self, scene_dir):
        """Reconstruct a saved scene as meshes: a copy of the part mesh at each stored GT pose
        plus the outer bin box (from scene_state.npz). Returns [(name, mesh), …]; empty when
        the mesh isn't imported or the npz is missing/unreadable. Pure geometry — no GUI."""
        npz_path = os.path.join(str(scene_dir), "scene_state.npz")
        if self.app.target_mesh is None or not os.path.exists(npz_path):
            return []
        try:
            state = np.load(npz_path, allow_pickle=True)
            T_gt = np.asarray(state["T_gt"], dtype=float)   # (n,4,4) part->world
            bin_geom = _reconstruct_bin_mesh(state["bin_dim"], state["bin_transform"])
        except Exception as e:
            print(f"[SCENE] failed to read {npz_path}: {e}")
            return []

        part_mesh = o3d.geometry.TriangleMesh(self.app.target_mesh)
        part_mesh.compute_vertex_normals()
        geoms = []
        for i in range(len(T_gt)):
            m = copy.deepcopy(part_mesh)
            m.transform(T_gt[i])
            geoms.append((f"disk_part_{i}", m))
        geoms.append(("disk_bin", bin_geom))
        return geoms

    def _preview_disk_scene(self, scene_name):
        if self.app.headless or not scene_name:
            return
        if self.app.target_mesh is None:
            print("[SCENE] Import a mesh before previewing saved scenes.")
            return
        scene_dir = os.path.join(str(self._synth_dir()), scene_name)
        geoms = self._build_disk_scene_geoms(scene_dir)
        if not geoms:
            print(f"[SCENE] nothing to preview for {scene_dir}")
            return

        def apply():
            self.app._clear_scene()
            for name, g in geoms:
                self.app.scene.scene.add_geometry(name, g, self.app.default_material)
            bbox = geoms[-1][1].get_axis_aligned_bounding_box()   # bin box frames the view
            if not bbox.is_empty():
                self.app.scene.setup_camera(60.0, bbox, bbox.get_center())
            self.app.scene.force_redraw()
        self.app.main_thread(apply)

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

        The packing factor is shape-aware (see obb_packing_factor): it tapers from
        PACKING_FACTOR_BASE by the OBB aspect ratio so elongated/flat parts — which pack far
        less densely — no longer over-count. We do NOT also multiply by solidity: the count
        divides by OBB volume, so the correct multiplier is the box-packing fraction (folding
        solidity in again would double-count)."""
        bw, bl, bh, _ = MAX_BIN_DIM
        usable_h = bh * (1.0 - BIN_TOP_MARGIN_FRAC)   # reserve top headroom to prevent spilling
        bin_vol  = bw * bl * usable_h
        mobb_vol = max(float(part_mesh.bounding_box_oriented.volume), 1e-9)
        packing_factor = obb_packing_factor(part_mesh)
        n = round(self.fill_rate * packing_factor * bin_vol / mobb_vol)
        rp(f"[PACKING] packing_factor={packing_factor:.3f}")
        return int(np.clip(n, MIN_AUTO_PARTS, MAX_AUTO_PARTS))

    def _resolve_bin_and_count(self, part_mesh):
        """Decide (bin_dim, n_parts) for this run — the stage owns this policy, physics owns
        the math.

        Dynamic sizing applies ONLY to a random arrangement in fill-rate mode with the Dynamic
        Bin checkbox on. Everything else keeps MAX_BIN_DIM: structured grids derive their own
        count from grid capacity (shrinking the bin would shrink the grid), and Override Count
        is a manual escape hatch that must stay exact.

        Sets self.num_targets as a side effect, since that is what worker() hands to
        MujocoBinScene.
        """
        use_dynamic = (self.dynamic_bin
                       and self.arrangement == "random"
                       and self.generate_mode == "fill_rate")
        if not use_dynamic:
            if self.generate_mode == "fill_rate" and self.arrangement == "random":
                self.num_targets = self._auto_part_count(part_mesh)   # legacy path, untouched
            return MAX_BIN_DIM, int(self.num_targets)

        bin_dim, n, layers = solve_bin_dim(part_mesh, self.fill_rate)
        self.num_targets = n
        rp(f"[BIN] {bin_dim[0]:.3f} x {bin_dim[1]:.3f} x {bin_dim[2]:.3f} m, "
           f"wall {bin_dim[3] * 1000:.1f} mm, n_parts={n}, "
           f"~{layers:.1f} layers @ fill {self.fill_rate:.0%}")
        if layers < MIN_USEFUL_LAYERS:
            # Stacking depth is fill x LAYERS_AT_FULL_FILL and does NOT depend on bin size, so
            # this is equally true of the fixed max bin — the operator just could not see it.
            rp(f"[BIN] near-monolayer ({layers:.1f} layers): raise fill rate to "
               f"{MIN_USEFUL_LAYERS / LAYERS_AT_FULL_FILL:.0%}+ for part-on-part stacking")
        return bin_dim, n

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
            self.app.main_thread(self.app._reframe)
            self.app.scene.force_redraw()
        self.app.main_thread(show)

    def _refresh_ui(self):
        if self.app.headless:
            return
        self._populate_disk_scenes()
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
        self.app.main_thread(self.app._reframe)

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
            self.dynamic_bin = bool(self.dynamic_bin_check.checked)

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
        # Size the bin to the part (dynamic) or the count to the fixed bin (legacy); structured
        # ignores n_parts either way (the grid sets it). See _resolve_bin_and_count.
        bin_dim, _ = self._resolve_bin_and_count(part_mesh)
        rp(f"[AUTO-COUNT] fill={self.fill_rate:.0%} "
           f"part OBB vol={part_mesh.bounding_box_oriented.volume:.2e} m³ "
           f"-> n_parts={self.num_targets}")
        self.mj_scene = MujocoBinScene(part_mesh, physics_convex, n_parts=self.num_targets,
                                       bin_dim=bin_dim,
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
