import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import trimesh
from enums import Stage
from stages.stage_base import BaseStage
from geometry.geom_utils import o3d_to_trimesh, trimesh_to_o3d, O3DSceneObject, rotation_aligning_vector_to_axis, face_facet_map
from geometry.file_utils import list_scene_dirs
from physics.mujoco_bin_scene import MujocoBinScene, MAX_BIN_DIM
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
        # Manual face-up picking (GUI structured modes): user picks the face to point up,
        # overriding the auto stable pose. See _pick_face_at / _enter_pick_mode.
        self.pick_mode = False
        self.picked_normal = None   # unit normal (mesh frame) of the last picked face
        self.user_R = None          # 3x3 rotation mapping picked_normal -> world +Z
        self._pick_down_xy = None    # mouse-down pixel, to tell a click from a camera drag
        # Pick-mode caches (built once in _enter_pick_mode): a reusable raycast scene + coplanar
        # facet grouping so hover/click highlight a whole flat patch, not a lone triangle.
        self._pick_rc = None
        self._pick_facets = None
        self._pick_facet_normal = None
        self._face_to_facet = None
        self._hover_facet = None     # last hovered facet key, so we only redraw on change

    def build_panel(self):
        if self.app.headless:
            return
        v = gui.Vert(4)

        # Single scene-mode selector. Drives BOTH self.arrangement (random vs structured)
        # and self.structure_type (none/partition/tray) — see the _is_random() /
        # _structure_type() helpers. Cluttered = random pile; Arranged = static grid at a
        # single pose; Partition = egg-crate dividers; Tray = molded pockets.
        self.arrangement_combo = self.register_widget(gui.Combobox())
        for item in ("Cluttered", "Arranged", "Partition", "Tray"):
            self.arrangement_combo.add_item(item)
        self.arrangement_combo.selected_index = 0
        self.arrangement_combo.set_on_selection_changed(self._on_arrangement_changed)

        # --- Cluttered (random) options: fill-rate % or override count ---
        # Grouped in a Vert so the whole block can be hidden when not in Cluttered mode.
        self.random_opts = gui.Vert(4)
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
        self.random_opts.add_child(self.radio_mode)
        self.random_opts.add_child(self.gen_label)
        self.random_opts.add_child(self.gen_slider)

        # --- Structured orientation: auto stable pose vs manual face-up pick ---
        # Visible for every structured mode (Arranged/Partition/Tray).
        self.orient_opts = gui.Vert(4)
        self.orient_radio = self.register_widget(gui.RadioButton(gui.RadioButton.HORIZ),
                                                 enabled_if=lambda: not self._is_random())
        self.orient_radio.set_items(["Auto pose", "Manual face-up"])
        self.orient_radio.selected_index = 0
        self.orient_radio.set_on_selection_changed(self._on_orient_changed)
        self.btn_pick_face = self.register_widget(
            gui.Button("Pick Face-Up"),
            lambda: (not self._is_random()) and self.orient_radio.selected_index == 1)
        self.btn_pick_face.set_on_clicked(self._enter_pick_mode)
        self.pick_status_label = gui.Label("Face-up: auto")   # plain label, not registered
        self.orient_opts.add_child(gui.Label("Part Orientation"))
        self.orient_opts.add_child(self.orient_radio)
        self.orient_opts.add_child(self.btn_pick_face)
        self.orient_opts.add_child(self.pick_status_label)

        # --- Partition / tray fixture options: height/depth + fit clearance ---
        self.structure_opts = gui.Vert(4)
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
        self.structure_opts.add_child(self.structure_slider_label)
        self.structure_opts.add_child(self.structure_slider)
        self.structure_opts.add_child(self.clearance_label)
        self.structure_opts.add_child(self.clearance_radio)

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
        v.add_child(gui.Label("Scene Mode"))
        v.add_child(self.arrangement_combo)
        v.add_child(self.random_opts)
        v.add_child(self.orient_opts)
        v.add_child(self.structure_opts)
        v.add_child(self.btn_generate)
        v.add_child(self.btn_reset)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label("Preview Saved Scene (mesh)"))
        v.add_child(self.combo_disk_scenes)
        v.add_child(self.btn_disk_refresh)
        v.add_child(gui.Label(""))
        v.add_child(self.btn_restart)

        # Initial visibility matches the default selection (Cluttered). set_needs_layout is
        # skipped here because the window doesn't exist yet at panel-build time.
        self._apply_option_visibility()

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
            self.app._reframe()
        self.app.main_thread(apply)

    # Merged scene-mode combo -> (arrangement, structure_type). Cluttered is the only
    # random mode; the other three are structured with an increasing amount of fixture.
    _SCENE_MODES = {
        "Cluttered": ("random", "none"),
        "Arranged":  ("structured", "none"),
        "Partition": ("structured", "partition"),
        "Tray":      ("structured", "tray"),
    }

    def _is_random(self) -> bool:
        return self._SCENE_MODES[self.arrangement_combo.selected_text][0] == "random"

    def _structure_type(self) -> str:
        return self._SCENE_MODES[self.arrangement_combo.selected_text][1]

    def _clearance_mode(self) -> str:
        return ["snug", "medium", "loose"][self.clearance_radio.selected_index]

    def _apply_structure_label(self):
        self.structure_slider_label.text = {
            "none": "Structure Height (%)", "partition": "Partition Height (%)",
            "tray": "Pocket Depth (%)"}[self._structure_type()]

    def _apply_option_visibility(self):
        """Show only the option block(s) relevant to the current scene mode. Hides (not just
        greys out) so the panel stays uncluttered. set_needs_layout is skipped until the
        window exists (panel is built before app.window in MeshSamplingApp.__init__)."""
        is_random = self._is_random()
        self.random_opts.visible    = is_random
        self.orient_opts.visible    = not is_random                              # any structured mode
        self.structure_opts.visible = (not is_random) and self._structure_type() != "none"
        if getattr(self.app, "window", None) is not None:
            self.app.window.set_needs_layout()

    def _on_arrangement_changed(self, text, idx):
        # The single combo now drives both arrangement and structure, so update the fixture
        # label, re-flow which option blocks are visible, and re-evaluate enable predicates.
        self._exit_pick_mode()          # a mode switch cancels any in-progress face pick
        self._apply_structure_label()
        self._apply_option_visibility()
        self.enable_widgets()

    def _on_orient_changed(self, idx: int):
        # Leaving manual mode cancels an in-progress pick; the Pick button enables in manual.
        if idx == 0:
            self._exit_pick_mode()
        self.enable_widgets()

    # ─────────────────────────────────────────────────────────────────────
    # Manual face-up picking (structured modes)
    # ─────────────────────────────────────────────────────────────────────
    def _enter_pick_mode(self):
        """Show the bare part mesh, build the pick caches, and arm hover+click picking. Hovering
        highlights the coplanar patch under the cursor; a click sets it as 'up'; a drag still
        rotates the camera (see _on_mouse_event)."""
        if self.app.headless or self.app.target_mesh is None:
            print("[SCENE] import a mesh before picking a face-up direction.")
            return
        self._remove_pick_overlays()
        self.app._clear_scene()
        mesh = o3d.geometry.TriangleMesh(self.app.target_mesh)
        mesh.compute_vertex_normals()
        self.app.scene.scene.add_geometry("pick_mesh", mesh, self.app.default_material)
        self.app._reframe()

        # Built once, reused for every hover/click: a raycast scene + coplanar facet grouping.
        self.pick_status_label.text = "Preparing pick…"
        self._pick_rc = o3d.t.geometry.RaycastingScene()
        self._pick_rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.app.target_mesh))
        tri = o3d_to_trimesh(self.app.target_mesh)
        tri.merge_vertices()   # restore adjacency on split-vertex STL meshes (no face reorder)
        self._face_to_facet, self._pick_facets, self._pick_facet_normal = face_facet_map(tri)
        self._hover_facet = None
        self.pick_mode = True
        self._pick_down_xy = None
        self.pick_status_label.text = "Hover a face; click to set it as 'up'."

    def _exit_pick_mode(self):
        """Stop the picking interaction (keeps any already-picked result and its overlay)."""
        self.pick_mode = False
        self._pick_down_xy = None
        self._hover_facet = None
        if not self.app.headless and getattr(self.app, "scene", None) is not None:
            self._remove_hover()
            self.app.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _remove_pick_overlays(self):
        if self.app.headless:
            return
        for name in ("hover_face", "picked_face", "pick_arrow"):
            if self.app.scene.scene.has_geometry(name):
                self.app.scene.scene.remove_geometry(name)

    def _remove_hover(self):
        if self.app.scene.scene.has_geometry("hover_face"):
            self.app.scene.scene.remove_geometry("hover_face")
            self.app.redraw()

    def _faces_normal_for(self, prim_id, ray_dir, tri_normal):
        """Resolve a hit triangle to its coplanar facet (faces + facet normal), or the single
        triangle for a curved surface. Returns (face_indices, unit up-normal facing the camera,
        facet_index) where facet_index is -1 for a singleton triangle."""
        fi = int(self._face_to_facet[prim_id])
        if fi >= 0:
            faces = np.asarray(self._pick_facets[fi])
            n = self._pick_facet_normal[fi].astype(float)
        else:
            faces = np.array([prim_id])
            n = np.asarray(tri_normal, dtype=float)
        norm = np.linalg.norm(n)
        n = n / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        if np.dot(n, ray_dir) > 0:   # geometric normal can point either way; face the camera
            n = -n
        return faces, n, fi

    def _add_facet_overlay(self, name, faces, normal, color):
        """Overlay the given faces of target_mesh in a flat `color`, nudged out along `normal` to
        avoid z-fighting. Only the used vertices are uploaded (cheap enough for per-hover redraw)."""
        verts = np.asarray(self.app.target_mesh.vertices)
        tris = np.asarray(self.app.target_mesh.triangles)[np.asarray(faces)]
        extent = float(np.linalg.norm(self.app.target_mesh.get_axis_aligned_bounding_box().get_extent()))
        offset = 0.002 * max(extent, 1e-6) * np.asarray(normal, dtype=float)
        used = np.unique(tris)
        hl = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts[used] + offset),
            o3d.utility.Vector3iVector(np.searchsorted(used, tris)))
        hl.compute_vertex_normals()
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = list(color) + [1.0]
        if self.app.scene.scene.has_geometry(name):
            self.app.scene.scene.remove_geometry(name)
        self.app.scene.scene.add_geometry(name, hl, mat)

    def _hover_face_at(self, x, y):
        """Highlight the coplanar patch under the cursor. Cheap (prebuilt raycast scene) and only
        redraws when the hovered facet changes, so it stays smooth on dense meshes."""
        if self._pick_rc is None:
            return
        ray = self._build_pick_ray(x, y)
        if ray is None:
            return
        cam_pos, ray_dir = ray
        rays = o3d.core.Tensor([[*cam_pos, *ray_dir]], dtype=o3d.core.Dtype.Float32)
        ans = self._pick_rc.cast_rays(rays)
        if not np.isfinite(float(ans["t_hit"].numpy()[0])):
            if self._hover_facet is not None:
                self._hover_facet = None
                self._remove_hover()
            return
        prim_id = int(ans["primitive_ids"].numpy()[0])
        tri_n = ans["primitive_normals"].numpy()[0].astype(float)
        faces, n_up, fi = self._faces_normal_for(prim_id, ray_dir, tri_n)
        key = fi if fi >= 0 else (-1 - prim_id)   # distinct key per singleton triangle
        if key == self._hover_facet:
            return
        self._hover_facet = key
        self._add_facet_overlay("hover_face", faces, n_up, (1.0, 0.85, 0.2))
        self.app.redraw()

    def _build_pick_ray(self, x, y):
        """World-space (origin, direction) for the click at window pixel (x, y). Mirrors the
        inverse proj@view unprojection in CropStage._get_selection_frustum_corners."""
        cam = self.app.scene.scene.camera
        view = np.asarray(cam.get_view_matrix())
        proj = np.asarray(cam.get_projection_matrix())
        inv_view = np.linalg.inv(view)
        inv_proj_view = np.linalg.inv(proj @ view)
        cam_pos = inv_view[:3, 3]

        # Widget-local pixel -> NDC (y flips). frame.x/y are 0 in this layout but subtract anyway.
        lx = x - self.app.scene.frame.x
        ly = y - self.app.scene.frame.y
        ndc_x = (2.0 * lx / self.app.scene.frame.width) - 1.0
        ndc_y = 1.0 - (2.0 * ly / self.app.scene.frame.height)
        near_world_h = inv_proj_view @ np.array([ndc_x, ndc_y, -1.0, 1.0])
        if abs(near_world_h[3]) < 1e-6:
            return None
        near_world = near_world_h[:3] / near_world_h[3]
        ray_dir = near_world - cam_pos
        length = np.linalg.norm(ray_dir)
        if length < 1e-6:
            return None
        return cam_pos, ray_dir / length

    def _pick_face_at(self, x, y):
        """Ray-cast the click, resolve its coplanar facet, take the facet normal as the new 'up'
        direction, store the aligning rotation, and draw the confirmation patch + arrow."""
        if self.app.target_mesh is None or self._pick_rc is None:
            return
        ray = self._build_pick_ray(x, y)
        if ray is None:
            return
        cam_pos, ray_dir = ray
        rays = o3d.core.Tensor([[*cam_pos, *ray_dir]], dtype=o3d.core.Dtype.Float32)
        ans = self._pick_rc.cast_rays(rays)
        if not np.isfinite(float(ans["t_hit"].numpy()[0])):
            self.pick_status_label.text = "Missed the part — click on a face."
            return

        prim_id = int(ans["primitive_ids"].numpy()[0])
        tri_n = ans["primitive_normals"].numpy()[0].astype(float)
        faces, n_up, _ = self._faces_normal_for(prim_id, ray_dir, tri_n)

        self.picked_normal = n_up
        self.user_R = rotation_aligning_vector_to_axis(n_up, (0.0, 0.0, 1.0))
        self.pick_mode = False   # one click = one pick; drag to rotate, click again to re-pick
        self._hover_facet = None
        self._highlight_picked_face(faces, n_up)
        self.pick_status_label.text = f"Face-up set: n=[{n_up[0]:+.2f} {n_up[1]:+.2f} {n_up[2]:+.2f}]"

    def _highlight_picked_face(self, faces, n_up):
        """Overlay the picked coplanar patch (cyan) plus a 3D arrow (blue) from the part centre along the
        chosen up direction."""
        self._remove_pick_overlays()
        self._add_facet_overlay("picked_face", faces, n_up, (0.0, 1.0, 1.0))

        bbox = self.app.target_mesh.get_axis_aligned_bounding_box()
        L = 0.6 * float(np.linalg.norm(bbox.get_extent()))
        if L < 1e-9:
            L = 1.0
        r = 0.02 * L
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=r, cone_radius=2 * r, cylinder_height=0.7 * L, cone_height=0.3 * L)
        arrow.rotate(rotation_aligning_vector_to_axis((0.0, 0.0, 1.0), n_up), center=(0.0, 0.0, 0.0))
        arrow.translate(self.app.target_mesh.get_center())
        arrow.compute_vertex_normals()
        arrow_mat = rendering.MaterialRecord()
        arrow_mat.shader = "defaultLit"
        arrow_mat.base_color = [0.0, 0.0, 1.0, 1.0]
        self.app.scene.scene.add_geometry("pick_arrow", arrow, arrow_mat)
        self.app.redraw()

    def _on_mouse_event(self, event):
        # Only active while picking a face-up direction. Hover highlights the patch under the
        # cursor; a near-stationary press-release is a pick; anything with drag falls through so
        # the camera controller can rotate the view.
        if not self.pick_mode:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.type == gui.MouseEvent.Type.MOVE:
            self._hover_face_at(event.x, event.y)
            return gui.Widget.EventCallbackResult.IGNORED
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            self._pick_down_xy = (event.x, event.y)
            return gui.Widget.EventCallbackResult.IGNORED
        if event.type == gui.MouseEvent.Type.BUTTON_UP and self._pick_down_xy is not None:
            dx = event.x - self._pick_down_xy[0]
            dy = event.y - self._pick_down_xy[1]
            self._pick_down_xy = None
            if (dx * dx + dy * dy) ** 0.5 < 5:
                self._pick_face_at(event.x, event.y)
                self.enable_widgets()
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

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
            self.app.main_thread(self.app._reframe)   # content swap -> reframe
        self._apply_structure_label()
        self._apply_option_visibility()
        self.enable_widgets()

    def on_enter(self):
        """Entry-only reset (not fired on worker completion): drop any stale face-up pick and
        start the orientation choice at Auto so a fresh visit is clean."""
        if self.app.headless:
            return
        self.pick_mode = False
        self.picked_normal = None
        self.user_R = None
        self._pick_down_xy = None
        self._pick_rc = None
        self._hover_facet = None
        self.orient_radio.selected_index = 0
        self.pick_status_label.text = "Face-up: auto"
        self._remove_pick_overlays()
        self._apply_option_visibility()
        self.enable_widgets()

    def on_clear(self):
        # Reset stage-local scratch alongside the app-level scene state. NOTE: do not clear
        # user_R / picked_normal here — worker() calls clear_state_from() at its start, so
        # wiping the pick here would drop the override before the same run reads it. The pick
        # is reset instead on stage entry (on_enter).
        self.o3d_scene = {}
        self.mj_scene = None
        self.worker_step = 0
        # No reframe here: this fires against a stale/empty scene. _refresh_ui reframes once the
        # replacement geometry is actually on screen.

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
            self.arrangement = "random" if self._is_random() else "structured"
            self.generate_mode = "fill_rate" if self.radio_mode.selected_index == 0 else "count"
            if self.generate_mode == "fill_rate":
                self.fill_rate = self.gen_slider.int_value / 100.0
            else:
                self.num_targets = int(self.gen_slider.int_value)
            self.structure_type = self._structure_type()
            self.structure_height_pct = int(self.structure_slider.int_value)
            self.clearance_mode = self._clearance_mode()
            # Optional manual override: when the user picked a face-up direction, force that
            # orientation instead of the auto stable pose. Otherwise None -> _generate_structured_scene
            # falls back to the most-probable stable pose (unchanged GUI behavior).
            if (not self._is_random()) and self.orient_radio.selected_index == 1 and self.user_R is not None:
                self.stable_pose_R = self.user_R
            else:
                self.stable_pose_R = None

        def _update_pb(message: str):
            if not self.app.headless:
                print(message)
            self.worker_step += 1
            self.app.update_progress(self.worker_step / TOTAL_STEPS, message)

        # MujocoBinScene requires mesh centered at its own origin: every rotation, spawn-height,
        # stable-pose, and tray-pocket calculation rotates vertices around (0,0,0). Center a
        # local copy without modifying app.target_mesh so the user's centering choice is preserved.
        # Anchored on the AABB centre (not the vertex mean) so r_max and bounding_sphere — both
        # measured about the body origin — are not skewed by tessellation density.
        mesh_center = np.asarray(self.app.target_mesh.get_axis_aligned_bounding_box().get_center())
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
        self.app.mj_scene = self.mj_scene   # handoff to RenderStage

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
        # Scene bounds work here even though parked bodies are hidden and still counted: they keep
        # an identity transform (see add_all above), so they sit at the part's own origin, and the
        # bin dominates the box either way.
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
            self.app.redraw()
        self.app.main_thread(apply)
