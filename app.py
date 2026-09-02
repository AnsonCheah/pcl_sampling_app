"""PCL Sampling App -- single entrypoint for GUI and headless use.

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
from open3d.geometry import Geometry3D, AxisAlignedBoundingBox
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
from stages.tuning_stage import TuningStage, SAMPLER_CHOICES
import threading
import time
from pathlib import Path
from rich import print as rp
from geometry.file_utils import pointcloud_to_ply, open_source_folder_dialog
from geometry.geom_utils import O3DSceneObject, camera_view_matrix, o3d_to_trimesh
from physics.mujoco_bin_scene import MujocoBinScene
# import cupy as cp
# print(cp.cuda.runtime.getDeviceCount())

# Extent framed when the scene is empty (launch, or after a clear). 1.0 m matches the axis length
# Open3D itself falls back to for degenerate bounds (Open3DScene.cpp, RecreateAxis), so the view and
# the axes agree instead of one overflowing the other.
DEFAULT_WORLD_EXTENT = 1.0

# How much the scene's bounding box must move or resize, as a fraction of its diagonal, before
# _reframe treats it as different content and moves the camera. Below this the camera is left alone
# so navigating between stages that show the same object does not throw away a manual orbit/zoom.
FRAME_REL_TOL = 0.05

# Pipeline order. Each class also declares the app.* state it owns in its `downstream` dict,
# which is what seeds those attributes before any panel is built.
STAGE_CLASSES = {
    Stage.IMPORT_MESH: ImportMeshStage,
    Stage.RAYCAST:     RaycastStage,
    Stage.CROP:        CropStage,
    Stage.DOWNSAMPLE:  DownsampleStage,
    Stage.SAVE:        SaveStage,
    Stage.DECOMPOSE:   DecomposeStage,
    Stage.SCENE:       SceneStage,
    Stage.RENDER:      RenderStage,
    Stage.TUNING:      TuningStage,
}


class MeshSamplingApp:

    def __init__(self, headless=False, mesh_path=None):
        self.headless = headless
        self.express_sampling_busy = False   # blocks the Express Sampling button while a run is in flight
        self.show_origin_frame = True        # read by _reframe on every scene swap
        self._framed_bbox = None             # (centre, diagonal) last framed; see _framing_changed

        self.mesh_path = Path(mesh_path) if mesh_path is not None else None
        if not self.headless:
            # Must happen after Application.initialize() and before create_window().
            self.axes_glyph = self._install_glyph_font()

            # === Scene widget ===
            self.window_width = 1440
            self.window_height = 900
            self.window = gui.Application.instance.create_window("Mesh Sampling Wizard", self.window_width, self.window_height)
            self.scene = gui.SceneWidget()
            self.scene.scene = rendering.Open3DScene(self.window.renderer)
            self.scene.scene.set_background([0.2, 0.2, 0.2, 1.0])
            self.scene.scene.show_axes(self.show_origin_frame)
            self.window.set_on_layout(self._on_layout)
            self.window.set_on_key(self._on_key)
            self.scene.set_on_mouse(self._on_mouse_event)
            self.window.add_child(self.scene)

            # Origin-axes toggle, floating over the bottom-right of the 3D view (positioned in
            # _on_layout). Added after self.scene so it draws on top. Intentionally NOT registered
            # through BaseStage.register_widget: set_stage force-resets is_on on every stage-owned
            # toggleable, which would switch the axes off each time the stage changes.
            self.btn_axes = gui.Button(self.axes_glyph)
            self.btn_axes.toggleable = True
            self.btn_axes.is_on = True
            self.btn_axes.horizontal_padding_em = 0.3
            self.btn_axes.vertical_padding_em = 0.3
            # No tooltip. Open3D's Button::Draw renders the tooltip *inside* the toggled-text-colour
            # push (button_on_text_color is literally black), so an ON toggleable button draws black
            # tooltip text on the unchanged grey popup background. Not fixable from Python: Theme
            # colours are read-only in the bindings and Button has no text-colour setter. The X key
            # binding below is the discoverability path instead.
            self.btn_axes.set_on_clicked(self._on_axes_toggled)
            self.window.add_child(self.btn_axes)

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

        # Stages are built AFTER the GUI shell and AFTER their owned state is seeded, because
        # BaseStage.__init__ calls build_panel(), which reads both. Seeding comes from the
        # classes' `downstream` dicts -- the same declarations _restart() replays -- so no stage
        # has to defend itself with getattr(app, "x", None) against its own construction order.
        for cls in STAGE_CLASSES.values():
            for attr, factory in cls.downstream.items():
                setattr(self, attr, factory())
        self.stages = {st: cls(self) for st, cls in STAGE_CLASSES.items()}
        # Back-reference to its own enum key so reset()/clear_state_from can locate a stage
        # without a reverse lookup.
        for st, inst in self.stages.items():
            inst.stage_key = st

        if not self.headless:
            for stage in self.stages.values():
                stage.panel.visible = False
                self.panel.add_child(stage.panel)

            # === Unified Back/Next navigation (pinned to panel bottom) ===
            # The buttons are declared once here, not per stage. set_stage drives their
            # labels/visibility from pipeline order; the leading stretch pushes them down.
            self._back_target = None
            self._next_target = None
            self.panel.add_stretch()
            self.nav_panel = gui.Vert(0.25 * em)
            self.btn_back = gui.Button("Back")
            self.btn_back.set_on_clicked(self._go_back)
            self.btn_next = gui.Button("Next")
            self.btn_next.set_on_clicked(self._go_next)
            self.nav_panel.add_child(self.btn_back)
            self.nav_panel.add_child(self.btn_next)
            self.panel.add_child(self.nav_panel)

            # === Progress Bar widget ===
            self.progress_panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
            self.progress_panel.visible = False
            self.progress_label = gui.Label("Processing...")
            self.progress_bar = gui.ProgressBar()
            self.progress_bar.value = 0.0  # range [0, 1]
            # Optional second line for per-step detail (e.g. current-eval error); any stage may set it.
            self.progress_detail = gui.Label("")
            self.progress_panel.add_child(self.progress_label)
            self.progress_panel.add_child(self.progress_bar)
            self.progress_panel.add_child(self.progress_detail)
            self.pb_panel_size = (400, 72)
            x=(self.window_width - self.pb_panel_size[0])>>1
            y=(self.window_height - self.pb_panel_size[1])>>1
            self.progress_panel.frame = gui.Rect(x, y, self.pb_panel_size[0], self.pb_panel_size[1])
            self.window.add_child(self.progress_panel)

        # Pre-seed the mesh path so headless callers skip the file dialog.
        if self.mesh_path is not None:
            self.stages[Stage.IMPORT_MESH].file_path = self.mesh_path

        self._restart()

    @staticmethod
    def _install_glyph_font():
        """Merge the axes glyph into the default UI font and return the button label.

        gui.Button carries no font id (only Label does), so a custom font cannot be attached to
        the widget -- the glyph has to go into DEFAULT_FONT_ID itself. The stock UI faces do not
        have it: segoeui.ttf and arial.ttf carry none of the candidate code points. DejaVuSans
        does, and ships inside the conda env via matplotlib, so it needs no OS font.

        Returns "XYZ" if anything fails: a missing glyph renders as a blank button, which is a
        worse outcome than three letters. Must be called after Application.initialize() and
        before any window is created.
        """
        glyph = "⊕"   # CIRCLED PLUS
        try:
            import matplotlib
            path = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
            if not path.exists():
                raise FileNotFoundError(path)
            fd = gui.FontDescription(gui.FontDescription.SANS_SERIF)
            fd.add_typeface_for_code_points(str(path), [ord(glyph)])
            gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, fd)
            return glyph
        except Exception as e:
            print(f"[UI] axes glyph font unavailable ({e}); using a text label instead")
            return "XYZ"

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
                inst.clear_downstream()

    def reset_all_state(self):
        """Reset every stage's owned state to defaults."""
        for inst in self.stages.values():
            inst.clear_downstream()

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
        self.stages[stage].on_enter()   # entry-only hook (safe for initial previews)
        self.window.title = f"Mesh Sampling Wizard | Stage: {self.stage.name}"

    # ===============================
    # Unified Back/Next navigation
    # ===============================
    @staticmethod
    def _stage_label(stage: Stage) -> str:
        """Human-readable button label derived from the stage enum name."""
        return stage.name.replace("_", " ").title()

    def _go_back(self):
        if self._back_target is not None:
            self.set_stage(self._back_target)

    def _go_next(self):
        if self._next_target is not None:
            target = self._next_target
            # Let the current stage intercept (e.g. confirm a skip) before advancing.
            self.stages[self.stage].request_next(lambda: self.set_stage(target))

    def _update_nav_buttons(self):
        """Re-derive Back/Next labels, visibility and the Next-enabled state from the
        current stage's position in the pipeline. Called on every enable_widgets pass."""
        if self.headless:
            return
        order = list(self.stages.keys())          # insertion order == Stage enum order
        idx = order.index(self.stage)
        # A stage can lock navigation while busy (e.g. an active tuning study).
        nav_ok = bool(self.stages[self.stage].nav_enabled())

        if idx > 0:
            self._back_target = order[idx - 1]
            self.btn_back.text = f"Back: {self._stage_label(self._back_target)}"
            self.btn_back.visible = True
            self.btn_back.enabled = nav_ok
        else:
            self._back_target = None
            self.btn_back.visible = False

        if idx < len(order) - 1:
            self._next_target = order[idx + 1]
            self.btn_next.text = f"Next: {self._stage_label(self._next_target)}"
            self.btn_next.visible = True
            self.btn_next.enabled = bool(nav_ok and self.stages[self.stage].next_enabled())
        else:
            self._next_target = None
            self.btn_next.visible = False

        self.window.set_needs_layout()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 300
        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
        # Floating origin-axes toggle, pinned to the bottom-right of the 3D view. Recomputed on
        # every layout (unlike progress_panel's one-shot frame) so it tracks window resizes.
        pref = self.btn_axes.calc_preferred_size(layout_context, gui.Widget.Constraints())
        margin = 8
        self.btn_axes.frame = gui.Rect(
            self.scene.frame.get_right() - pref.width - margin,
            self.scene.frame.get_bottom() - pref.height - margin,
            pref.width, pref.height)

    # ===============================
    # UI helpers
    # ===============================
    @staticmethod
    def _framing_key(bbox):
        """(centre, diagonal) -- the shape-and-place summary _framing_changed compares."""
        lo = np.asarray(bbox.get_min_bound(), dtype=float)
        hi = np.asarray(bbox.get_max_bound(), dtype=float)
        return (lo + hi) / 2.0, float(np.linalg.norm(hi - lo))

    def _remember_framing(self, bbox):
        self._framed_bbox = self._framing_key(bbox)

    def _framing_changed(self, bbox) -> bool:
        """Has the viewport's content actually changed since the last framing?

        Compared as centre + diagonal rather than raw corners, so re-showing the same object in a
        slightly different form counts as unchanged -- a cloud sampled from a mesh has almost the
        same box, just not to the millimetre. Both are relative to the diagonal, so the test scales
        from a 5 cm part to a 76 cm bin.
        """
        if self._framed_bbox is None:
            return True
        centre, diag = self._framing_key(bbox)
        prev_centre, prev_diag = self._framed_bbox
        scale = max(diag, prev_diag, 1e-9)
        moved = float(np.linalg.norm(centre - prev_centre)) > FRAME_REL_TOL * scale
        resized = abs(diag - prev_diag) > FRAME_REL_TOL * scale
        return moved or resized

    def _reframe(self, fov_deg=41.1, margin=1.15, force=False):
        """Frame the camera on whatever is currently in the viewport, from a fixed isometric angle.

        The renderer's own accumulated bounding box is the only source of bounds -- no stage state,
        no per-caller overrides. Deriving the view from app state instead of scene contents is what
        made the face-up picker frame the bin after a scene had been generated.

        That box covers "all the items in the scene, visible and invisible" (Open3D's wording), so
        hidden geometry counts. That is fine here: the only geometry we hide is the parked bodies in
        the live settling preview, and those keep an identity transform (see
        SceneStage._setup_live_preview), so they sit at the part's own origin inside the bin.
        """
        if self.headless:
            return
        bbox = self.scene.scene.bounding_box
        if bbox is None or bbox.is_empty() or np.linalg.norm(bbox.get_extent()) < 1e-9:
            bbox = self._default_world_bbox()   # nothing loaded -> frame the world axes
        diag = float(np.linalg.norm(bbox.get_extent()))
        look_at = bbox.get_center()
        # Distance that actually fits `diag` in the vertical FOV, with `margin` headroom.
        distance = margin * 0.5 * diag / np.tan(np.radians(fov_deg) * 0.5)
        cam_pos = look_at + distance * (np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0))
        up = camera_view_matrix(cam_pos, look_at)[:3, 1]

        # Resize the origin axes here, not in _clear_scene. Open3D only rebuilds them inside
        # show_axes() and only while its internal dirty flag is set, so calling it straight after
        # clear_geometry rebuilds from the just-emptied bounds and pins the axes at their 1 m
        # degenerate fallback for good. Called after the geometry is in, it sizes them to the
        # content. RecreateAxis adds the axis to the low-level Scene, bypassing Open3DScene's
        # bounds_, so the axes never feed back into the bbox read above.
        self.scene.scene.show_axes(self.show_origin_frame)

        # Only move the camera when the viewport is actually showing something else. Stages re-add
        # the same object as you navigate (Raycast -> Decompose both show the part), and resetting
        # the orbit there would discard whatever view the user had set up. `R` passes force=True.
        if not force and not self._framing_changed(bbox):
            self.redraw()
            return
        self._remember_framing(bbox)

        self.scene.center_of_rotation = look_at
        self.scene.setup_camera(fov_deg, bbox, look_at)
        self.scene.scene.camera.look_at(look_at, cam_pos, up)
        self.redraw()

    def redraw(self):
        """Repaint the viewport. Use this, never `scene.force_redraw()` on its own.

        `SceneWidget::ForceRedraw` opens with `if (!scene_caching_enabled_) return;` and caching
        defaults to off (we never enable it), so calling it alone is a no-op -- which is why
        geometry added from a finished worker only appeared once the mouse moved and generated a
        real input event. `window.post_redraw()` is the call that actually queues a repaint; the
        `post_to_main_thread` binding documents it ("you will need to manually request a redraw of
        the window with w.post_redraw()"). Both are issued here, in that order, so this stays
        correct if scene caching is ever turned on -- the same pairing O3DVisualizer uses.
        """
        if self.headless:
            return
        self.scene.force_redraw()
        self.window.post_redraw()

    @staticmethod
    def _default_world_bbox():
        """Bounds to frame when nothing is loaded: the world axes themselves, with a little margin
        behind the origin. Open3D's axis runs from the origin out to +length on each axis, so this
        is deliberately asymmetric."""
        lo = -0.15 * DEFAULT_WORLD_EXTENT
        hi = DEFAULT_WORLD_EXTENT
        return AxisAlignedBoundingBox((lo, lo, lo), (hi, hi, hi))

    def _set_origin_axes(self, show: bool):
        """Single entry point for the axes toggle, shared by the button and the X key, so the two
        can never disagree. Deliberately does not reframe -- the camera must not jump."""
        self.show_origin_frame = bool(show)
        self.btn_axes.is_on = self.show_origin_frame
        self.scene.scene.show_axes(self.show_origin_frame)
        self.redraw()

    def _on_axes_toggled(self):
        # Button callback: Open3D has already flipped is_on for us.
        self._set_origin_axes(self.btn_axes.is_on)

    def show_progress(self, text="Processing...", detail=""):
        if self.headless:
            return
        def _show():
            self.progress_label.text = text
            self.progress_detail.text = detail
            self.progress_bar.value = 0.0
            self.progress_panel.visible = True
        gui.Application.instance.post_to_main_thread(self.window, _show)

    def update_progress(self, value=None, text=None, detail=None):
        """Update any subset of the progress panel. A None argument leaves that part
        unchanged, so callers can drive the bar/text and the detail line independently."""
        if self.headless:
            return
        clamped = None if value is None else max(0.0, min(1.0, value))
        def _update():
            if clamped is not None:
                self.progress_bar.value = clamped
            if text is not None:
                self.progress_label.text = text
            if detail is not None:
                self.progress_detail.text = detail
        gui.Application.instance.post_to_main_thread(self.window, _update)

    def hide_progress(self):
        if self.headless:
            return
        def _hide():
            self.progress_panel.visible = False
        gui.Application.instance.post_to_main_thread(self.window, _hide)

    def _clear_scene(self):
        self.scene.scene.clear_geometry()
        # Deliberately no show_axes() here -- see the note in _reframe. Calling it at this point
        # rebuilds the axes from the bounds clear_geometry just emptied, pinning them at 1 m.

    def main_thread(self, fn):
        if self.headless:
            return
        gui.Application.instance.post_to_main_thread(self.window, fn)

    # ===============================
    # Reusable modal dialogs
    # ===============================
    def choice_dialog(self, message, options, title="Confirm"):
        """Modal dialog with one button per (label, callback) in `options`, plus Cancel.
        Each button closes the dialog, then runs its callback. Headless runs the first
        option's callback (no dialog)."""
        if self.headless:
            if options:
                options[0][1]()
            return
        em  = self.window.theme.font_size
        dlg = gui.Dialog(title)
        v = gui.Vert(em, gui.Margins(em, em, em, em))
        v.add_child(gui.Label(message))
        h = gui.Horiz(0.5 * em)
        h.add_stretch()

        def _make(cb):
            def _clicked():
                self.window.close_dialog()
                if cb is not None:
                    try:
                        cb()
                    except Exception as e:
                        print(f"[UI] dialog action failed: {e}")
            return _clicked

        for label, cb in options:
            btn = gui.Button(label)
            btn.set_on_clicked(_make(cb))
            h.add_child(btn)
        cancel = gui.Button("Cancel")
        cancel.set_on_clicked(lambda: self.window.close_dialog())
        h.add_child(cancel)

        v.add_child(gui.Label(""))
        v.add_child(h)
        dlg.add_child(v)
        self.window.show_dialog(dlg)

    def confirm_dialog(self, message, on_ok, title="Confirm"):
        """Modal OK/Cancel confirm. Headless just runs on_ok (no dialog)."""
        if self.headless:
            on_ok()
            return
        # Reuse choice_dialog; its Cancel button already closes the dialog and does nothing.
        self.choice_dialog(message, [("OK", on_ok)], title=title)

    # ===============================
    # Keybindings
    # ===============================
    def _on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return False

        key = event.key

        # --- Global ---
        if key == gui.KeyName.R:
            self._reframe(force=True)   # explicit user request: always re-frame
            return True
        if key == gui.KeyName.X:
            self._set_origin_axes(not self.show_origin_frame)
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
        # Express recentres into the ambiguity frame without asking, so the analysis it needs
        # must be on. Forced here rather than in `_express_sampling_worker`, which is shared
        # with `bench/generate_scenes.py` where `--no-ambiguity` is a deliberate choice.
        downsample = self.stages[Stage.DOWNSAMPLE]
        downsample.run_ambiguity = True
        if not self.headless:
            downsample.chk_ambiguity.checked = True
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
                self.main_thread(self._reframe)   # GUI op from a worker thread -> must be posted
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
        # Uniform, matching the GUI default and `bench/generate_scenes.py`. Adaptive
        # produced a p90/p10 spacing spread of ~4.3, and the ambiguity analysis derives a
        # single global epsilon from the median spacing -- one number that is simultaneously
        # too tight for the sparse regions and too loose for the dense ones. It also meant
        # the express path and the bench sweep built different clouds for the same part,
        # so any comparison between them varied two things at once.
        downsample.use_adaptive = False
        # Express recentres into the ambiguity frame, so the analysis has to run.
        downsample.run_ambiguity = True
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
        downsample.run_ambiguity = ask_yes_no(
            "Analyse pose ambiguity?", default=True,
            hint="Costs 1-3 min per part. With it on, the model frame is built around the "
                 "dominant ambiguity axis so MechVision's rotationStrategy can address it; "
                 "with it off the cloud is recentred into the PCA frame instead.")

    rp("[yellow]Note: CROP stage is GUI-only and is skipped in headless mode.[/yellow]")

    t0 = time.time()
    app._express_sampling_worker()   # raycast -> downsample -> recenter -> set_stage(SAVE)
    rp(f"[green]Sampling took {time.time() - t0:.1f}s[/green]")
    rp(f"mean point count = {raycast.point_count_mean}")
    rp(f"point count range = {raycast.point_count_range}")

    # Set the stage DIRECTLY. `_express_sampling_worker` routes its `set_stage(SAVE)` through
    # `main_thread`, which is a no-op headless -- so `app.stage` was still IMPORT_MESH here,
    # `SaveStage.worker()` matched neither of its two branches, and the reference bundle was
    # silently never written. `bench/generate_scenes.py` has always carried this line; this
    # driver did not, and claimed in a comment that the stage was already SAVE.
    app.set_stage(Stage.SAVE)
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

    # 8. Optional MechVision tuning (whole-pipeline end-to-end, headless).
    if ask_yes_no("\nRun MechVision tuning now?", default=express,
                  hint="Runs the joint Optuna study against a live MechVision instance for the "
                       "scenes just generated. Requires MechMind Hub running at 127.0.0.1:5307."):
        banner("MechVision tuning")
        tuning = app.stages[Stage.TUNING]
        # Headless callers set the study budget directly on the stage (no GUI sliders).
        tuning.n_trials = ask("Trial budget", tuning.n_trials, int,
                              hint="Number of round-0 Optuna trials.")
        tuning.n_rounds = ask("Rounds", tuning.n_rounds, int)
        tuning.sampler  = ask_choice("Sampler", list(SAMPLER_CHOICES), tuning.sampler)
        app.set_stage(Stage.TUNING)
        t0 = time.time()
        tuning._run_worker()
        rp(f"[green]Tuning took {time.time() - t0:.0f}s[/green]")
        rp(f"Pareto front : {len(tuning._pareto)} configs")

    # 9. Report output locations.
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
