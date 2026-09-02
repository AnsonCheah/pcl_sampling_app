from abc import ABC, abstractmethod
import colorsys
import open3d as o3d
import open3d.visualization.rendering as rendering
from stages.widget_base import WidgetBinding
import threading

class BaseStage(ABC):
    """
    Base class for all stages.
    """

    # Maps each app.* attribute this stage owns -> a zero-arg factory returning its
    # reset default. Factories (not literals) so every clear gets a fresh mutable
    # default. Stages that own nothing (e.g. CROP) inherit the empty dict.
    downstream = {}

    def __init__(self, app):
        self.app = app
        self.widgets = []
        self.panel = self.build_panel()
        self.worker_thread = None
        self.stage_key = None   # set by MeshSamplingApp.__init__ after the stages dict is built

    # -------------------------
    # Required interface
    # -------------------------

    @abstractmethod
    def build_panel(self):
        """Create GUI widgets and return panel"""
        pass

    @abstractmethod
    def _refresh_ui(self):
        """Called when entering stage or refreshing scene"""
        pass

    @abstractmethod
    def worker(self):
        """Heavy processing logic"""
        pass

    # -------------------------
    # State clearing
    # -------------------------

    def clear_downstream(self):
        """Reset every app.* attribute this stage owns to its declared default,
        then run any GUI-side cleanup tied to this stage's data."""
        for attr, factory in self.downstream.items():
            setattr(self.app, attr, factory())
        self.on_clear()

    def on_clear(self):
        """Hook: GUI cleanup when this stage's products are cleared. Default no-op."""
        pass

    def reset(self):
        """Clear this stage's data plus every later stage's data, then refresh UI.
        Stages with special restore semantics (e.g. CROP) override this."""
        self.app.clear_state_from(self.stage_key)
        self._refresh_ui()

    # -------------------------
    # Thread handling
    # -------------------------

    def start(self, run_on_main: bool = False):
        """
        Called by GUI button.
        If run_on_main=True, _run_worker is executed on the GUI main thread.
        Otherwise, it runs in a background worker thread.
        """
        if not self.app.headless:
            self.disable_widgets()
            self.app.show_progress()

        if run_on_main:
            # GUI-safe execution (dialogs, Open3D gui, Tk, etc.)
            self.app.main_thread(self._run_worker)
        else:
            # Background computation
            self.worker_thread = threading.Thread(target=self._run_worker, daemon=True)
            self.worker_thread.start()
            # Lock Back/Next now that the worker is alive (re-enabled in _on_worker_done).
            if not self.app.headless:
                self.app._update_nav_buttons()

    def _on_worker_start(self):
        pass

    def _run_worker(self):
        try:
            # self.app.main_thread(self._on_worker_start)
            self._on_worker_start()
            self.worker()
        finally:
            self.app.main_thread(self._on_worker_done)

    def _on_worker_done(self):
        if self.app.headless:
            return
        self.app.hide_progress()
        self._refresh_ui()
        self.enable_widgets()

    # -------------------------
    # Widget control
    # -------------------------

    def add_meshes_in_distinct_colors(self, meshes, prefix):
        """Add `meshes` to the 3D scene as `<prefix>_<i>`, each a distinct HSV hue.

        Main-thread only, like every other scene mutation.
        """
        n = max(len(meshes), 1)
        for i, mesh in enumerate(meshes):
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.base_color = list(colorsys.hsv_to_rgb(i / n, 0.6, 0.9)) + [1.0]
            geom = o3d.geometry.TriangleMesh(mesh)
            geom.compute_vertex_normals()
            self.app.scene.scene.add_geometry(f"{prefix}_{i}", geom, material)

    def register_widget(self, widget, enabled_if=lambda: True):
        """
        Register a widget and optionally its enable condition.
        """
        if self.app.headless:
            return
        self.widgets.append(WidgetBinding(widget, enabled_if))
        return widget

    def next_enabled(self) -> bool:
        """Whether the unified Next button is clickable while on this stage.
        Override in stages that require produced state before advancing."""
        return True

    def request_next(self, proceed):
        """Called by app._go_next before advancing to the next stage. Default: advance
        immediately. Stages override to intercept -- e.g. show a confirm dialog and call
        `proceed()` only on OK (used to confirm skipping a stage)."""
        proceed()

    def worker_running(self) -> bool:
        """True while this stage's background worker thread is alive."""
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def nav_enabled(self) -> bool:
        """Whether Back/Next navigation is allowed while on this stage. Locked automatically
        while a background worker is running so the user can't leave mid-run."""
        return not self.worker_running()

    def on_enter(self):
        """Hook fired once by app.set_stage when this stage becomes active (after _refresh_ui).
        Unlike _refresh_ui it is NOT called on worker completion, so it is the safe place to set
        up an initial 3D preview without clobbering a worker's result. Default no-op."""
        pass

    def enable_widgets(self):
        """
        Re-evaluate all widget enable conditions.
        Must be called from main thread.
        """
        if self.app.headless:
            return
        for binding in self.widgets:
            try:
                binding.widget.enabled = bool(binding.enabled_if())
            except Exception as e:
                print(f"[UI] Condition failed: {e}")
                binding.widget.enabled = False
        # Keep the app-level Back/Next buttons in sync with the latest state.
        self.app._update_nav_buttons()

    def disable_widgets(self):
        if self.app.headless:
            return
        for w in self.widgets:
            w.widget.enabled = False

    def _on_key(self,event):
        pass

    def _on_mouse_event(self,event):
        pass
