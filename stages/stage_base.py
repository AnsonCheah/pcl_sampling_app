from abc import ABC, abstractmethod
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
