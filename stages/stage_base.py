from abc import ABC, abstractmethod
import threading

class BaseStage(ABC):
    """
    Base class for all stages.
    """

    def __init__(self, app):
        self.app = app
        self.widgets = []
        self.panel = self.build_panel()
        self.worker_thread = None

    # -------------------------
    # Required interface
    # -------------------------

    @abstractmethod
    def build_panel(self):
        """Create GUI widgets and return panel"""
        pass

    @abstractmethod
    def init(self):
        """Called when entering stage or refreshing scene"""
        pass

    @abstractmethod
    def worker(self):
        """Heavy processing logic"""
        pass

    @abstractmethod
    def reset(self):
        """Clear stage data and restore previous stage"""
        pass

    # -------------------------
    # Thread handling
    # -------------------------

    def start(self, run_on_main: bool = False):
        """
        Called by GUI button.
        If run_on_main=True, _run_worker is executed on the GUI main thread.
        Otherwise, it runs in a background worker thread.
        """
        self.disable_widgets()
        self.app.show_progress()

        if run_on_main:
            # GUI-safe execution (dialogs, Open3D gui, Tk, etc.)
            self.app.main_thread(self._run_worker)
        else:
            # Background computation
            self.worker_thread = threading.Thread(target=self._run_worker, daemon=True)
            self.worker_thread.start()

    def _run_worker(self):
        try:
            # self.app.main_thread(self._on_worker_start)
            self._on_worker_start()
            self.worker()
        finally:
            self.app.main_thread(self._on_worker_done)

    def _on_worker_start(self):
        pass

    def _on_worker_done(self):
        self.enable_widgets()
        self.app.hide_progress()
        self.init()

    # -------------------------
    # Widget control
    # -------------------------

    def register_widget(self, widget):
        self.widgets.append(widget)

    def disable_widgets(self):
        for w in self.widgets:
            w.enabled = False

    def enable_widgets(self, widget=[]):
        for w in self.widgets:
            w.enabled = True
    
    def _on_key(self,event):
        pass

    def _on_mouse_event(self,event):
        pass
