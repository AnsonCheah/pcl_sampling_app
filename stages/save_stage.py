import open3d.visualization.gui as gui
from enums import Stage
from pathlib import Path
from stages.stage_base import BaseStage
from utilities import pointcloud_to_ply

class SaveStage(BaseStage):
    def __init__(self, app):
        self.name = Stage.SAVE.name
        super().__init__(app)

    def build_panel(self):
        v = gui.Vert(4)

        self.btn_save = self.register_widget(gui.Button("Export Point Cloud"))
        self.btn_save.set_on_clicked(self.start)

        self.btn_next = self.register_widget(gui.Button("Next: Synthetic Target"))
        self.btn_next.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value + 1)))
        self.btn_back = self.register_widget(gui.Button("Back: Downsample"))
        self.btn_back.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value - 1)))

        self.btn_restart = self.register_widget(gui.Button("Restart"))
        self.btn_restart.set_on_clicked(lambda: self.app._restart())

        v.add_child(gui.Label("Export PCL"))
        v.add_child(gui.Label(""))
        v.add_child(self.btn_save)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(self.btn_back)
        v.add_child(self.btn_next)
        v.add_child(self.btn_restart)

        print("loaded save panel")
        return v
        
    def _refresh_ui(self):
        if self.app.headless:
            return
        if self.app.down_pcd != None:
            self.app.main_thread(lambda: self.app._clear_scene())
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("down_pcd", self.app.down_pcd, self.app.default_point_material))
            self.app.scene.force_redraw()
        self.enable_widgets()

    def worker(self):
        if not self.app.headless:
            if self.app.down_pcd is None:
                print("[WARN] No pointcloud to save")
                return
        try:
            folder_path = Path.cwd() / "reference_pcd"
            folder_path.mkdir(parents=True, exist_ok=True)
            pcd_path = folder_path / (self.app.mesh_basename + ".ply")
            pointcloud_to_ply(self.app.down_pcd, str(pcd_path))
            print(f"[INFO] Point cloud saved to {pcd_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save PLY: {e}")
            return

    def reset(self):
        pass
