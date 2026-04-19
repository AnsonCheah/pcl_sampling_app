import open3d.visualization.gui as gui
from enums import Stage
from pathlib import Path
from stages.stage_base import BaseStage
from geometry.file_utils import pointcloud_to_ply
from scipy.spatial.transform import Rotation as R
# from open3d.io import write_triangle_mesh

class SaveStage(BaseStage):
    def __init__(self, app):
        self.name = Stage.SAVE.name
        super().__init__(app)

    def build_panel(self):
        if self.app.headless:
            return
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

    def worker(self, path=None):
        if not self.app.headless:
            if self.app.down_pcd is None:
                print("[WARN] No pointcloud to save")
                return
        try:
            if self.app.stage == Stage.SAVE:
                pcd_folder_path = Path.cwd() / "reference_pcd"
                pcd_folder_path.mkdir(parents=True, exist_ok=True)
                pcd_path = pcd_folder_path / (self.app.mesh_basename + ".ply")
                self.app.output_pcd_path = pcd_path
            elif self.app.stage == Stage.SYNTHETIC:
                if path==None: 
                    print(f"[SAVE] worker: path is not provided in synthetic stage.")
                pcd_path = path / "reference_cloud.ply"

            center_quat = R.from_matrix(self.app.geocenter[:3,:3]).as_quat()
            comments = [
                f"geocenter_x {self.app.geocenter[3, 0]}",
                f"geocenter_y {self.app.geocenter[3, 1]}",
                f"geocenter_z {self.app.geocenter[3, 2]}",
                f"geocenter_qx {center_quat[0]}",
                f"geocenter_qy {center_quat[1]}",
                f"geocenter_qz {center_quat[2]}",
                f"geocenter_qw {center_quat[3]}"
                ]
            pointcloud_to_ply(self.app.down_pcd, str(pcd_path), comments=comments)
            print(f"[INFO] Point cloud saved to {pcd_path}")
            # mesh_folder_path = Path.cwd() / "processed_mesh"
            # mesh_folder_path.mkdir(parents=True, exist_ok=True)
            # mesh_path = mesh_folder_path / (self.app.mesh_basename + ".stl")
            # write_triangle_mesh(mesh_path, self.app.target_mesh, write_ascii=False, print_progress=True)
            # print(f"[INFO] Mesh saved to {mesh_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save PLY: {e}")
            return

    def reset(self):
        pass
