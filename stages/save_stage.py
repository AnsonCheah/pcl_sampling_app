import json
import open3d as o3d
import open3d.visualization.gui as gui
from enums import Stage
from pathlib import Path
from stages.stage_base import BaseStage
from geometry.file_utils import pointcloud_to_ply
from geometry.ambiguity import save_ambiguity_profile
from scipy.spatial.transform import Rotation as R

_IDENTITY_POSE = [0, 0, 0, 1, 0, 0, 0]
AMBIGUITY_SIDECAR = "ambiguity_profile.json"

class SaveStage(BaseStage):
    downstream = {
        "output_pcd_path": lambda: None,
    }

    def __init__(self, app):
        self.name = Stage.SAVE.name
        super().__init__(app)

    def build_panel(self):
        if self.app.headless:
            return
        v = gui.Vert(4)

        self.btn_save = self.register_widget(gui.Button("Export Point Cloud"))
        self.btn_save.set_on_clicked(self.start)

        self.btn_restart = self.register_widget(gui.Button("Restart"))
        self.btn_restart.set_on_clicked(lambda: self.app._restart())

        v.add_child(gui.Label("Export PCL"))
        v.add_child(gui.Label(""))
        v.add_child(self.btn_save)
        v.add_child(gui.Label(""))
        v.add_child(self.btn_restart)

        print("loaded save panel")
        return v

    def _refresh_ui(self):
        if self.app.headless:
            return
        if self.app.down_pcd is not None:
            self.app.main_thread(lambda: self.app._clear_scene())
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("down_pcd", self.app.down_pcd, self.app.default_point_material))
            self.app.main_thread(self.app._reframe)   # content swap -> reframe (posts a redraw too)
        self.enable_widgets()

    def _save_cloud_bundle(self, pcd, folder_path, stem, cloud_type):
        """Save PLY + sidecar files for one cloud type into folder_path."""
        folder_path.mkdir(parents=True, exist_ok=True)
        ply_path = folder_path / f"{stem}_{cloud_type}.ply"

        center_quat = R.from_matrix(self.app.geocenter[:3, :3]).as_quat()
        comments = [
            f"geocenter_x {self.app.geocenter[3, 0]}",
            f"geocenter_y {self.app.geocenter[3, 1]}",
            f"geocenter_z {self.app.geocenter[3, 2]}",
            f"geocenter_qx {center_quat[0]}",
            f"geocenter_qy {center_quat[1]}",
            f"geocenter_qz {center_quat[2]}",
            f"geocenter_qw {center_quat[3]}"
        ]
        pointcloud_to_ply(pcd, str(ply_path), comments=comments)

        with open(folder_path / "geo_center.json", "w") as f:
            json.dump([_IDENTITY_POSE], f, indent=4)

        with open(folder_path / "pick_points.json", "w") as f:
            json.dump([], f, indent=4)

        with open(folder_path / "pick_points_labels.json", "w") as f:
            json.dump([], f, indent=4)

        poses = [{
            "label": "0",
            "name": f"{stem}_{cloud_type}_geocenter",
            "pose": _IDENTITY_POSE,
            "pose_type": 2
        }]
        with open(folder_path / "poses.poses", "w") as f:
            json.dump(poses, f, indent=4)

        # Sidecar goes next to every cloud variant so the tuner finds it beside whichever
        # model it loads. The axis is already expressed in this cloud's frame.
        #
        # The per-point heat map is written ONLY for the surface cloud: that is the cloud the
        # analysis ran on, so it is the only one the array is index-aligned with. Writing it
        # beside the edge / feature / flat variants would produce a file that looks usable
        # and silently scores the wrong points.
        profile = getattr(self.app, "ambiguity_profile", None)
        if profile is not None:
            save_ambiguity_profile(profile, folder_path / AMBIGUITY_SIDECAR,
                                   per_point=(cloud_type == "surface"))

        print(f"[INFO] Saved {cloud_type} cloud to {folder_path}")

    def worker(self, path=None):
        if not self.app.headless:
            if self.app.down_pcd is None:
                print("[WARN] No pointcloud to save")
                return
        try:
            if self.app.stage == Stage.SAVE:
                stem = self.app.mesh_basename
                base_path = Path(__file__).resolve().parent.parent / "output" / "reference_pcd" / stem

                surface_pcd = self.app.down_pcd_surface if self.app.down_pcd_surface is not None else self.app.down_pcd
                self._save_cloud_bundle(surface_pcd, base_path / f"{stem}_surface", stem, "surface")

                if self.app.target_mesh is not None:
                    stl_path = base_path / f"{stem}.stl"
                    o3d.io.write_triangle_mesh(str(stl_path), self.app.target_mesh)
                    print(f"[INFO] Saved mesh to {stl_path}")

                if self.app.down_pcd_edge is not None:
                    self._save_cloud_bundle(self.app.down_pcd_edge, base_path / f"{stem}_edge", stem, "edge")
                else:
                    print("[WARN] Edge cloud not available, skipping edge export")

                if self.app.feature_pcd is not None:
                    self._save_cloud_bundle(self.app.feature_pcd, base_path / f"{stem}_feature", stem, "feature")
                if self.app.pcd_flat is not None:
                    self._save_cloud_bundle(self.app.pcd_flat, base_path / f"{stem}_flat", stem, "flat")

                self.app.output_pcd_path = base_path / f"{stem}_surface" / f"{stem}_surface.ply"

            elif self.app.stage == Stage.RENDER:
                if path is None:
                    print(f"[SAVE] worker: path is not provided in render stage.")
                pcd_path = path / "reference_cloud.ply"

                center_quat = R.from_matrix(self.app.geocenter[:3, :3]).as_quat()
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

                # Also drop the ambiguity sidecar into the scene directory. Without it a
                # consumer of these scenes has to re-run the analysis from the mesh (a minute
                # per part) just to weight votes by discriminability. `down_pcd` shares point
                # order with `down_pcd_surface` in both uniform and adaptive modes, so the
                # per-point array is aligned with the cloud written just above.
                profile = getattr(self.app, "ambiguity_profile", None)
                if profile is not None:
                    save_ambiguity_profile(profile, path / AMBIGUITY_SIDECAR, per_point=True)
                    print(f"[INFO] Ambiguity sidecar saved to {path / AMBIGUITY_SIDECAR}")

        except Exception as e:
            print(f"[ERROR] Failed to save PLY: {e}")
            return
