import json
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from enums import Stage
from pathlib import Path
from stages.stage_base import BaseStage
from geometry.file_utils import pointcloud_to_ply
from scipy.spatial.transform import Rotation as R

_IDENTITY_POSE = [0, 0, 0, 1, 0, 0, 0]

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

    def _geocenter_comments(self):
        """The `geocenter_*` PLY header comments: this cloud's model-frame provenance.

        `app.geocenter` is the transform that has been APPLIED to the geometry, so its
        inverse is the model frame expressed in the pre-recentre frame -- i.e. where the
        geocenter origin sat, and how its axes were oriented, before `recenter_mesh_pcd`
        moved everything. That is what these key names have always claimed to hold.

        They did not hold it. The old code read `geocenter[3, 0..2]`, but `pcd_geocenter`
        returns `inv([R|o])`, whose row 3 is always `[0, 0, 0, 1]` -- so `geocenter_x/y/z`
        was `0.0` for every part ever exported, whatever the frame. The quaternion was
        likewise taken from the *inverse* rotation. Getting that inversion wrong by hand in
        `geo_center.json` is the "180 degree flip in X" incident in
        MM_Optimizer/project_state_log.md, which cost two debugging sessions chasing a
        phantom symmetry problem.

        An un-recentred export gives identity, hence zeros and a unit quaternion -- which is
        honest rather than uninformative: nothing was applied.
        """
        G = np.linalg.inv(np.asarray(self.app.geocenter, dtype=float))
        quat = R.from_matrix(G[:3, :3]).as_quat()
        return [
            f"geocenter_x {G[0, 3]}",
            f"geocenter_y {G[1, 3]}",
            f"geocenter_z {G[2, 3]}",
            f"geocenter_qx {quat[0]}",
            f"geocenter_qy {quat[1]}",
            f"geocenter_qz {quat[2]}",
            f"geocenter_qw {quat[3]}",
        ]

    def _save_cloud_bundle(self, pcd, folder_path, stem, cloud_type):
        """Save PLY + sidecar files for one cloud type into folder_path."""
        folder_path.mkdir(parents=True, exist_ok=True)
        ply_path = folder_path / f"{stem}_{cloud_type}.ply"

        pointcloud_to_ply(pcd, str(ply_path), comments=self._geocenter_comments())

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

        print(f"[INFO] Saved {cloud_type} cloud to {folder_path}")

    def _warn_if_axis_not_addressable(self, axis_tol_deg=1.0, origin_tol_m=1e-4):
        """Loudly flag a bundle whose ambiguity axis MechVision cannot rotate about.

        `rotationStrategy` can only rotate about an axis of the geocenter frame, through
        that frame's origin, and `geo_center.json` is always written as identity -- so the
        frame has to be the exported cloud's own. This checks the thing that actually has
        to be true: the dominant axis is a canonical frame axis and passes through the
        origin. If it does not, the symmetry search sweeps the wrong line at any angleStep.

        This deliberately reads the profile rather than `app.geocenter`. The old check tested
        `geocenter != identity` on the assumption that `worker()` left a *pending* transform
        there and `recenter_mesh_pcd` cleared it to identity. `geocenter` now records what
        has been APPLIED, so that test would fire on exactly the recentred bundles it was
        meant to bless. Asking the geometry is also strictly better: it catches a frame that
        is wrong for any reason, not just one specific unapplied-transform bookkeeping state.

        Warn rather than block: recentring stays a deliberate user action (the GUI button),
        and the tuner re-checks frame agreement before spending a run.
        """
        profile = getattr(self.app, "ambiguity_profile", None)
        dominant = profile.dominant if profile is not None else None
        if dominant is None:
            return                       # no axis to address; nothing to be wrong about
        d = np.asarray(dominant.direction, dtype=float)
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        p = np.asarray(dominant.point, dtype=float)
        off_axis = float(np.linalg.norm(p - float(p @ d) * d))
        aligned = float(np.abs(d).max()) >= np.cos(np.deg2rad(axis_tol_deg))
        if aligned and off_axis <= origin_tol_m:
            return
        print("[WARN] " + "=" * 66)
        print("[WARN] Exporting a cloud whose ambiguity axis is NOT a frame axis through")
        print("[WARN] the origin, so MechVision's rotationStrategy cannot address it.")
        print(f"[WARN]   axis direction {np.round(d, 4)} (off-axis by "
              f"{np.degrees(np.arccos(min(1.0, float(np.abs(d).max())))):.2f} deg)")
        print(f"[WARN]   axis passes {off_axis * 1000:.3f} mm from the origin")
        print("[WARN] Click 'Recenter to Ambiguity Axis' in the Downsample stage first.")
        print("[WARN] " + "=" * 66)

    def worker(self, path=None):
        if not self.app.headless:
            if self.app.down_pcd is None:
                print("[WARN] No pointcloud to save")
                return
        try:
            if self.app.stage == Stage.SAVE:
                self._warn_if_axis_not_addressable()
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

                pointcloud_to_ply(self.app.down_pcd, str(pcd_path),
                                  comments=self._geocenter_comments())
                print(f"[INFO] Point cloud saved to {pcd_path}")

        except Exception as e:
            print(f"[ERROR] Failed to save PLY: {e}")
            return
