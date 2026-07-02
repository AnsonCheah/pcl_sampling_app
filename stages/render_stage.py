import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from enums import Stage
from stages.stage_base import BaseStage
from pathlib import Path
from geometry.file_utils import pointcloud_to_ply
from geometry.geom_utils import camera_view_matrix, compute_overlap, O3DSceneObject
from sensor.scene_render import (
    scene_render,
    compute_dropout_mask,
    add_projector_nonuniformity,
    add_specular_patch_missing,
    add_edge_artifacts,
    add_multipath_outliers, add_pepper_noise, subset_render,
    add_sensor_noise,
    add_surface_noise,
    add_image_space_effects,
    add_scan_line_banding
)
from sensor.segment_instances import segment_point_cloud
import copy
import colorsys
from rich import print as rp
np.set_printoptions(precision=6, suppress=True)


def _passes_2d_filter(aspect, area_ratio, aspect_range, area_range) -> bool:
    """Mirror the real Mask2Former post-filter: an instance is a valid candidate iff its
    2D-mask elongation and image-area fraction both fall inside the auto-derived per-part
    ranges. If a range is missing (None), that dimension is not gated."""
    if aspect_range is not None and not (aspect_range[0] <= aspect <= aspect_range[1]):
        return False
    if area_range is not None and not (area_range[0] <= area_ratio <= area_range[1]):
        return False
    return True


class RenderStage(BaseStage):
    """Simulate the 3D sensor scanning the physical scene produced by SceneStage: raycast → noise
    pipeline → instance segmentation → per-target export. Reads app.o3d_scene + app.mj_scene; writes
    app.synthetic_targets + app.synthetic_scenes."""

    downstream = {
        "synthetic_targets": lambda: {},
        "synthetic_scenes": lambda: {},
    }

    def __init__(self, app):
        self.name = Stage.RENDER.name
        super().__init__(app)
        self.fov_deg = 41.11
        self.res_width = 1920
        self.res_height = 1200
        self.verbose = True
        self.T_cam = np.eye(4)      # world->camera view matrix; set per-scene in worker(), exported in save
        self.worker_step = 0

    def build_panel(self):
        if self.app.headless:
            return
        v = gui.Vert(4)

        self.btn_generate = self.register_widget(gui.Button("Generate Data"),
                                                 lambda: len(self.app.o3d_scene) > 0)
        self.btn_generate.set_on_clicked(self.start)
        self.combobox_targets = self.register_widget(gui.Combobox(), lambda: len(self.app.synthetic_targets) > 0)
        self.combobox_targets.set_on_selection_changed(self.preview_synthetic_target)
        self.combobox_scenes = self.register_widget(gui.Combobox(), lambda: len(self.app.synthetic_scenes) > 0)
        self.combobox_scenes.set_on_selection_changed(self.preview_synthetic_scenes)
        self.btn_export = self.register_widget(gui.Button("Export Synthetic Targets"), lambda: len(self.app.synthetic_targets) > 0)
        self.btn_export.set_on_clicked(self.save_synthetic_targets)

        self.btn_restart = self.register_widget(gui.Button("Restart"))
        self.btn_restart.set_on_clicked(lambda: self.app._restart())

        v.add_child(gui.Label("Render Synthetic Data"))
        v.add_child(gui.Label(""))
        v.add_child(self.btn_generate)
        v.add_child(self.combobox_targets)
        v.add_child(self.combobox_scenes)
        v.add_child(self.btn_export)
        v.add_child(gui.Label(""))
        v.add_child(self.btn_restart)

        print("loaded render panel")
        return v

    def _refresh_ui(self):
        if self.app.headless:
            return
        if len(self.app.synthetic_targets) > 0:
            # Targets already generated — keep the segmented preview on screen.
            self.show_segmented_scene()
        elif self.app.scene_mesh is not None:
            # Scene built but not yet rendered — show the physical scene as context.
            self.app.main_thread(lambda: self.app._clear_scene())
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry(
                "scene_mesh", self.app.scene_mesh, self.app.default_material))
        else:
            self.app.main_thread(lambda: self.app._clear_scene())
        self.enable_widgets()

    def on_clear(self):
        # GUI + stage-local scratch tied to the synthetic outputs.
        if not self.app.headless:
            self.combobox_targets.clear_items()
            self.combobox_scenes.clear_items()
        self.worker_step = 0

    def worker(self):
        TOTAL_STEPS = 9
        self.app.clear_state_from(self.stage_key)

        if not self.app.o3d_scene or self.app.mj_scene is None:
            print("[WARN] No scene available — run SCENE before RENDER.")
            return

        if not self.app.headless:
            self.app.show_progress("Rendering synthetic scene...")
            self.app.main_thread(lambda: self.app._clear_scene())

        def add_to_render_scene(name: str, geom):
            self.app.synthetic_scenes[name] = O3DSceneObject(geom)
            if self.app.headless:
                return
            self.combobox_scenes.selected_text = name
            self.combobox_scenes.add_item(name)
            self.app.main_thread(lambda: self.app._clear_scene())
            if type(geom) == o3d.geometry.PointCloud:
                material = self.app.default_point_material
            if type(geom) == o3d.geometry.TriangleMesh:
                material = self.app.default_material
            self.app.synthetic_scenes[name].material = material
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry(name, geom, material))

        def _update_pb(message: str):
            if not self.app.headless:
                print(message)
            self.worker_step += 1
            self.app.update_progress(self.worker_step / TOTAL_STEPS, message)

        cam_pos = np.asarray([0.0, 0.0, self.app.mj_scene.camera_distance])  # above bin
        look_at = np.asarray([0.0, 0.0, 0.0])                                # bin floor centre
        T_cam = camera_view_matrix(cam_pos, look_at, up=np.array([0.0, 1.0, 0.0]))
        self.T_cam = T_cam   # kept for scene-state export in save_synthetic_targets

        self.app._reframe()

        render = scene_render(self.app.o3d_scene, T_cam, look_at, self.fov_deg, self.res_width, self.res_height, verbose=self.verbose)
        pts = render["points"]
        add_to_render_scene("canonical_scene", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
        _update_pb("Synthesizing image space noise...")

        keep = compute_dropout_mask(render, roughness=0.4, density_cos_ref=0.7)
        render = add_projector_nonuniformity(render, verbose=self.verbose)
        keep   = add_specular_patch_missing(render, keep, verbose=self.verbose)
        render = add_image_space_effects(render, keep, smooth_sigma_px=0.5, sigma_fringe_corr=0.0001, verbose=self.verbose)

        add_to_render_scene("image_space_noise_scene", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(render["points"])))
        _update_pb("Synthesizing edge artifacts...")

        pts, nrm = add_edge_artifacts(render, keep, verbose=self.verbose)
        add_to_render_scene("edge_artifact_scene", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
        _update_pb("Synthesizing outliers...")

        r = subset_render(render, keep)
        mp, mn = add_multipath_outliers(r, verbose=self.verbose)
        pp, pn = add_pepper_noise(r, verbose=self.verbose)
        pts = np.vstack([pts, mp, pp])
        nrm = np.vstack([nrm, mn, pn])
        add_to_render_scene("outliers_scene", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
        _update_pb("Synthesizing banding artifacts...")
        n_kept    = keep.sum()
        n_outlier = len(pts) - n_kept
        pix_all   = np.concatenate([render["pixel_idx"][keep], np.full(n_outlier, -1, np.int64)])
        cproj_all = np.concatenate([render["cos_proj"][keep], np.ones(n_outlier)])

        pts = add_scan_line_banding(pts, nrm, pix_all, render["res"], render["sensor_origin"], verbose=self.verbose)
        add_to_render_scene("scan_banding_scene", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
        _update_pb("Adding sensor noises...")

        pts = add_sensor_noise(pts, nrm, render["sensor_origin"], pixel_idx=pix_all, res=render["res"], cos_proj=cproj_all, verbose=self.verbose)
        add_to_render_scene("sensor_noise_scene", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
        _update_pb("Adding surface noises...")

        pts = add_surface_noise(pts, nrm, verbose=self.verbose)
        surface_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        surface_pcd.normals = o3d.utility.Vector3dVector(nrm)   # kept so the full scene can be exported as PLY
        add_to_render_scene("surface_noise_scene", surface_pcd)
        _update_pb("Synthesizing segmentation erosion/dilation...")

        label_masks, seg_metrics = segment_point_cloud(  # masks: {geom_id: (N,) bool}; metrics: 2D mask stats
            render, pts, pix_all,
            erosion_px=5.0,
            dilation_px=5.0,
            confusion_depth_sigma=0.015,   # ~15 mm — tune to your part height spread
            confusion_boundary_px=4,
            occlusion_loss_px=2,
            boundary_noise_px=10.0,
            seed=0,
            return_metrics=True,
            verbose=self.verbose
        )
        _update_pb("Filtering targets by 2D aspect/area...")

        unique_id_list = list(label_masks.keys())
        rp(f"length of unique id list: {len(unique_id_list)} \n {unique_id_list}")
        valid_count = 0
        tf_by_id = {value.id: value.T_gt for value in self.app.o3d_scene.values()}
        bin_geom_id = self.app.o3d_scene["bin"].id  # set by scene_render; robust to any part count

        voxel_size =  0.001
        ref_xyz = np.asarray(self.app.down_pcd.points)
        min_overlap = 0.1
        bin_pcd = None

        # 2D candidate filter (mirrors real Mask2Former post-filter): per-part aspect-ratio
        # and area-ratio ranges auto-derived in RaycastStage. area_ratio scales as 1/d^2, so
        # rescale the scene measurement to the reference camera distance before comparing.
        aspect_range = getattr(self.app, "aspect_ratio_range", None)
        area_range   = getattr(self.app, "area_ratio_range", None)
        ref_cam_d    = getattr(self.app, "ref_cam_distance", None)
        if (aspect_range is None or area_range is None or ref_cam_d is None):
            print("[WARN] 2D filter ranges unavailable (run RAYCAST) — skipping aspect/area gate.")
            area_scale = 1.0
        else:
            area_scale = (self.app.mj_scene.camera_distance / ref_cam_d) ** 2
        for _, inst_id in enumerate(unique_id_list):
            inst_pts = pts[label_masks[inst_id]]
            inst_nrm = nrm[label_masks[inst_id]]
            inst_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inst_pts))
            inst_pcd.normals = o3d.utility.Vector3dVector(inst_nrm)
            inst_pcd_downsampled = inst_pcd.voxel_down_sample(voxel_size)

            if inst_id == bin_geom_id:
                bin_pcd = copy.deepcopy(inst_pcd_downsampled)
                self.app.synthetic_scenes["bin_pcd"] = bin_pcd
                continue

            xyz_inst = np.asarray(inst_pcd_downsampled.points)
            inst_rmat = tf_by_id[inst_id][:3, :3]
            inst_trans = tf_by_id[inst_id][:3, 3]
            xyz_ref_in_scene = (inst_rmat @ ref_xyz.T + inst_trans[:, None]).T
            try:
                overlap = compute_overlap(xyz_ref_in_scene, xyz_inst, threshold=voxel_size * 2.5)
            except Exception as e:
                print(f"[WARN] Failed to compute overlap for instance {inst_id} with {len(xyz_inst)} points: {e}")
                overlap = 0.0
            precheck_pass = True

            # --- Legacy 3D gates (point count + overlap) — kept for reference, disabled. ---
            # if not (min(self.app.point_count_range)<=len(xyz_inst)<=max(self.app.point_count_range)):
            #     print(f"[skip] Instance {inst_id} point count out of threshold {self.app.point_count_range}: {len(xyz_inst)}")
            #     precheck_pass = False
            # if overlap < min_overlap:
            #     print(f"[skip] Instance {inst_id}: overlap={overlap:.2f} < {min_overlap}")
            #     precheck_pass = False

            # --- 2D candidate filter (aspect-ratio + area-ratio), as the real network does. ---
            m = seg_metrics.get(inst_id)
            if m is None:
                print(f"[skip] Instance {inst_id}: no 2D mask metrics")
                precheck_pass = False
            elif (aspect_range is not None and area_range is not None):
                aspect     = m["aspect_ratio"]
                area_ratio = m["area_ratio"] * area_scale
                if not _passes_2d_filter(aspect, area_ratio, aspect_range, area_range):
                    print(f"[skip] Instance {inst_id}: aspect={aspect:.2f} range={tuple(round(v,2) for v in aspect_range)}, "
                          f"area_ratio={area_ratio:.5f} range={tuple(round(v,5) for v in area_range)}")
                    precheck_pass = False
            if not precheck_pass: continue
            valid_count += 1
            inst_name = f"synthetic_sample_{valid_count}"
            self.app.synthetic_targets[inst_name] = O3DSceneObject(
                geom=inst_pcd_downsampled,
                ref_geom=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_ref_in_scene)),
                id=int(inst_id),
                T_gt=tf_by_id[inst_id],
                xyz0=xyz_ref_in_scene,
                xyz1=xyz_inst,
                overlap=overlap
            )
            if not self.app.headless: self.combobox_targets.add_item(inst_name)

        # default_point_material only exists in GUI mode; omit it when headless.
        self.app.synthetic_scenes["bin_pcd"] = O3DSceneObject(
            geom=bin_pcd,
            material=None if self.app.headless else self.app.default_point_material)

        if not self.app.headless:
            bin_pcd.paint_uniform_color([1., 1., 1.])
            saturation, value = 0.4, 0.9
            hues = np.linspace(0, 1, len(self.app.synthetic_targets), endpoint=False).tolist()
            colors = [list(colorsys.hsv_to_rgb(h, saturation, value)) for h in hues]
            self.app.main_thread(lambda: self.app._clear_scene())
            for index, (key, value) in enumerate(self.app.synthetic_targets.items()):
                material = rendering.MaterialRecord()
                material.point_size = 1.5
                material.base_color = colors[index] + [1.0]
                value.material = material
                self.app.scene.scene.add_geometry(key, value.geom, value.material)
            self.combobox_scenes.selected_text = "segmented_bin_scene"
            self.combobox_scenes.add_item("segmented_bin_scene")
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("bin_pcd", bin_pcd, self.app.default_point_material))
            _update_pb("Compiling results...")
            self.show_segmented_scene()

    def save_synthetic_targets(self):
        try:
            scene_num = 0
            out_dir = Path.cwd() / "output" / "synthetic_target" / self.app.mesh_basename
            while Path.exists(out_dir / f"scene_{scene_num:05}"):
                scene_num += 1
            out_dir = out_dir / f"scene_{scene_num:05}"
            out_dir.mkdir(parents=True, exist_ok=True)
            self.app.stages[Stage.SAVE].worker(path=(out_dir))

            # Full unsegmented scene (all instances + bin + outliers, post-noise) alongside the per-instance clouds.
            scene_obj = self.app.synthetic_scenes.get("surface_noise_scene")
            if scene_obj is not None:
                pointcloud_to_ply(scene_obj.geom, out_dir / "scene.ply")

            # Final scene state: per-part GT poses + bin geometry (from the sim) + camera matrix.
            if getattr(self.app, "mj_scene", None) is not None:
                state = self.app.mj_scene.export_scene_state()
                state.update(
                    T_cam      = np.asarray(self.T_cam, dtype=np.float64),  # world -> camera view matrix
                    fov_deg    = np.float64(self.fov_deg),
                    res_width  = np.int64(self.res_width),
                    res_height = np.int64(self.res_height),
                    source     = str(out_dir).encode(),
                )
                np.savez(out_dir / "scene_state.npz", **state)

            for i, value in enumerate(self.app.synthetic_targets.values()):
                tf = value.T_gt # this is in 4x4 matrix
                gt_comments = []
                scene_pcd = copy.deepcopy(value.geom)
                for (dim, val) in zip(["gt_x", "gt_y", "gt_z"], tf[:3, 3]):
                    gt_comments.append(f"{dim} {val}")
                for (dim, val) in zip(["gt_qx", "gt_qy", "gt_qz", "gt_w"], R.from_matrix(tf[:3, :3]).as_quat()):
                    gt_comments.append(f"{dim} {val}")
                inst_stem = f"sample_{i}"
                pointcloud_to_ply(scene_pcd, out_dir / f"{inst_stem}.ply", gt_comments)
                np.savez(
                    out_dir / f"{inst_stem}.npz",
                    xyz0    = value.xyz0,                     # reference in part frame
                    xyz1    = value.xyz1,                    # noisy instance in scene frame
                    T_gt    = value.T_gt,                        # part_frame → scene_frame
                    overlap = np.float32(value.overlap),
                    source  = str(out_dir).encode(),
                )
        except Exception as e:
            print(e)

    def preview_synthetic_target(self, selected_text: str, selected_index: int) -> None:
        if self.app.headless: return

        self.app.main_thread(lambda: self.app._clear_scene())
        self.app.main_thread(lambda: self.app.scene.scene.add_geometry(
            selected_text,
            self.app.synthetic_targets[selected_text].geom,
            self.app.synthetic_targets[selected_text].material))
        self.app.scene.force_redraw()

    def preview_synthetic_scenes(self, selected_text: str, selected_index: int) -> None:
        if self.app.headless: return
        self.app.main_thread(lambda: self.app._clear_scene())
        if selected_text == "segmented_bin_scene": self.show_segmented_scene()
        else:
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry(
                selected_text,
                self.app.synthetic_scenes[selected_text].geom,
                self.app.synthetic_scenes[selected_text].material))
        self.app.scene.force_redraw()

    def show_segmented_scene(self):
        if self.app.headless: return
        self.app.main_thread(lambda: self.app._clear_scene())
        def add_geoms():
            for target_name in self.app.synthetic_targets.keys():
                self.app.scene.scene.add_geometry(
                    target_name,
                    self.app.synthetic_targets[target_name].geom,
                    self.app.synthetic_targets[target_name].material
                )
            self.app.scene.scene.add_geometry(
                "bin_pcd",
                self.app.synthetic_scenes["bin_pcd"].geom,
                self.app.synthetic_scenes["bin_pcd"].material
            )
        self.app.main_thread(add_geoms)
