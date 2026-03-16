import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from enums import Stage
from stages.stage_base import BaseStage
from pathlib import Path
from file_utils import pointcloud_to_ply
# from trimesh.collision import CollisionManager
from geom_utils import o3d_to_trimesh, trimesh_to_o3d, camera_view_matrix, O3DSceneObject
from scene_render import (
    scene_render,
    compute_dropout_mask,
    add_edge_artifacts,
    add_multipath_outliers, add_pepper_noise, subset_render,
    add_sensor_noise,
    add_surface_noise,
    add_image_space_effects,
    add_scan_line_banding
)
from segment_instances import segment_point_cloud
import copy
from mujoco_bin_scene import MujocoBinScene
import colorsys

class SyntheticStage(BaseStage):
    def __init__(self, app):
        self.name = Stage.SYNTHETIC.name
        super().__init__(app)
        self.fov_deg = 41.11
        self.res_width = 1920
        self.res_height = 1200
        self.app.synthetic_targets = []
        self.app.synthetic_scenes = []
        self.scene_objects = {}

    def build_panel(self):
        if self.app.headless:
            return
        v = gui.Vert(4)

        self.num_targets_slider = self.register_widget(gui.Slider(gui.Slider.INT))
        self.num_targets_slider.set_limits(2, 200)
        self.num_targets_slider.int_value = 6
        self.btn_generate = self.register_widget(gui.Button("Generate Synthetic Targets"))
        self.btn_generate.set_on_clicked(self.start)
        self.btn_reset = self.register_widget(gui.Button("Clear Synthetic Targets"), lambda: len(self.app.synthetic_targets)>0)
        self.btn_reset.set_on_clicked(self.reset)
        self.combobox_targets = self.register_widget(gui.Combobox(), lambda: len(self.app.synthetic_targets)>0)
        self.combobox_targets.set_on_selection_changed(self.preview_synthetic_target)
        self.combobox_scenes = self.register_widget(gui.Combobox(), lambda: len(self.app.synthetic_scenes)>0)
        self.combobox_scenes.set_on_selection_changed(self.preview_synthetic_scenes)
        self.btn_export = self.register_widget(gui.Button("Export Synthetic Targets"), lambda: len(self.app.synthetic_targets)>0)
        self.btn_export.set_on_clicked(self.save_synthetic_targets)

        self.btn_back = self.register_widget(gui.Button("Back: SAVE"))
        self.btn_back.set_on_clicked(lambda: self.app.set_stage(Stage(self.app.stage.value - 1)))

        self.btn_restart = self.register_widget(gui.Button("Restart"))
        self.btn_restart.set_on_clicked(lambda: self.app._restart())
        v.add_child(gui.Label("Generate Synthetic Targets"))
        v.add_child(gui.Label(""))
        v.add_child(self.num_targets_slider)
        v.add_child(self.btn_generate)
        v.add_child(self.btn_reset)
        v.add_child(self.combobox_targets)
        v.add_child(self.combobox_scenes)
        v.add_child(self.btn_export)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(gui.Label(""))
        v.add_child(self.btn_back)
        v.add_child(self.btn_restart)

        print("loaded synthetic panel")
        return v

    def _refresh_ui(self):
        if self.app.headless:
            return
        self.enable_widgets()

    def reset(self):
        self.app.synthetic_targets = []
        self.scene_objects = {}
        self.combobox_targets.clear_items()
        self._refresh_ui()

    def worker(self):
        if not self.app.headless:
            self.app.show_progress("Simulating synthetic scene...")

        def add_to_render_scene(name:str, geom, type:str, color=[1.0, 1.0, 1.0]):
            material = rendering.MaterialRecord()
            material.point_size = 1.5
            material.base_color = color + [1.0]
            if type=="scene": 
                self.app.synthetic_scenes.append(geom)
                self.combobox_scenes.add_item(name)
                self.hide_geoms_in_scene()
            if type=="target": 
                self.combobox_targets.add_item(name)
                pass
            self.scene_objects[name] = O3DSceneObject(geom, material)
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry(name, geom, material))

        TOTAL_STEPS = 10
        self.app.synthetic_targets = []
        self.app.main_thread(lambda: self.app._clear_scene())

        current_step = 0
        self.num_targets =  self.num_targets_slider.int_value
        rendering_flag = False
        verbose = True
        part_mesh = o3d_to_trimesh(self.app.target_mesh)
        scene = MujocoBinScene(part_mesh, self.app.convex_meshes, n_parts=self.num_targets, render=rendering_flag)
        scene.simulate(realtime=rendering_flag)
        scene_state = scene.extract_scene_state()
        o3d_scene = scene.mujoco_scene_to_open3d(scene_state)

        current_step += 1
        self.app.update_progress(current_step/TOTAL_STEPS, f"Generating synthetic scene...")

        cam_pos = np.asarray([0,0,1.5])
        look_at = np.zeros(3)
        T_cam = camera_view_matrix(cam_pos, look_at)

        render = scene_render(o3d_scene, T_cam, look_at, self.fov_deg, self.res_width, self.res_height, verbose=verbose)
        pts = render["points"]
        add_to_render_scene("canonical_scene", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)), type="scene")

        current_step += 1
        self.app.update_progress(current_step/TOTAL_STEPS, f"Synthesizing image space noise...")

        keep = compute_dropout_mask(render, roughness=0.4, density_cos_ref=0.7)
        render = add_image_space_effects(render, keep, smooth_sigma_px=0.5, sigma_fringe_corr=0.0001, verbose=verbose)

        add_to_render_scene("image_space_noise_scene", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(render["points"])), type="scene")
        current_step += 1
        self.app.update_progress(current_step/TOTAL_STEPS, f"Synthesizing edge artifacts...")
        
        pts, nrm = add_edge_artifacts(render, keep, verbose=verbose)
        add_to_render_scene("edge_artifact_scene", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)), type="scene")
        current_step += 1
        self.app.update_progress(current_step/TOTAL_STEPS, f"Synthesizing outliers...")

        r = subset_render(render, keep)
        mp, mn = add_multipath_outliers(r, verbose=verbose)
        pp, pn = add_pepper_noise(r, verbose=verbose)
        pts = np.vstack([pts, mp, pp])
        nrm = np.vstack([nrm, mn, pn])
        add_to_render_scene("outliers_scene", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)), type="scene")
        current_step += 1
        self.app.update_progress(current_step/TOTAL_STEPS, f"Synthesizing banding artifacts...")
        # Build per-point metadata for the full combined cloud.
        n_kept    = keep.sum()
        n_outlier = len(pts) - n_kept
        pix_all   = np.concatenate([render["pixel_idx"][keep], np.full(n_outlier, -1, np.int64)])
        cproj_all = np.concatenate([render["cos_proj"][keep], np.ones(n_outlier)])

        pts = add_scan_line_banding(pts, nrm, pix_all, render["res"], render["sensor_origin"], verbose=verbose)
        add_to_render_scene("scan_banding_scene", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)), type="scene")
        current_step += 1
        self.app.update_progress(current_step/TOTAL_STEPS, f"Adding sensor noises...")

        pts = add_sensor_noise(pts, nrm, render["sensor_origin"], pixel_idx=pix_all, res=render["res"], cos_proj=cproj_all, verbose=verbose)
        add_to_render_scene("sensor_noise_scene", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)), type="scene")
        current_step += 1
        self.app.update_progress(current_step/TOTAL_STEPS, f"Adding surface noises...")

        pts = add_surface_noise(pts, nrm, verbose=verbose)
        add_to_render_scene("surface_noise_scene", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)), type="scene")
        current_step += 1
        self.app.update_progress(current_step/TOTAL_STEPS, f"Synthesizing segmentation erosion/dilation...")

        labels = segment_point_cloud( # labels : (N,) int32 — geom_id per point, -1 = unassigned
            render, pts, pix_all,
            erosion_px=3.0,
            dilation_px=1.5,
            confusion_depth_sigma=0.015,   # ~15 mm — tune to your part height spread
            confusion_boundary_px=4,
            occlusion_loss_px=2,
            boundary_noise_px=6.0,
            seed=0,
            verbose=verbose
        )
        current_step += 1
        self.app.update_progress(current_step/TOTAL_STEPS, f"Filtering targets by point counts...")

        unique_id_list = np.unique(labels[labels >= 0])
        for _, inst_id in enumerate(unique_id_list):
            inst_pts = pts[labels == inst_id]
            inst_nrm = nrm[labels == inst_id]
            inst_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inst_pts))
            inst_pcd.normals = o3d.utility.Vector3dVector(inst_nrm)
            inst_pcd_downsampled = inst_pcd.voxel_down_sample(0.001)
            if inst_id == len(unique_id_list)-1: # box will always be at last
                bin_pcd = copy.deepcopy(inst_pcd_downsampled)
                bin_pcd.paint_uniform_color([1., 1., 1.])
                continue
            if min(self.app.point_count_range)<=len(inst_pcd_downsampled.points)<=max(self.app.point_count_range):
                print(f"Instance {inst_id} point count within threshold {self.app.point_count_range}: {len(inst_pcd_downsampled.points)}")
                self.app.synthetic_targets.append(inst_pcd_downsampled)
                continue
            print(f"Instance {inst_id} point count out of threshold {self.app.point_count_range}: {len(inst_pcd_downsampled.points)}")

        saturation, value = 0.4, 0.9
        hues = np.linspace(0, 1, len(self.app.synthetic_targets), endpoint=False).tolist()
        colors = [list(colorsys.hsv_to_rgb(h, saturation, value)) for h in hues]
        self.hide_geoms_in_scene()
        for index, pcd in enumerate(self.app.synthetic_targets):
            add_to_render_scene(f"synthetic_sample_{index}", pcd, type="target", color=colors[index])

        self.combobox_scenes.add_item("segmented_bin_scene")
        self.scene_objects["bin_pcd"] = O3DSceneObject(geom=bin_pcd, material=self.app.default_point_material)
        self.app.synthetic_scenes.append("segmented_bin_scene")
        self.app.main_thread(lambda: self.app.scene.scene.add_geometry("bin_pcd", bin_pcd, self.app.default_point_material))
        current_step += 1
        self.app.update_progress(current_step/TOTAL_STEPS, f"Compiling results...")
        self.show_segmented_scene()

    def save_synthetic_targets(self):
        try:
            base_path = Path.cwd() / "synthetic_target" / self.app.mesh_basename
            for start, prefix in enumerate(["train", "test"]):
                folder_path = base_path / prefix
                folder_path.mkdir(parents=True, exist_ok=True)
                for i, pcd in enumerate(self.app.synthetic_targets[start::2]):
                    pointcloud_to_ply(pcd, folder_path / f"{prefix}_sample_{i}.ply")
        except Exception as e:
            print(e)

    def preview_synthetic_target(self, selected_text: str, selected_index: int) -> None:
        if self.app.headless:
            return
        self.app.main_thread(lambda: self.app._clear_scene())
        if not isinstance(self.scene_objects[selected_text], O3DSceneObject):
            return
        self.app.main_thread(lambda: self.app.scene.scene.add_geometry(
            selected_text, 
            self.scene_objects[selected_text].geom, 
            self.scene_objects[selected_text].material))
        self.app.scene.force_redraw()

    def preview_synthetic_scenes(self, selected_text: str, selected_index: int) -> None:
        if self.app.headless:
            return
        self.app.main_thread(lambda: self.app._clear_scene())
        if selected_text == "segmented_bin_scene":
            geom_list = ["bin_pcd"] + [f"synthetic_sample_{i}" for i,_ in enumerate(self.app.synthetic_targets)]
            def add_geoms():
                for geom_name in geom_list:
                    self.app.scene.scene.add_geometry(geom_name, self.scene_objects[geom_name].geom, self.scene_objects[geom_name].material)
            self.app.main_thread(add_geoms)
        else:
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry(
                selected_text, 
                self.scene_objects[selected_text].geom, 
                self.scene_objects[selected_text].material))
        self.app.scene.force_redraw()

    def hide_geoms_in_scene(self, geoms=None):
        def hide_geoms():
            for key in self.scene_objects.keys():
                self.app.scene.scene.show_geometry(name=key, show=False)
        self.app.main_thread(hide_geoms)
        
    def show_segmented_scene(self):
        geom_list = ["bin_pcd"] + [f"synthetic_sample_{i}" for i,_ in enumerate(self.app.synthetic_targets)]
        def show_geoms():
            for name in geom_list:
                self.app.scene.scene.show_geometry(name, show=True)
        self.app.main_thread(show_geoms)
