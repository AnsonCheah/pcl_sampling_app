import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
# from trimesh.boolean import union
import trimesh
from enums import Stage
from stages.stage_base import BaseStage
from pathlib import Path
from geometry.file_utils import pointcloud_to_ply
from geometry.geom_utils import o3d_to_trimesh, trimesh_to_o3d, camera_view_matrix, compute_overlap, O3DSceneObject
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
from physics.mujoco_bin_scene import MujocoBinScene, MAX_BIN_DIM
import colorsys
from rich import print as rp
np.set_printoptions(precision=6, suppress=True)

SCENE_FILL_RATE     = 0.6    # default fill rate (fraction of safe capacity)
# Shape-aware random-packing efficiency for part OBB volumes. A fixed factor is only valid for
# cube-like parts; random packing fraction falls ~1/aspect_ratio for elongated/flat parts
# (Philipse random-contact scaling; see _auto_part_count). We taper from a cube anchor by the
# OBB edge ratio so big elongated parts no longer over-count.
PACKING_FACTOR_BASE  = 0.62  # cube/blocky parts (sphere random-close-pack band)
PACKING_FACTOR_FLOOR = 0.18  # keep very thin/long parts from collapsing toward 0
BIN_TOP_MARGIN_FRAC = 0.20   # reserve top 20% of bin height as spill headroom
MIN_AUTO_PARTS      = 2      # never fewer than this
MAX_AUTO_PARTS      = 500    # safety cap (tune): tiny parts otherwise explode the sim
FILL_SLIDER_MIN, FILL_SLIDER_MAX   = 20, 100   # fill-rate slider range (percent)
COUNT_SLIDER_MIN, COUNT_SLIDER_MAX = 2, 500    # override-count slider range (parts)
PREVIEW_PARKED_Z = 1.0   # m — live-preview bodies above this z are still parked (PARKING_Z≈10);
#                          hide them until release_batch() drops them onto the pile (bin ~0.25 m tall)

class SyntheticStage(BaseStage):
    def __init__(self, app):
        self.name = Stage.SYNTHETIC.name
        # Set before super().__init__(): it calls build_panel(), which reads these.
        self.num_targets = 6
        self.arrangement = "random"        # headless callers may set before _run_worker()
        self.generate_mode = "fill_rate"   # "fill_rate" (auto-size to part) or "count" (manual override)
        self.fill_rate = SCENE_FILL_RATE   # used when generate_mode == "fill_rate"
        super().__init__(app)
        self.fov_deg = 41.11
        self.res_width = 1920
        self.res_height = 1200
        self.app.synthetic_targets = {}
        self.app.synthetic_scenes = {}
        self.rendering_flag = False
        self.verbose = True
        self.o3d_scene = {}
        self.T_cam = np.eye(4)      # world->camera view matrix; set per-scene in worker(), exported in save
        self.stable_pose_R = None   # optional forced stable orientation (structured mode, set per-pose by headless driver)

    def build_panel(self):
        if self.app.headless:
            return
        v = gui.Vert(4)

        self.arrangement_combo = self.register_widget(gui.Combobox())
        self.arrangement_combo.add_item("Random")
        self.arrangement_combo.add_item("Structured")
        self.arrangement_combo.selected_index = 0
        self.arrangement_combo.set_on_selection_changed(self._on_arrangement_changed)
        # One slider, repurposed by the radio: fill-rate % (auto) or override count.
        self.radio_mode = self.register_widget(gui.RadioButton(gui.RadioButton.HORIZ),
                                               enabled_if=lambda: self._is_random())
        self.radio_mode.set_items(["By Fill Rate", "Override Count"])
        self.radio_mode.selected_index = 0
        self.radio_mode.set_on_selection_changed(self._on_generate_mode_changed)
        self.gen_label = gui.Label("Fill Rate (%)")   # plain label, not registered
        self.gen_slider = self.register_widget(gui.Slider(gui.Slider.INT),
                                               enabled_if=lambda: self._is_random())
        self._apply_slider_mode(0)                     # init in fill-rate mode
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
        v.add_child(self.arrangement_combo)
        v.add_child(self.radio_mode)
        v.add_child(self.gen_label)
        v.add_child(self.gen_slider)
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

    def _is_random(self) -> bool:
        return self.arrangement_combo.selected_text == "Random"

    def _on_arrangement_changed(self, text, idx):
        # Predicates drive enable/disable; just re-evaluate them.
        self.enable_widgets()

    def _apply_slider_mode(self, idx: int):
        """Repurpose the single slider: idx 0 = fill-rate %, idx 1 = override count."""
        if idx == 0:
            self.gen_label.text = "Fill Rate (%)"
            self.gen_slider.set_limits(FILL_SLIDER_MIN, FILL_SLIDER_MAX)   # limits BEFORE value
            self.gen_slider.int_value = int(round(self.fill_rate * 100))
        else:
            self.gen_label.text = "Override Count"
            self.gen_slider.set_limits(COUNT_SLIDER_MIN, COUNT_SLIDER_MAX)
            self.gen_slider.int_value = int(self.num_targets)

    def _on_generate_mode_changed(self, idx: int):
        # Persist the value from the mode being left, then repurpose the slider.
        if idx == 0:                                   # leaving count -> fill
            self.num_targets = int(self.gen_slider.int_value)
        else:                                          # leaving fill -> count
            self.fill_rate = self.gen_slider.int_value / 100.0
        self.generate_mode = "fill_rate" if idx == 0 else "count"
        self._apply_slider_mode(idx)
        self.enable_widgets()

    def _auto_part_count(self, part_mesh) -> int:
        """Size the part count to the part so the fixed max-size bin reaches a
        consistent volumetric fill across all parts, reserving BIN_TOP_MARGIN_FRAC
        of the bin height as spill headroom. See module constants.

        The packing factor is shape-aware: it tapers from PACKING_FACTOR_BASE by the OBB
        aspect ratio (longest/shortest edge) so elongated/flat parts — which pack far less
        densely — no longer over-count. We do NOT also multiply by solidity: the count divides
        by OBB volume, so the correct multiplier is the box-packing fraction (folding solidity
        in again would double-count). Anchors: AR 1->0.62, 2->0.44, 4->0.31, >=12->0.18 floor."""
        bw, bl, bh, _ = MAX_BIN_DIM
        usable_h = bh * (1.0 - BIN_TOP_MARGIN_FRAC)   # reserve top headroom to prevent spilling
        bin_vol  = bw * bl * usable_h
        mobb_vol = max(float(part_mesh.bounding_box_oriented.volume), 1e-9)
        ext = np.sort(part_mesh.bounding_box_oriented.extents)[::-1]   # e1 >= e2 >= e3
        aspect_ratio   = float(ext[0] / max(ext[2], 1e-9))
        packing_factor = max(PACKING_FACTOR_FLOOR, PACKING_FACTOR_BASE / np.sqrt(aspect_ratio))
        n = round(self.fill_rate * packing_factor * bin_vol / mobb_vol)
        rp(f"[PACKING] OBB aspect ratio={aspect_ratio:.2f} -> packing_factor={packing_factor:.3f}")
        return int(np.clip(n, MIN_AUTO_PARTS, MAX_AUTO_PARTS))

    def _display_convex_meshes(self):
        """GUI helper: show each convex hull in a distinct HSV colour."""
        n = len(self.app.convex_meshes)
        for i, mesh in enumerate(self.app.convex_meshes):
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            rgb = colorsys.hsv_to_rgb(i / max(n, 1), 0.6, 0.9)
            material.base_color = list(rgb) + [1.0]
            geom = o3d.geometry.TriangleMesh(mesh)
            geom.compute_vertex_normals()
            self.app.scene.scene.add_geometry(f"convex_{i}", geom, material)

    def _refresh_ui(self):
        if self.app.headless:
            return
        if len(self.app.synthetic_targets) > 0:
            # Targets already generated — keep the segmented preview on screen.
            self.show_segmented_scene()
        else:
            # Stage entry, before generation: show the decomposed convex hulls.
            self.app.main_thread(lambda: self.app._clear_scene())
            if len(self.app.convex_meshes) > 0:
                self.app.main_thread(self._display_convex_meshes)
            elif self.app.target_mesh is not None:
                self.app.main_thread(lambda: self.app.scene.scene.add_geometry(
                    "mesh", self.app.target_mesh, self.app.default_material))
        self.enable_widgets()

    def reset(self):
        if not self.app.headless:
            self.combobox_targets.clear_items()
            self.combobox_scenes.clear_items()
        self.app.synthetic_targets = {}
        self.app.synthetic_scenes = {}
        self.o3d_scene = {}
        self.worker_step = 0
        self._refresh_ui()

    def worker(self):
        TOTAL_STEPS = 10
        self.reset()

        # The MuJoCo passive viewer is blocking and must never run headless/batch.
        if self.app.headless:
            self.rendering_flag = False
        if not self.app.convex_meshes:
            print("[WARN] No convex meshes available — run DecomposeStage before SYNTHETIC; "
                  "simulation collisions will be degraded.")

        if not self.app.headless:
            self.app.show_progress("Simulating synthetic scene...")
            self.app.main_thread(lambda: self.app._clear_scene())
            self.arrangement = self.arrangement_combo.selected_text.lower()
            self.generate_mode = "fill_rate" if self.radio_mode.selected_index == 0 else "count"
            if self.generate_mode == "fill_rate":
                self.fill_rate = self.gen_slider.int_value / 100.0
            else:
                self.num_targets = int(self.gen_slider.int_value)

        def add_to_render_scene(name:str, geom):
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

        def _update_pb(message:str):
            if not self.app.headless:
                print(message)
            self.worker_step += 1
            self.app.update_progress(self.worker_step/TOTAL_STEPS, message)

        part_mesh = o3d_to_trimesh(self.app.target_mesh)
        # Fill-rate mode auto-sizes count to the part; structured ignores n_parts (grid sets it).
        if self.generate_mode == "fill_rate" and self.arrangement == "random":
            self.num_targets = self._auto_part_count(part_mesh)
            rp(f"[AUTO-COUNT] fill={self.fill_rate:.0%} "
               f"part OBB vol={part_mesh.bounding_box_oriented.volume:.2e} m³ "
               f"-> n_parts={self.num_targets}")
        self.mj_scene = MujocoBinScene(part_mesh, self.app.convex_meshes, n_parts=self.num_targets, render=self.rendering_flag, arrangement=self.arrangement, stable_pose_R=self.stable_pose_R)
        # Live mesh preview: only meaningful in GUI + random (structured simulate() is a no-op).
        # The callback runs inside simulate()'s step loop (this thread) — no MjData race.
        if not self.app.headless and self.arrangement == "random":
            self._setup_live_preview()
            self.mj_scene.simulate(on_step=self._live_preview_update)
        else:
            self.mj_scene.simulate()
        self.mj_scene.verify_parts_in_bin()
        scene_state = self.mj_scene.extract_scene_state()
        self.o3d_scene = self.mj_scene.mujoco_scene_to_o3d(scene_state)
        mesh_list = []
        for obj in self.o3d_scene.values():
            mesh = copy.deepcopy(obj.geom)
            tri_mesh = o3d_to_trimesh(mesh.transform(obj.T_gt))
            tri_mesh.process(validate=True)
            mesh_list.append(tri_mesh)
        final_mesh = trimesh_to_o3d(trimesh.boolean.union(mesh_list, engine="manifold", check_volume=False))
        add_to_render_scene("mesh_scene", final_mesh)
        
        self.worker_step += 1
        self.app.update_progress(self.worker_step/TOTAL_STEPS, f"Generating synthetic scene...")

        cam_pos = np.asarray([0.0, 0.0, self.mj_scene.camera_distance])  # above bin
        look_at = np.asarray([0.0, 0.0, 0.0])                           # bin floor centre
        T_cam = camera_view_matrix(cam_pos, look_at, up=np.array([0.0, 1.0, 0.0]))
        self.T_cam = T_cam   # kept for scene-state export in save_synthetic_targets

        self.app._reframe()

        render = scene_render(self.o3d_scene, T_cam, look_at, self.fov_deg, self.res_width, self.res_height, verbose=self.verbose)
        pts = render["points"]
        # rp(np.unique(render["geom_ids"]))
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

        label_masks = segment_point_cloud(  # {geom_id: (N,) bool} — one mask per instance, may overlap
            render, pts, pix_all,
            erosion_px=5.0,
            dilation_px=5.0,
            confusion_depth_sigma=0.015,   # ~15 mm — tune to your part height spread
            confusion_boundary_px=4,
            occlusion_loss_px=2,
            boundary_noise_px=10.0,
            seed=0,
            verbose=self.verbose
        )
        _update_pb("Filtering targets by point counts...")

        unique_id_list = list(label_masks.keys())
        rp(f"length of unique id list: {len(unique_id_list)} \n {unique_id_list}")
        valid_count = 0
        tf_by_id = {value.id: value.T_gt for value in self.o3d_scene.values()}
        bin_geom_id = self.o3d_scene["bin"].id  # set by scene_render; robust to any part count

        voxel_size =  0.001
        ref_xyz = np.asarray(self.app.down_pcd.points)
        min_overlap = 0.1
        bin_pcd = None
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

            if not (min(self.app.point_count_range)<=len(xyz_inst)<=max(self.app.point_count_range)):
                print(f"[skip] Instance {inst_id} point count out of threshold {self.app.point_count_range}: {len(xyz_inst)}")
                precheck_pass = False
                # continue
            if overlap < min_overlap:
                print(f"[skip] Instance {inst_id}: overlap={overlap:.2f} < {min_overlap}")
                precheck_pass = False
                # continue
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
            self.worker_step += 1
            self.app.update_progress(self.worker_step/TOTAL_STEPS, f"Compiling results...")
            self.show_segmented_scene()

    # ===============================
    # Live mesh preview during settling (GUI + random only)
    # ===============================
    @staticmethod
    def _pose_to_T(pos, quat):
        """Build a fresh 4x4 transform from a MuJoCo position + wxyz quaternion. Returns a new
        array (safe to hand to the GUI thread; does not alias data.xpos/xquat)."""
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
        T[:3, 3] = pos
        return T

    def _setup_live_preview(self):
        """Add one part mesh per body + the bin to the scene once, framed on the bin. Subsequent
        per-step transforms are pushed by _live_preview_update(). Parked (not-yet-released) bodies
        start hidden. Must run before mj_scene.simulate()."""
        import mujoco
        # Populate xpos/xquat for the initial spawn poses (pure kinematics; does not advance the
        # sim — simulate() recomputes everything from qpos/qvel via mj_step).
        mujoco.mj_forward(self.mj_scene.model, self.mj_scene.data)
        state = self.mj_scene.extract_scene_state()
        part_mesh_o3d = trimesh_to_o3d(self.mj_scene.part_mesh)
        bin_mesh = self.mj_scene.bin_mesh

        def add_all():
            self.app._clear_scene()
            for body_name, bd in state.items():
                pos = bd["position"]
                geom = o3d.geometry.TriangleMesh(part_mesh_o3d)   # one copy per body
                self.app.scene.scene.add_geometry(body_name, geom, self.app.default_material)
                parked = float(pos[2]) > PREVIEW_PARKED_Z
                self.app.scene.scene.show_geometry(body_name, not parked)
                if not parked:
                    self.app.scene.scene.set_geometry_transform(
                        body_name, self._pose_to_T(pos, bd["quaternion"]))
            self.app.scene.scene.add_geometry("bin", bin_mesh, self.app.default_material)
        self.app.main_thread(add_all)
        self.app._reframe()

    def _live_preview_update(self):
        """Called from inside simulate()'s step loop (worker thread). Snapshots body poses into
        fresh transforms and posts a single GUI update; parked bodies are hidden until released."""
        state = self.mj_scene.extract_scene_state()
        updates = {
            name: (self._pose_to_T(bd["position"], bd["quaternion"]),
                   float(bd["position"][2]) > PREVIEW_PARKED_Z)
            for name, bd in state.items()
        }

        def apply():
            for name, (T, parked) in updates.items():
                if not self.app.scene.scene.has_geometry(name):
                    continue
                self.app.scene.scene.show_geometry(name, not parked)
                if not parked:
                    self.app.scene.scene.set_geometry_transform(name, T)
            self.app.scene.force_redraw()
        self.app.main_thread(apply)

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
            if getattr(self, "mj_scene", None) is not None:
                state = self.mj_scene.export_scene_state()
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