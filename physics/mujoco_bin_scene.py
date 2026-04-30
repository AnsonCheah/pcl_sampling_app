import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import time
import numpy as np
from dataclasses import dataclass
import open3d as o3d
from rich import print as rp
from scipy.spatial.transform import Rotation as R
import copy
from geometry.geom_utils import o3d_to_trimesh, trimesh_to_o3d, camera_view_matrix, o3d_display, init_open3d, O3DSceneObject
from trimesh.collision import CollisionManager

@dataclass
class SceneObject:
    name: str
    body_name: str

class MujocoBinScene:
    def __init__(self, part_mesh, part_convex_meshes, n_parts=1, bin_dim=(0.76, 0.585, 0.25, 0.005), settle_time=10.0, render=True, arrangement: str = "random"):
        self.timestep = 0.002
        self.lvel_threshold = 0.03
        self.avel_threshold = 0.5
        self.stable_duration = 0.5
        self._stable_count = 0
        self.rendering_flag = True
        self.verbose = True
        self.stable_steps = int(self.stable_duration / self.timestep)
        self.camera_distance = 1.5
        self.bin_transform = np.eye(4)
        self.bin_transform[:3, :3] = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
        self.bin_transform[:3,  3] = np.asarray([0, 0, self.camera_distance])


        self.part_mesh = part_mesh
        self.part_convex_meshes = part_convex_meshes
        self.n_parts = n_parts
        self.settle_time = settle_time
        self.scene_objects = []
        self.model = None
        self.data = None
        self.render_flag = render
        self.viewer = None
        self.arrangement = arrangement

        self.spec = mujoco.MjSpec()
        self.spec.option.timestep = self.timestep
        self.spec.option.gravity = [0,0,9.81]
        self.spec.option.o_margin = 0.001
        self.spec.option.iterations = 200
        self.spec.memory = 3000*1024*1024
        default = self.spec.default.geom
        default.condim = 6
        default.friction = [1.5, 0.005, 0.0001]
        default.solref= [0.002, 1]
        default.contype = 1
        default.conaffinity = 1

        self.world = self.spec.worldbody

        self.bin_dim = bin_dim
        self.hx = self.bin_dim[0] / 2
        self.hy = self.bin_dim[1] / 2
        self.hh = self.bin_dim[2] / 2
        self._load_convex_assets()
        self.bin_mesh = self._build_bin()
        self.generate_scene()

    def _load_convex_assets(self):
        self.convex_mesh_names = []
        for i, convex_mesh in enumerate(self.part_convex_meshes):
            mesh_name = f"convex_mesh_{i}"
            mesh = self.spec.add_mesh()
            mesh.name = mesh_name
            mesh.uservert = np.asarray(convex_mesh.vertices).flatten().tolist()
            mesh.userface = np.asarray(convex_mesh.triangles).flatten().tolist()
            self.convex_mesh_names.append(mesh_name)

    def _build_bin(self):
        boxes = [
            ([self.hx, self.hy, self.bin_dim[3]], [0, 0, -self.bin_dim[3]]),                                   # floor
            ([self.bin_dim[3], self.hy, self.hh], [self.hx - self.bin_dim[3], 0, self.hh - self.bin_dim[3]]),     # +x wall
            ([self.bin_dim[3], self.hy, self.hh], [-self.hx + self.bin_dim[3], 0, self.hh - self.bin_dim[3]]),    # -x wall
            ([self.hx, self.bin_dim[3], self.hh], [0, self.hy - self.bin_dim[3], self.hh - self.bin_dim[3]]),     # +y wall
            ([self.hx, self.bin_dim[3], self.hh], [0, -self.hy + self.bin_dim[3], self.hh - self.bin_dim[3]]),    # -y wall
        ]


        # Extract rotation and translation for geom application
        # R_bin = self.bin_transform[:3, :3]   # (3,3)
        # t_bin = self.bin_transform[:3,  3]   # (3,)

        # Precompute the quaternion for the bin rotation (MuJoCo: [w, x, y, z])
        geom_quat = np.array(R.from_matrix(self.bin_transform[:3, :3]).as_quat(scalar_first=True))        # scipy: [x, y, z, w]

        # Build Open3D mesh with transform applied
        combined = o3d.geometry.TriangleMesh()
        for half_sizes, pos in boxes:
            box = o3d.geometry.TriangleMesh.create_box(
                width=half_sizes[0] * 2,
                height=half_sizes[1] * 2,
                depth=half_sizes[2] * 2,
            )
            box.translate(np.array(pos) - np.array(half_sizes))
            combined += box

        combined.merge_close_vertices(1e-6)
        combined.compute_vertex_normals()
        combined.transform(self.bin_transform)

        # Build MuJoCo geoms with transformed positions and orientations
        bin_body = self.world.add_body()
        bin_body.name = "bin"
        bin_body.pos = [0, 0, 0]

        for half_sizes, pos in boxes:
            # Transform geom center position into new frame
            bin_pos = np.eye(4)
            bin_pos[:3, 3] = pos
            transformed_pos = self.bin_transform @ np.array(bin_pos)

            geom = bin_body.add_geom()
            geom.type  = mujoco.mjtGeom.mjGEOM_BOX
            geom.size  = half_sizes
            geom.pos   = transformed_pos[:3, 3].tolist()
            geom.quat  = geom_quat.tolist()   # apply bin rotation to every geom
            geom.mass  = 0.5
            geom.rgba  = [0.6, 0.6, 0.6, 0.2]

        return combined
    
    def _find_stable_pose(self) -> np.ndarray:
        """Return 3x3 rotation matrix orienting part_mesh stably on the XY floor (analytical)."""
        import trimesh.poses
        transforms, probs = trimesh.poses.compute_stable_poses(self.part_mesh)
        return transforms[int(np.argmax(probs))][:3, :3]

    def _compute_structured_grid(self, R_stable):
        """
        Compute centered rectangular grid positions and a yaw-aligned rotation.

        Returns xs, ys (1-D arrays of grid coords), R_aligned (3x3), raw_pos_z (float).
        raw_pos_z places the part centre 10 mm above the bin floor (floor at z=0 raw).
        Yaw is auto-selected so the long axis fits the bin and grid slot count is maximised.
        """
        verts = np.asarray(self.part_mesh.vertices)
        sv = (R_stable @ verts.T).T
        mins, maxs = sv.min(axis=0), sv.max(axis=0)
        fp_x = maxs[0] - mins[0]
        fp_y = maxs[1] - mins[1]
        # Floor surface in MuJoCo world is at camera_distance (1.5 m).
        # After R_bin (180° Rx), the part's max world-z extent = -mins[2].
        # So part bottom in world = raw_pos_z + (-mins[2]) = camera_distance - gap.
        raw_pos_z = float(self.camera_distance + mins[2] - 0.01)  # 10 mm gap above floor

        bin_x = 2 * self.hx   # 0.76
        bin_y = 2 * self.hy   # 0.585
        gap = max(fp_x, fp_y) * 0.1

        R_yaw90 = R.from_euler("z", 90, degrees=True).as_matrix()
        apply_yaw90 = False

        if max(fp_x, fp_y) > bin_y:
            # Long axis must go along x; rotate if it is currently along y
            if fp_y > fp_x:
                apply_yaw90 = True
        else:
            # Both orientations fit — pick whichever packs more slots
            nx_a = max(1, int(bin_x / (fp_x + gap)))
            ny_a = max(1, int(bin_y / (fp_y + gap)))
            nx_b = max(1, int(bin_x / (fp_y + gap)))
            ny_b = max(1, int(bin_y / (fp_x + gap)))
            if nx_b * ny_b > nx_a * ny_a:
                apply_yaw90 = True

        if apply_yaw90:
            R_aligned = R_yaw90 @ R_stable
            fp_x, fp_y = fp_y, fp_x
        else:
            R_aligned = R_stable

        pitch_x = fp_x + gap
        pitch_y = fp_y + gap
        nx = max(1, int(bin_x / pitch_x))
        ny = max(1, int(bin_y / pitch_y))

        xs = np.linspace(-(nx - 1) * pitch_x / 2, (nx - 1) * pitch_x / 2, nx)
        ys = np.linspace(-(ny - 1) * pitch_y / 2, (ny - 1) * pitch_y / 2, ny)

        rp(f"[STRUCTURED] footprint {fp_x:.3f}x{fp_y:.3f} m  pitch {pitch_x:.3f}x{pitch_y:.3f} m  grid {nx}x{ny}={'(yaw-corrected)' if apply_yaw90 else ''}")
        return xs, ys, R_aligned, raw_pos_z

    def _spawn_bodies(self, valid_poses, static=False):
        """Add part bodies to the MjSpec. static=True omits the freejoint (structured mode)."""
        for pose in valid_poses:
            body = self.world.add_body()
            body.name = f"part_{self.part_counter}"
            body.pos = pose[:3]
            body.quat = pose[3:]
            if not static:
                body.add_freejoint()
            for mesh_name in self.convex_mesh_names:
                geom = body.add_geom()
                geom.type = mujoco.mjtGeom.mjGEOM_MESH
                geom.meshname = mesh_name
                geom.mass = 0.1
            self.scene_objects.append(SceneObject(body.name, body.name))
            self.part_counter += 1

    def _generate_random_scene(self):
        collision_manager = CollisionManager()
        radius = self.part_mesh.bounding_sphere.primitive.radius
        batch_size = min(10, int((2 * self.hx) // (2 * radius)) * int((2 * self.hy) // (2 * radius)))
        valid_poses = np.zeros((self.n_parts, 7))
        for i in range(self.n_parts):
            success = False
            candidate_trimesh = copy.deepcopy(self.part_mesh)
            for _ in range(200):
                layer = i // batch_size
                x = np.random.uniform(-self.hx + radius*2, self.hx - radius*2)
                y = np.random.uniform(-self.hy + radius*2, self.hy - radius*2)
                z = self.camera_distance - max(radius, 0.5) - layer * (2.5 * radius) - np.random.uniform(-radius, radius)
                pos = np.asarray([x, y, z])
                T = np.eye(4)
                T[:3, 3] = pos
                T[:3, :3] = R.random().as_matrix()
                T = self.bin_transform @ T
                quat = R.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
                is_collision, _, _ = collision_manager.in_collision_single(candidate_trimesh, transform=T, return_names=True, return_data=True)
                if is_collision:
                    continue
                valid_poses[i, :3] = pos
                valid_poses[i, 3:] = quat
                collision_manager.add_object(f"part_{i}", candidate_trimesh, transform=T)
                success = True
                print(f"Generated part {i+1}/{self.n_parts}")
                break
            if not success:
                print(f"[WARNING] Could not place part {i} without intersection")
        print(f"[INFO] Successfully placed {len(valid_poses)} parts")
        self._spawn_bodies(valid_poses)

    def _generate_structured_scene(self):
        R_stable = self._find_stable_pose()
        xs, ys, R_aligned, raw_pos_z = self._compute_structured_grid(R_stable)

        grid_slots = [(x, y) for x in xs for y in ys]
        self.n_parts = len(grid_slots)

        valid_poses = np.zeros((self.n_parts, 7))
        for i, (x, y) in enumerate(grid_slots):
            R_jitter = R.from_euler("xyz", np.random.uniform(-5, 5, 3), degrees=True).as_matrix()
            R_local = R_aligned @ R_jitter
            T_local = np.eye(4)
            T_local[:3, :3] = R_local
            T_local[:3, 3] = [x, y, raw_pos_z]
            T = self.bin_transform @ T_local
            quat = R.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
            valid_poses[i, :3] = [x, y, raw_pos_z]
            valid_poses[i, 3:] = quat
        # Static bodies — no freejoint; non-intersection guaranteed by grid pitch
        self._spawn_bodies(valid_poses, static=True)

    def generate_scene(self):
        self.part_counter = 0
        if self.arrangement == "structured":
            self._generate_structured_scene()
        else:
            self._generate_random_scene()

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        if self.arrangement == "structured":
            mujoco.mj_forward(self.model, self.data)  # populate xpos/xquat without dynamics

        if self.render_flag and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 1.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = 30
            self.viewer.cam.lookat[:] = [0, 0, 1.5]

    def is_settled(self) -> bool:
        cvel = self.data.cvel[1:]  # skip worldbody, shape: (nbody-1, 6)
        ang_speeds = np.linalg.norm(cvel[:, :3], axis=1)
        lin_speeds = np.linalg.norm(cvel[:, 3:], axis=1)
        max_lin = lin_speeds.max() if len(lin_speeds) else 0.0
        max_ang = ang_speeds.max() if len(ang_speeds) else 0.0
        all_slow = (max_lin < self.lvel_threshold) and (max_ang < self.avel_threshold)
        self._stable_count = self._stable_count + 1 if all_slow else 0
        return self._stable_count >= self.stable_steps

    def simulate(self, realtime=False):
        if self.arrangement == "structured":
            return  # poses are final; mj_forward already called in generate_scene
        steps = int(self.settle_time / self.model.opt.timestep)
        
        for i in range(steps):
            print(f"[t={self.data.time:.2f}s] step={i}/{steps}") if i%int(1.0 / self.model.opt.timestep)==0 else None
            step_start = time.time()
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None: self.viewer.sync()
            if self.is_settled():
                print(f"[t={self.data.time:.2f}s] Settled at step {i}")
                break
            if realtime:
                elapsed = time.time() - step_start
                remaining = max(0, self.model.opt.timestep - elapsed)
                time.sleep(remaining)
        if self.viewer is not None: self.viewer.close()

    def verify_parts_in_bin(self) -> dict:
        """
        After simulation, report how many parts remain inside the bin footprint.

        A part is considered escaped if its centre of mass is more than one
        bounding-sphere radius beyond the bin x/y wall extent, or more than
        one full bin-height below the bin opening (i.e. clearly in free space,
        not just stacked above the rim).

        Returns
        -------
        dict with keys:
            in_bin     : list of body names whose centres are inside the bin
            out_of_bin : list of body names whose centres are outside the bin
            n_in       : count inside
            n_out      : count outside
        """
        r_part  = self.part_mesh.bounding_sphere.primitive.radius
        bin_tx  = self.bin_transform[0, 3]
        bin_ty  = self.bin_transform[1, 3]
        bin_tz  = self.bin_transform[2, 3]

        # x/y: allow one part-radius beyond the physical wall before calling it escaped
        x_limit = self.hx + r_part
        y_limit = self.hy + r_part
        # z: parts stacked above the bin rim are fine (they extend toward camera at z=0).
        # Only flag a part if it is more than one bin-height beyond the opening —
        # i.e. clearly in free space, not just cresting the rim.
        z_escape = bin_tz - 2 * self.hh   # bin_tz(1.5) - hh(0.125)*2 = 1.25

        in_bin, out_of_bin = [], []

        for obj in self.scene_objects:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, obj.body_name
            )
            pos = self.data.xpos[body_id]
            escaped = (
                abs(pos[0] - bin_tx) > x_limit or
                abs(pos[1] - bin_ty) > y_limit or
                pos[2] < z_escape
            )
            (out_of_bin if escaped else in_bin).append(obj.body_name)

        n_in  = len(in_bin)
        n_out = len(out_of_bin)
        rp(f"[BIN CHECK] {n_in}/{n_in + n_out} parts inside bin, {n_out} escaped")
        if n_out > 0:
            rp(f"  Escaped: {out_of_bin[:10]}{'  …' if n_out > 10 else ''}")
        return {"in_bin": in_bin, "out_of_bin": out_of_bin, "n_in": n_in, "n_out": n_out}

    def extract_scene_state(self):
        scene_dict = {}
        for obj in self.scene_objects:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj.body_name)
            pos = self.data.xpos[body_id]
            quat = self.data.xquat[body_id]  # w x y z
            scene_dict[obj.body_name] = {
                "position": pos,
                "quaternion": quat,
            }
        return scene_dict

    def mujoco_scene_to_o3d(self, scene_dict):
        """
        scene_dict : dict
            {
                body_name: {
                    position: xyz
                    quaternion: wxyz
                }
            }
        -------
        o3d_mesh_list : list of open3d.geometry.TriangleMesh
        """
        o3d_mesh_dict = {}
        for key, body_data in scene_dict.items():
            pos = body_data["position"]
            quat = body_data["quaternion"]
            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
            T[:3, 3] = pos

            mesh = trimesh_to_o3d(self.part_mesh)
            if not mesh.has_vertices():
                print(f"[WARNING] Part Mesh failed to convert to o3d")
                continue
            mesh.compute_vertex_normals()
            o3d_mesh_dict[key] = O3DSceneObject(geom=mesh, T_gt=T)
        o3d_mesh_dict["bin"] = O3DSceneObject(geom=self.bin_mesh, T_gt=np.eye(4))
        return o3d_mesh_dict
    
    def _get_camera_lookat(self, x_multiple=1.5, z_multiple=5.0):
        """
        Compute center, eye, up for Open3D look_at
        such that the camera views the bin diagonally from above.
        --- Eye position ---
        Diagonal offset: step out along bin's local +x and -z (above),
        then transform to world frame.
        Tune diagonal_factor and height_factor to taste.
        """
        R_bin = self.bin_transform[:3, :3]
        t_bin = self.bin_transform[:3,  3]

        # --- Bin center in world frame ---
        # Bin interior center in local frame: floor at z=0 after flip,
        # so interior midpoint is at half the bin height along local +z
        bin_local_center = np.array([0.0, 0.0, self.hh * 0.5])
        center = R_bin @ bin_local_center + t_bin
        eye_local = np.array([
            self.hx  * 0,    # step out in +x
            self.hy  * x_multiple,    # step out in +y (true diagonal)
            self.hh  * z_multiple,      # step up in +z
        ])
        eye = R_bin @ eye_local + t_bin

        # --- Up vector ---
        # Local up is +z in the bin's own frame.
        # After Rx(180) this flips — derive it from R_bin to stay correct.
        local_up = np.array([0.0, 0.0, 1.0])
        up = R_bin @ local_up
        up = up / np.linalg.norm(up)       # normalise — look_at expects unit vector

        return center, eye, up

if __name__ == "__main__":
    from app_v2 import MeshSamplingApp
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
    from geometry.geom_utils import o3d_to_trimesh, trimesh_to_o3d, camera_view_matrix, compute_overlap, O3DSceneObject
    from enums import Stage
    import copy
    import open3d.visualization.rendering as rendering

    app = MeshSamplingApp(headless=True)

    app.stages[Stage.IMPORT_MESH]._run_worker()
    app._express_sampling_worker()
    part_mesh = o3d_to_trimesh(app.target_mesh)


    init_open3d()
    scene = MujocoBinScene(part_mesh, app.convex_meshes, n_parts=10, render=True)
    scene.simulate(realtime=scene.rendering_flag)
    scene_state = scene.extract_scene_state()

    collision_manager = CollisionManager()
    for part_name, status in scene_state.items():
        T = np.eye(4)
        T[:3, :3] = R.from_quat(status["quaternion"], scalar_first=True).as_matrix()
        T[:3, 3] = status["position"]
        collision_manager.add_object(part_name, part_mesh, transform=T)
    is_collision = collision_manager.in_collision_internal()
    print("Collision detected by trimesh!!") if is_collision else None
    o3d_scene = scene.mujoco_scene_to_o3d(scene_state)
    geom_list = []
    for obj in o3d_scene.values():
        mesh = copy.deepcopy(obj.geom)
        geom_list.append(mesh.transform(obj.T_gt))
    vis = o3d_display(geom_list)
    vis.run()
    vis.destroy_window()
    cam_pos = np.zeros(3)
    look_at = np.asarray([0,0,1.5])
    T_cam = camera_view_matrix(cam_pos, look_at)

    fov = 41.11
    W, H = 1920, 1200

    render = scene_render(o3d_scene, T_cam, look_at, fov, W, H, verbose=scene.verbose)
    pts = render["points"]
    nrm = render["normals"]
    geom_ids = render["geom_ids"]
    pix_all   = render["pixel_idx"]
    bin_pts = pts[geom_ids==0]
    bin_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bin_pts))
    bin_pcd = bin_pcd.voxel_down_sample(0.001)

    keep   = compute_dropout_mask(render, roughness=0.4,
                                 albedo_per_geom_id={2: 0.04},  # black rubber part
                                 density_cos_ref=0.7,           # oblique density thinning
                                 verbose=scene.verbose)
    render = add_projector_nonuniformity(render, verbose=scene.verbose)
    keep   = add_specular_patch_missing(render, keep, verbose=scene.verbose)
    render = add_image_space_effects(render, keep,
                                     smooth_sigma_px=0.5,
                                     sigma_fringe_corr=0.0001, 
                                     verbose=scene.verbose)
    pts, nrm = add_edge_artifacts(render, keep, verbose=scene.verbose)
    r = subset_render(render, keep, verbose=scene.verbose)
    mp, mn = add_multipath_outliers(r, verbose=scene.verbose)
    pp, pn = add_pepper_noise(r, verbose=scene.verbose)
    pts = np.vstack([pts, mp, pp])
    nrm = np.vstack([nrm, mn, pn])

    # Build per-point metadata for the full combined cloud.
    n_kept    = keep.sum()
    n_outlier = len(pts) - n_kept
    pix_all   = np.concatenate([render["pixel_idx"][keep], np.full(n_outlier, -1, np.int64)])
    cproj_all = np.concatenate([render["cos_proj"][keep], np.ones(n_outlier)])

    pts = add_scan_line_banding(pts, nrm, pix_all, render["res"], render["sensor_origin"], verbose=scene.verbose)
    pts = add_sensor_noise(pts, nrm, render["sensor_origin"], pixel_idx=pix_all, res=render["res"], cos_proj=cproj_all, verbose=scene.verbose)
    pts = add_surface_noise(pts, nrm, verbose=scene.verbose)
    scene_noisy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(0.001)
    # vis = o3d_display([scene_noisy])
    # vis.run()
    # vis.destroy_window()

    # ── Instance segmentation ──────────────────────────────────────────────────
    # render still holds the full canonical geom_id image (built from the
    # shadow-visible set before dropout), which is the correct substrate for
    # mask generation.  pix_all maps every point — canonical and injected —
    # to its image pixel (-1 for injected points, which receive label -1).
    label_masks = segment_point_cloud(  # {geom_id: (N,) bool} — one mask per instance, may overlap
        render, pts, pix_all,
        erosion_px=5.0,
        dilation_px=5.0,
        confusion_depth_sigma=0.015,   # ~15 mm — tune to your part height spread
        confusion_boundary_px=4,
        occlusion_loss_px=2,
        boundary_noise_px=10.0,
        seed=0,
        verbose=scene.verbose
    )
    print("segmented scene point virtually")

    unique_id_list = list(label_masks.keys())
    rp(f"length of unique id list: {len(unique_id_list)} \n {unique_id_list}")
    valid_count = 0
    tf_by_id = {value.id: value.T_gt for value in o3d_scene.values()}
    bin_geom_id = o3d_scene["bin"].id  # set by scene_render; robust to any part count

    voxel_size =  0.001
    ref_xyz = np.asarray(app.down_pcd.points)
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
            app.synthetic_scenes["bin_pcd"] = bin_pcd
            continue
        
        xyz_inst = np.asarray(inst_pcd_downsampled.points)
        # if not (min(app.point_count_range)<=len(xyz_inst)<=max(app.point_count_range)):
        #     print(f"Instance {inst_id} point count out of threshold {app.point_count_range}: {len(xyz_inst)}")
        #     continue
        
        inst_rmat = tf_by_id[inst_id][:3, :3]
        inst_trans = tf_by_id[inst_id][:3, 3]
        xyz_ref_in_scene = (inst_rmat @ ref_xyz.T + inst_trans[:, None]).T
        overlap = compute_overlap(xyz_ref_in_scene, xyz_inst, threshold=voxel_size * 2.5)
        # print(f"Instance {inst_id} overlap is {overlap}")
        # if overlap < min_overlap:
        #     print(f"  [skip] Instance {inst_id}: overlap={overlap:.2f} < {min_overlap}")
        #     continue

        # print(f"Instance {inst_id} point count within threshold {app.point_count_range}: {len(xyz_inst)}")
        valid_count += 1
        inst_name = f"synthetic_sample_{valid_count}"
        app.synthetic_targets[inst_name] = O3DSceneObject(
            geom=inst_pcd_downsampled, 
            ref_geom=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_ref_in_scene)),
            id=int(inst_id), 
            T_gt=tf_by_id[inst_id],
            xyz0=xyz_ref_in_scene,
            xyz1=xyz_inst,
            overlap=overlap
        )

    
    default_point_material = rendering.MaterialRecord()
    default_point_material.point_size = 1.5
    default_point_material.base_color = [1.0, 1.0, 1.0, 1.0]
    bin_pcd.paint_uniform_color([0.6, 0.6, 0.6])

    pcds = []
    for key in app.synthetic_targets:
        geom = app.synthetic_targets[key].geom
        geom.transform(scene.bin_transform)
        pcds.append(geom)
    vis = o3d_display(pcds, dynamic_color=True)
    bin_pcd.transform(scene.bin_transform)
    vis.add_geometry(bin_pcd)
    vis.run()
    vis.destroy_window()