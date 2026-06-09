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
import trimesh
from trimesh.collision import CollisionManager
from pathlib import Path


JITTER_DEG = [3,3,3]
HOPPER_INWARD_OFFSET = 0.02   # hopper walls inset this far inside the bin wall plane
HOPPER_TOP_Z = 100            # hopper walls extend to this z; generous upper bound

# Dynamic bin sizing (random arrangement only, when fill_rate is provided)
BIN_XY_SCALE_FACTOR  = 10.0   # bin side ≈ BIN_XY_SCALE_FACTOR × bounding-sphere radius
BIN_LAYER_MARGIN     = 1.5    # 50% headroom on stack height
PACKING_FACTOR       = 0.6    # random packing efficiency for MOBB volumes
PART_TO_BIN_OVERSIZE = 0.5    # part_diameter > this × max bin extent → use static max bin

# Batched-release settling (random arrangement only)
PARKING_Z             = 10.0  # m — z of parked (not-yet-released) bodies, well below HOPPER_TOP_Z
BATCH_DROP_OFFSET     = 0.03  # m — gap between current pile top and the next batch's spawn z
BATCH_MAX_SETTLE_TIME = 1.5   # s — cap per-batch settle time so wedged scenes can't hang
BATCH_SIZE_LAYERS     = 1     # number of "layers" (n_per_layer parts) per batch


@dataclass
class SceneObject:
    name: str
    body_name: str

class MujocoBinScene:
    def __init__(self, part_mesh, part_convex_meshes, n_parts: int = None, fill_rate: float = None, bin_dim=(0.76, 0.585, 0.25, 0.005), settle_time=10.0, render=True, arrangement: str = "random", stable_pose_R: np.ndarray = None):
        self.timestep = 0.002
        self.lvel_threshold = 0.03
        self.avel_threshold = 0.5
        self.stable_duration = 0.5
        self._stable_count = 0
        # self.rendering_flag = True
        self.verbose = True
        self.stable_steps = int(self.stable_duration / self.timestep)
        self.camera_distance = 1.5
        # Bin floor at world origin, camera hovering above at z=camera_distance.
        # Gravity -Z (natural). No coordinate flip needed.
        self.bin_transform = np.eye(4)

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
        self.stable_pose_R = stable_pose_R  # optional forced stable orientation for structured mode
        self._cached_valid_poses = None     # populated when fill_rate triggers dynamic sizing
        self._use_layer_drop = False        # set by _compute_dynamic_bin oversize branch
        self._batch_of_body  = []           # per-body batch index (populated in placement)
        self._geom_ids_of_body = []         # per-body list of geom IDs (filled after compile)
        self._adr_of_body     = []          # per-body (qpos_adr, qvel_adr) of the freejoint
        self._n_per_layer     = 1           # used for batch sizing in legacy n_parts mode

        if n_parts is None and fill_rate is None:
            raise ValueError("MujocoBinScene requires either n_parts or fill_rate")

        self.spec = mujoco.MjSpec()
        self.spec.option.timestep = self.timestep
        self.spec.option.gravity = [0, 0, -9.81]
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

        # ── Dynamic bin sizing: only when fill_rate is given for random mode ──
        if fill_rate is not None and arrangement == "random":
            if n_parts is not None:
                rp(f"[WARN] Both n_parts={n_parts} and fill_rate={fill_rate} provided; fill_rate wins")
            # Bootstrap self.hh with the max bin so _compute_valid_stable_poses
            # has a generous clearance; _compute_dynamic_bin then overrides bin_dim.
            self.hh = bin_dim[2] / 2
            self._cached_valid_poses = self._compute_valid_stable_poses()
            bin_dim, n_parts = self._compute_dynamic_bin(fill_rate, bin_dim, self._cached_valid_poses)

        self.bin_dim = bin_dim
        self.n_parts = n_parts
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

        # --- Hopper extension walls (simulation only, not in bin_mesh) ---
        # These invisible walls extend the bin walls up to camera_distance, acting
        # as a hopper that guides falling parts into the bin during random simulation.
        # They are disabled via contype/conaffinity after settling in simulate().
        wall_top = 2 * self.hh - self.bin_dim[3]       # z where bin walls end
        hopper_top = HOPPER_TOP_Z      # generous upper bound
        hopper_half_h = (hopper_top - wall_top) / 2
        hopper_center_z = wall_top + hopper_half_h
        t = self.bin_dim[3]                             # wall thickness alias
        off = HOPPER_INWARD_OFFSET
        hopper_defs = [
            ("hopper_px", [t, self.hy, hopper_half_h], [ self.hx - t - off,  0,             hopper_center_z]),
            ("hopper_mx", [t, self.hy, hopper_half_h], [-self.hx + t + off,  0,             hopper_center_z]),
            ("hopper_py", [self.hx, t, hopper_half_h], [ 0,            self.hy - t - off,   hopper_center_z]),
            ("hopper_my", [self.hx, t, hopper_half_h], [ 0,           -self.hy + t + off,   hopper_center_z]),
        ]
        self._hopper_geom_names = []
        for name, half_sizes, pos in hopper_defs:
            hgeom = bin_body.add_geom()
            hgeom.name  = name
            hgeom.type  = mujoco.mjtGeom.mjGEOM_BOX
            hgeom.size  = half_sizes
            hgeom.pos   = pos
            hgeom.quat  = geom_quat.tolist()
            hgeom.mass  = 0.0
            hgeom.rgba  = [0.0, 0.0, 0.0, 0.0]  # invisible
            self._hopper_geom_names.append(name)

        # Infinite safety floor — catches parts that escape the bin during physics.
        # Placed one camera_distance below the bin floor so it never interferes with
        # normal in-bin contact, but stops runaway parts from falling to infinity.
        sf = bin_body.add_geom()
        sf.name = "safety_floor"
        sf.type = mujoco.mjtGeom.mjGEOM_PLANE
        sf.pos  = [0, 0, -self.camera_distance]
        sf.size = [0, 0, 0.1]   # infinite extent; 0.1 is grid spacing (visual only)
        sf.mass = 0.0
        sf.rgba = [0.5, 0.5, 0.5, 0.0]  # invisible

        return combined
    
    @staticmethod
    def get_stable_poses(
        part_mesh,
        min_face_area_frac: float = 0.01,
        angular_tol_deg: float = 15.0,
    ) -> list:
        """
        Return [(R_3x3, probability), ...] for all stable resting poses,
        enumerated from the convex hull faces and deduplicated by resting face.
        Probability is proportional to the face area (larger face = more stable).

        Filtering:
          min_face_area_frac — skip faces with < this fraction of total hull area (sliver filter).
          angular_tol_deg    — cluster poses whose resting "up" direction in part frame
                               differs by < angular_tol_deg; keep the highest-probability
                               representative.  Collapses near-parallel faces on faceted
                               parts that would otherwise produce near-identical scenes.

        Results are sorted by descending probability.
        """
        hull = part_mesh.convex_hull
        face_areas   = hull.area_faces
        face_normals = hull.face_normals
        total_area   = float(face_areas.sum())
        cos_thresh   = float(np.cos(np.radians(angular_tol_deg)))

        z_down = np.array([0., 0., -1.])
        candidates = []

        for normal, area in zip(face_normals, face_areas):
            prob = float(area / total_area)
            if prob < min_face_area_frac:
                continue

            n = np.array(normal, dtype=float)
            n /= np.linalg.norm(n)
            cos_a = float(np.clip(np.dot(n, z_down), -1., 1.))

            if cos_a > 1.0 - 1e-6:
                R_stable = np.eye(3)
            elif cos_a < -1.0 + 1e-6:
                R_stable = R.from_euler('x', 180, degrees=True).as_matrix()
            else:
                axis = np.cross(n, z_down)
                axis /= np.linalg.norm(axis)
                R_stable = R.from_rotvec(axis * np.arccos(cos_a)).as_matrix()

            candidates.append((R_stable, prob))

        if not candidates:
            return [(np.eye(3), 1.0)]

        # Cluster by resting-face direction in part frame (R.T @ world-up).
        # Process highest-probability first; merge subsequent poses that are within
        # angular_tol_deg of an already-accepted representative.
        up_world = np.array([0., 0., 1.])
        deduped: list = []
        for R_s, prob in sorted(candidates, key=lambda x: -x[1]):
            up_part = R_s.T @ up_world
            up_part /= np.linalg.norm(up_part)
            is_dup = any(
                np.dot(up_part,
                       (R_u.T @ up_world) / np.linalg.norm(R_u.T @ up_world)
                       ) >= cos_thresh
                for R_u, _ in deduped
            )
            if not is_dup:
                deduped.append((R_s, prob))

        deduped.sort(key=lambda x: -x[1])
        return deduped

    def _find_stable_pose(self) -> np.ndarray:
        """Return 3x3 rotation matrix for the most probable stable pose."""
        return self.get_stable_poses(self.part_mesh)[0][0]

    def _compute_valid_stable_poses(self) -> list:
        """
        Filter stable poses to those whose z-extent fits within 0.9 × bin_height.
        Returns list of (R_3x3, prob, lz) — private 3-tuple.
        Falls back to the minimum-lz pose if all exceed the threshold.
        """
        bin_height = 2 * self.hh
        threshold  = 0.9 * bin_height
        verts      = np.asarray(self.part_mesh.vertices)

        valid = []
        for R_stable, prob in self.get_stable_poses(self.part_mesh):
            rotated = (R_stable @ verts.T).T
            lz = float(rotated[:, 2].max() - rotated[:, 2].min())
            if lz <= threshold:
                valid.append((R_stable, prob, lz))

        if not valid:
            all_with_lz = sorted(
                [(R_s, p, float((R_s @ verts.T).T[:, 2].max() - (R_s @ verts.T).T[:, 2].min()))
                 for R_s, p in self.get_stable_poses(self.part_mesh)],
                key=lambda x: x[2],
            )
            print(f"[WARNING] No stable pose fits in bin height {bin_height:.3f} m. "
                  f"Using minimum-lz pose ({all_with_lz[0][2]:.3f} m).")
            valid = [all_with_lz[0]]

        return valid

    def _compute_max_tilted_height(self, valid_poses: list, clearance: float = None) -> tuple:
        """
        Compute the worst-case height a part can achieve in any constrained orientation.
        Used for layer spacing in random spawning to ensure parts don't overlap.

        Returns (max_h, max_footprint_area) across all valid stable poses at their
        respective theta_max. `clearance` defaults to 0.9 × current bin height; pass
        an explicit value when computing layer stats before self.hh is finalised
        (e.g. during dynamic bin sizing).
        """
        max_h = 0.0
        max_footprint = 0.0
        verts   = np.asarray(self.part_mesh.vertices)
        if clearance is None:
            clearance = 0.9 * 2 * self.hh

        for R_stable, _, lz in valid_poses:
            rotated = (R_stable @ verts.T).T
            lx = float(rotated[:, 0].max() - rotated[:, 0].min())
            ly = float(rotated[:, 1].max() - rotated[:, 1].min())

            if max(lx, ly, lz) <= clearance:
                h = max(lx, ly, lz)
            else:
                diag      = np.sqrt(lx ** 2 + ly ** 2)
                theta_max = np.arcsin(np.clip((clearance - lz) / diag, 0.0, 1.0))
                h = lz + max(lx, ly) * np.sin(theta_max)

            max_h = max(max_h, float(h))
            max_footprint = max(max_footprint, lx * ly)

        return max_h, max_footprint

    def _compute_dynamic_bin(self, fill_rate: float, max_bin_dim: tuple, valid_poses: list) -> tuple:
        """
        Size the bin to the part. Returns ((bin_w, bin_l, bin_h, wall_t), n_parts).

        Volume basis: the part's MOBB (minimum oriented bounding box). Mesh volume
        under-counts a part's settled footprint (non-convex parts and elongated parts
        leave gaps between neighbours that the mesh doesn't see). MOBB is a much better
        proxy for the box each part actually claims in a pile.

        Bin shape policy: bin_h is sized ONCE from the fill=1.0 pile geometry (capped
        at the max bin) and reused for ALL fill rates — so the 5 fill_rate scenes for a
        given part share the same bin, with only n_parts varying. fill_rate is then
        interpreted as fraction of the physical-max capacity for that bin (fill=1.0 =
        random-packed bin). Visual MOBB-volume ratio ≈ fill_rate × PACKING_FACTOR.

        Oversized parts (2r > PART_TO_BIN_OVERSIZE × max XY extent) fall back to
        the full max bin so very large parts behave like the legacy fixed-bin path.
        """
        r_sphere = float(self.part_mesh.bounding_sphere.primitive.radius)
        mobb_vol = max(float(self.part_mesh.bounding_box_oriented.volume), 1e-9)

        if 2 * r_sphere > PART_TO_BIN_OVERSIZE * max(max_bin_dim[0], max_bin_dim[1]):
            n_at_max = int(np.floor(max_bin_dim[0] * max_bin_dim[1] * max_bin_dim[2]
                                    * PACKING_FACTOR / mobb_vol))
            n_parts  = max(1, int(round(fill_rate * max(n_at_max, 1))))
            self._use_layer_drop = True
            rp(f"[INFO] Oversize part (2r={2*r_sphere:.3f} m > "
               f"{PART_TO_BIN_OVERSIZE}*{max(max_bin_dim[0], max_bin_dim[1]):.3f} m); "
               f"using max bin + layer-drop. n_parts={n_parts}")
            return tuple(max_bin_dim), n_parts

        bin_w = float(min(BIN_XY_SCALE_FACTOR * r_sphere, max_bin_dim[0]))
        bin_l = float(min(BIN_XY_SCALE_FACTOR * r_sphere, max_bin_dim[1]))

        h_layer, footprint = self._compute_max_tilted_height(valid_poses, clearance=max_bin_dim[2])
        footprint = max(footprint, 1e-9)

        # Size bin_h once for the fill=1.0 pile. Stays constant across fill rates.
        n_per_layer    = max(1, int(np.floor(bin_w * bin_l * PACKING_FACTOR / footprint)))
        self._n_per_layer = n_per_layer
        n_max_vol      = max(1, int(np.floor(PACKING_FACTOR * bin_w * bin_l * max_bin_dim[2] / mobb_vol)))
        n_layers_full  = max(1, int(np.ceil(n_max_vol / n_per_layer)))
        bin_h_target   = max(2 * h_layer, n_layers_full * h_layer * BIN_LAYER_MARGIN)
        bin_h          = min(max_bin_dim[2], bin_h_target)

        if bin_h_target > max_bin_dim[2]:
            rp(f"[WARN] bin_h clipped: raw={bin_h_target:.3f} m -> {bin_h:.3f} m. "
               f"Hopper extension walls will contain parts during phase-1 settling.")

        # n_parts scales with fill_rate against the FINAL bin (visual fill ≈ fill × PACKING).
        bin_vol = bin_w * bin_l * bin_h
        n_parts = max(1, int(round(fill_rate * PACKING_FACTOR * bin_vol / mobb_vol)))

        visual_fill = n_parts * mobb_vol / bin_vol
        rp(f"[INFO] Dynamic bin ({bin_w:.3f}, {bin_l:.3f}, {bin_h:.3f}) m  "
           f"n_parts={n_parts}  fill={fill_rate:.0%}  visual_mobb={visual_fill:.0%}  "
           f"n_per_layer={n_per_layer}  mobb={mobb_vol:.2e}")
        return (bin_w, bin_l, bin_h, max_bin_dim[3]), n_parts

    def _sample_constrained_rotation(self, valid_poses: list) -> np.ndarray:
        """
        Sample a random rotation constrained so the part z-extent stays within
        0.9 × bin_height.  Decomposed as: unrestricted yaw around world-Z, then
        a tilt of at most theta_max away from world-Z (cone sampling).

        theta_max derivation (conservative):
            lz + diag·sin(θ) ≤ 0.9·bin_height
            θ_max = arcsin(clip((clearance − lz) / diag, 0, 1))
        where diag = sqrt(lx² + ly²) is the AABB footprint diagonal.
        """
        probs = np.array([p for _, p, _ in valid_poses], dtype=float)
        probs /= probs.sum()
        idx = np.random.choice(len(valid_poses), p=probs)
        R_stable, _, lz = valid_poses[idx]

        verts   = np.asarray(self.part_mesh.vertices)
        rotated = (R_stable @ verts.T).T
        lx = float(rotated[:, 0].max() - rotated[:, 0].min())
        ly = float(rotated[:, 1].max() - rotated[:, 1].min())

        clearance = 0.9 * 2 * self.hh

        if max(lx, ly, lz) <= clearance:
            theta_max = np.pi / 2          # any orientation fits — full hemisphere
        else:
            diag      = np.sqrt(lx ** 2 + ly ** 2)
            theta_max = np.arcsin(np.clip((clearance - lz) / diag, 0.0, 1.0))

        R_yaw  = R.from_euler('z', np.random.uniform(0.0, 2 * np.pi)).as_matrix()
        phi    = np.random.uniform(0.0, 2 * np.pi)
        theta  = np.random.uniform(0.0, theta_max)
        R_tilt = R.from_rotvec(np.array([np.cos(phi), np.sin(phi), 0.0]) * theta).as_matrix()

        return R_tilt @ R_yaw @ R_stable

    def _compute_structured_grid(self, R_stable):
        """
        Compute centered rectangular grid positions and a yaw-aligned rotation.

        Returns xs, ys (1-D arrays of grid coords), R_aligned (3x3), raw_pos_z (float).
        raw_pos_z places the part centre just above the bin floor at z=0, with enough
        clearance to absorb a +-5 deg orientation jitter without floor intersection.
        Yaw is auto-selected so the long axis fits the bin and grid slot count is maximised.
        Grid bounds account for wall thickness and jitter-induced footprint expansion so
        the last row/column cannot clip the bin walls.
        """
        verts = np.asarray(self.part_mesh.vertices)
        sv = (R_stable @ verts.T).T
        mins, maxs = sv.min(axis=0), sv.max(axis=0)
        fp_x = maxs[0] - mins[0]
        fp_y = maxs[1] - mins[1]

        # Worst-case vertex displacement from compound "xyz" Euler jitter of ±5° per axis.
        # Effective rotation angle ≈ √3·5° = 8.66° (RMS of three independent axes).
        # Using 10° adds a small safety buffer above that bound.
        r_max = float(np.sqrt((sv ** 2).sum(axis=1)).max())
        jitter_margin = r_max * np.sin(np.radians(max(JITTER_DEG)))
        raw_pos_z = float(-mins[2] + 0.01 + jitter_margin)

        # Inner usable extent: subtract wall thickness from both sides.
        t = self.bin_dim[3]
        inner_hx = self.hx - 2 * t
        inner_hy = self.hy - 2 * t

        # For each orientation the maximum valid part-centre coordinate is:
        #   inner_h? - half_footprint - jitter_margin
        # (the jitter can push the part edge outward by ~jitter_margin in X/Y)
        gap = max(fp_x, fp_y) * 0.1

        def _slot_count(fx, fy):
            px = fx + gap
            py = fy + gap
            xmc = inner_hx - fx / 2 - jitter_margin
            ymc = inner_hy - fy / 2 - jitter_margin
            if xmc <= 0 or ymc <= 0:
                return 0
            return (int(2 * xmc / px) + 1) * (int(2 * ymc / py) + 1)

        R_yaw90 = R.from_euler("z", 90, degrees=True).as_matrix()
        apply_yaw90 = False

        if max(fp_x, fp_y) > 2 * inner_hy:
            # Long axis must go along x; rotate if it is currently along y
            if fp_y > fp_x:
                apply_yaw90 = True
        else:
            # Both orientations fit — pick whichever packs more slots
            if _slot_count(fp_y, fp_x) > _slot_count(fp_x, fp_y):
                apply_yaw90 = True

        if apply_yaw90:
            R_aligned = R_yaw90 @ R_stable
            fp_x, fp_y = fp_y, fp_x
        else:
            R_aligned = R_stable

        pitch_x = fp_x + gap
        pitch_y = fp_y + gap

        x_max_center = max(0.0, inner_hx - fp_x / 2 - jitter_margin)
        y_max_center = max(0.0, inner_hy - fp_y / 2 - jitter_margin)

        nx = int(2 * x_max_center / pitch_x) + 1
        ny = int(2 * y_max_center / pitch_y) + 1

        xs = np.linspace(-(nx - 1) * pitch_x / 2, (nx - 1) * pitch_x / 2, nx)
        ys = np.linspace(-(ny - 1) * pitch_y / 2, (ny - 1) * pitch_y / 2, ny)

        # For asymmetric meshes the AABB centre (where load_part anchors the mesh)
        # does not coincide with the footprint centroid after R_aligned is applied.
        # Shift every body position so the footprint midpoint lands on the grid point.
        sv_aligned = (R_aligned @ verts.T).T
        cx = float((sv_aligned[:, 0].min() + sv_aligned[:, 0].max()) / 2)
        cy = float((sv_aligned[:, 1].min() + sv_aligned[:, 1].max()) / 2)
        xs = xs - cx
        ys = ys - cy

        rp(f"[STRUCTURED] footprint {fp_x:.3f}x{fp_y:.3f} m  pitch {pitch_x:.3f}x{pitch_y:.3f} m  grid {nx}x{ny}{'  (yaw-rotated)' if apply_yaw90 else ''}")
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

    def _build_bin_collision_trimesh(self):
        """Trimesh of bin floor/walls + hopper extensions for spawn-time collision checks."""
        t = self.bin_dim[3]
        wall_top  = 2 * self.hh - t
        hopper_half_h   = (HOPPER_TOP_Z - wall_top) / 2
        hopper_center_z =  wall_top + hopper_half_h
        box_defs = [
            # floor
            ([self.hx, self.hy, t],        [0,              0,             -t]),
            # bin walls
            ([t, self.hy, self.hh],        [ self.hx - t,   0,              self.hh - t]),
            ([t, self.hy, self.hh],        [-self.hx + t,   0,              self.hh - t]),
            ([self.hx, t, self.hh],        [ 0,             self.hy - t,    self.hh - t]),
            ([self.hx, t, self.hh],        [ 0,            -self.hy + t,    self.hh - t]),
            # hopper extension walls (inset by HOPPER_INWARD_OFFSET to match MuJoCo geoms)
            ([t, self.hy, hopper_half_h],  [ self.hx - t - HOPPER_INWARD_OFFSET,   0,              hopper_center_z]),
            ([t, self.hy, hopper_half_h],  [-self.hx + t + HOPPER_INWARD_OFFSET,   0,              hopper_center_z]),
            ([self.hx, t, hopper_half_h],  [ 0,             self.hy - t - HOPPER_INWARD_OFFSET,    hopper_center_z]),
            ([self.hx, t, hopper_half_h],  [ 0,            -self.hy + t + HOPPER_INWARD_OFFSET,    hopper_center_z]),
        ]
        meshes = []
        for half_sizes, pos in box_defs:
            box = trimesh.creation.box(extents=(np.array(half_sizes) * 2).tolist())
            box.apply_translation(pos)
            meshes.append(box)
        return trimesh.util.concatenate(meshes)

    def _generate_random_scene(self):
        bin_ct = self._build_bin_collision_trimesh()
        collision_manager = CollisionManager()
        collision_manager.add_object("bin", bin_ct)
        radius = self.part_mesh.bounding_sphere.primitive.radius
        valid_poses_for_rotation = self._cached_valid_poses if self._cached_valid_poses is not None else self._compute_valid_stable_poses()
        h_layer, footprint = self._compute_max_tilted_height(valid_poses_for_rotation)
        footprint = max(footprint, 1e-9)

        # In legacy n_parts mode, _compute_dynamic_bin never ran so _n_per_layer is still 1.
        # Compute it here from current bin dimensions + footprint.
        if self._n_per_layer <= 1:
            self._n_per_layer = max(1, int(np.floor(2 * self.hx * 2 * self.hy * PACKING_FACTOR / footprint)))

        batch_size = max(1, self._n_per_layer * BATCH_SIZE_LAYERS)
        n_batches  = int(np.ceil(self.n_parts / batch_size))

        valid_poses = np.zeros((self.n_parts, 7))
        self._batch_of_body = [i // batch_size for i in range(self.n_parts)]

        # Batch 0 spawns near the bin floor via collision-aware placement; later batches
        # are parked at PARKING_Z (contype=0/conaffinity=0 is set after compile) and will
        # be teleported into drop position by release_batch().
        active_z_lo = max(h_layer, 0.05)
        active_z_hi = active_z_lo + h_layer * 0.5
        for i in range(self.n_parts):
            placed_active = False
            if self._batch_of_body[i] == 0:
                candidate_trimesh = copy.deepcopy(self.part_mesh)
                for _ in range(200):
                    x = np.random.uniform(-self.hx + radius*2, self.hx - radius*2)
                    y = np.random.uniform(-self.hy + radius*2, self.hy - radius*2)
                    z = np.random.uniform(active_z_lo, active_z_hi)
                    pos = np.asarray([x, y, z])
                    T = np.eye(4)
                    T[:3, 3] = pos
                    T[:3, :3] = self._sample_constrained_rotation(valid_poses_for_rotation)
                    T = self.bin_transform @ T
                    quat = R.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
                    is_collision, _, _ = collision_manager.in_collision_single(candidate_trimesh, transform=T, return_names=True, return_data=True)
                    if is_collision:
                        continue
                    valid_poses[i, :3] = pos
                    valid_poses[i, 3:] = quat
                    collision_manager.add_object(f"part_{i}", candidate_trimesh, transform=T)
                    placed_active = True
                    break
                if not placed_active:
                    # Could not place in active zone — demote to last parked batch;
                    # release_batch() will teleport it back into the bin.
                    self._batch_of_body[i] = max(1, n_batches - 1)
                    print(f"[WARNING] Could not place part {i} in batch 0; demoted to batch {self._batch_of_body[i]}")

            if not placed_active:
                # Park (random XY for visualization, batch-dependent z spread)
                x = np.random.uniform(-self.hx + radius*2, self.hx - radius*2)
                y = np.random.uniform(-self.hy + radius*2, self.hy - radius*2)
                z = PARKING_Z + self._batch_of_body[i] * 0.2
                R_mat = self._sample_constrained_rotation(valid_poses_for_rotation)
                quat = R.from_matrix(R_mat).as_quat(scalar_first=True)
                valid_poses[i, :3] = [x, y, z]
                valid_poses[i, 3:] = quat
        print(f"[INFO] {self.n_parts} parts in {n_batches} batches (batch_size={batch_size})")
        self._spawn_bodies(valid_poses)

    def _generate_structured_scene(self):
        R_stable = self.stable_pose_R if self.stable_pose_R is not None else self._find_stable_pose()
        xs, ys, R_aligned, raw_pos_z = self._compute_structured_grid(R_stable)

        grid_slots = [(x, y) for x in xs for y in ys]
        self.n_parts = len(grid_slots)

        valid_poses = np.zeros((self.n_parts, 7))
        for i, (x, y) in enumerate(grid_slots):
            R_jitter = R.from_euler("xyz", np.random.uniform(*JITTER_DEG), degrees=True).as_matrix()
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

    def _generate_oversize_layer_scene(self):
        """
        Hybrid structured/random layout for oversize parts (triggered by
        PART_TO_BIN_OVERSIZE in _compute_dynamic_bin). Place parts in a stable-pose
        grid above the bin floor — stacking layers if n_parts > per-layer grid
        capacity — with random yaw and ±JITTER_DEG tilt per part. Physics drop is
        small (just above raw_pos_z) so settling is fast and avoids the high-velocity
        interlocking seen with the random columnar drop.
        All parts spawn as a single active batch (no parking) — oversize implies low
        n_parts, so the batched-release machinery is unnecessary here.
        """
        valid_poses_for_rotation = (self._cached_valid_poses
                                    if self._cached_valid_poses is not None
                                    else self._compute_valid_stable_poses())
        R_stable = valid_poses_for_rotation[0][0]
        xs, ys, R_aligned, raw_pos_z = self._compute_structured_grid(R_stable)
        h_layer, _ = self._compute_max_tilted_height(valid_poses_for_rotation)

        grid_slots       = [(x, y) for x in xs for y in ys]
        n_per_layer_grid = max(1, len(grid_slots))
        n_layers         = int(np.ceil(self.n_parts / n_per_layer_grid))

        valid_poses          = np.zeros((self.n_parts, 7))
        self._batch_of_body  = [0] * self.n_parts   # single-shot drop

        print(f"[INFO] Oversize layer drop: n_parts={self.n_parts}, grid={n_per_layer_grid}/layer, n_layers={n_layers}")

        i = 0
        for layer in range(n_layers):
            for (x, y) in grid_slots:
                if i >= self.n_parts:
                    break
                yaw_R    = R.from_euler('z', np.random.uniform(0, 2 * np.pi)).as_matrix()
                jitter_R = R.from_euler("xyz", np.random.uniform(*JITTER_DEG), degrees=True).as_matrix()
                R_local  = R_aligned @ yaw_R @ jitter_R
                z = raw_pos_z + layer * (h_layer * BIN_LAYER_MARGIN)
                T_local = np.eye(4)
                T_local[:3, :3] = R_local
                T_local[:3, 3]  = [x, y, z]
                T = self.bin_transform @ T_local
                quat = R.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
                valid_poses[i, :3] = [x, y, z]
                valid_poses[i, 3:] = quat
                i += 1
            if i >= self.n_parts:
                break

        self._spawn_bodies(valid_poses)

    def _cache_body_topology(self):
        """
        Cache per-part body ID, geom IDs, and (qpos_adr, qvel_adr) of the freejoint.
        Also disables collisions on parked (batch > 0) bodies so they free-fall in
        isolation until released by release_batch().
        Must be called after self.spec.compile() and self.data = MjData(...).
        """
        self._body_ids         = []
        self._geom_ids_of_body = []
        self._adr_of_body      = []
        for obj in self.scene_objects:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj.body_name)
            if bid < 0:
                raise RuntimeError(f"Body {obj.body_name} not found in compiled model")
            self._body_ids.append(bid)
            g_start = int(self.model.body_geomadr[bid])
            g_count = int(self.model.body_geomnum[bid])
            self._geom_ids_of_body.append(list(range(g_start, g_start + g_count)))
            jnt = int(self.model.body_jntadr[bid])
            qadr = int(self.model.jnt_qposadr[jnt])
            vadr = int(self.model.jnt_dofadr[jnt])
            self._adr_of_body.append((qadr, vadr))

        if not self._batch_of_body:
            return
        for i, batch in enumerate(self._batch_of_body):
            if batch > 0:
                for gid in self._geom_ids_of_body[i]:
                    self.model.geom_contype[gid] = 0
                    self.model.geom_conaffinity[gid] = 0

    def release_batch(self, k: int, valid_poses_for_rotation):
        """
        Enable collisions for every part assigned to batch k and teleport each one to
        a drop position above the current pile top. Resets qvel to zero so the
        teleport itself does not impart velocity.
        """
        batch_indices = [i for i, b in enumerate(self._batch_of_body) if b == k]
        if not batch_indices:
            return

        released_so_far = [self._body_ids[i] for i, b in enumerate(self._batch_of_body) if b < k]
        if released_so_far:
            pile_top = float(self.data.xpos[released_so_far, 2].max())
        else:
            pile_top = 2 * self.bin_dim[3]  # ~ bin floor + wall thickness

        h_layer, _ = self._compute_max_tilted_height(valid_poses_for_rotation)
        drop_z = pile_top + BATCH_DROP_OFFSET + h_layer * 0.5

        radius = self.part_mesh.bounding_sphere.primitive.radius
        for i in batch_indices:
            for gid in self._geom_ids_of_body[i]:
                self.model.geom_contype[gid] = 1
                self.model.geom_conaffinity[gid] = 1
            x = np.random.uniform(-self.hx + radius*2, self.hx - radius*2)
            y = np.random.uniform(-self.hy + radius*2, self.hy - radius*2)
            z = drop_z + np.random.uniform(0, h_layer * 0.3)
            R_mat = self._sample_constrained_rotation(valid_poses_for_rotation)
            quat = R.from_matrix(R_mat).as_quat(scalar_first=True)
            qadr, vadr = self._adr_of_body[i]
            self.data.qpos[qadr:qadr+3] = [x, y, z]
            self.data.qpos[qadr+3:qadr+7] = quat   # [w, x, y, z]
            self.data.qvel[vadr:vadr+6] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def generate_scene(self):
        self.part_counter = 0
        if self.arrangement == "structured":
            self._generate_structured_scene()
        elif self._use_layer_drop:
            self._generate_oversize_layer_scene()
        else:
            self._generate_random_scene()

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        if self.arrangement == "structured":
            mujoco.mj_forward(self.model, self.data)  # populate xpos/xquat without dynamics
            return

        self._cache_body_topology()

        if self.render_flag and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = self.camera_distance
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -30
            self.viewer.cam.lookat[:] = [0, 0, self.hh * 0.5]  # centre of bin interior

    def is_settled(self, body_id_subset=None) -> bool:
        """
        body_id_subset: optional iterable of body IDs to consider. When set, only those
        bodies' velocities count toward the settle check (used by batched release so
        parked bodies and not-yet-released batches are ignored).
        """
        if body_id_subset is not None:
            ids = list(body_id_subset)
            if not ids:
                self._stable_count = self._stable_count + 1
                return self._stable_count >= self.stable_steps
            cvel = self.data.cvel[ids]
        else:
            cvel = self.data.cvel[1:]  # skip worldbody, shape: (nbody-1, 6)
        ang_speeds = np.linalg.norm(cvel[:, :3], axis=1)
        lin_speeds = np.linalg.norm(cvel[:, 3:], axis=1)
        max_lin = lin_speeds.max() if len(lin_speeds) else 0.0
        max_ang = ang_speeds.max() if len(ang_speeds) else 0.0
        all_slow = (max_lin < self.lvel_threshold) and (max_ang < self.avel_threshold)
        self._stable_count = self._stable_count + 1 if all_slow else 0
        return self._stable_count >= self.stable_steps

    def simulate(self):
        if self.arrangement == "structured":
            print("[INFO] Structured arrangement: skipping simulation and settling")
            return  # poses are final; mj_forward already called in generate_scene
        # input("Press Enter to start simulation...")

        n_batches = (max(self._batch_of_body) + 1) if self._batch_of_body else 1
        valid_poses_for_rotation = (self._cached_valid_poses
                                    if self._cached_valid_poses is not None
                                    else self._compute_valid_stable_poses())

        # Snapshot parking qpos so we can hold parked bodies in place every step.
        # Without this, gravity still pulls them down (contype=0 prevents collisions,
        # not gravity) and they free-fall through the bin region — visually confusing.
        parking_qpos = {}
        for i, b in enumerate(self._batch_of_body):
            if b > 0:
                qadr, _ = self._adr_of_body[i]
                parking_qpos[i] = self.data.qpos[qadr:qadr+7].copy()

        def freeze_parked(parked_ids):
            for i in parked_ids:
                qadr, vadr = self._adr_of_body[i]
                self.data.qpos[qadr:qadr+7] = parking_qpos[i]
                self.data.qvel[vadr:vadr+6] = 0.0

        def run_settle(max_steps: int, label: str, body_subset=None, parked_ids=()):
            self._stable_count = 0
            tick = max(1, int(1.0 / self.model.opt.timestep))
            for i in range(max_steps):
                if i % tick == 0:
                    print(f"[t={self.data.time:.2f}s] {label} step={i}/{max_steps}")
                step_start = time.time()
                mujoco.mj_step(self.model, self.data)
                if parked_ids:
                    freeze_parked(parked_ids)
                if self.viewer is not None: self.viewer.sync()
                if self.is_settled(body_id_subset=body_subset):
                    print(f"[t={self.data.time:.2f}s] {label} settled at step {i}")
                    return
                if self.render_flag:
                    elapsed = time.time() - step_start
                    remaining = max(0, self.model.opt.timestep - elapsed)
                    time.sleep(remaining)

        batch_settle_steps = int(BATCH_MAX_SETTLE_TIME / self.model.opt.timestep)

        # Batch 0 is already at low z with collisions enabled — settle it first.
        parked_now   = [i for i, b in enumerate(self._batch_of_body) if b > 0]
        released_ids = [self._body_ids[i] for i, b in enumerate(self._batch_of_body) if b == 0]
        run_settle(batch_settle_steps, "batch-0", body_subset=released_ids, parked_ids=parked_now)

        for k in range(1, n_batches):
            self.release_batch(k, valid_poses_for_rotation)
            # Drop the just-released bodies from parking and from parking_qpos.
            for i in [idx for idx, b in enumerate(self._batch_of_body) if b == k]:
                parking_qpos.pop(i, None)
            parked_now   = [i for i, b in enumerate(self._batch_of_body) if b > k]
            released_ids.extend(self._body_ids[i] for i, b in enumerate(self._batch_of_body) if b == k)
            run_settle(batch_settle_steps, f"batch-{k}", body_subset=released_ids, parked_ids=parked_now)

        # Final phase-1 settle over all released bodies (hopper still active)
        final_phase1_steps = int(self.settle_time / self.model.opt.timestep)
        run_settle(final_phase1_steps, "phase-1-final", body_subset=None)

        # Disable hopper extensions so parts leaning on them can fall flat.
        for name in self._hopper_geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.model.geom_contype[gid] = 0
                self.model.geom_conaffinity[gid] = 0

        # Phase 2: short re-settle after hopper walls removed
        settle2_steps = int(2.0 / self.model.opt.timestep)
        run_settle(settle2_steps, "phase-2", body_subset=None)

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
        # z: bin floor at z=0, walls to z=2*hh=0.25 m; parts may stack above the rim.
        # Flag escape only when a part goes more than one bin-height BELOW the floor
        # (e.g. fell through due to tunneling) — bin_tz(0) - 2*hh(0.125) = -0.25.
        z_escape = bin_tz - 2 * self.hh

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
    
    def _get_camera_lookat(self, y_multiple=1.5, z_multiple=5.0):
        """
        Compute (center, eye, up) for Open3D look_at.
        Camera is above the bin looking down at a diagonal.
        Bin floor at world origin; +Z is up.
        """
        center = np.array([0.0, 0.0, self.hh * 0.5])  # centre of bin interior
        eye = np.array([
            0.0,
            self.hy * y_multiple,       # step out in +y
            self.hh * z_multiple,       # step up in +z (above bin)
        ])
        up = np.array([0.0, 0.0, 1.0])  # world Z is up
        return center, eye, up

# -- Mesh loading --------------------------------------------------------------

def load_part(mesh_path=None, verbose: bool = True):
    """Load STL, scale mm->m, centre, convex-decompose. No GUI or app required."""
    if mesh_path is None:
        mesh_path = input("Enter path to STL file: ").strip().strip('"').strip("'")

    path = Path(mesh_path)
    if verbose:
        rp(f"[bold]Loading {path.name}...[/bold]")

    mesh_o3d = o3d.io.read_triangle_mesh(str(path))
    if mesh_o3d.is_empty():
        raise ValueError(f"Could not load mesh from {path}")

    extent_max = mesh_o3d.get_axis_aligned_bounding_box().get_extent().max()
    if 5.0 < extent_max < 5000.0:
        if verbose:
            rp("  Converting mm -> m")
        mesh_o3d.scale(0.001, center=(0, 0, 0))

    mesh_o3d.compute_vertex_normals()
    mesh_o3d.translate(-mesh_o3d.get_center())

    part_mesh = o3d_to_trimesh(mesh_o3d)
    if verbose:
        rp(f"  vertices: {len(part_mesh.vertices):,}  faces: {len(part_mesh.faces):,}")
        rp(f"  bounding sphere r = {part_mesh.bounding_sphere.primitive.radius:.4f} m")

    if verbose:
        rp("  Running convex decomposition...")
    decomposed = trimesh.decomposition.convex_decomposition(part_mesh)
    convex_meshes = [
        o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(h["vertices"]),
            triangles=o3d.utility.Vector3iVector(h["faces"]),
        )
        for h in decomposed
    ]
    if verbose:
        rp(f"  Convex pieces: {len(convex_meshes)}")

    return part_mesh, convex_meshes

def _display_scene(scene, scene_state):
    o3d_scene = scene.mujoco_scene_to_o3d(scene_state)
    geom_list = []
    for obj in o3d_scene.values():
        mesh = copy.deepcopy(obj.geom)
        geom_list.append(mesh.transform(obj.T_gt))
    vis = o3d_display(geom_list, dynamic_color=True)
    vis.run()
    vis.destroy_window()

def _build_primitive(name: str):
    """Return (part_mesh, convex_meshes) for a reproducible trimesh primitive."""
    if name == "sphere_small":
        part_mesh = trimesh.creation.icosphere(radius=0.005, subdivisions=2)
    elif name == "sphere_large":
        part_mesh = trimesh.creation.icosphere(radius=0.05, subdivisions=2)
    elif name == "rod":
        part_mesh = trimesh.creation.cylinder(radius=0.005, height=0.20, sections=24)
    elif name == "plate":
        part_mesh = trimesh.creation.box(extents=(0.15, 0.10, 0.005))
    elif name == "cube":
        part_mesh = trimesh.creation.box(extents=(0.30, 0.30, 0.30))
    else:
        raise ValueError(f"unknown primitive '{name}'")
    convex = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(part_mesh.vertices)),
        triangles=o3d.utility.Vector3iVector(np.asarray(part_mesh.faces)),
    )
    rp(f"[PRIMITIVE {name}] verts={len(part_mesh.vertices)}  "
       f"bounding_sphere_r={part_mesh.bounding_sphere.primitive.radius:.4f} m  "
       f"volume={part_mesh.volume:.6e} m^3")
    return part_mesh, [convex]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bin scene generation tests")
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Path to STL mesh file (skips file dialog)",
    )
    src.add_argument(
        "--primitive",
        choices=["sphere_small", "sphere_large", "rod", "plate", "cube"],
        default=None,
        help="Use a built-in trimesh primitive instead of loading an STL",
    )
    parser.add_argument(
        "--arrangement",
        choices=["random", "structured", "both"],
        default="both",
        help="Which arrangement to test (default: both)",
    )
    # Random-mode sizing: --fill_rate uses dynamic bin (default); --n_parts uses static max bin.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fill_rate",
        type=float,
        default=None,
        help="Fill rate (0-1) for dynamic bin sizing in random mode. Default if --n_parts not set: 0.6",
    )
    mode.add_argument(
        "--n_parts",
        type=int,
        default=None,
        help="Number of parts for random mode with the static max bin (legacy path)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Open Open3D viewer after each test",
    )
    args = parser.parse_args()
    if args.fill_rate is None and args.n_parts is None:
        args.fill_rate = 1   # default to dynamic bin sizing in random mode

    init_open3d()
    if args.primitive is not None:
        part_mesh, convex_meshes = _build_primitive(args.primitive)
    else:
        part_mesh, convex_meshes = load_part(mesh_path=args.mesh)

    # Enumerate all stable poses once; structured tests iterate over them.
    stable_poses = MujocoBinScene.get_stable_poses(part_mesh)
    rp(f"\nFound {len(stable_poses)} stable pose(s) for this part:")
    for i, (_, p) in enumerate(stable_poses):
        rp(f"  pose {i}: probability = {p:.4f}")

    if args.arrangement in ("random", "both"):
        use_fill = args.fill_rate is not None
        scene = MujocoBinScene(
            part_mesh, convex_meshes,
            n_parts   = None             if use_fill else args.n_parts,
            fill_rate = args.fill_rate   if use_fill else None,
            render    = args.display,
            arrangement = "random",
        )
        rp(f"[RANDOM] bin_dim={scene.bin_dim}  n_parts={scene.n_parts}")
        scene.simulate()
        scene_state = scene.extract_scene_state()
        if args.display:
            rp("  Opening viewer...")
            _display_scene(scene, scene_state)

    if args.arrangement in ("structured", "both"):
        for i, (R_stable, prob) in enumerate(stable_poses):
            scene = MujocoBinScene(
                part_mesh, convex_meshes,
                n_parts=1,           # ignored - grid capacity overrides
                render=args.display,
                arrangement="structured",
                stable_pose_R=R_stable,
            )
            scene.simulate()         # should be a no-op
            scene_state = scene.extract_scene_state()

            if args.display:
                rp("  Opening viewer...")
                _display_scene(scene, scene_state)

    # passed = []
    # failed = []

    # def run(name, fn, *a, **kw):
    #     try:
    #         fn(*a, **kw)
    #         passed.append(name)
    #     except Exception as e:
    #         print(f"  FAIL - {name}: {e}")
    #         failed.append(name)

    # if args.arrangement in ("random", "both"):
    #     run("random", test_random, part_mesh, convex_meshes, args.n_parts, args.display)

    # if args.arrangement in ("structured", "both"):
    #     for i, (R_stable, prob) in enumerate(stable_poses):
    #         run(f"structured_pose_{i}", test_structured,
    #             part_mesh, convex_meshes, R_stable, i, prob, args.display)

    # rp(f"\n{'-'*40}")
    # rp(f"Results: {len(passed)} passed, {len(failed)} failed")
    # if failed:
    #     rp(f"[red]Failed: {failed}[/red]")
    #     sys.exit(1)
    # else:
    #     rp("[green bold]All tests passed.[/green bold]")


    # from app_v2 import MeshSamplingApp
    # from sensor.scene_render import (
    #     scene_render,
    #     compute_dropout_mask,
    #     add_projector_nonuniformity,
    #     add_specular_patch_missing,
    #     add_edge_artifacts,
    #     add_multipath_outliers, add_pepper_noise, subset_render,
    #     add_sensor_noise,
    #     add_surface_noise,
    #     add_image_space_effects,
    #     add_scan_line_banding
    # )
    # from sensor.segment_instances import segment_point_cloud
    # from geometry.geom_utils import o3d_to_trimesh, trimesh_to_o3d, camera_view_matrix, compute_overlap, O3DSceneObject
    # from enums import Stage
    # import copy
    # import open3d.visualization.rendering as rendering

    # app = MeshSamplingApp(headless=True)

    # app.stages[Stage.IMPORT_MESH]._run_worker()
    # app._express_sampling_worker()
    # part_mesh = o3d_to_trimesh(app.target_mesh)


    # init_open3d()
    # scene = MujocoBinScene(part_mesh, app.convex_meshes, n_parts=10, render=True)
    # scene.simulate(realtime=scene.rendering_flag)
    # scene_state = scene.extract_scene_state()

    # collision_manager = CollisionManager()
    # for part_name, status in scene_state.items():
    #     T = np.eye(4)
    #     T[:3, :3] = R.from_quat(status["quaternion"], scalar_first=True).as_matrix()
    #     T[:3, 3] = status["position"]
    #     collision_manager.add_object(part_name, part_mesh, transform=T)
    # is_collision = collision_manager.in_collision_internal()
    # print("Collision detected by trimesh!!") if is_collision else None
    # o3d_scene = scene.mujoco_scene_to_o3d(scene_state)
    # geom_list = []
    # for obj in o3d_scene.values():
    #     mesh = copy.deepcopy(obj.geom)
    #     geom_list.append(mesh.transform(obj.T_gt))
    # vis = o3d_display(geom_list)
    # vis.run()
    # vis.destroy_window()
    # cam_pos = np.asarray([0.0, 0.0, scene.camera_distance])  # above bin
    # look_at = np.asarray([0.0, 0.0, 0.0])                   # bin floor centre
    # T_cam = camera_view_matrix(cam_pos, look_at, up=np.array([0.0, 1.0, 0.0]))

    # fov = 41.11
    # W, H = 1920, 1200

    # render = scene_render(o3d_scene, T_cam, look_at, fov, W, H, verbose=scene.verbose)
    # pts = render["points"]
    # nrm = render["normals"]
    # geom_ids = render["geom_ids"]
    # pix_all   = render["pixel_idx"]
    # bin_pts = pts[geom_ids==0]
    # bin_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bin_pts))
    # bin_pcd = bin_pcd.voxel_down_sample(0.001)

    # keep   = compute_dropout_mask(render, roughness=0.4,
    #                              albedo_per_geom_id={2: 0.04},  # black rubber part
    #                              density_cos_ref=0.7,           # oblique density thinning
    #                              verbose=scene.verbose)
    # render = add_projector_nonuniformity(render, verbose=scene.verbose)
    # keep   = add_specular_patch_missing(render, keep, verbose=scene.verbose)
    # render = add_image_space_effects(render, keep,
    #                                  smooth_sigma_px=0.5,
    #                                  sigma_fringe_corr=0.0001, 
    #                                  verbose=scene.verbose)
    # pts, nrm = add_edge_artifacts(render, keep, verbose=scene.verbose)
    # r = subset_render(render, keep, verbose=scene.verbose)
    # mp, mn = add_multipath_outliers(r, verbose=scene.verbose)
    # pp, pn = add_pepper_noise(r, verbose=scene.verbose)
    # pts = np.vstack([pts, mp, pp])
    # nrm = np.vstack([nrm, mn, pn])

    # # Build per-point metadata for the full combined cloud.
    # n_kept    = keep.sum()
    # n_outlier = len(pts) - n_kept
    # pix_all   = np.concatenate([render["pixel_idx"][keep], np.full(n_outlier, -1, np.int64)])
    # cproj_all = np.concatenate([render["cos_proj"][keep], np.ones(n_outlier)])

    # pts = add_scan_line_banding(pts, nrm, pix_all, render["res"], render["sensor_origin"], verbose=scene.verbose)
    # pts = add_sensor_noise(pts, nrm, render["sensor_origin"], pixel_idx=pix_all, res=render["res"], cos_proj=cproj_all, verbose=scene.verbose)
    # pts = add_surface_noise(pts, nrm, verbose=scene.verbose)
    # scene_noisy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(0.001)
    # # vis = o3d_display([scene_noisy])
    # # vis.run()
    # # vis.destroy_window()

    # # ── Instance segmentation ──────────────────────────────────────────────────
    # # render still holds the full canonical geom_id image (built from the
    # # shadow-visible set before dropout), which is the correct substrate for
    # # mask generation.  pix_all maps every point — canonical and injected —
    # # to its image pixel (-1 for injected points, which receive label -1).
    # label_masks = segment_point_cloud(  # {geom_id: (N,) bool} — one mask per instance, may overlap
    #     render, pts, pix_all,
    #     erosion_px=5.0,
    #     dilation_px=5.0,
    #     confusion_depth_sigma=0.015,   # ~15 mm — tune to your part height spread
    #     confusion_boundary_px=4,
    #     occlusion_loss_px=2,
    #     boundary_noise_px=10.0,
    #     seed=0,
    #     verbose=scene.verbose
    # )
    # print("segmented scene point virtually")

    # unique_id_list = list(label_masks.keys())
    # rp(f"length of unique id list: {len(unique_id_list)} \n {unique_id_list}")
    # valid_count = 0
    # tf_by_id = {value.id: value.T_gt for value in o3d_scene.values()}
    # bin_geom_id = o3d_scene["bin"].id  # set by scene_render; robust to any part count

    # voxel_size =  0.001
    # ref_xyz = np.asarray(app.down_pcd.points)
    # min_overlap = 0.1
    # bin_pcd = None
    # for _, inst_id in enumerate(unique_id_list):
    #     inst_pts = pts[label_masks[inst_id]]
    #     inst_nrm = nrm[label_masks[inst_id]]
    #     inst_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inst_pts))
    #     inst_pcd.normals = o3d.utility.Vector3dVector(inst_nrm)
    #     inst_pcd_downsampled = inst_pcd.voxel_down_sample(voxel_size)

    #     if inst_id == bin_geom_id:
    #         bin_pcd = copy.deepcopy(inst_pcd_downsampled)
    #         app.synthetic_scenes["bin_pcd"] = bin_pcd
    #         continue
        
    #     xyz_inst = np.asarray(inst_pcd_downsampled.points)
    #     # if not (min(app.point_count_range)<=len(xyz_inst)<=max(app.point_count_range)):
    #     #     print(f"Instance {inst_id} point count out of threshold {app.point_count_range}: {len(xyz_inst)}")
    #     #     continue
        
    #     inst_rmat = tf_by_id[inst_id][:3, :3]
    #     inst_trans = tf_by_id[inst_id][:3, 3]
    #     xyz_ref_in_scene = (inst_rmat @ ref_xyz.T + inst_trans[:, None]).T
    #     overlap = compute_overlap(xyz_ref_in_scene, xyz_inst, threshold=voxel_size * 2.5)
    #     # print(f"Instance {inst_id} overlap is {overlap}")
    #     # if overlap < min_overlap:
    #     #     print(f"  [skip] Instance {inst_id}: overlap={overlap:.2f} < {min_overlap}")
    #     #     continue

    #     # print(f"Instance {inst_id} point count within threshold {app.point_count_range}: {len(xyz_inst)}")
    #     valid_count += 1
    #     inst_name = f"synthetic_sample_{valid_count}"
    #     app.synthetic_targets[inst_name] = O3DSceneObject(
    #         geom=inst_pcd_downsampled, 
    #         ref_geom=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_ref_in_scene)),
    #         id=int(inst_id), 
    #         T_gt=tf_by_id[inst_id],
    #         xyz0=xyz_ref_in_scene,
    #         xyz1=xyz_inst,
    #         overlap=overlap
    #     )

    
    # default_point_material = rendering.MaterialRecord()
    # default_point_material.point_size = 1.5
    # default_point_material.base_color = [1.0, 1.0, 1.0, 1.0]
    # bin_pcd.paint_uniform_color([0.6, 0.6, 0.6])

    # pcds = []
    # for key in app.synthetic_targets:
    #     geom = app.synthetic_targets[key].geom
    #     geom.transform(scene.bin_transform)
    #     pcds.append(geom)
    # vis = o3d_display(pcds, dynamic_color=True)
    # bin_pcd.transform(scene.bin_transform)
    # vis.add_geometry(bin_pcd)
    # vis.run()
    # vis.destroy_window()