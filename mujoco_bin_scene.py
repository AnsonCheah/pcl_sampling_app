import mujoco
import mujoco.viewer
import time
import numpy as np
import random
from dataclasses import dataclass
import open3d as o3d
from pathlib import Path
from rich import print as rp
from scipy.spatial.transform import Rotation as R
import copy
from utilities import o3d_to_trimesh, trimesh_to_o3d
import trimesh
from trimesh.collision import CollisionManager

@dataclass
class SceneObject:
    name: str
    body_name: str

class MujocoBinScene:
    def __init__(self, part_mesh, part_convex_meshes, n_parts=1, settle_time=10.0, render=True):
        self.timestep = 0.002
        self.lvel_threshold = 0.03
        self.avel_threshold = 0.5
        self.stable_duration = 0.5
        self._stable_count = 0
        self.stable_steps = int(self.stable_duration / self.timestep)

        self.part_mesh = part_mesh
        self.part_convex_meshes = part_convex_meshes
        self.n_parts = n_parts
        self.settle_time = settle_time
        self.scene_objects = []
        self.model = None
        self.data = None
        self.render_flag = render
        
        self.spec = mujoco.MjSpec()
        self.spec.option.timestep = self.timestep
        self.spec.option.gravity = [0,0,-9.81]
        self.spec.option.o_margin = 0.001
        self.spec.option.iterations = 200
        self.spec.memory = 1000*1024*1024
        default = self.spec.default.geom
        default.condim = 6
        default.friction = [1.5, 0.005, 0.0001]
        default.solref= [0.002, 1]
        default.contype = 1
        default.conaffinity = 1

        self.world = self.spec.worldbody
        self.part_counter = 0

        self._load_convex_assets()
        self._build_bin()
        poses = self.generate_collision_free_poses()
        for pose in poses:
            self.add_part(pose)
        self.compile()
        if self.render_flag:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 1.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -30
            self.viewer.cam.lookat[:] = [0, 0, 0]

    def _build_bin(self):
        self.hx = 0.76 / 2
        self.hy = 0.585 / 2
        hh = 0.3 / 2
        wt = 0.01 / 2
        ft = 0.01 / 2

        bin_body = self.world.add_body()
        bin_body.name = "bin"
        bin_body.pos = [0,0,0]

        def add_box(size, pos):
            geom = bin_body.add_geom()
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = size
            geom.pos = pos
            geom.mass = 0.5
            geom.contype=1
            geom.conaffinity=1
            geom.rgba=[0.6,0.6,0.6,0.2]

        add_box([self.hx,self.hy,ft],[0,0,-ft])
        add_box([wt,self.hy,hh],[self.hx-wt,0,hh-ft])
        add_box([wt,self.hy,hh],[-self.hx+wt,0,hh-ft])
        add_box([self.hx,wt,hh],[0,self.hy-wt,hh-ft])
        add_box([self.hx,wt,hh],[0,-self.hy+wt,hh-ft])

    def _load_convex_assets(self):
        self.convex_mesh_names = []

        for i, convex_mesh in enumerate(self.part_convex_meshes):
            mesh_name = f"convex_mesh_{i}"
            mesh = self.spec.add_mesh()
            mesh.name = mesh_name
            mesh.uservert = convex_mesh.vertices.flatten().tolist()
            mesh.userface = convex_mesh.faces.flatten().tolist()
            self.convex_mesh_names.append(mesh_name)

    def add_part(self, pose:np.ndarray):
        body = self.world.add_body()
        body.name = f"part_{self.part_counter}"
        body.pos = pose[:3]
        body.quat = pose[3:]
        body.add_freejoint()

        for mesh_name in self.convex_mesh_names:
            geom = body.add_geom()
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = mesh_name
            geom.mass = 0.1

        self.scene_objects.append(SceneObject(body.name,body.name))
        self.part_counter += 1

    def mujoco_scene_to_open3d(self, scene_dict):
        """
        scene_dict : dict
            {
                body_name: {
                    position: xyz
                    quaternion: wxyz
                }
            }
        Returns
        -------
        o3d_mesh_list : list of open3d.geometry.TriangleMesh
        """

        o3d_mesh_list = []
        for _, body_data in scene_dict.items():
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
            mesh.transform(T)
            o3d_mesh_list.append(mesh)

        return o3d_mesh_list

    def extract_scene_state(self):
        scene_dict = {}
        for obj in self.scene_objects:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj.body_name)
            pos = self.data.xpos[body_id]
            quat = self.data.xquat[body_id]  # w x y z
            scene_dict[obj.body_name] = {
                "position": pos,
                "quaternion": quat
            }
        return scene_dict

    def is_settled(self) -> bool:
        cvel = self.data.cvel[1:]  # skip worldbody, shape: (nbody-1, 6)
        ang_speeds = np.linalg.norm(cvel[:, :3], axis=1)
        lin_speeds = np.linalg.norm(cvel[:, 3:], axis=1)
        max_lin = lin_speeds.max() if len(lin_speeds) else 0.0
        max_ang = ang_speeds.max() if len(ang_speeds) else 0.0
        all_slow = (max_lin < self.lvel_threshold) and (max_ang < self.avel_threshold)
        self._stable_count = self._stable_count + 1 if all_slow else 0
        return self._stable_count >= self.stable_steps
    
    def compile(self):
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

    def simulate(self, realtime=False):
        steps = int(self.settle_time / self.model.opt.timestep)
        input("enter")
        for i in range(steps):
            print(f"[t={self.data.time:.2f}s] step={i}/{steps}") if i%int(1.0 / self.model.opt.timestep)==0 else None
            step_start = time.time()
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            if self.is_settled():
                print(f"[t={self.data.time:.2f}s] Settled at step {i}")
                break
            if realtime:
                elapsed = time.time() - step_start
                remaining = max(0, self.model.opt.timestep - elapsed)
                time.sleep(remaining)
        self.viewer.close()

    def generate_collision_free_poses(self):
        collision_manager = CollisionManager()
        original_mesh = self.part_mesh
        radius = original_mesh.bounding_sphere.primitive.radius
        batch_size = min(10, int((2 * self.hx) // (2 * radius)) * int((2 * self.hy) // (2 * radius)))
        valid_poses = np.zeros((self.n_parts, 7))
        for i in range(self.n_parts):
            success = False
            candidate_trimesh = copy.deepcopy(original_mesh)

            for _ in range(200):
                layer = i // batch_size
                x = np.random.uniform(-self.hx + radius*2, self.hx - radius*2)
                y = np.random.uniform(-self.hy + radius*2, self.hy - radius*2)
                z = max(radius, 0.5) + layer * (2.5 * radius) + np.random.uniform(-radius, radius)
                pos = np.asarray([x, y, z])
                T = np.eye(4)
                T[:3, 3] = pos
                T[:3, :3] = R.random().as_matrix()
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
        return valid_poses

def standardize_mesh(mesh_path:Path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh_basename = mesh_path.stem
    bbox = mesh.get_axis_aligned_bounding_box()
    extent_max = bbox.get_extent().max()
    if 5.0 < extent_max < 5000.0:
        print(f"[INFO] Converting units mm → m:")
        mesh.scale(0.001, center=(0, 0, 0))
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())
    processed_mesh_path = Path(f"mj_stl/{mesh_basename}.stl")
    o3d.io.write_triangle_mesh(processed_mesh_path, mesh, write_ascii=False)
    return processed_mesh_path

def init_open3d():
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.destroy_window()

if __name__ == "__main__":
    bin_mesh = standardize_mesh(Path("bin.stl"))
    part_mesh_path = standardize_mesh(Path("25333MB000.stl"))
    part_mesh = trimesh.load(str(part_mesh_path))

    decomposed_convex_list = trimesh.decomposition.convex_decomposition(part_mesh)
    decomposed_mesh_list = [trimesh.Trimesh(vertices=h["vertices"], faces=h["faces"], process=False) for h in decomposed_convex_list]
    print(f"trimesh decomposed to {len(decomposed_convex_list)} convex hulls")

    rendering_flag = True
    init_open3d()
    scene = MujocoBinScene(part_mesh, decomposed_mesh_list, n_parts=200, render=rendering_flag)
    scene.simulate(realtime=rendering_flag)
    scene_state = scene.extract_scene_state()

    collision_manager = CollisionManager()
    for part_name, status in scene_state.items():
        T = np.eye(4)
        T[:3, :3] = R.from_quat(status["quaternion"], scalar_first=True).as_matrix()
        T[:3, 3] = status["position"]
        collision_manager.add_object(part_name, part_mesh, transform=T)
    is_collision = collision_manager.in_collision_internal()
    print("Collision detected by trimesh!!") if is_collision else None
    o3d_scene = scene.mujoco_scene_to_open3d(scene_state)
    o3d.visualization.draw_geometries(o3d_scene, width=1080, height=720)