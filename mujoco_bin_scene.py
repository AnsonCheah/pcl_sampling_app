import mujoco
import mujoco.viewer
import time
import numpy as np
import random
from dataclasses import dataclass
from utilities import random_quaternion
import open3d as o3d
from pathlib import Path
from rich import print as rp
from scipy.spatial.transform import Rotation as R
import copy
from utilities import o3d_to_trimesh, meshes_intersect, trimesh_to_o3d
import trimesh
from trimesh.collision import CollisionManager

@dataclass
class SceneObject:
    name: str
    mesh_path: str
    body_name: str

class MujocoBinScene:
    def __init__(self,
                 bin_mesh_path,
                 part_mesh_path,
                 n_parts=1,
                 settle_time=30.0,
                 render=True):
        self.timestep = 0.002
        self.lvel_threshold = 0.03
        self.avel_threshold = 0.5
        self.stable_duration = 0.5
        self._stable_count = 0
        self.stable_steps = int(self.stable_duration / self.timestep)
        print(f"stable_steps: {self.stable_steps}")

        self.bin_mesh_path = bin_mesh_path
        self.part_mesh_path = part_mesh_path
        self.n_parts = n_parts
        self.settle_time = settle_time
        self.scene_objects = []
        self.model = None
        self.data = None
        self.convex_dir = "convex_output"
        self.render_flag = render
        self.build()
        mujoco.mj_step(self.model, self.data)
        self.step_count = 1

    def _generate_mjcf(self):
        mesh_assets = []
        bodies = []
        original_mesh = trimesh.load(str(self.part_mesh_path))
        radius = original_mesh.bounding_sphere.primitive.radius
        valid_poses = []
        max_attempts = 200
        print("[INFO] Generating intersection-free initial poses...")
        bin_size_x = 0.76
        bin_size_y = 0.585
        bin_height = 0.3
        wall_thickness = 0.01
        floor_thickness = 0.01

        # Half sizes (MuJoCo uses half extents)
        hx = bin_size_x / 2
        hy = bin_size_y / 2
        hh = bin_height / 2
        wt = wall_thickness / 2
        ft = floor_thickness / 2


        collision_manager = CollisionManager()
        for i in range(self.n_parts):
            success = False
            candidate_trimesh = copy.deepcopy(original_mesh)
            for attempt in range(max_attempts):
                x = np.random.uniform(-hx + radius, hx - radius)
                y = np.random.uniform(-hy + radius, hy - radius)
                z = np.random.uniform(0.2, 0.2 + bin_height)
                rmat = R.random().as_matrix()
                quat = R.from_matrix(rmat).as_quat(scalar_first=True)
                T = np.eye(4)
                T[:3, 3] = [x, y, z]
                T[:3, :3] = rmat

                is_collision, _, _ = collision_manager.in_collision_single(candidate_trimesh, transform=T, return_names=True, return_data=True)
                if is_collision:
                    print(f"Intersection detected at attempt {attempt} for part {i}")
                    continue

                valid_poses.append((x, y, z, quat))
                collision_manager.add_object(f"part_{i}", candidate_trimesh, transform=T)
                success = True
                break


            if not success:
                print(f"[WARNING] Could not place part {i} without intersection")

        print(f"[INFO] Successfully placed {len(valid_poses)} parts")


        box_geom_configs = """
                mass="0.5"
                contype="1"
                conaffinity="1"
                condim="6"
                friction="1.5 0.005 0.0001"
                solref="0.002 1"
                solimp="0.9 0.95 0.001"
                rgba="0.6 0.6 0.6 0.2"""
        
        bodies.append(f"""
        <body name="bin" pos="0 0 0">
            <!-- Bottom -->
            <geom type="box"
                size="{hx} {hy} {ft}"
                pos="0 0 {-ft}"
                {box_geom_configs}"/>

            <!-- +X Wall -->
            <geom type="box"
                size="{wt} {hy} {hh}"
                pos="{hx - wt} 0 {hh - ft}"
                {box_geom_configs}"/>

            <!-- -X Wall -->
            <geom type="box"
                size="{wt} {hy} {hh}"
                pos="{-hx + wt} 0 {hh - ft}"
                {box_geom_configs}"/>

            <!-- +Y Wall -->
            <geom type="box"
                size="{hx} {wt} {hh}"
                pos="0 {hy - wt} {hh - ft}"
                {box_geom_configs}"/>

            <!-- -Y Wall -->
            <geom type="box"
                size="{hx} {wt} {hh}"
                pos="0 {-hy + wt} {hh - ft}"
                {box_geom_configs}"/>

        </body>
        """)

        convex_files = sorted(Path(self.convex_dir).glob("*.stl"))
        convex_mesh_names = []
        for idx, file in enumerate(convex_files):
            mesh_name = f"convex_mesh_{idx}"
            convex_mesh_names.append(mesh_name)
            mesh_assets.append(
                f'<mesh name="{mesh_name}" file="{file}"  />\n'
            )

        geom_block_template = "\n".join([f'<geom type="mesh" mesh="{name}"/>' for name in convex_mesh_names])

        for i, (x, y, z, quat) in enumerate(valid_poses):
            bodies.append(f"""
            <body name="part_{i}"
                pos="{x} {y} {z}"
                quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">
                <freejoint/>
                {geom_block_template}
            </body>
            """)
            self.scene_objects.append(SceneObject(f"part_{i}", self.part_mesh_path, f"part_{i}"))

        mjcf = f"""
        <mujoco>
            <option timestep="{self.timestep}" o_margin="0.001"/>
            <size   memory="1000M"/>
            <default>
                <geom 
                    condim="6"
                />
            </default>

            <asset>
                {''.join(mesh_assets)}
            </asset>

            <worldbody>
                <geom type="plane" size="1 1 0.1"/>
                {''.join(bodies)}
            </worldbody>

        </mujoco>
        """

        return mjcf

    def build(self):
        xml = self._generate_mjcf()
        with open("xml.xml","w") as file:
            file.write(xml)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

    def simulate(self, realtime=False):
        if self.model is None:
            raise RuntimeError("Call build() first.")

        steps = int(self.settle_time / self.model.opt.timestep)
        if self.render_flag:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # Optional: camera configuration
                viewer.cam.distance = 0.6
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -30
                viewer.cam.lookat[:] = [0, 0, 0]

                for _ in range(steps):

                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)

                    viewer.sync()
                    if realtime:
                        # Maintain real-time simulation speed
                        elapsed = time.time() - step_start
                        remaining = self.model.opt.timestep - elapsed
                        if remaining > 0:
                            time.sleep(remaining)
                        if self.is_settled():
                            break
        else:
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)
                if self.is_settled():
                    break

    def mujoco_scene_to_open3d(self, scene_dict, visualize=True):
        """
        Convert current MuJoCo scene state into Open3D scene.
        Parameters
        ----------
        model : mujoco.MjModel
        data  : mujoco.MjData
        scene_dict : dict
            {
                body_name: {
                        mesh_path: str
                        position: xyz
                        quaternion: wxyz
                    }
            }
            Snapshot of scene after settling
        visualize : bool
        Returns
        -------
        o3d_mesh_list : list of open3d.geometry.TriangleMesh
        """

        o3d_mesh_list = []
        for body_name, body_data in scene_dict.items():
            pos = body_data["position"]
            quat = body_data["quaternion"]
            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
            T[:3, 3] = pos

            mesh = o3d.io.read_triangle_mesh(body_data["mesh_path"])
            if not mesh.has_vertices():
                print(f"[WARNING] Mesh {body_data["mesh_path"]} failed to load")
                continue
            mesh.compute_vertex_normals()
            mesh.transform(T)
            o3d_mesh_list.append(mesh)

        return o3d_mesh_list

    def extract_scene_state(self):
        scene_dict = {}
        for obj in self.scene_objects:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                obj.body_name
            )
            pos = self.data.xpos[body_id]
            quat = self.data.xquat[body_id]  # w x y z
            scene_dict[obj.body_name] = {
                "mesh_path": obj.mesh_path,
                "position": pos.tolist(),
                "quaternion": quat.tolist()
            }

        return scene_dict

    def is_settled(self) -> bool:
        cvel = self.data.cvel[1:]  # skip worldbody, shape: (nbody-1, 6)
        ang_speeds = np.linalg.norm(cvel[:, :3], axis=1)
        lin_speeds = np.linalg.norm(cvel[:, 3:], axis=1)
        max_lin = lin_speeds.max() if len(lin_speeds) else 0.0
        max_ang = ang_speeds.max() if len(ang_speeds) else 0.0
        # print(f"max_lin: {max_lin}, max_ang: {max_ang}")
        all_slow = (max_lin < self.lvel_threshold) and (max_ang < self.avel_threshold)
        self._stable_count = self._stable_count + 1 if all_slow else 0
        return self._stable_count >= self.stable_steps
    
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

    # decomposed_convex_list = trimesh.decomposition.convex_decomposition(part_mesh)
    # print(f"trimesh decomposed to {len(decomposed_convex_list)} convex hulls")
    # decomposed_mesh_list = [trimesh.Trimesh(vertices=h["vertices"], faces=h["faces"], process=False) for h in decomposed_convex_list]
    # decomposed_open3d_mesh_list=[]
    # print(f"trimesh converted back to {len(decomposed_convex_list)} meshes")
    # for m in decomposed_mesh_list:
    #     o3d_mesh = trimesh_to_o3d(m)
    #     o3d_mesh.paint_uniform_color(np.random.uniform(0.0, 1.0, (3,1)).tolist())
    #     decomposed_open3d_mesh_list.append(o3d_mesh)
    # o3d.visualization.draw_geometries(decomposed_open3d_mesh_list, width=1080, height=720)

    rendering_flag = True
    init_open3d()
    scene = MujocoBinScene(
        bin_mesh_path=bin_mesh,
        part_mesh_path=part_mesh_path,
        n_parts=50,
        render=rendering_flag
    )
    scene.simulate(realtime=rendering_flag)
    scene_state = scene.extract_scene_state()

    collision_manager = CollisionManager()
    for part_name, status in scene_state.items():
        T = np.eye(4)
        T[:3, :3] = R.from_quat(status["quaternion"], scalar_first=True).as_matrix()
        T[:3, 3] = status["position"]
        collision_manager.add_object(part_name, part_mesh, transform=T)
    is_collision = collision_manager.in_collision_internal()
    if is_collision:
        print("Collision detected by trimesh!!")
        o3d_scene = scene.mujoco_scene_to_open3d(scene_state)
        o3d.visualization.draw_geometries(o3d_scene, width=1080, height=720)
