"""Does the baked GT put the exported reference cloud where the simulator actually put the part?

This is the invariant the Optuna tuning stage rests on. `MM_Optimizer` scores every MechVision
pose against `read_gt_pose_from_ply(sample_i.ply)` and matches it with
`output/reference_pcd/<part>/<part>_surface/<part>_surface.ply`. If the GT is expressed in a
different frame from the reference cloud, every trial is scored against a fiction: coverage and
pose error come out wrong, no error is raised, and the study happily optimises toward it.

`reference_frames_agree` (geometry/geom_utils.py) already guards the *other* half of this — that
the bundle and each scene's `reference_cloud.ply` are the same cloud in the same frame. It never
touches `T_gt`, so the hop that actually bakes the GT was uncovered.

That hop is `SceneStage.worker`. MuJoCo requires a body-origin-centred mesh, so the stage centres
a local copy on the mesh AABB centre and afterwards remaps the physics poses back into the
original mesh frame::

    T_gt_exported = T_gt_physics @ [[I, -mesh_center], [0, 1]]

Neither the PCA frame nor the ambiguity frame puts the mesh AABB centre on the origin -- the
ambiguity frame deliberately does not (see `pcd_geocenter`) -- so `mesh_center` is essentially
always non-zero and this remap always fires. These tests pin it down in both frames.

The check is deliberately tolerance-free where it can be: the reference cloud's distance to the
part surface is a rigid invariant, so measuring it in the model frame and again in world against
the *simulator's own* geometry must give the same numbers. Nothing here compares noisy rendered
points; the question is only whether the mesh the physics settled and the cloud the tuner matches
against live in one frame.

Fast subset:  "$PCL_PY" -m pytest stages/tests/test_gt_frame_agreement.py -q -m "not slow"
"""

import copy

import numpy as np
import open3d as o3d
import pytest
import trimesh
from scipy.spatial.transform import Rotation as R

import stages.scene_stage as scene_stage_module
from enums import Stage
from geometry.ambiguity import AmbiguityAxis, AmbiguityProfile
from geometry.geom_utils import O3DSceneObject, trimesh_to_o3d

# Distance comparisons go through Open3D's float32 raycasting scene, so ~1e-8 m of numerical
# noise on a 0.1 m part is expected. Any real frame disagreement is millimetres or more.
SURFACE_ATOL_M = 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def l_mesh_factory():
    """An L-shaped part: two boxes, so the surface-area-weighted cloud mean is nowhere near
    the mesh AABB centre.

    A plain box is useless here. Its cloud mean and AABB centre coincide, so a PCA recentre
    leaves `mesh_center` at ~0, `needs_centering` goes False, and the remap under test never
    runs -- the test would pass without executing the code it exists to cover. It is also
    PCA-degenerate, so the frame axes would be arbitrary.
    """
    def _make():
        base = trimesh.creation.box(extents=(0.10, 0.03, 0.02))
        upright = trimesh.creation.box(extents=(0.03, 0.03, 0.06))
        upright.apply_translation((0.035, 0.0, 0.04))
        tri = trimesh.util.concatenate([base, upright])
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(tri.vertices),
            triangles=o3d.utility.Vector3iVector(tri.faces),
        )
        mesh.compute_vertex_normals()
        return mesh
    return _make


@pytest.fixture
def l_mesh(l_mesh_factory):
    """Function-scoped: recentring transforms the mesh in place."""
    return l_mesh_factory()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aabb_center(mesh):
    return np.asarray(mesh.get_axis_aligned_bounding_box().get_center())


def _apply(T, pts):
    T = np.asarray(T, dtype=float)
    return np.asarray(pts, dtype=float) @ T[:3, :3].T + T[:3, 3]


def _pose_matrix(pos, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(quat_wxyz, dtype=float), scalar_first=True).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def _dist_to_surface(points, mesh):
    """Unsigned distance from each point to the surface of `mesh`."""
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    q = o3d.core.Tensor(np.asarray(points, dtype=np.float32), dtype=o3d.core.Dtype.Float32)
    return sc.compute_distance(q).numpy().astype(np.float64)


def _transformed_mesh(mesh, T):
    m = copy.deepcopy(mesh)
    m.transform(np.asarray(T, dtype=float))
    return m


def _inject_ambiguity_profile(app, pcd):
    """A dominant axis deliberately offset from the centroid.

    That offset is the point: `pcd_geocenter(axis=...)` puts the *axis* on the origin, not the
    part, so the recentred part sits off-origin by exactly this much -- which is what makes the
    mesh AABB centre non-zero and forces `SceneStage`'s physics recentring to do real work.
    """
    centroid = np.asarray(pcd.points).mean(axis=0)
    axis = AmbiguityAxis(
        direction=np.array([0.0, 1.0, 0.0]),
        point=centroid + np.array([0.012, 0.0, 0.006]),
        fold=2, angles_deg=[180.0], is_global=True,
        view_fraction=1.0, area_fraction=0.9,
    )
    app.ambiguity_profile = AmbiguityProfile(axes=[axis], dominant=axis)


def _downsample_and_recenter(app, mesh, frame_mode, n_points=20000):
    """Run DOWNSAMPLE and apply the requested model frame, as the real pipeline does.

    `run_ambiguity=False` with a hand-built profile rather than a real `analyse_ambiguity`
    sweep: the analysis costs 10-60 s and its output is not the thing under test here. What is
    under test is that whatever frame comes out of it survives the trip into the physics scene.
    """
    app.target_mesh = mesh
    app.cropped_pcd = mesh.sample_points_uniformly(n_points)
    stage = app.stages[Stage.DOWNSAMPLE]
    stage.use_adaptive = False
    stage.run_ambiguity = False
    stage.worker()

    if frame_mode == "ambiguity":
        _inject_ambiguity_profile(app, app.down_pcd_surface)
        stage.run_ambiguity = True
    stage.recenter_mesh_pcd()
    assert stage._use_ambiguity_frame() is (frame_mode == "ambiguity")
    return stage


# ─────────────────────────────────────────────────────────────────────────────
# Fake physics layer — lets the frame algebra be checked without running MuJoCo
# ─────────────────────────────────────────────────────────────────────────────

class _FakeMjScene:
    """Stand-in for `MujocoBinScene` that records what `SceneStage` handed the physics layer.

    `mujoco_scene_to_o3d` mirrors the real one exactly (geom = the *centred* mesh it was
    constructed with, `T_gt` = the raw body pose), because that is precisely the state
    `SceneStage` then has to remap.
    """

    last = None

    #: settled poses the fake reports; arbitrary but non-trivial (rotation AND translation).
    POSES = {
        "part_0": (np.array([0.031, -0.017, 0.044]),
                   R.from_euler("xyz", [21.0, -37.0, 63.0], degrees=True)
                    .as_quat(scalar_first=True)),
        "part_1": (np.array([-0.052, 0.023, 0.019]),
                   R.from_euler("xyz", [-8.0, 74.0, -15.0], degrees=True)
                    .as_quat(scalar_first=True)),
    }

    def __init__(self, part_mesh, part_convex_meshes, n_parts=2, **kwargs):
        self.part_mesh = copy.deepcopy(part_mesh)                 # trimesh, as received
        self.part_convex_meshes = [copy.deepcopy(m) for m in (part_convex_meshes or [])]
        self.n_parts = n_parts
        self.kwargs = kwargs
        self.bin_mesh = o3d.geometry.TriangleMesh.create_box(0.4, 0.4, 0.2)
        self.camera_distance = 1.0
        _FakeMjScene.last = self

    def simulate(self, on_step=None):
        pass

    def verify_parts_in_bin(self):
        pass

    def extract_scene_state(self):
        return {name: {"position": pos, "quaternion": quat}
                for name, (pos, quat) in list(self.POSES.items())[:self.n_parts]}

    def mujoco_scene_to_o3d(self, scene_dict):
        out = {}
        for key, bd in scene_dict.items():
            mesh = trimesh_to_o3d(self.part_mesh)
            mesh.compute_vertex_normals()
            out[key] = O3DSceneObject(geom=mesh,
                                      T_gt=_pose_matrix(bd["position"], bd["quaternion"]))
        out["bin"] = O3DSceneObject(geom=self.bin_mesh, T_gt=np.eye(4))
        return out


def _run_scene_with_fake_physics(app, monkeypatch, n_parts=2):
    monkeypatch.setattr(scene_stage_module, "MujocoBinScene", _FakeMjScene)
    stage = app.stages[Stage.SCENE]
    stage.arrangement = "random"
    stage.generate_mode = "count"
    stage.num_targets = n_parts
    stage.structure_type = "none"
    stage.stable_pose_R = None
    stage.worker()
    return _FakeMjScene.last


# ─────────────────────────────────────────────────────────────────────────────
# Fast tests — the physics-recentring remap, in both model frames
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("frame_mode", ["pca", "ambiguity"])
def test_exported_T_gt_undoes_the_physics_recentring(headless_app, l_mesh, monkeypatch,
                                                     frame_mode):
    """The exported GT must place the ORIGINAL-frame mesh exactly where MuJoCo placed the
    CENTRED one.

    This is the arithmetic behind `T_gt_adj = T_gt_phys @ T_shift`. Get it wrong and every
    baked pose is offset by the mesh AABB centre -- a constant, plausible-looking error that
    survives every downstream check and simply makes the tuner optimise against a shifted
    target.
    """
    app = headless_app
    _downsample_and_recenter(app, l_mesh, frame_mode)
    app.convex_meshes = [copy.deepcopy(app.target_mesh)]

    mesh_center = _aabb_center(app.target_mesh)
    assert np.linalg.norm(mesh_center) > 1e-9, \
        "fixture must leave the AABB centre off-origin, or the remap under test never runs"
    verts_original = np.asarray(app.target_mesh.vertices).copy()

    fake = _run_scene_with_fake_physics(app, monkeypatch)

    # What the physics layer was actually given: the mesh, centred on its AABB centre.
    verts_physics = np.asarray(fake.part_mesh.vertices)
    assert np.allclose(verts_physics, verts_original - mesh_center, atol=1e-12)

    for name, (pos, quat) in list(_FakeMjScene.POSES.items())[:2]:
        obj = app.o3d_scene[name]
        # The exported geom is the original mesh, untouched — not the centred physics copy.
        assert np.allclose(np.asarray(obj.geom.vertices), verts_original, atol=1e-12)
        world_from_export = _apply(obj.T_gt, verts_original)
        world_from_physics = _apply(_pose_matrix(pos, quat), verts_physics)
        assert np.allclose(world_from_export, world_from_physics, atol=1e-12), (
            f"{name}: exported T_gt does not reproduce the simulated pose; max error "
            f"{np.abs(world_from_export - world_from_physics).max() * 1000:.4f} mm")


@pytest.mark.parametrize("frame_mode", ["pca", "ambiguity"])
def test_reference_cloud_lands_on_the_simulated_part(headless_app, l_mesh, monkeypatch,
                                                     frame_mode):
    """The tuner's own operation: take the exported reference cloud, apply the baked GT, and
    check it lands on the part the simulator settled.

    Measured against the physics geometry (centred mesh at its raw body pose), which is
    derived without touching the exported `T_gt` — so this cannot pass by construction.

    Tolerance-free: point-to-surface distance is a rigid invariant, so the numbers must match
    the model-frame baseline exactly. A frame disagreement would not merely inflate them, it
    would change them.
    """
    app = headless_app
    _downsample_and_recenter(app, l_mesh, frame_mode)
    app.convex_meshes = [copy.deepcopy(app.target_mesh)]

    ref_pts = np.asarray(app.down_pcd_surface.points).copy()
    baseline = _dist_to_surface(ref_pts, app.target_mesh)

    fake = _run_scene_with_fake_physics(app, monkeypatch)
    physics_mesh_o3d = trimesh_to_o3d(fake.part_mesh)

    for name, (pos, quat) in list(_FakeMjScene.POSES.items())[:2]:
        obj = app.o3d_scene[name]
        placed = _dist_to_surface(_apply(obj.T_gt, ref_pts),
                                  _transformed_mesh(physics_mesh_o3d, _pose_matrix(pos, quat)))
        assert np.allclose(placed, baseline, atol=SURFACE_ATOL_M), (
            f"{name}: reference cloud placed by the baked GT does not sit on the simulated "
            f"part; p99 distance {np.percentile(placed, 99) * 1000:.4f} mm vs baseline "
            f"{np.percentile(baseline, 99) * 1000:.4f} mm")


@pytest.mark.parametrize("frame_mode", ["pca", "ambiguity"])
def test_collision_hulls_are_shifted_with_the_part_mesh(headless_app, l_mesh, monkeypatch,
                                                        frame_mode):
    """Convex hulls must be centred by exactly the same offset as the part mesh.

    They are separate objects on `app.convex_meshes` and shifted in a separate branch, so
    nothing but a test couples them. Shift them differently and MuJoCo settles the part on
    collision geometry that does not match its visual mesh: the poses are self-consistent, the
    GT is self-consistent, and the parts simply rest in places the real part could not.
    """
    app = headless_app
    _downsample_and_recenter(app, l_mesh, frame_mode)
    hull = copy.deepcopy(app.target_mesh)
    app.convex_meshes = [hull]
    hull_verts = np.asarray(hull.vertices).copy()

    mesh_center = _aabb_center(app.target_mesh)
    fake = _run_scene_with_fake_physics(app, monkeypatch)

    assert len(fake.part_convex_meshes) == 1
    assert np.allclose(np.asarray(fake.part_convex_meshes[0].vertices),
                       hull_verts - mesh_center, atol=1e-12)
    # The stage must not mutate the app-owned hulls while making its physics copies.
    assert np.allclose(np.asarray(app.convex_meshes[0].vertices), hull_verts, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# scene_state.npz — the scene-level GT must be in the same frame as the per-sample GT
# ─────────────────────────────────────────────────────────────────────────────

def _bare_scene(body_offset):
    """A `MujocoBinScene` with only the attributes `export_scene_state` reads.

    Built with `object.__new__` and a stubbed `extract_scene_state` so the frame arithmetic can
    be tested without compiling a MuJoCo model — the offset composition is pure bookkeeping and
    has nothing to do with physics.
    """
    from physics.mujoco_bin_scene import MujocoBinScene

    sc = object.__new__(MujocoBinScene)
    sc.body_offset = np.zeros(3) if body_offset is None else np.asarray(body_offset, float)
    sc.bin_dim = (0.4, 0.4, 0.2, 0.005)
    sc.bin_transform = np.eye(4)
    sc.camera_distance = 1.5
    sc.arrangement = "random"
    sc.structure_type = "none"
    sc.structure_height_frac = 0.75
    sc._clearance_m = 0.0025
    sc.extract_scene_state = lambda: {
        name: {"position": pos, "quaternion": quat}
        for name, (pos, quat) in _FakeMjScene.POSES.items()}
    return sc


def test_export_scene_state_reports_poses_in_the_model_frame():
    """`scene_state.npz`'s T_gt must mean the same thing as `sample_i.ply`'s `gt_*`.

    It used to mean something else: raw physics body poses, while `SceneStage` fixed up only
    the per-instance copies. Two GT arrays in one scene directory disagreeing by a silent few
    millimetres is exactly the failure that scores a tuning run against a fiction.
    """
    offset = np.array([-0.0058, -0.0010, 0.0001])
    raw, shifted = _bare_scene(None), _bare_scene(offset)

    s_raw, s_shifted = raw.export_scene_state(), shifted.export_scene_state()
    assert list(s_raw["body_names"]) == list(s_shifted["body_names"])

    for i in range(len(s_raw["body_names"])):
        T_body, T_model = s_raw["T_gt"][i], s_shifted["T_gt"][i]
        # p_world = T_body @ (p_model - offset)
        assert np.allclose(T_model[:3, :3], T_body[:3, :3], atol=1e-12), \
            "a pure-translation offset must not touch the rotation"
        assert np.allclose(T_model[:3, 3], T_body[:3, 3] - T_body[:3, :3] @ offset, atol=1e-12)
        # positions must follow T_gt, or the npz contradicts itself.
        assert np.allclose(s_shifted["positions"][i], T_model[:3, 3], atol=1e-12)

    # Rotations are frame-invariant here, so the quaternions are untouched.
    assert np.allclose(s_raw["quaternions_wxyz"], s_shifted["quaternions_wxyz"], atol=1e-12)
    # The version marker downstream readers use to tell old files from new ones.
    assert np.allclose(s_shifted["body_offset"], offset, atol=1e-12)
    assert np.allclose(s_raw["body_offset"], 0.0, atol=1e-12)


@pytest.mark.parametrize("frame_mode", ["pca", "ambiguity"])
def test_scene_stage_hands_the_centring_offset_to_the_physics_layer(headless_app, l_mesh,
                                                                    monkeypatch, frame_mode):
    """The wiring, not the arithmetic: `SceneStage` must tell `MujocoBinScene` what it
    subtracted. Forget the kwarg and `export_scene_state` silently reverts to body-frame
    poses while every other export stays in the model frame."""
    app = headless_app
    _downsample_and_recenter(app, l_mesh, frame_mode)
    app.convex_meshes = [copy.deepcopy(app.target_mesh)]
    mesh_center = _aabb_center(app.target_mesh)

    fake = _run_scene_with_fake_physics(app, monkeypatch)

    assert np.allclose(fake.kwargs.get("body_offset"), mesh_center, atol=1e-12), \
        "SceneStage must pass the centring offset through to MujocoBinScene"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end — real MuJoCo, and the PLY the tuner actually reads
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("frame_mode", ["pca", "ambiguity"])
def test_real_sim_gt_agrees_with_the_reference_cloud(headless_app, l_mesh, frame_mode):
    """The same invariant with MuJoCo and VHACD in the loop, not a fake.

    The fast tests fix the algebra; this one catches anything the real physics layer does to
    the mesh on the way through (`MujocoBinScene` keeps its own copy, rotates about the body
    origin, and reports poses for a body whose frame the stage has to reason about).
    """
    app = headless_app
    app.mesh_basename = "pytest_l_part"
    _downsample_and_recenter(app, l_mesh, frame_mode)

    app.stages[Stage.DECOMPOSE].worker()
    assert len(app.convex_meshes) > 0

    scene = app.stages[Stage.SCENE]
    scene.arrangement = "random"
    scene.generate_mode = "count"
    scene.num_targets = 2
    scene.structure_type = "none"
    scene.stable_pose_R = None
    scene.worker()

    mesh_center = _aabb_center(app.target_mesh)
    verts_original = np.asarray(app.target_mesh.vertices)
    physics_verts = np.asarray(app.mj_scene.part_mesh.vertices)
    assert np.allclose(physics_verts, verts_original - mesh_center, atol=1e-9)

    ref_pts = np.asarray(app.down_pcd_surface.points).copy()
    baseline = _dist_to_surface(ref_pts, app.target_mesh)
    physics_mesh_o3d = trimesh_to_o3d(app.mj_scene.part_mesh)

    # The scene-level GT array must be the same frame as the per-instance one it ships beside.
    exported = app.mj_scene.export_scene_state()
    assert np.allclose(exported["body_offset"], mesh_center, atol=1e-9)
    for i, name in enumerate(exported["body_names"]):
        assert np.allclose(exported["T_gt"][i], app.o3d_scene[str(name)].T_gt, atol=1e-9), \
            f"{name}: scene_state T_gt disagrees with the per-instance T_gt"

    # extract_scene_state() re-reads the settled MjData; no stepping happened since worker().
    truth = app.mj_scene.extract_scene_state()
    assert truth, "sim produced no bodies"
    for name, bd in truth.items():
        obj = app.o3d_scene[name]
        T_phys = _pose_matrix(bd["position"], bd["quaternion"])
        assert np.allclose(_apply(obj.T_gt, verts_original),
                           _apply(T_phys, physics_verts), atol=1e-9), \
            f"{name}: exported T_gt does not reproduce the settled pose"
        placed = _dist_to_surface(_apply(obj.T_gt, ref_pts),
                                  _transformed_mesh(physics_mesh_o3d, T_phys))
        assert np.allclose(placed, baseline, atol=SURFACE_ATOL_M), (
            f"{name}: reference cloud placed by the baked GT is off the settled part; "
            f"p99 {np.percentile(placed, 99) * 1000:.4f} mm vs baseline "
            f"{np.percentile(baseline, 99) * 1000:.4f} mm")


@pytest.mark.slow
@pytest.mark.parametrize("frame_mode", ["pca", "ambiguity"])
def test_gt_baked_into_the_exported_ply_survives_the_round_trip(headless_app, l_mesh,
                                                                monkeypatch, tmp_path,
                                                                frame_mode):
    """The exact bytes the tuner consumes.

    `MM_Optimizer.optimizer_utils.read_gt_pose_from_ply` parses the `gt_*` header comments of
    `sample_i.ply` and that pose is what every trial is scored against. This runs the whole
    pipeline to disk and reads it back the way the optimizer does, so a formatting slip
    (quaternion convention, key name, precision) is caught here rather than as inexplicable
    pose error hours into a study.

    Only the pose is checked against the reference cloud and the simulated geometry -- the
    noisy rendered points in the same file are a sensor question, not a frame question.
    """
    from MM_Optimizer.optimizer_utils import read_gt_pose_from_ply

    app = headless_app
    app.mesh_basename = "pytest_l_part"
    _downsample_and_recenter(app, l_mesh, frame_mode)

    app.stages[Stage.DECOMPOSE].worker()
    scene = app.stages[Stage.SCENE]
    scene.arrangement = "random"
    scene.generate_mode = "count"
    scene.num_targets = 3
    scene.structure_type = "none"
    scene.stable_pose_R = None
    scene.worker()

    app.set_stage(Stage.RENDER)
    render = app.stages[Stage.RENDER]
    render.worker()
    if not app.synthetic_targets:
        pytest.skip("no instance passed the 2D candidate filter in this sim roll")

    monkeypatch.chdir(tmp_path)          # save_synthetic_targets writes under Path.cwd()
    render.save_synthetic_targets()
    out_dir = tmp_path / "output" / "synthetic_target" / app.mesh_basename / "scene_00000"
    assert out_dir.is_dir(), "scene directory was not written"

    # The reference cloud the tuner will match against, straight from the same export.
    ref_ply = out_dir / "reference_cloud.ply"
    assert ref_ply.exists()
    ref_pts = np.asarray(o3d.io.read_point_cloud(str(ref_ply)).points)
    baseline = _dist_to_surface(ref_pts, app.target_mesh)

    physics_mesh_o3d = trimesh_to_o3d(app.mj_scene.part_mesh)
    truth = {name: _pose_matrix(bd["position"], bd["quaternion"])
             for name, bd in app.mj_scene.extract_scene_state().items()}

    # scene_state.npz ships in the same directory and must be in the same frame as the
    # per-sample GT, so a reader that picks either one gets the same answer.
    with np.load(out_dir / "scene_state.npz", allow_pickle=True) as z:
        assert "body_offset" in z.files, "new scenes must carry the model-frame marker"
        npz_T_gt = np.asarray(z["T_gt"], dtype=float)
        npz_names = [str(n) for n in z["body_names"]]
    for i, name in enumerate(npz_names):
        placed = _dist_to_surface(_apply(npz_T_gt[i], ref_pts),
                                  _transformed_mesh(physics_mesh_o3d, truth[name]))
        assert np.allclose(placed, baseline, atol=SURFACE_ATOL_M), (
            f"{name}: scene_state.npz T_gt does not place the exported reference cloud on "
            f"the settled part; p99 {np.percentile(placed, 99) * 1000:.4f} mm vs baseline "
            f"{np.percentile(baseline, 99) * 1000:.4f} mm")

    samples = sorted(out_dir.glob("sample_*.ply"))
    assert samples, "no candidate PLY was exported"
    for ply in samples:
        x, y, z, qw, qx, qy, qz = read_gt_pose_from_ply(str(ply), scalar_first=True)
        T_read = _pose_matrix((x, y, z), (qw, qx, qy, qz))

        # The GT must name one of the settled bodies, and place the reference cloud on it.
        placed = [_dist_to_surface(_apply(T_read, ref_pts),
                                   _transformed_mesh(physics_mesh_o3d, T_phys))
                  for T_phys in truth.values()]
        best = min(placed, key=lambda d: float(np.percentile(d, 99)))
        assert np.allclose(best, baseline, atol=SURFACE_ATOL_M), (
            f"{ply.name}: GT read back from the PLY header does not place the exported "
            f"reference cloud on any settled part; best p99 "
            f"{np.percentile(best, 99) * 1000:.4f} mm vs baseline "
            f"{np.percentile(baseline, 99) * 1000:.4f} mm")
