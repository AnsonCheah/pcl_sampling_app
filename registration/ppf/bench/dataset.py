"""Loader for the synthetic scenes under ``output/synthetic_target/<part>/scene_*/``.

Three things about this data are easy to get wrong, and all three are load-bearing:

* **Use ``sample_i.npz["T_gt"]``, not ``scene_state.npz["T_gt"]``, on any scene generated
  before the two were unified.**  ``MujocoBinScene.export_scene_state`` used to write poses
  straight from the physics bodies, while ``SceneStage.worker`` post-multiplied a
  mesh-recentring shift onto the per-instance poses whenever the imported mesh was not
  already centred on its bounding box (which is the normal case -- ``center_mesh`` is a manual
  button).  Only the per-sample one was expressed in the same frame as
  ``reference_cloud.ply``.  ``export_scene_state`` now composes the same shift, so both agree;
  the marker is the ``body_offset`` key, present only on scenes written since.  This loader
  reads the per-sample pose either way, which is correct for old and new directories alike --
  do not "simplify" it to the scene-level array.

* **``sample_<i>`` does not necessarily correspond to ``part_<i>``.**  Samples are numbered
  by a running counter over instances that pass the 2D aspect/area filter, so the mapping is
  identity only when every instance passes.  Never join the two by index.

* **Normals.**  ``sample_*.ply`` carries the raycast surface normals, which are exact -- the
  sensor noise chain perturbs the *points* but those normals came from the mesh.  A real
  sensor delivers depth, and normals get estimated from the noisy points, so benchmarking
  against the stored ones flatters the matcher.  ``normals="estimated"`` (the default) is
  the honest setting; ``"stored"`` is available to isolate normal-estimation error from
  matcher error.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d

__all__ = ["Instance", "Scene", "list_parts", "list_scenes", "load_reference", "load_scene",
           "SYNTH_ROOT", "default_scenes_root"]


def default_scenes_root() -> str:
    """Where scenes live, resolved without assuming any particular repository layout.

    The loader reads a *directory format*, not a specific project's output tree, so the
    location has to come from the caller.  ``PPF_SCENES_ROOT`` wins if set; otherwise this
    falls back to ``./output/synthetic_target`` under the current working directory, which is
    what the generator in the surrounding repo happens to write.  Every public function also
    takes ``root`` explicitly, so nothing here depends on getting this default right.
    """
    env = os.environ.get("PPF_SCENES_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(os.getcwd(), "output", "synthetic_target"))


SYNTH_ROOT = default_scenes_root()


@dataclass
class Instance:
    """One segmented cluster plus its ground-truth pose."""

    index: int
    points: np.ndarray               # (N, 3) scene frame
    normals: np.ndarray              # (N, 3) unit
    T_gt: np.ndarray                 # (4, 4) model frame -> scene frame
    overlap: float                   # fraction of the reference visible; 0.10-0.48 typical


@dataclass
class Scene:
    directory: str
    instances: List[Instance] = field(default_factory=list)
    ref_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    ref_normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    scene_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    state: Dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(self.directory)

    @property
    def bin_width(self) -> float:
        dim = self.state.get("bin_dim")
        return float(dim[0]) if dim is not None else 1.0

    @property
    def camera_position(self) -> np.ndarray:
        return np.array([0.0, 0.0, float(self.state.get("camera_distance", 1.5))])


def list_parts(root: Optional[str] = None) -> List[str]:
    # Resolved per call, not bound at import: the default depends on the working directory
    # and on PPF_SCENES_ROOT, either of which can legitimately change after import.
    root = root or default_scenes_root()
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def list_scenes(part: str, root: Optional[str] = None) -> List[str]:
    """Absolute paths of a part's ``scene_NNNNN`` directories, in order."""
    part_dir = part if os.path.isdir(part) else os.path.join(root or default_scenes_root(), part)
    if not os.path.isdir(part_dir):
        return []
    return [os.path.join(part_dir, d) for d in sorted(os.listdir(part_dir))
            if d.startswith("scene_") and os.path.isdir(os.path.join(part_dir, d))]


def _sample_index(path: str) -> int:
    return int(re.search(r"(\d+)", os.path.basename(path)).group(1))


def load_reference(scene_dir: str):
    """Just the reference cloud -- ``(points, normals)``.

    Split out so a caller can derive its matching parameters before loading the instances,
    because the normal-estimation radius has to come *from* those parameters. See
    ``load_scene``.
    """
    ref = o3d.io.read_point_cloud(os.path.join(scene_dir, "reference_cloud.ply"))
    return (np.asarray(ref.points, dtype=np.float64),
            np.asarray(ref.normals, dtype=np.float64))


def load_scene(scene_dir: str,
               normal_radius: float,
               normals: str = "estimated",
               load_scene_cloud: bool = True,
               max_instances: Optional[int] = None) -> Scene:
    """Load one scene directory.

    ``normal_radius`` is **required and not defaulted**, because it is not a cosmetic
    preprocessing choice -- it moves the answer.  Measured on this bunny scene, switching it
    from 2*tau (14 mm) to 4x the reference spacing (2.8 mm) moved recall @5mm/10deg from
    0.77 to 0.62 on the unweighted arm, and *reordered* the weighting arms against each
    other.  A quietly-defaulted value here would silently become the most influential
    untracked parameter in the benchmark.

    Pass ``2 * cfg.tau``: tau is the scale at which the matcher quantises distances, so
    estimating normals over that same neighbourhood is what makes the scene's normals
    comparable to the model's.  It is also the radius ``PPFConfig.derive`` already assumes
    when it converts depth noise into an angular bin width, so anything else makes that
    derivation describe a cloud that was never built.
    """
    scene_dir = os.path.abspath(scene_dir)
    ref_pts, ref_nrm = load_reference(scene_dir)

    state: Dict = {}
    state_path = os.path.join(scene_dir, "scene_state.npz")
    if os.path.exists(state_path):
        with np.load(state_path, allow_pickle=True) as z:
            state = {k: z[k] for k in z.files}

    cam = np.array([0.0, 0.0, float(state.get("camera_distance", 1.5))])

    plys = sorted((os.path.join(scene_dir, f) for f in os.listdir(scene_dir)
                   if f.startswith("sample_") and f.endswith(".ply")), key=_sample_index)
    if max_instances:
        plys = plys[:max_instances]

    instances: List[Instance] = []
    for ply in plys:
        i = _sample_index(ply)
        npz = os.path.join(scene_dir, f"sample_{i}.npz")
        if not os.path.exists(npz):
            continue
        with np.load(npz, allow_pickle=True) as z:
            T_gt = np.asarray(z["T_gt"], dtype=np.float64)
            overlap = float(z["overlap"])

        pcd = o3d.io.read_point_cloud(ply)
        if normals == "estimated" or not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
            # Orientation matters as much as direction: PPF's features use the signed angle
            # between a normal and the pair direction, so a flipped normal is a different
            # feature, not a near one. The sensor saw everything from the camera.
            pcd.orient_normals_towards_camera_location(cam)
        n = np.asarray(pcd.normals, dtype=np.float64)
        instances.append(Instance(index=i, points=np.asarray(pcd.points, dtype=np.float64),
                                  normals=n, T_gt=T_gt, overlap=overlap))

    scene_pts = np.empty((0, 3))
    scene_ply = os.path.join(scene_dir, "scene.ply")
    if load_scene_cloud and os.path.exists(scene_ply):
        scene_pts = np.asarray(o3d.io.read_point_cloud(scene_ply).points, dtype=np.float64)

    return Scene(directory=scene_dir, instances=instances,
                 ref_points=ref_pts, ref_normals=ref_nrm,
                 scene_points=scene_pts, state=state)
