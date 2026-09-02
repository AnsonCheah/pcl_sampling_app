"""
_fixtures.py -- Shared synthetic scene helpers for sensor/ unit tests.

Builds minimal render dicts using real scene_render() so tests exercise the
actual pipeline without requiring any file I/O or MuJoCo.

Usage
-----
    from sensor.tests._fixtures import make_render_dict, make_two_plane_render_dict

    render, keep = make_render_dict()              # single face-on plane
    render, keep = make_render_dict(tilt_deg=45.)  # tilted plane (specular lobe mismatch)
    render, keep = make_two_plane_render_dict()    # two adjacent planes, distinct geom_ids
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import open3d as o3d
from geometry.geom_utils import O3DSceneObject, camera_view_matrix
from sensor.scene_render import scene_render, compute_dropout_mask

# -- Scene constants ------------------------------------------------------------
_H, _W   = 120, 160          # image resolution -- small for speed
_FOV     = 41.11             # degrees, matches production pipeline
_CAM_POS = np.array([0., 0., 0.])
_LOOK_AT = np.array([0., 0., 1.5])   # plane centre, 1.5 m range (mid of 1-2 m robotics range)


# -- Private mesh builder -------------------------------------------------------

def _plane_mesh(cx: float = 0., cy: float = 0., cz: float = 1.5,
                w: float = 0.25, h: float = 0.25,
                tilt_deg: float = 0.) -> o3d.geometry.TriangleMesh:
    """
    Thin rectangular box (0.002 m deep) centred at (cx, cy, cz).
    Optional tilt about the X-axis (positive = top of plane tilts away from camera).
    """
    mesh = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=0.002)
    # create_box places corner at origin; shift so centre is at (cx, cy, cz)
    mesh.translate([-w / 2 + cx, -h / 2 + cy, cz - 0.001])
    if tilt_deg != 0.:
        R = mesh.get_rotation_matrix_from_xyz((np.deg2rad(tilt_deg), 0., 0.))
        mesh.rotate(R, center=(cx, cy, cz))
    mesh.compute_vertex_normals()
    return mesh


# -- Public fixtures ------------------------------------------------------------

def make_render_dict(tilt_deg: float = 0., seed: int = 0):
    """
    Single flat plane at z=1.5 m, optionally tilted about the X-axis.

    Parameters
    ----------
    tilt_deg : float
        X-axis tilt in degrees. 0 = face-on (normal points at camera).
        45 = tilted away (specular lobe points off-camera -> patch missing trigger).
    seed : int
        Passed to compute_dropout_mask.

    Returns
    -------
    render : dict from scene_render()
    keep   : (N,) bool mask from compute_dropout_mask()
    """
    mesh  = _plane_mesh(tilt_deg=tilt_deg)
    T_cam = camera_view_matrix(_CAM_POS, _LOOK_AT)
    meshes = {"plane": O3DSceneObject(geom=mesh, T_gt=np.eye(4))}
    # normal_radius=0.020 m: at 120x160 resolution the point spacing at 1.5 m is
    # ~6.7 mm, so the default 5 mm radius gives zero neighbours and wrong normals.
    # 20 mm (~3x point spacing) produces correct PCA normals for tilted surfaces.
    render = scene_render(meshes, T_cam, _LOOK_AT, _FOV, _W, _H, normal_radius=0.020)
    assert render is not None, (
        "Fixture plane not visible -- check _CAM_POS / _LOOK_AT geometry."
    )
    keep = compute_dropout_mask(render, roughness=0.4, seed=seed)
    return render, keep


def make_two_plane_render_dict(seed: int = 0):
    """
    Two adjacent planes side-by-side in image space, with distinct geom_ids.

    Plane L: centred at x = -0.10 m  (left half of image)
    Plane R: centred at x = +0.10 m  (right half of image)

    Their projected silhouettes touch at the image centre, providing a shared
    boundary zone for segmentation overlap / confusion tests.

    Returns
    -------
    render : dict from scene_render() -- geom_ids will be 0 (left) and 1 (right)
    keep   : (N,) bool mask from compute_dropout_mask()
    """
    mesh_l = _plane_mesh(cx=-0.10, w=0.15)
    mesh_r = _plane_mesh(cx=+0.10, w=0.15)
    T_cam  = camera_view_matrix(_CAM_POS, _LOOK_AT)
    meshes = {
        "plane_l": O3DSceneObject(geom=mesh_l, T_gt=np.eye(4)),
        "plane_r": O3DSceneObject(geom=mesh_r, T_gt=np.eye(4)),
    }
    render = scene_render(meshes, T_cam, _LOOK_AT, _FOV, _W, _H, normal_radius=0.020)
    assert render is not None, (
        "Fixture planes not visible -- check scene geometry."
    )
    keep = compute_dropout_mask(render, roughness=0.4, seed=seed)
    return render, keep


def make_multi_instance_render_dict(layout: str = "grid", W: int = _W, H: int = _H,
                                    seed: int = 0):
    """
    Several small planes with distinct geom_ids -- for confusion bbox-pruning and
    scaling tests.

    layout : "grid" -- 2x2 block of planes whose silhouettes sit close together so
                       neighbouring pairs share a boundary zone (pruning keeps them).
             "far"  -- the same planes pushed apart so most pairs' padded bounding
                       boxes do NOT overlap (pruning skips them). Pruned and
                       un-pruned confusion must give identical masks either way.
    W, H   : render resolution. Pixel separation between instances scales with
             resolution while the px crop margin is fixed by the noise params, so a
             higher resolution is needed for the "far" layout to actually exercise
             the bbox-prune path (see the segmentation pruning test).

    Returns
    -------
    render : dict from scene_render() -- geom_ids 0..3
    keep   : (N,) bool mask from compute_dropout_mask()
    """
    spread = 0.06 if layout == "grid" else 0.16
    w = 0.07
    centres = [(-spread, -spread), (spread, -spread),
               (-spread,  spread), (spread,  spread)]
    T_cam  = camera_view_matrix(_CAM_POS, _LOOK_AT)
    meshes = {
        f"plane_{i}": O3DSceneObject(geom=_plane_mesh(cx=cx, cy=cy, w=w, h=w), T_gt=np.eye(4))
        for i, (cx, cy) in enumerate(centres)
    }
    render = scene_render(meshes, T_cam, _LOOK_AT, _FOV, W, H, normal_radius=0.020)
    assert render is not None, "Multi-instance fixture not visible -- check geometry."
    keep = compute_dropout_mask(render, roughness=0.4, seed=seed)
    return render, keep
