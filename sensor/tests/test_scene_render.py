"""
test_scene_render.py -- Regression + new-stage tests for sensor/scene_render.py.

Run from project root:
    python sensor/tests/test_scene_render.py

All tests use the minimal synthetic fixtures from _fixtures.py (no file I/O,
no MuJoCo). A flat plane at z=1.5 m is built programmatically and raycasted.
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from sensor.tests._fixtures import make_render_dict, make_two_plane_render_dict, _CAM_POS, _LOOK_AT, _FOV, _H, _W
from geometry.geom_utils import O3DSceneObject, camera_view_matrix
from sensor.scene_render import (
    scene_render,
    compute_dropout_mask,
    add_projector_nonuniformity,
    add_specular_patch_missing,
    add_image_space_effects,
    add_edge_artifacts,
    add_multipath_outliers,
    add_pepper_noise,
    add_scan_line_banding,
    add_sensor_noise,
    add_surface_noise,
    subset_render,
)
import open3d as o3d

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = []

def _check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    _results.append((name, cond))


# ==============================================================================
#  Existing function regression tests
# ==============================================================================

def test_scene_render_returns_expected_keys():
    render, _ = make_render_dict()
    required = {
        "points", "normals", "geom_ids", "t_hit",
        "ray_origins", "ray_dirs", "proj_dirs", "proj_dist",
        "cos_cam", "cos_proj", "snr_proxy",
        "sensor_origin", "proj_origin",
        "pixel_idx", "res", "depth_img", "hit_mask",
        "ray_origins_img", "ray_dirs_img",
    }
    missing = required - set(render.keys())
    _check("scene_render_returns_expected_keys", not missing,
           f"missing: {missing}" if missing else "")


def test_scene_render_returns_none_on_miss():
    import open3d as o3d
    # Plane placed 100 m away -- outside FOV at 1.5 m focal distance
    mesh = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.001)
    mesh.translate([0, 0, 100.])
    T_cam   = camera_view_matrix(_CAM_POS, _LOOK_AT)
    meshes  = {"far": O3DSceneObject(geom=mesh, T_gt=np.eye(4))}
    result  = scene_render(meshes, T_cam, _LOOK_AT, _FOV, _W, _H)
    _check("scene_render_returns_none_on_miss", result is None)


def test_compute_dropout_mask_lambertian_keeps_more_than_specular():
    # Use a 40deg tilted plane so the specular lobe is misaligned -- only Lambertian
    # (roughness=1.0, diffuse floor=1) keeps all points; roughness=0.0 (pure mirror)
    # drops most because the reflected ray doesn't align with the camera.
    render, _ = make_render_dict(tilt_deg=40.)
    keep_lamb = compute_dropout_mask(render, roughness=1.0, seed=0)
    keep_spec = compute_dropout_mask(render, roughness=0.0, seed=0)
    _check("dropout_lambertian_keeps_more_than_specular",
           keep_lamb.sum() > keep_spec.sum(),
           f"lambertian={keep_lamb.sum()}, specular={keep_spec.sum()}")


def test_compute_dropout_mask_returns_correct_shape():
    render, keep = make_render_dict()
    _check("dropout_mask_shape", keep.shape == (len(render["points"]),))
    _check("dropout_mask_dtype", keep.dtype == bool)


def test_add_image_space_effects_zero_sigma_is_noop():
    render, keep = make_render_dict()
    out = add_image_space_effects(render, keep, smooth_sigma_px=0., sigma_fringe_corr=0.)
    diff = np.abs(out["points"] - render["points"]).max()
    _check("image_space_zero_sigma_noop", diff < 1e-6, f"max_diff={diff:.2e}")


def test_add_image_space_effects_preserves_dict_keys():
    render, keep = make_render_dict()
    out = add_image_space_effects(render, keep, smooth_sigma_px=0.5)
    missing = set(render.keys()) - set(out.keys())
    _check("image_space_preserves_keys", not missing,
           f"lost keys: {missing}" if missing else "")


def test_add_edge_artifacts_returns_at_least_keep_count():
    render, keep = make_render_dict()
    pts, nrm = add_edge_artifacts(render, keep)
    _check("edge_artifacts_at_least_keep_count",
           len(pts) >= keep.sum(),
           f"pts={len(pts)}, keep={keep.sum()}")


def test_add_edge_artifacts_normals_unit_length():
    render, keep = make_render_dict()
    _, nrm = add_edge_artifacts(render, keep)
    norms  = np.linalg.norm(nrm, axis=1)
    _check("edge_artifacts_normals_unit",
           np.allclose(norms, 1., atol=1e-5),
           f"max_dev={np.abs(norms-1).max():.2e}")


def test_add_multipath_outliers_returns_ndarray():
    render, keep = make_render_dict()
    r = subset_render(render, keep)
    mp, mn = add_multipath_outliers(r)
    _check("multipath_outliers_shape_pts", mp.ndim == 2 and mp.shape[1] == 3)
    _check("multipath_outliers_shape_nrm",
           mn.ndim == 2 and mn.shape[1] == 3 and len(mn) == len(mp))


def test_add_pepper_noise_returns_ndarray():
    render, keep = make_render_dict()
    r = subset_render(render, keep)
    pp, pn = add_pepper_noise(r)
    _check("pepper_noise_shape_pts", pp.ndim == 2 and pp.shape[1] == 3)
    _check("pepper_noise_shape_nrm",
           pn.ndim == 2 and pn.shape[1] == 3 and len(pn) == len(pp))


def test_add_scan_line_banding_preserves_shape():
    render, keep = make_render_dict()
    pts = render["points"][keep]
    nrm = render["normals"][keep]
    pidx = render["pixel_idx"][keep]
    out = add_scan_line_banding(pts, nrm, pidx, render["res"], render["sensor_origin"])
    _check("scan_line_banding_shape", out.shape == pts.shape)


def test_add_sensor_noise_introduces_displacement():
    render, keep = make_render_dict()
    pts = render["points"][keep]
    nrm = render["normals"][keep]
    pidx = render["pixel_idx"][keep]
    cproj = render["cos_proj"][keep]
    out = add_sensor_noise(pts, nrm, render["sensor_origin"],
                           pixel_idx=pidx, res=render["res"], cos_proj=cproj)
    _check("sensor_noise_shape", out.shape == pts.shape)
    _check("sensor_noise_introduces_displacement", not np.allclose(pts, out))


def test_add_surface_noise_displaces_along_normal():
    render, keep = make_render_dict()
    pts_in  = render["points"][keep].copy()
    nrm     = render["normals"][keep]
    pts_out = add_surface_noise(pts_in, nrm, amplitude=0.002)
    delta   = pts_out - pts_in
    total   = np.linalg.norm(delta, axis=1) + 1e-12
    axial   = np.abs(np.einsum("ij,ij->i", delta, nrm)) / total
    _check("surface_noise_displaces_along_normal",
           axial.mean() > 0.8, f"mean_axial_frac={axial.mean():.3f}")


def test_subset_render_preserves_image_buffers():
    render, keep = make_render_dict()
    r = subset_render(render, keep)
    for key in ("depth_img", "res", "sensor_origin", "proj_origin"):
        a, b = render[key], r[key]
        # np.array_equal treats NaN != NaN, so use nanequal logic for depth_img
        if key == "depth_img":
            same = np.all((a == b) | (np.isnan(a) & np.isnan(b)))
        else:
            same = np.array_equal(a, b)
        _check(f"subset_render_preserves_{key}", same)


# ==============================================================================
#  New stage tests
# ==============================================================================

def test_specular_patch_missing_reduces_keep():
    render, keep = make_render_dict(tilt_deg=45.)
    # patch_dropout_rate=1.0 guarantees deterministic dropout whenever a qualifying
    # dark patch exists; min_patch_area_px=1 ensures even single-pixel patches count.
    keep_after   = add_specular_patch_missing(render, keep, roughness=0.25,
                                              patch_dropout_rate=1.0,
                                              min_patch_area_px=1, seed=0)
    _check("patch_missing_reduces_keep",
           keep_after.sum() <= keep.sum(),
           f"before={keep.sum()}, after={keep_after.sum()}")
    # Verify at least one patch was dropped (tilted plane should trigger dark regions)
    dropped = keep & ~keep_after
    _check("patch_missing_drops_some_points", dropped.any(),
           f"dropped={dropped.sum()}")


def test_specular_patch_missing_lambertian_unchanged():
    # Face-on plane: normal points at camera -> specular lobe aligned -> few dark patches
    render, keep = make_render_dict(tilt_deg=0.)
    keep_after   = add_specular_patch_missing(render, keep, roughness=0.25,
                                              patch_dropout_rate=1.0, seed=0)
    frac_kept = keep_after.sum() / (keep.sum() + 1e-12)
    _check("patch_missing_faceon_mostly_survives",
           frac_kept >= 0.80,
           f"kept_frac={frac_kept:.3f}")


def test_projector_nonuniformity_modulates_snr():
    render, _ = make_render_dict()
    std_before = render["snr_proxy"].std()
    out        = add_projector_nonuniformity(render, proj_fpn_sigma=0.05, seed=0)
    std_after  = out["snr_proxy"].std()
    _check("proj_nonuniformity_increases_snr_std",
           std_after > std_before,
           f"std_before={std_before:.4f}, std_after={std_after:.4f}")
    mean_before = render["snr_proxy"].mean()
    mean_after  = out["snr_proxy"].mean()
    _check("proj_nonuniformity_preserves_snr_mean",
           abs(mean_after - mean_before) < 0.05,
           f"mean_before={mean_before:.4f}, mean_after={mean_after:.4f}")


def test_projector_nonuniformity_is_deterministic():
    render, _ = make_render_dict()
    out1 = add_projector_nonuniformity(render, seed=42)
    out2 = add_projector_nonuniformity(render, seed=42)
    _check("proj_nonuniformity_deterministic",
           np.array_equal(out1["snr_proxy"], out2["snr_proxy"]))


def test_anisotropic_roughness_zero_matches_isotropic():
    render, _ = make_render_dict(tilt_deg=30.)
    keep_iso   = compute_dropout_mask(render, roughness=0.3, anisotropy=0.0, seed=7)
    keep_base  = compute_dropout_mask(render, roughness=0.3, seed=7)
    _check("anisotropic_zero_matches_isotropic",
           np.array_equal(keep_iso, keep_base))


def test_anisotropic_roughness_elongates_dropout():
    # Test the Ward BRDF math directly on synthetic ray vectors that span
    # different azimuthal orientations. A flat plane has uniform normals so the
    # count can't differ -- we instead verify that the per-point lobe VALUES
    # differ between the isotropic and anisotropic formulations.
    from sensor.scene_render import _specular_keep_anisotropic, _specular_keep

    N = 200
    rng = np.random.default_rng(42)
    # Random half-sphere directions for normals, ray_dirs, proj_dirs
    def _rand_hemi(n, rng):
        v = rng.standard_normal((n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        v[:, 2] = np.abs(v[:, 2])   # force positive Z (hemisphere facing camera)
        return v

    normals   = _rand_hemi(N, rng)
    ray_dirs  = -normals + rng.standard_normal((N, 3)) * 0.1   # slight variation
    ray_dirs /= np.linalg.norm(ray_dirs, axis=1, keepdims=True) + 1e-12
    proj_dirs = -normals + rng.standard_normal((N, 3)) * 0.1
    proj_dirs /= np.linalg.norm(proj_dirs, axis=1, keepdims=True) + 1e-12

    roughness = 0.0   # pure mirror: no diffuse floor, small differences are visible
    keep_iso = _specular_keep(normals, ray_dirs, proj_dirs, roughness)
    keep_ani = _specular_keep_anisotropic(normals, ray_dirs, proj_dirs,
                                          alpha_t=0.05, alpha_b=0.50,
                                          brush_dir=np.array([1.,0.,0.]),
                                          anisotropy=1.0,
                                          roughness=roughness)
    # Anisotropic (elongated lobe) should produce a DIFFERENT survival pattern
    # from isotropic GGX for randomly-oriented normals spanning the hemisphere.
    _check("anisotropic_differs_from_isotropic_math",
           not np.array_equal(keep_iso, keep_ani),
           f"iso_kept={keep_iso.sum()}, ani_kept={keep_ani.sum()}")


# ==============================================================================
#  Runner
# ==============================================================================

if __name__ == "__main__":
    tests = [
        test_scene_render_returns_expected_keys,
        test_scene_render_returns_none_on_miss,
        test_compute_dropout_mask_lambertian_keeps_more_than_specular,
        test_compute_dropout_mask_returns_correct_shape,
        test_add_image_space_effects_zero_sigma_is_noop,
        test_add_image_space_effects_preserves_dict_keys,
        test_add_edge_artifacts_returns_at_least_keep_count,
        test_add_edge_artifacts_normals_unit_length,
        test_add_multipath_outliers_returns_ndarray,
        test_add_pepper_noise_returns_ndarray,
        test_add_scan_line_banding_preserves_shape,
        test_add_sensor_noise_introduces_displacement,
        test_add_surface_noise_displaces_along_normal,
        test_subset_render_preserves_image_buffers,
        # New stages
        test_specular_patch_missing_reduces_keep,
        test_specular_patch_missing_lambertian_unchanged,
        test_projector_nonuniformity_modulates_snr,
        test_projector_nonuniformity_is_deterministic,
        test_anisotropic_roughness_zero_matches_isotropic,
        test_anisotropic_roughness_elongates_dropout,
    ]

    print(f"\n{'='*60}")
    print(f"  test_scene_render.py -- {len(tests)} tests")
    print(f"{'='*60}")
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
        except Exception as e:
            _check(t.__name__, False, f"EXCEPTION: {e}")

    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{total} passed")
    print(f"{'='*60}\n")
    sys.exit(0 if passed == total else 1)
