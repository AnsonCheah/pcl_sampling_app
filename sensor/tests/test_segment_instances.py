"""
test_segment_instances.py — Regression + multi-label tests for segment_instances.py.

Run from project root:
    python sensor/tests/test_segment_instances.py
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from sensor.tests._fixtures import make_render_dict, make_two_plane_render_dict
from sensor.segment_instances import (
    build_geom_id_image,
    build_depth_image,
    build_perturbed_masks,
    segment_point_cloud,
    segmentation_stats,
    _apply_erosion_bias,
    _apply_dilation_into_background,
    _apply_boundary_noise,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = []

def _check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    _results.append((name, cond))


# ══════════════════════════════════════════════════════════════════════════════
#  Existing function regression tests
# ══════════════════════════════════════════════════════════════════════════════

def test_build_geom_id_image_shape_and_background():
    render, _ = make_render_dict()
    H, W = render["res"]
    img  = build_geom_id_image(render)
    _check("geom_id_image_shape", img.shape == (H, W))
    _check("geom_id_image_dtype", img.dtype == np.int32)
    # Background pixels are -1
    bg_count = (img == -1).sum()
    _check("geom_id_image_has_background", bg_count > 0,
           f"bg_pixels={bg_count}")
    # All visible-point pixels hold a valid (≥0) geom_id
    valid_ids = set(render["geom_ids"].tolist())
    img_ids   = set(img[img >= 0].tolist())
    _check("geom_id_image_ids_match_render", img_ids.issubset(valid_ids),
           f"img_ids={img_ids}, render_ids={valid_ids}")


def test_build_depth_image_nan_at_background():
    render, _ = make_render_dict()
    H, W = render["res"]
    depth = build_depth_image(render)
    _check("depth_image_shape", depth.shape == (H, W))
    # Background pixels are NaN
    visible_flat = np.full(H * W, False)
    visible_flat[render["pixel_idx"]] = True
    bg_flat = ~visible_flat
    _check("depth_image_nan_at_background",
           np.all(np.isnan(depth.ravel()[bg_flat])))
    # Visible pixels are finite
    _check("depth_image_finite_at_visible",
           np.all(np.isfinite(depth.ravel()[render["pixel_idx"]])))


def test_apply_erosion_bias_is_subset():
    render, _ = make_render_dict()
    rng       = np.random.default_rng(0)
    geom_img  = build_geom_id_image(render)
    gid       = int(np.unique(render["geom_ids"])[0])
    mask_orig = (geom_img == gid)
    mask_eros = _apply_erosion_bias(mask_orig, erosion_px=3.0, rng=rng)
    _check("erosion_is_subset",
           mask_eros.sum() <= mask_orig.sum(),
           f"orig={mask_orig.sum()}, eroded={mask_eros.sum()}")


def test_apply_dilation_into_background_is_superset():
    render, _ = make_render_dict()
    rng       = np.random.default_rng(0)
    geom_img  = build_geom_id_image(render)
    gid       = int(np.unique(render["geom_ids"])[0])
    mask_orig = (geom_img == gid)
    mask_dil  = _apply_dilation_into_background(mask_orig, geom_img, dilation_px=2.0, rng=rng)
    # All original pixels must remain True
    _check("dilation_keeps_original_pixels",
           np.all(mask_orig[mask_orig] == mask_dil[mask_orig]))
    # Mask can only grow into background
    _check("dilation_grows_into_background",
           mask_dil.sum() >= mask_orig.sum(),
           f"orig={mask_orig.sum()}, dilated={mask_dil.sum()}")


def test_apply_boundary_noise_preserves_approximate_area():
    render, _ = make_render_dict()
    rng       = np.random.default_rng(0)
    geom_img  = build_geom_id_image(render)
    gid       = int(np.unique(render["geom_ids"])[0])
    mask_orig = (geom_img == gid)
    mask_noisy = _apply_boundary_noise(mask_orig, noise_px=6.0, rng=rng)
    orig_area  = mask_orig.sum()
    noisy_area = mask_noisy.sum()
    frac_change = abs(noisy_area - orig_area) / (orig_area + 1e-12)
    _check("boundary_noise_area_within_10pct",
           frac_change < 0.10,
           f"change={frac_change:.3f}")


def test_segmentation_stats_returns_expected_keys():
    canonical   = np.array([0, 0, 1, 1, -1], dtype=np.int32)
    perturbed   = np.array([0, 0, 1, 1, -1], dtype=np.int32)
    stats       = segmentation_stats(perturbed, canonical)
    required    = {"n_instances", "mean_iou", "mean_boundary_loss_frac",
                   "mean_confusion_frac", "unassigned_frac"}
    _check("stats_has_expected_keys", required == set(stats.keys()))
    for k, v in stats.items():
        if k != "n_instances":
            _check(f"stats_{k}_in_01", 0. <= v <= 1., f"{k}={v:.4f}")


def test_segmentation_stats_perfect_match():
    labels = np.array([0, 0, 0, 1, 1, 1, -1], dtype=np.int32)
    stats  = segmentation_stats(labels, labels)
    _check("stats_perfect_iou",
           abs(stats["mean_iou"] - 1.0) < 1e-6,
           f"iou={stats['mean_iou']:.4f}")
    _check("stats_perfect_no_confusion",
           stats["mean_confusion_frac"] < 1e-6,
           f"confusion={stats['mean_confusion_frac']:.4f}")


def test_build_perturbed_masks_toggles_work():
    render, _ = make_render_dict()
    # With all effects off, masks should closely match the canonical geom_id image
    masks, geom_img = build_perturbed_masks(
        render,
        apply_erosion=False,
        apply_dilation=False,
        apply_confusion=False,
        apply_occlusion_loss=False,
        apply_boundary_noise=False,
    )
    for gid, m in masks.items():
        canon  = (geom_img == gid)
        iou    = (m & canon).sum() / ((m | canon).sum() + 1e-12)
        _check(f"perturbed_mask_noop_iou_gid{gid}",
               iou > 0.90, f"iou={iou:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-label segmentation tests
# ══════════════════════════════════════════════════════════════════════════════

def test_segment_returns_dict():
    render, keep = make_render_dict()
    pts  = render["points"][keep]
    pidx = render["pixel_idx"][keep]
    result = segment_point_cloud(render, pts, pidx)
    _check("segment_returns_dict", isinstance(result, dict))
    _check("segment_keys_are_int", all(isinstance(k, (int, np.integer)) for k in result.keys()))
    for g, arr in result.items():
        _check(f"segment_value_gid{g}_is_bool_array",
               isinstance(arr, np.ndarray) and arr.dtype == bool and arr.ndim == 1)
        _check(f"segment_value_gid{g}_length",
               len(arr) == len(pts))


def test_no_resolve_conflicts_allows_overlap():
    render, keep = make_two_plane_render_dict()
    pts  = render["points"][keep]
    pidx = render["pixel_idx"][keep]
    result = segment_point_cloud(render, pts, pidx,
                                 dilation_px=4.0,        # generous dilation → forces overlap
                                 apply_confusion=False)   # isolate the dilation effect
    ids = list(result.keys())
    if len(ids) < 2:
        _check("two_plane_overlap_skipped_single_instance",
               True, "only 1 instance survived segmentation — skip overlap check")
        return
    # At least one pair should have overlapping pixels in image space
    masks_img, _ = build_perturbed_masks(render, dilation_px=4.0,
                                         apply_confusion=False,
                                         resolve_conflicts=False)
    pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i+1, len(ids))]
    any_overlap = any(
        (masks_img[a] & masks_img[b]).any()
        for a, b in pairs if a in masks_img and b in masks_img
    )
    _check("no_resolve_conflicts_allows_image_overlap", any_overlap,
           f"checked {len(pairs)} pairs")


def test_resolve_conflicts_flag_eliminates_overlap():
    render, _ = make_two_plane_render_dict()
    masks_resolved, _ = build_perturbed_masks(render, dilation_px=4.0,
                                              apply_confusion=False,
                                              resolve_conflicts=True)
    ids = list(masks_resolved.keys())
    if len(ids) < 2:
        _check("resolve_conflicts_skipped_single_instance", True)
        return
    overlap_free = all(
        not (masks_resolved[ids[i]] & masks_resolved[ids[j]]).any()
        for i in range(len(ids)) for j in range(i+1, len(ids))
    )
    _check("resolve_conflicts_eliminates_overlap", overlap_free)


def test_injected_points_are_false_in_all_masks():
    render, keep = make_render_dict()
    pts  = render["points"][keep]
    # Append 10 injected points with pixel_idx = -1
    n_injected = 10
    injected_pts = np.zeros((n_injected, 3), dtype=pts.dtype)
    all_pts  = np.vstack([pts, injected_pts])
    all_pidx = np.concatenate([render["pixel_idx"][keep],
                                np.full(n_injected, -1, dtype=np.int64)])
    result = segment_point_cloud(render, all_pts, all_pidx)
    for g, arr in result.items():
        injected_vals = arr[-n_injected:]
        _check(f"injected_false_in_mask_gid{g}",
               not injected_vals.any(),
               f"injected_true_count={injected_vals.sum()}")


def test_single_instance_no_overlap():
    render, keep = make_render_dict()
    pts  = render["points"][keep]
    pidx = render["pixel_idx"][keep]
    result = segment_point_cloud(render, pts, pidx)
    _check("single_instance_one_key", len(result) == 1,
           f"keys={list(result.keys())}")
    # With a single instance, overlap is impossible
    gid  = list(result.keys())[0]
    mask = result[gid]
    _check("single_instance_some_points_assigned", mask.any(),
           f"assigned={mask.sum()}")


# ══════════════════════════════════════════════════════════════════════════════
#  segmentation_stats — dict-format path
# ══════════════════════════════════════════════════════════════════════════════

def test_segmentation_stats_dict_perfect_match():
    # {geom_id: bool_mask} with no errors → iou=1, confusion=0, boundary_loss=0
    canonical  = np.array([0, 0, 0, 1, 1, 1, -1], dtype=np.int32)
    perturbed  = {
        0: np.array([True, True, True, False, False, False, False]),
        1: np.array([False, False, False, True, True, True, False]),
    }
    stats = segmentation_stats(perturbed, canonical)
    _check("stats_dict_perfect_iou",
           abs(stats["mean_iou"] - 1.0) < 1e-6,
           f"iou={stats['mean_iou']:.4f}")
    _check("stats_dict_perfect_no_confusion",
           stats["mean_confusion_frac"] < 1e-6,
           f"confusion={stats['mean_confusion_frac']:.4f}")
    _check("stats_dict_perfect_no_loss",
           stats["mean_boundary_loss_frac"] < 1e-6,
           f"loss={stats['mean_boundary_loss_frac']:.4f}")
    _check("stats_dict_perfect_no_unassigned",
           stats["unassigned_frac"] < 1e-6,
           f"unassigned={stats['unassigned_frac']:.4f}")


def test_segmentation_stats_dict_confusion():
    # Point 1 (canonical inst 0) absorbed into inst 1 → confusion detected
    canonical  = np.array([0, 0, 1, 1], dtype=np.int32)
    perturbed  = {
        0: np.array([True,  False, False, False]),
        1: np.array([False, True,  True,  True ]),
    }
    stats = segmentation_stats(perturbed, canonical)
    # Inst 0 confused frac = 1/2 = 0.5; inst 1 = 0 → mean = 0.25
    _check("stats_dict_confusion_detected",
           stats["mean_confusion_frac"] > 0.0,
           f"confusion={stats['mean_confusion_frac']:.4f}")
    _check("stats_dict_no_unassigned",
           stats["unassigned_frac"] < 1e-6,
           f"unassigned={stats['unassigned_frac']:.4f}")


def test_segmentation_stats_dict_unassigned():
    # Point 1 (canonical inst 0) appears in no mask → boundary loss + unassigned
    canonical  = np.array([0, 0, 1, 1], dtype=np.int32)
    perturbed  = {
        0: np.array([True,  False, False, False]),
        1: np.array([False, False, True,  True ]),
    }
    stats = segmentation_stats(perturbed, canonical)
    _check("stats_dict_boundary_loss_detected",
           stats["mean_boundary_loss_frac"] > 0.0,
           f"loss={stats['mean_boundary_loss_frac']:.4f}")
    _check("stats_dict_unassigned_detected",
           stats["unassigned_frac"] > 0.0,
           f"unassigned={stats['unassigned_frac']:.4f}")
    _check("stats_dict_no_confusion",
           stats["mean_confusion_frac"] < 1e-6,
           f"confusion={stats['mean_confusion_frac']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        # Existing function regression
        test_build_geom_id_image_shape_and_background,
        test_build_depth_image_nan_at_background,
        test_apply_erosion_bias_is_subset,
        test_apply_dilation_into_background_is_superset,
        test_apply_boundary_noise_preserves_approximate_area,
        test_segmentation_stats_returns_expected_keys,
        test_segmentation_stats_perfect_match,
        test_build_perturbed_masks_toggles_work,
        # Multi-label
        test_segment_returns_dict,
        test_no_resolve_conflicts_allows_overlap,
        test_resolve_conflicts_flag_eliminates_overlap,
        test_injected_points_are_false_in_all_masks,
        test_single_instance_no_overlap,
        # segmentation_stats dict path
        test_segmentation_stats_dict_perfect_match,
        test_segmentation_stats_dict_confusion,
        test_segmentation_stats_dict_unassigned,
    ]

    print(f"\n{'='*60}")
    print(f"  test_segment_instances.py — {len(tests)} tests")
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
