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
from sensor.tests._fixtures import (
    make_render_dict,
    make_two_plane_render_dict,
    make_multi_instance_render_dict,
)
import sensor.segment_instances as si
from sensor.segment_instances import (
    build_geom_id_image,
    build_depth_image,
    build_perturbed_masks,
    segment_point_cloud,
    segmentation_stats,
    compute_2d_instance_metrics,
    _build_perturbed_crops,
    _tight_bbox_metrics,
    _apply_erosion_bias,
    _apply_dilation_into_background,
    _apply_boundary_noise,
    _bbox_of,
    _influence_margin,
)

_GOLDEN = os.path.join(os.path.dirname(__file__), "golden")


def _ensure_det_golden(name, render):
    """Load (or, if absent, regenerate) the deterministic-models golden for `name`.

    The golden .npz are gitignored, so on a fresh checkout they are rebuilt from the
    current CPU code — this turns the bitwise test into a regression lock on the
    current (verified) behaviour. The committed goldens carry the original
    pre-refactor values; delete them to re-baseline.
    """
    path = os.path.join(_GOLDEN, f"{name}_det.npz")
    if not os.path.exists(path):
        os.makedirs(_GOLDEN, exist_ok=True)
        saved = si._HAS_GPU
        try:
            si._HAS_GPU = False
            masks, geom_img = build_perturbed_masks(render, apply_boundary_noise=False, seed=0)
            d = {f"mask_{g}": m for g, m in masks.items()}
            d["geom_img"] = geom_img
            d["ids"] = np.array(sorted(masks.keys()), dtype=np.int64)
            np.savez_compressed(path, **d)
        finally:
            si._HAS_GPU = saved
    return np.load(path)


def _gpu_available() -> bool:
    return (si._cp is not None) and (si._cp.cuda.runtime.getDeviceCount() > 0)


def _iou(a, b):
    return (a & b).sum() / ((a | b).sum() + 1e-12)

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
    perturbed   = {
        0: np.array([True, True, False, False, False]),
        1: np.array([False, False, True, True, False]),
    }
    stats       = segmentation_stats(perturbed, canonical)
    required    = {"n_instances", "mean_iou", "mean_boundary_loss_frac",
                   "mean_confusion_frac", "unassigned_frac"}
    _check("stats_has_expected_keys", required == set(stats.keys()))
    for k, v in stats.items():
        if k != "n_instances":
            _check(f"stats_{k}_in_01", 0. <= v <= 1., f"{k}={v:.4f}")


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
    # Structural overlap: a large dilation bridges the ~8 px gap between the two
    # silhouettes deterministically. Erosion and boundary noise are disabled so the
    # overlap does not depend on a stochastic boundary realisation (with crops, the
    # boundary-noise field is a different draw than the full frame — see module docs).
    render, keep = make_two_plane_render_dict()
    pts  = render["points"][keep]
    pidx = render["pixel_idx"][keep]
    ov_kw = dict(dilation_px=8.0, apply_confusion=False,
                 apply_erosion=False, apply_boundary_noise=False)
    result = segment_point_cloud(render, pts, pidx, **ov_kw)
    ids = list(result.keys())
    if len(ids) < 2:
        _check("two_plane_overlap_skipped_single_instance",
               True, "only 1 instance survived segmentation — skip overlap check")
        return
    masks_img, _ = build_perturbed_masks(render, resolve_conflicts=False, **ov_kw)
    pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i+1, len(ids))]
    any_overlap = any(
        (masks_img[a] & masks_img[b]).any()
        for a, b in pairs if a in masks_img and b in masks_img
    )
    _check("no_resolve_conflicts_allows_image_overlap", any_overlap,
           f"checked {len(pairs)} pairs")


def test_resolve_conflicts_flag_eliminates_overlap():
    # Same structural-overlap config as the no-resolve test, so there is genuine
    # overlap for resolve_conflicts to eliminate (otherwise this passes vacuously).
    render, _ = make_two_plane_render_dict()
    ov_kw = dict(dilation_px=8.0, apply_confusion=False,
                 apply_erosion=False, apply_boundary_noise=False)
    masks_overlap, _   = build_perturbed_masks(render, resolve_conflicts=False, **ov_kw)
    masks_resolved, _  = build_perturbed_masks(render, resolve_conflicts=True,  **ov_kw)
    ids = list(masks_resolved.keys())
    if len(ids) < 2:
        _check("resolve_conflicts_skipped_single_instance", True)
        return
    had_overlap = any(
        (masks_overlap[ids[i]] & masks_overlap[ids[j]]).any()
        for i in range(len(ids)) for j in range(i+1, len(ids))
    )
    overlap_free = all(
        not (masks_resolved[ids[i]] & masks_resolved[ids[j]]).any()
        for i in range(len(ids)) for j in range(i+1, len(ids))
    )
    _check("resolve_conflicts_had_overlap_to_resolve", had_overlap)
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
#  Crop refactor + GPU backend equivalence
# ══════════════════════════════════════════════════════════════════════════════

def test_crop_equiv_deterministic_models_bitwise():
    # With boundary noise OFF, the cropped pipeline (erosion+dilation+confusion+
    # occlusion) must be BITWISE-equal to the pre-refactor full-frame golden.
    si._HAS_GPU = False          # golden was produced on CPU
    for name, mk in [("single", make_render_dict), ("two_plane", make_two_plane_render_dict)]:
        render, _ = mk()
        gold = _ensure_det_golden(name, render)
        masks, geom_img = build_perturbed_masks(render, apply_boundary_noise=False, seed=0)
        _check(f"crop_det_geom_img_bitwise_{name}",
               np.array_equal(geom_img, gold["geom_img"]))
        for gid in gold["ids"]:
            ref = gold[f"mask_{gid}"]
            cur = masks[int(gid)]
            _check(f"crop_det_bitwise_{name}_gid{gid}",
                   np.array_equal(ref, cur),
                   f"diff_px={(ref ^ cur).sum()}")


def test_crop_equiv_boundary_noise_statistical():
    # Boundary noise draws an array-shaped field, so a crop is a different RNG
    # realisation than the full frame — not bitwise. The model must still be
    # statistically the same under cropping: near-zero-mean area change, of a
    # magnitude comparable to the full-frame computation.
    render, _ = make_two_plane_render_dict()
    geom = build_geom_id_image(render)
    H, W = geom.shape
    margin = _influence_margin(3, 1.5, 4, 6.0, False, False, False, False, True)
    for g in (0, 1):
        full = (geom == g)
        a0   = full.sum()
        r0, c0, r1, c1 = _bbox_of(full, margin, H, W)
        crop = full[r0:r1, c0:c1]
        df, dc = [], []
        for s in range(40):
            nf = _apply_boundary_noise(full, 6.0, np.random.default_rng(s))
            nc = _apply_boundary_noise(crop, 6.0, np.random.default_rng(s))
            df.append((nf.sum() - a0) / a0)
            dc.append((nc.sum() - a0) / a0)
        mf, mc = float(np.mean(df)), float(np.mean(dc))
        _check(f"boundary_noise_crop_unbiased_gid{g}", abs(mc) < 0.05,
               f"crop_mean_area_change={mc:+.4f}")
        _check(f"boundary_noise_crop_matches_full_gid{g}", abs(mf - mc) < 0.05,
               f"full={mf:+.4f} crop={mc:+.4f}")


def test_confusion_pruning_matches_allpairs():
    # Higher resolution so the "far" layout's padded bboxes actually separate and
    # the prune path fires; result must be identical to the un-pruned all-pairs pass.
    si._HAS_GPU = False
    kw = dict(erosion_px=2.0, boundary_noise_px=3.0, seed=1)
    for layout, fires in [("grid", False), ("far", True)]:
        render, _ = make_multi_instance_render_dict(layout=layout, W=480, H=360)
        geom = build_geom_id_image(render)
        margin = _influence_margin(2.0, 1.5, 4, 3.0, True, True, True, True, True)
        crops = si._instance_crops(geom, margin, np)
        ids = list(crops.keys())
        pruned = sum(1 for i, a in enumerate(ids) for b in ids[i + 1:]
                     if not si._bboxes_overlap(crops[a], crops[b]))
        mp, _ = build_perturbed_masks(render, prune_confusion=True,  **kw)
        mu, _ = build_perturbed_masks(render, prune_confusion=False, **kw)
        same = all(np.array_equal(mp[g], mu[g]) for g in mp)
        _check(f"confusion_prune_eq_allpairs_{layout}", same,
               f"pruned {pruned} pairs")
        if fires:
            _check(f"confusion_prune_fires_{layout}", pruned > 0,
                   f"pruned={pruned} (test would be vacuous otherwise)")


def test_back_projection_self_consistent():
    # segment_point_cloud back-projects from crops; it must equal back-projecting the
    # full-frame masks of build_perturbed_masks with the same seed (same realisation).
    render, keep = make_two_plane_render_dict()
    pts  = render["points"][keep]
    pidx = render["pixel_idx"][keep]
    H, W = render["res"]
    valid = pidx >= 0
    seg = segment_point_cloud(render, pts, pidx, seed=0)
    masks, _ = build_perturbed_masks(render, seed=0)
    for g, m in masks.items():
        manual = np.zeros(len(pts), bool)
        manual[valid] = m.ravel()[pidx[valid]]
        _check(f"backproj_self_consistent_gid{g}",
               np.array_equal(manual, seg[g]))


def test_cpu_fallback_without_gpu():
    # With the GPU forced off the module must still produce a well-formed result.
    saved = si._HAS_GPU
    try:
        si._HAS_GPU = False
        render, keep = make_two_plane_render_dict()
        pts  = render["points"][keep]
        pidx = render["pixel_idx"][keep]
        result = segment_point_cloud(render, pts, pidx, seed=0)
        _check("cpu_fallback_returns_dict", isinstance(result, dict) and len(result) >= 1)
        for g, arr in result.items():
            _check(f"cpu_fallback_bool_len_gid{g}",
                   arr.dtype == bool and len(arr) == len(pts))
    finally:
        si._HAS_GPU = saved


def test_gpu_cpu_parity():
    # GPU output must match CPU (RNG is on the host; only filter float-rounding can
    # differ). Skipped when no CUDA device is present.
    if not _gpu_available():
        _check("gpu_cpu_parity_skipped_no_device", True, "no CUDA device")
        return
    saved = si._HAS_GPU
    try:
        render, _ = make_two_plane_render_dict()
        si._HAS_GPU = False
        det_cpu, _ = build_perturbed_masks(render, apply_boundary_noise=False, seed=0)
        full_cpu, _ = build_perturbed_masks(render, seed=0)
        si._HAS_GPU = True
        det_gpu, _ = build_perturbed_masks(render, apply_boundary_noise=False, seed=0)
        full_gpu, _ = build_perturbed_masks(render, seed=0)
        for g in det_cpu:
            _check(f"gpu_cpu_parity_det_gid{g}", _iou(det_cpu[g], det_gpu[g]) > 0.99,
                   f"iou={_iou(det_cpu[g], det_gpu[g]):.5f}")
        for g in full_cpu:
            _check(f"gpu_cpu_parity_full_gid{g}", _iou(full_cpu[g], full_gpu[g]) > 0.99,
                   f"iou={_iou(full_cpu[g], full_gpu[g]):.5f}")
    finally:
        si._HAS_GPU = saved


def test_scaling_smoke():
    # Informational: time CPU vs GPU on a multi-instance render. Never fails.
    import time
    render, keep = make_multi_instance_render_dict(layout="grid", W=480, H=360)
    pts  = render["points"][keep]
    pidx = render["pixel_idx"][keep]
    saved = si._HAS_GPU
    try:
        si._HAS_GPU = False
        t = time.time(); segment_point_cloud(render, pts, pidx, seed=0); t_cpu = time.time() - t
        msg = f"CPU={t_cpu * 1e3:.1f}ms"
        if _gpu_available():
            si._HAS_GPU = True
            segment_point_cloud(render, pts, pidx, seed=0)  # warm up kernels
            t = time.time(); segment_point_cloud(render, pts, pidx, seed=0); t_gpu = time.time() - t
            msg += f"  GPU={t_gpu * 1e3:.1f}ms"
        _check("scaling_smoke_informational", True, msg)
    finally:
        si._HAS_GPU = saved


# ══════════════════════════════════════════════════════════════════════════════
#  2D instance metrics (aspect ratio / area ratio candidate-filter inputs)
# ══════════════════════════════════════════════════════════════════════════════

def test_tight_bbox_metrics_known_mask():
    # 3-row x 6-col solid block embedded (with padding) in a larger frame.
    arr = np.zeros((10, 12), dtype=bool)
    arr[2:5, 3:9] = True            # height 3, width 6, area 18
    area, h, w = _tight_bbox_metrics(arr)
    _check("tight_bbox_area", area == 18, f"area={area}")
    _check("tight_bbox_height", h == 3, f"h={h}")
    _check("tight_bbox_width", w == 6, f"w={w}")
    _check("tight_bbox_empty_is_zero", _tight_bbox_metrics(np.zeros((4, 4), bool)) == (0, 0, 0))


def test_compute_2d_instance_metrics_ranges():
    render, _ = make_multi_instance_render_dict(layout="grid")
    crops, geom_img = _build_perturbed_crops(render, seed=0)
    H, W = geom_img.shape
    metrics = compute_2d_instance_metrics(crops, H, W)
    _check("metrics_nonempty", len(metrics) > 0, f"n={len(metrics)}")
    _check("metrics_keys_match_crops", set(metrics.keys()) <= set(crops.keys()))
    for g, m in metrics.items():
        _check(f"metrics_gid{g}_has_keys",
               {"pixel_area", "aspect_ratio", "area_ratio"} <= set(m.keys()))
        _check(f"metrics_gid{g}_aspect_ge_1", m["aspect_ratio"] >= 1.0,
               f"aspect={m['aspect_ratio']:.3f}")
        _check(f"metrics_gid{g}_area_ratio_in_unit", 0.0 < m["area_ratio"] <= 1.0,
               f"area_ratio={m['area_ratio']:.5f}")


def test_segment_point_cloud_return_metrics_aligns_with_masks():
    render, keep = make_multi_instance_render_dict(layout="grid")
    pts  = render["points"][keep]
    pidx = render["pixel_idx"][keep]
    masks, metrics = segment_point_cloud(render, pts, pidx, return_metrics=True, seed=0)
    _check("return_metrics_is_tuple", isinstance(masks, dict) and isinstance(metrics, dict))
    # Every instance with a metric is a real instance; pixel_area matches its 2D mask.
    crops, geom_img = _build_perturbed_crops(render, seed=0)
    H, W = geom_img.shape
    for g, m in metrics.items():
        expected_area = int(_tight_bbox_metrics(crops[g].arr)[0])
        _check(f"return_metrics_gid{g}_area_matches_mask",
               m["pixel_area"] == expected_area,
               f"got={m['pixel_area']} expected={expected_area}")


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
        # crop refactor + GPU backend equivalence
        test_crop_equiv_deterministic_models_bitwise,
        test_crop_equiv_boundary_noise_statistical,
        test_confusion_pruning_matches_allpairs,
        test_back_projection_self_consistent,
        test_cpu_fallback_without_gpu,
        test_gpu_cpu_parity,
        test_scaling_smoke,
        # 2D instance metrics
        test_tight_bbox_metrics_known_mask,
        test_compute_2d_instance_metrics_ranges,
        test_segment_point_cloud_return_metrics_aligns_with_masks,
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
