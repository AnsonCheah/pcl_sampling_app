"""
segment_instances.py
────────────────────
Realistic 2D instance segmentation simulation for synthetic bin-picking scenes.

Given a render dict from scene_render.py (which contains a canonical geom_id
image), this module produces per-point instance labels that replicate the
boundary errors of a real 2D segmentation network.

Five physical error sources are modelled in image space before back-projection:

  1. Global erosion bias        — mask contour sits 2-5 px inside true silhouette
  2. Dilation into background   — mask bleeds 1-3 px outward into bin floor/walls
  3. Inter-object confusion     — boundary pixels near adjacent objects are
                                  probabilistically re-assigned; confusion rate
                                  scales with exp(-depth_gap / σ_gap)
  4. Occlusion-edge gap         — the occluded object's mask is over-eroded where
                                  it disappears behind another instance
  5. Soft-mask threshold noise  — Gaussian-correlated pixel-level boundary noise
                                  at the scale of a network's boundary receptive
                                  field (~8-16 px)

Typical call
────────────
    from segment_instances import segment_point_cloud

    labels = segment_point_cloud(render, points, pixel_idx)
    # labels : (N,) int  -1 = unassigned background, ≥0 = instance id

    for inst_id in np.unique(labels[labels >= 0]):
        inst_pts = points[labels == inst_id]
        ...

All errors are applied in image space on the geom_id image, then the resulting
label map is back-projected to the surviving point cloud via pixel_idx.

Performance
───────────
Each instance is processed on its own padded **bounding-box crop** rather than the
full (H, W) frame, and inter-object confusion only touches pairs whose padded
bboxes overlap. The five error models are local operators, so a crop padded by the
operator support radius produces results identical to the full-frame computation
(see `_influence_margin`). When CuPy + a CUDA device are available the crop ops run
on the GPU (`cupyx.scipy.ndimage`); otherwise they fall back to SciPy/NumPy
transparently — there is no hard GPU dependency.
"""

import numpy as np
import scipy.ndimage as _scipy_ndi
from dataclasses import dataclass
import time
import sys
from rich import print as rp

# ── Optional GPU backend ────────────────────────────────────────────────────────
# Resolved lazily via _backend() so tests can monkeypatch _HAS_GPU. The morphology
# / filter ops below are routed through the selected `ndi` module and array ops
# through `xp`, so the same code runs on CuPy or NumPy/SciPy unchanged.
try:
    import cupy as _cp
    import cupyx.scipy.ndimage as _cupy_ndi
    _HAS_GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _cp = None
    _cupy_ndi = None
    _HAS_GPU = False


def _backend():
    """Return (xp, ndi) — CuPy/cupyx when a device is present, else NumPy/SciPy."""
    if _HAS_GPU and _cp is not None:
        return _cp, _cupy_ndi
    return np, _scipy_ndi


def _to_cpu(arr):
    """Bring an array back to host NumPy (no-op for NumPy input)."""
    if _cp is not None and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return np.asarray(arr)


# ══════════════════════════════════════════════════════════════════════════════
#  Crop representation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Crop:
    """A per-instance mask stored as a small sub-image plus its top-left offset.

    `arr` is the (h, w) boolean mask for the instance; the full-frame pixel (r, c)
    corresponds to local index (r - r0, c - c0). Pixels outside [r0:r1, c0:c1] are
    implicitly False. `arr` may be a NumPy or CuPy array.
    """
    r0: int
    c0: int
    arr: "np.ndarray"

    @property
    def r1(self) -> int:
        return self.r0 + self.arr.shape[0]

    @property
    def c1(self) -> int:
        return self.c0 + self.arr.shape[1]

    def any(self) -> bool:
        return bool(self.arr.any())


def _bbox_of(full_bool, margin: int, H: int, W: int):
    """(r0, c0, r1, c1) bounding box of True pixels, padded by `margin`, clipped to
    the image. Returns None if the mask is empty. `full_bool` is a NumPy array."""
    rows = np.any(full_bool, axis=1)
    cols = np.any(full_bool, axis=0)
    if not rows.any():
        return None
    r0 = int(np.argmax(rows)); r1 = int(H - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols)); c1 = int(W - np.argmax(cols[::-1]))
    r0 = max(0, r0 - margin); c0 = max(0, c0 - margin)
    r1 = min(H, r1 + margin); c1 = min(W, c1 + margin)
    return r0, c0, r1, c1


def _expand_to_full(crop: _Crop, H: int, W: int) -> np.ndarray:
    """Re-embed a crop into a full (H, W) NumPy bool frame (background False)."""
    out = np.zeros((H, W), dtype=bool)
    out[crop.r0:crop.r1, crop.c0:crop.c1] = _to_cpu(crop.arr)
    return out


def _tight_bbox_metrics(arr) -> "tuple[int, int, int]":
    """
    (pixel_area, height, width) of the TRUE pixels in a 2D bool array.

    `arr` is a NumPy or CuPy array (e.g. a padded _Crop.arr); the height/width are
    the tight extent of the True region, ignoring the surrounding padding. Returns
    (0, 0, 0) for an empty mask. Uses the same np.any(...argmax...) pattern as
    `_bbox_of`, on the host.
    """
    a = _to_cpu(arr)
    rows = np.any(a, axis=1)
    cols = np.any(a, axis=0)
    if not rows.any():
        return 0, 0, 0
    r0 = int(np.argmax(rows)); r1 = int(len(rows) - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols)); c1 = int(len(cols) - np.argmax(cols[::-1]))
    return int(a.sum()), r1 - r0, c1 - c0


def _metrics_from_mask(arr, H: int, W: int) -> "dict | None":
    """
    {pixel_area, aspect_ratio, area_ratio} for a single 2D bool mask, or None if
    the mask is empty. `arr` may be a padded crop array or a full-frame mask —
    only the tight extent of the True region matters. area_ratio is normalised by
    the full image (H * W).
    """
    area, h, w = _tight_bbox_metrics(arr)
    if area == 0:
        return None
    long_side  = max(h, w)
    short_side = max(min(h, w), 1)
    return {
        "pixel_area":   area,
        "aspect_ratio": long_side / short_side,
        "area_ratio":   area / float(H * W),
    }


def compute_2d_instance_metrics(
    crops: "dict[int, _Crop]", H: int, W: int,
) -> "dict[int, dict]":
    """
    Per-instance 2D mask statistics for the (perturbed) crops, measured in image
    space exactly as a real 2D segmentation network's post-filter would see them.

    Returns
    -------
    dict[geom_id] -> {
        "pixel_area"   : int    count of True pixels in the mask,
        "aspect_ratio" : float  max(h, w) / max(min(h, w), 1)  — elongation (>= 1),
                                 orientation-agnostic,
        "area_ratio"   : float  pixel_area / (H * W)  — fraction of the full image.
    }
    Empty masks are skipped.
    """
    out: "dict[int, dict]" = {}
    for g, c in crops.items():
        m = _metrics_from_mask(c.arr, H, W)
        if m is not None:
            out[g] = m
    return out


def _influence_margin(
    erosion_px: float, dilation_px: float, confusion_boundary_px: int,
    boundary_noise_px: float,
    apply_erosion: bool, apply_dilation: bool, apply_confusion: bool,
    apply_occlusion_loss: bool, apply_boundary_noise: bool,
) -> int:
    """Crop padding (px) guaranteeing the cropped result equals the full-frame
    result. Each model is a local operator; the margin is the sum of every enabled
    model's outward influence radius so the operator support never reaches the crop
    edge. The erosion/boundary-noise Gaussians use SciPy's default truncate=4.0, so
    their support is 4σ (σ_erosion = 1.3·erosion_px/0.42 at max jitter)."""
    m = 1
    if apply_erosion:
        sigma_max = 1.3 * erosion_px / 0.42
        m += int(np.ceil(4.0 * sigma_max))
    if apply_dilation:
        m += int(np.ceil(1.3 * dilation_px)) + 1
    if apply_confusion:
        m += int(confusion_boundary_px) + 1
    if apply_occlusion_loss:
        m += 2
    if apply_boundary_noise:
        m += int(np.ceil(4.0 * boundary_noise_px))
    return int(m)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Build the canonical geom_id / depth images
# ══════════════════════════════════════════════════════════════════════════════

def build_geom_id_image(render: dict) -> np.ndarray:
    """
    (H, W) int32 image where each pixel holds the geom_id of the visible surface
    point that maps to it. Pixels with no visible surface point are set to -1.

    Built from render["pixel_idx"] and render["geom_ids"], which are the
    projector-illuminated, shadow-free points from scene_render().

    This is the canonical ground-truth segmentation mask before any perturbation.
    """
    H, W = render["res"]
    img  = np.full(H * W, -1, dtype=np.int32)
    img[render["pixel_idx"]] = render["geom_ids"].astype(np.int32)
    return img.reshape(H, W)


def build_depth_image(render: dict) -> np.ndarray:
    """
    (H, W) float32 visible-surface depth image (NaN for background pixels).
    Used to compute inter-object depth gaps at shared boundaries.
    """
    H, W = render["res"]
    img  = np.full(H * W, np.nan, dtype=np.float32)
    img[render["pixel_idx"]] = render["t_hit"].astype(np.float32)
    return img.reshape(H, W)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Per-instance crop helpers
# ══════════════════════════════════════════════════════════════════════════════

def _instance_crops(geom_img: np.ndarray, margin: int, xp) -> "dict[int, _Crop]":
    """
    Returns {geom_id: _Crop} for every object in the scene, each as a padded
    bounding-box crop on backend `xp`. Background (-1) is excluded. Ids are in
    ascending order (np.unique) so RNG draw order matches the full-frame reference.
    """
    H, W = geom_img.shape
    ids  = np.unique(geom_img)
    crops: "dict[int, _Crop]" = {}
    for g in ids:
        if g < 0:
            continue
        full = (geom_img == g)
        bbox = _bbox_of(full, margin, H, W)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        sub = full[r0:r1, c0:c1]
        crops[int(g)] = _Crop(r0, c0, xp.asarray(sub))
    return crops


def _boundary_map(mask, ndi, xp, width: int = 1):
    """
    bool — True where `mask` has a boundary within `width` pixels.
    Computed as: mask AND NOT eroded(mask, width).
    """
    struct = xp.ones((2 * width + 1, 2 * width + 1), bool)
    return mask & ~ndi.binary_erosion(mask, structure=struct)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Five error models (operate on a single instance's crop array)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_erosion_bias(mask, erosion_px: float, rng, *, xp=np, ndi=_scipy_ndi):
    """
    Model 1: Global erosion bias.

    The network is conservative at boundaries — the predicted mask contour
    sits slightly inside the true object silhouette. Modelled as a soft erosion:
    Gaussian blur the binary mask, then threshold above 0.5.

        eroded = gaussian_blur(mask, σ) > 0.5

    Using a Gaussian blur instead of morphological erosion gives smooth,
    rounded boundary errors that match network output statistics. The effective
    erosion radius is ≈ 0.42 · σ (for a threshold of 0.5).

    erosion_px : target erosion radius in pixels. σ = erosion_px / 0.42.
    A small random jitter (±30%) is added per call to avoid identical masks
    across frames.
    """
    jitter = float(rng.uniform(0.7, 1.3))
    sigma  = jitter * erosion_px / 0.42
    soft   = ndi.gaussian_filter(mask.astype(xp.float32), sigma=sigma)
    return soft > 0.50


def _apply_dilation_into_background(mask, geom_img, dilation_px: float, rng,
                                    *, xp=np, ndi=_scipy_ndi):
    """
    Model 2: Dilation into background.

    At edges where the object meets the bin floor or walls (background pixels),
    the mask bleeds outward slightly. This models the network including
    low-confidence boundary pixels that technically belong to the background.

    Only pixels that are currently background (geom_img == -1) can be added;
    we do not steal pixels from other objects here (that is handled by model 3).

    `geom_img` must be the geom_id image cropped to the same window as `mask`.
    """
    jitter   = float(rng.uniform(0.7, 1.3))
    radius   = max(1, int(round(jitter * dilation_px)))
    struct   = _disk_struct(radius, xp)
    dilated  = ndi.binary_dilation(mask, structure=struct)
    bg_only  = geom_img == -1
    # New pixels = dilated AND background (don't touch other objects yet)
    return mask | (dilated & bg_only)


def _apply_boundary_noise(mask, noise_px: float, rng, *, xp=np, ndi=_scipy_ndi):
    """
    Model 5: Soft-mask threshold noise.

    The raw network output is a probability map thresholded at 0.5. Near the
    boundary, the probability is close to 0.5, making pixel-level membership
    noisy. This noise is spatially correlated at the scale of the network's
    boundary receptive field (noise_px, typically 4-16 px).

    Modelled as: perturb the mask by adding Gaussian-correlated noise to the
    signed distance transform, then re-threshold at zero.

        sdt    = signed_distance_transform(mask)   (positive inside, negative outside)
        sdt   += gaussian_noise(σ=noise_amplitude, spatial_σ=noise_px)
        result = sdt > 0

    The noise amplitude is set to noise_px / 3 so that the 3σ range spans
    noise_px pixels, matching the boundary region where the mask is uncertain.
    """
    dist_in  = ndi.distance_transform_edt(mask).astype(xp.float32)
    dist_out = ndi.distance_transform_edt(~mask).astype(xp.float32)
    sdt      = dist_in - dist_out

    # Draw the white field on the host (NumPy) so results are reproducible and
    # backend-independent, then move to the compute device.
    white     = xp.asarray(rng.standard_normal(tuple(mask.shape)).astype(np.float32))
    corr      = ndi.gaussian_filter(white, sigma=noise_px)
    corr     /= (corr.std() + 1e-12)
    amplitude = noise_px / 3.0
    return (sdt + amplitude * corr) > 0


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3b — Cross-instance models (confusion, occlusion)
# ══════════════════════════════════════════════════════════════════════════════

def _bboxes_overlap(a: _Crop, b: _Crop) -> bool:
    """True if the two crop windows intersect (half-open boxes)."""
    return not (a.r1 <= b.r0 or b.r1 <= a.r0 or a.c1 <= b.c0 or b.c1 <= a.c0)


def _place_into(window_shape, win_r0, win_c0, crop: _Crop, xp):
    """Embed a crop's array into a zeroed window array at the correct offset."""
    out = xp.zeros(window_shape, dtype=bool)
    rr = crop.r0 - win_r0
    cc = crop.c0 - win_c0
    out[rr:rr + crop.arr.shape[0], cc:cc + crop.arr.shape[1]] = crop.arr
    return out


def _apply_interobject_confusion(
    crops: "dict[int, _Crop]",
    boundary_zones: "dict[int, _Crop]",
    depth_img,
    depth_sigma: float,
    boundary_px: int,
    rng,
    xp,
    ndi,
    prune: bool = True,
) -> "dict[int, _Crop]":
    """
    Model 3: Inter-object boundary confusion.

    Where two object silhouettes are adjacent in image space, boundary pixels
    are probabilistically re-assigned to the wrong instance. The confusion
    probability scales with the depth similarity between the two objects:

        p_flip = exp(-|z_i - z_j| / depth_sigma)

    Each pair is processed inside the union of their two padded crop windows; when
    `prune` is True, pairs whose windows do not overlap are skipped (they have no
    shared boundary zone, so the full-frame reference would skip them too — the
    result is identical, the cost is not). `boundary_zones[g]` is the pre-confusion
    boundary band for instance g (same window as crops[g]).

    Flips stay within each instance's padded crop, so only the affected crop arrays
    are mutated in place.
    """
    ids    = list(crops.keys())
    result = {g: _Crop(c.r0, c.c0, c.arr.copy()) for g, c in crops.items()}

    for i, id_i in enumerate(ids):
        ci = result[id_i]
        bzi = boundary_zones[id_i]
        for id_j in ids[i + 1:]:
            cj = result[id_j]
            bzj = boundary_zones[id_j]
            if prune and not _bboxes_overlap(bzi, bzj):
                continue

            # Union window of the two padded crops.
            wr0 = min(ci.r0, cj.r0); wr1 = max(ci.r1, cj.r1)
            wc0 = min(ci.c0, cj.c0); wc1 = max(ci.c1, cj.c1)
            wshape = (wr1 - wr0, wc1 - wc0)

            bz_i_w = _place_into(wshape, wr0, wc0, bzi, xp)
            bz_j_w = _place_into(wshape, wr0, wc0, bzj, xp)
            zone   = bz_i_w & bz_j_w
            if not bool(zone.any()):
                continue

            res_i_w = _place_into(wshape, wr0, wc0, ci, xp)
            res_j_w = _place_into(wshape, wr0, wc0, cj, xp)
            depth_w = depth_img[wr0:wr1, wc0:wc1]

            rows, cols = xp.where(zone)
            gap  = _local_depth_gap(depth_w, rows, cols, res_i_w, res_j_w, xp, ndi)
            p    = xp.exp(-xp.abs(gap) / (depth_sigma + 1e-6))
            flip = xp.asarray(rng.random(int(rows.shape[0]))) < p

            in_i = res_i_w[rows, cols] & flip
            in_j = res_j_w[rows, cols] & flip
            res_i_w[rows[in_i], cols[in_i]] = False
            res_j_w[rows[in_i], cols[in_i]] = True
            res_j_w[rows[in_j], cols[in_j]] = False
            res_i_w[rows[in_j], cols[in_j]] = True

            # Write the (only) changed sub-window back into each instance crop.
            ci.arr[:] = res_i_w[ci.r0 - wr0:ci.r1 - wr0, ci.c0 - wc0:ci.c1 - wc0]
            cj.arr[:] = res_j_w[cj.r0 - wr0:cj.r1 - wr0, cj.c0 - wc0:cj.c1 - wc0]

    return result


def _apply_occlusion_edge_loss(
    crops: "dict[int, _Crop]",
    depth_img,
    geom_img,
    loss_px: int,
    rng,
    xp,
    ndi,
) -> "dict[int, _Crop]":
    """
    Model 4: Occlusion edge erosion.

    Where one object is occluded by another (it disappears behind a closer object
    at a depth discontinuity), the occluded object's mask is over-eroded along that
    boundary. An 'occlusion edge' pixel of object i is a boundary pixel of mask_i
    adjacent to a closer, different object j.

    Each instance is processed within its own crop window; neighbour lookups use the
    full geom_img / depth_img (indexed in full-frame coordinates) so the result is
    identical to the full-frame computation.
    """
    H, W   = geom_img.shape
    result = {g: _Crop(c.r0, c.c0, c.arr.copy()) for g, c in crops.items()}

    for id_i, ci in result.items():
        boundary = _boundary_map(ci.arr, ndi, xp, width=1)
        lr, lc = xp.where(boundary)
        if lr.shape[0] == 0:
            continue
        brows = lr + ci.r0
        bcols = lc + ci.c0

        occ_rows, occ_cols = [], []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = xp.clip(brows + dr, 0, H - 1)
                nc = xp.clip(bcols + dc, 0, W - 1)
                nb_geom    = geom_img[nr, nc]
                nb_depth   = depth_img[nr, nc]
                self_depth = depth_img[brows, bcols]
                is_occluder = (
                    (nb_geom >= 0) &
                    (nb_geom != id_i) &
                    xp.isfinite(nb_depth) &
                    xp.isfinite(self_depth) &
                    (nb_depth < self_depth - 0.001)
                )
                occ_rows.append(brows[is_occluder])
                occ_cols.append(bcols[is_occluder])

        occ_rows = xp.concatenate(occ_rows)
        occ_cols = xp.concatenate(occ_cols)
        if occ_rows.shape[0] == 0:
            continue

        nn       = int(occ_rows.shape[0])
        p_remove = xp.asarray(rng.uniform(0.5, 0.9, nn))
        remove   = xp.asarray(rng.random(nn)) < p_remove
        rr = occ_rows[remove] - ci.r0
        cc = occ_cols[remove] - ci.c0
        ci.arr[rr, cc] = False

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def _disk_struct(radius: int, xp=np):
    """Circular binary structuring element of given radius (backend `xp`)."""
    r  = max(1, int(radius))
    y, x = xp.ogrid[-r:r + 1, -r:r + 1]
    return (x ** 2 + y ** 2) <= r ** 2


def _local_depth_gap(depth_img, rows, cols, mask_i, mask_j, xp, ndi):
    """
    Estimate the depth gap between objects i and j at each (row, col) position in
    the shared boundary zone, using a 5×5 window mean (uniform_filter) of each
    object's depth. Within a smooth depth surface the 5×5 spread is negligible.
    """
    d = xp.where(xp.isfinite(depth_img), depth_img, 0.0)
    wi = mask_i.astype(xp.float64)
    wj = mask_j.astype(xp.float64)
    sum_di = ndi.uniform_filter(d * wi, size=5, mode="constant")
    cnt_i  = ndi.uniform_filter(wi,     size=5, mode="constant")
    sum_dj = ndi.uniform_filter(d * wj, size=5, mode="constant")
    cnt_j  = ndi.uniform_filter(wj,     size=5, mode="constant")
    mean_i = xp.where(cnt_i > 1e-6, sum_di / (cnt_i + 1e-12), 0.0)
    mean_j = xp.where(cnt_j > 1e-6, sum_dj / (cnt_j + 1e-12), 0.0)
    return (mean_i[rows, cols] - mean_j[rows, cols]).astype(xp.float32)


def _resolve_conflicts_full(masks: "dict[int, np.ndarray]") -> "dict[int, np.ndarray]":
    """
    After all perturbations some pixels may belong to multiple masks. Resolve by
    assigning each contested pixel to the mask with the largest interior distance
    (the one for which this pixel is most 'interior'). Operates on full-frame NumPy
    masks (used only when resolve_conflicts=True, which is not the production path).
    """
    ids = [g for g, m in masks.items() if m.any()]
    if not ids:
        return {}
    H, W = next(iter(masks.values())).shape
    best_id   = np.full((H, W), -1,  dtype=np.int32)
    best_dist = np.full((H, W), -1.0, dtype=np.float32)
    for g in ids:
        m    = masks[g]
        dist = _scipy_ndi.distance_transform_edt(m).astype(np.float32)
        better = m & (dist > best_dist)
        best_id[better]   = g
        best_dist[better] = dist[better]
    return {g: (best_id == g) for g in ids}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Main API
# ══════════════════════════════════════════════════════════════════════════════

def _build_perturbed_crops(
    render:              dict,
    erosion_px:          float = 3.0,
    dilation_px:         float = 1.5,
    confusion_depth_sigma: float = 0.015,
    confusion_boundary_px: int  = 4,
    occlusion_loss_px:   int   = 2,
    boundary_noise_px:   float = 6.0,
    apply_erosion:       bool  = True,
    apply_dilation:      bool  = True,
    apply_confusion:     bool  = True,
    apply_occlusion_loss: bool = True,
    apply_boundary_noise: bool = True,
    prune_confusion:     bool  = True,
    seed:                int   = 0,
):
    """Crop-based core of build_perturbed_masks. Returns ({geom_id: _Crop}, geom_img).

    See build_perturbed_masks for parameter documentation. `prune_confusion`
    controls the bbox-overlap skip in Model 3 (exposed for the equivalence test;
    pruned and un-pruned results are identical).
    """
    xp, ndi  = _backend()
    # RNG always on the host: identical draws regardless of compute backend, so the CPU
    # golden stays valid and CPU/GPU differ only by floating-point rounding in the filters.
    rng      = np.random.default_rng(seed)

    geom_img_np  = build_geom_id_image(render)
    depth_img_np = build_depth_image(render)

    margin = _influence_margin(
        erosion_px, dilation_px, confusion_boundary_px, boundary_noise_px,
        apply_erosion, apply_dilation, apply_confusion,
        apply_occlusion_loss, apply_boundary_noise,
    )

    crops = _instance_crops(geom_img_np, margin, xp)
    if not crops:
        return {}, geom_img_np

    geom_img = xp.asarray(geom_img_np)
    depth_img = xp.asarray(depth_img_np.astype(np.float32))

    # ── Model 1: Erosion bias ────────────────────────────────────────────────
    if apply_erosion:
        for g, c in crops.items():
            c.arr = _apply_erosion_bias(c.arr, erosion_px, rng, xp=xp, ndi=ndi)

    # ── Model 2: Dilation into background ────────────────────────────────────
    if apply_dilation:
        for g, c in crops.items():
            geom_crop = geom_img[c.r0:c.r1, c.c0:c.c1]
            c.arr = _apply_dilation_into_background(c.arr, geom_crop, dilation_px,
                                                    rng, xp=xp, ndi=ndi)

    # ── Model 3: Inter-object confusion ──────────────────────────────────────
    if apply_confusion and len(crops) > 1:
        struct = _disk_struct(confusion_boundary_px, xp)
        boundary_zones = {
            g: _Crop(c.r0, c.c0, ndi.binary_dilation(c.arr, structure=struct) & ~c.arr)
            for g, c in crops.items()
        }
        crops = _apply_interobject_confusion(
            crops, boundary_zones, depth_img, confusion_depth_sigma,
            confusion_boundary_px, rng, xp, ndi, prune=prune_confusion,
        )

    # ── Model 4: Occlusion edge loss ──────────────────────────────────────────
    if apply_occlusion_loss:
        crops = _apply_occlusion_edge_loss(
            crops, depth_img, geom_img, occlusion_loss_px, rng, xp, ndi,
        )

    # ── Model 5: Boundary threshold noise ────────────────────────────────────
    if apply_boundary_noise:
        for g, c in crops.items():
            c.arr = _apply_boundary_noise(c.arr, boundary_noise_px, rng, xp=xp, ndi=ndi)

    crops = {g: c for g, c in crops.items() if c.any()}
    return crops, geom_img_np


def build_perturbed_masks(
    render:              dict,
    erosion_px:          float = 3.0,
    dilation_px:         float = 1.5,
    confusion_depth_sigma: float = 0.015,
    confusion_boundary_px: int  = 4,
    occlusion_loss_px:   int   = 2,
    boundary_noise_px:   float = 6.0,
    apply_erosion:       bool  = True,
    apply_dilation:      bool  = True,
    apply_confusion:     bool  = True,
    apply_occlusion_loss: bool = True,
    apply_boundary_noise: bool = True,
    resolve_conflicts:   bool  = False,
    prune_confusion:     bool  = True,
    seed:                int   = 0,
) -> "tuple[dict[int, np.ndarray], np.ndarray]":
    """
    Build per-instance perturbed binary masks in image space.

    Applies the five structured error models to the canonical geom_id image
    from scene_render, producing masks that replicate the boundary behaviour
    of a real 2D instance segmentation network.

    Internally each instance is processed on its padded bounding-box crop (and on
    the GPU when available); the returned masks are re-expanded to full (H, W)
    NumPy arrays for backward compatibility.

    Parameters
    ----------
    render                 : dict from scene_render()
    erosion_px             : target erosion radius (pixels). Typical: 2-5 px.
    dilation_px            : dilation into background radius (pixels). Typical: 1-3 px.
    confusion_depth_sigma  : depth gap scale for inter-object confusion (metres).
    confusion_boundary_px  : width of the shared boundary zone (pixels). Typical: 3-6 px.
    occlusion_loss_px      : erosion strength at occlusion edges. Typical: 1-3 px.
    boundary_noise_px      : spatial correlation length of boundary threshold noise (px).
    apply_*                : toggle each error model independently.
    resolve_conflicts      : if True, run conflict resolution so every pixel belongs
                             to at most one instance (single-label output). Default
                             False — masks may overlap, matching a real network.
    prune_confusion        : skip non-overlapping pairs in Model 3 (default True;
                             identical result, lower cost).
    seed                   : RNG seed for reproducibility.

    Returns
    -------
    masks   : {geom_id: (H, W) bool} — perturbed instance masks in image space.
    geom_img: (H, W) int32 — canonical geom_id image (before perturbation).
    """
    crops, geom_img = _build_perturbed_crops(
        render,
        erosion_px=erosion_px, dilation_px=dilation_px,
        confusion_depth_sigma=confusion_depth_sigma,
        confusion_boundary_px=confusion_boundary_px,
        occlusion_loss_px=occlusion_loss_px,
        boundary_noise_px=boundary_noise_px,
        apply_erosion=apply_erosion, apply_dilation=apply_dilation,
        apply_confusion=apply_confusion, apply_occlusion_loss=apply_occlusion_loss,
        apply_boundary_noise=apply_boundary_noise,
        prune_confusion=prune_confusion, seed=seed,
    )
    if not crops:
        return {}, geom_img

    H, W = geom_img.shape
    masks = {g: _expand_to_full(c, H, W) for g, c in crops.items()}

    if resolve_conflicts:
        masks = _resolve_conflicts_full(masks)
    masks = {g: m for g, m in masks.items() if m.any()}
    return masks, geom_img


def segment_point_cloud(
    render:    dict,
    points:    np.ndarray,
    pixel_idx: np.ndarray,
    verbose: bool = False,
    return_metrics: bool = False,
    **kwargs,
) -> "dict[int, np.ndarray] | tuple[dict[int, np.ndarray], dict[int, dict]]":
    """
    Assign per-instance boolean masks to a point cloud using perturbed 2D masks.

    This is the main entry point. It calls _build_perturbed_crops() and
    back-projects each per-instance crop to 3D via pixel_idx, never materialising a
    full-frame mask per instance.

    A real 2D segmentation network outputs one independent binary mask per
    instance — masks CAN overlap at shared boundary pixels. This function
    replicates that behaviour by default (resolve_conflicts=False).

    Parameters
    ----------
    render    : dict from scene_render()
    points    : (N, 3) point cloud (after noise/dropout pipeline)
    pixel_idx : (N,) flat pixel index for each point.
                  For canonical points: render["pixel_idx"][keep_mask]
                  For injected points (flying pixels, outliers): use -1.
                  Injected points (pixel_idx == -1) are False in all masks.
    return_metrics : bool
        If True, also return per-instance 2D mask metrics (see
        compute_2d_instance_metrics) measured on the perturbed image-space masks.
    **kwargs  : forwarded to build_perturbed_masks()

    Returns
    -------
    instance_masks : dict[int, np.ndarray]
        {geom_id: (N,) bool} — one boolean mask per instance.
        A point may be True in multiple masks (overlap at boundaries).
        Points with pixel_idx == -1 are False in every mask.
    metrics : dict[int, dict]  (only when return_metrics=True)
        {geom_id: {pixel_area, aspect_ratio, area_ratio}} from the 2D masks.
    """
    start = time.time()

    # resolve_conflicts needs full-frame arbitration; route through the full path.
    if kwargs.get("resolve_conflicts", False):
        masks, _ = build_perturbed_masks(render, **kwargs)
        N    = len(points)
        pidx = np.asarray(pixel_idx)
        valid = pidx >= 0
        instance_masks = {}
        for g, m in masks.items():
            arr = np.zeros(N, dtype=bool)
            arr[valid] = m.ravel()[pidx[valid]]
            instance_masks[g] = arr
        if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")
        if return_metrics:
            H, W = render["res"]
            metrics = {g: m_ for g, m_ in (
                (g, _metrics_from_mask(m, H, W)) for g, m in masks.items())
                if m_ is not None}
            return instance_masks, metrics
        return instance_masks

    crops, geom_img = _build_perturbed_crops(render, **kwargs)
    H, W = geom_img.shape

    N    = len(points)
    pidx = np.asarray(pixel_idx)
    valid = pidx >= 0
    vidx = pidx[valid]
    vrow = vidx // W
    vcol = vidx % W

    instance_masks: "dict[int, np.ndarray]" = {}
    for g, c in crops.items():
        arr_cpu = _to_cpu(c.arr)
        sub = np.zeros(vidx.shape[0], dtype=bool)
        inb = (vrow >= c.r0) & (vrow < c.r1) & (vcol >= c.c0) & (vcol < c.c1)
        sub[inb] = arr_cpu[vrow[inb] - c.r0, vcol[inb] - c.c0]
        full = np.zeros(N, dtype=bool)
        full[valid] = sub
        instance_masks[g] = full

    if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")
    if return_metrics:
        return instance_masks, compute_2d_instance_metrics(crops, H, W)
    return instance_masks


def segmentation_stats(
    labels_perturbed: "dict[int, np.ndarray] | np.ndarray",
    labels_canonical: np.ndarray,
) -> dict:
    """
    Compute per-instance boundary error statistics comparing perturbed labels
    to canonical geom_id labels.

    Useful for calibrating parameters against a target sensor/network pair.

    Parameters
    ----------
    labels_perturbed : dict[int, (N,) bool] or (N,) int32
        Perturbed segmentation.  Dict format ``{geom_id: bool_mask}`` as
        returned by ``segment_point_cloud()``.  Int32 flat-label array is
        accepted for backward compatibility.
    labels_canonical : (N,) int32
        Canonical geom_id labels (ground truth).

    Returns
    -------
    dict with keys:
        n_instances            : number of canonical instances
        mean_iou               : mean per-instance IoU (perturbed vs canonical)
        mean_boundary_loss_frac: fraction of canonical boundary points lost
        mean_confusion_frac    : fraction of points re-assigned to wrong instance
        unassigned_frac        : fraction of canonical points now label -1
    """
    inst_ids = np.unique(labels_canonical[labels_canonical >= 0])
    if not len(inst_ids):
        return {}

    ious, bl_fracs, conf_fracs = [], [], []

    if isinstance(labels_perturbed, dict):
        N = len(labels_canonical)
        any_assigned = np.zeros(N, bool)
        for mask in labels_perturbed.values():
            any_assigned |= mask

        for g in inst_ids:
            canon = labels_canonical == g
            pert  = labels_perturbed.get(g, np.zeros(N, bool))

            inter = (canon & pert).sum()
            union = (canon | pert).sum()
            ious.append(inter / (union + 1e-12))

            bl = (canon & ~any_assigned).sum()
            bl_fracs.append(bl / (canon.sum() + 1e-12))

            conf = (canon & ~pert & any_assigned).sum()
            conf_fracs.append(conf / (canon.sum() + 1e-12))

        unassigned_frac = ((labels_canonical >= 0) & ~any_assigned).sum() / \
                          ((labels_canonical >= 0).sum() + 1e-12)
    else:
        for g in inst_ids:
            canon = labels_canonical == g
            pert  = labels_perturbed  == g

            inter = (canon & pert).sum()
            union = (canon | pert).sum()
            ious.append(inter / (union + 1e-12))

            bl = (canon & (labels_perturbed == -1)).sum()
            bl_fracs.append(bl / (canon.sum() + 1e-12))

            conf = (canon & (labels_perturbed >= 0) & ~pert).sum()
            conf_fracs.append(conf / (canon.sum() + 1e-12))

        unassigned_frac = ((labels_canonical >= 0) & (labels_perturbed == -1)).sum() / \
                          ((labels_canonical >= 0).sum() + 1e-12)

    return {
        "n_instances":             len(inst_ids),
        "mean_iou":                float(np.mean(ious)),
        "mean_boundary_loss_frac": float(np.mean(bl_fracs)),
        "mean_confusion_frac":     float(np.mean(conf_fracs)),
        "unassigned_frac":         float(unassigned_frac),
    }
