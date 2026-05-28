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
"""

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    binary_dilation,
    binary_erosion,
    gaussian_filter,
    label as connected_components,
    uniform_filter,
)
import time
import sys
from rich import print as rp

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Build the canonical geom_id image
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
#  SECTION 2 — Per-instance mask helpers
# ══════════════════════════════════════════════════════════════════════════════

def _instance_masks(geom_img: np.ndarray) -> dict[int, np.ndarray]:
    """
    Returns {geom_id: (H,W) bool mask} for every object in the scene.
    Background (-1) is excluded.
    """
    ids = np.unique(geom_img)
    return {int(g): (geom_img == g) for g in ids if g >= 0}


def _boundary_map(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """
    (H, W) bool — True where the mask has a boundary within `width` pixels.
    Computed as: mask XOR eroded(mask, width).
    """
    struct = np.ones((2 * width + 1, 2 * width + 1), bool)
    return mask & ~binary_erosion(mask, structure=struct)


def _dist_to_boundary(mask: np.ndarray) -> np.ndarray:
    """
    (H, W) float32 — Euclidean distance (pixels) from each True pixel in `mask`
    to the nearest boundary pixel of that mask.
    Pixels outside the mask return 0.
    """
    interior = binary_erosion(mask, np.ones((3, 3)))
    dist = distance_transform_edt(interior).astype(np.float32)
    return np.where(mask, dist, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Five error models
# ══════════════════════════════════════════════════════════════════════════════

def _apply_erosion_bias(
    mask:         np.ndarray,
    erosion_px:   float,
    rng:          np.random.Generator,
) -> np.ndarray:
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
    jitter = rng.uniform(0.7, 1.3)
    sigma  = jitter * erosion_px / 0.42
    soft   = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return soft > 0.50


def _apply_dilation_into_background(
    mask:        np.ndarray,
    geom_img:    np.ndarray,
    dilation_px: float,
    rng:         np.random.Generator,
) -> np.ndarray:
    """
    Model 2: Dilation into background.

    At edges where the object meets the bin floor or walls (background pixels),
    the mask bleeds outward slightly. This models the network including
    low-confidence boundary pixels that technically belong to the background.

    Only pixels that are currently background (geom_img == -1) can be added;
    we do not steal pixels from other objects here (that is handled by model 3).

    Implemented as morphological dilation of the mask, then AND with the
    background region.
    """
    jitter   = rng.uniform(0.7, 1.3)
    radius   = max(1, int(round(jitter * dilation_px)))
    struct   = _disk_struct(radius)
    dilated  = binary_dilation(mask, structure=struct)
    bg_only  = geom_img == -1
    # New pixels = dilated AND background (don't touch other objects yet)
    return mask | (dilated & bg_only)


def _apply_interobject_confusion(
    masks:       dict[int, np.ndarray],
    depth_img:   np.ndarray,
    depth_sigma: float,
    boundary_px: int,
    rng:         np.random.Generator,
) -> dict[int, np.ndarray]:
    """
    Model 3: Inter-object boundary confusion.

    Where two object silhouettes are adjacent in image space, boundary pixels
    are probabilistically re-assigned to the wrong instance. The confusion
    probability scales with the depth similarity between the two objects:

        p_flip(i→j) = exp(-|z_i - z_j| / depth_sigma)

    A small depth gap (nearly co-planar objects) → p_flip near 1.
    A large depth gap (clearly separated) → p_flip near 0.

    boundary_px  : how many pixels from each object's boundary to consider.
    depth_sigma  : depth-gap scale for confusion (metres). Objects within
                  depth_sigma of each other are maximally confused.

    Implementation
    ──────────────
    For each pair of adjacent objects (i, j):
      1. Identify pixels within boundary_px of object i AND within boundary_px
         of object j (the shared ambiguous zone).
      2. Sample the depth gap at those pixels.
      3. Draw Bernoulli(p_flip) and flip the label from i to j for True draws.

    Because flipping is applied symmetrically, the total number of boundary
    pixels is conserved.
    """
    ids     = list(masks.keys())
    result  = {g: m.copy() for g, m in masks.items()}
    struct  = _disk_struct(boundary_px)

    # Pre-compute dilated boundary zones for all instances
    boundary_zones = {
        g: binary_dilation(m, structure=struct) & ~m
        for g, m in masks.items()
    }

    for i, id_i in enumerate(ids):
        for id_j in ids[i+1:]:
            # Shared ambiguous zone: within boundary_px of BOTH objects
            zone = boundary_zones[id_i] & boundary_zones[id_j]
            if not zone.any():
                continue

            rows, cols = np.where(zone)
            # Depth gap at each ambiguous pixel
            z_i   = depth_img[rows, cols]
            z_j   = depth_img[rows, cols]
            # Use the nearest object pixel's depth as representative
            # (the zone pixels may be background — look at surrounding depths)
            gap   = _local_depth_gap(depth_img, rows, cols, result[id_i], result[id_j])
            p     = np.exp(-np.abs(gap) / (depth_sigma + 1e-6))
            flip  = rng.random(len(rows)) < p

            # Flip i→j for pixels currently assigned to i
            in_i   = result[id_i][rows, cols] & flip
            in_j   = result[id_j][rows, cols] & flip
            result[id_i][rows[in_i], cols[in_i]] = False
            result[id_j][rows[in_i], cols[in_i]] = True
            result[id_j][rows[in_j], cols[in_j]] = False
            result[id_i][rows[in_j], cols[in_j]] = True

    return result


def _apply_occlusion_edge_loss(
    masks:     dict[int, np.ndarray],
    depth_img: np.ndarray,
    geom_img:  np.ndarray,
    loss_px:   int,
    rng:       np.random.Generator,
) -> dict[int, np.ndarray]:
    """
    Model 4: Occlusion edge erosion.

    Where one object is occluded by another (the occluded object disappears
    behind the occluder at a depth discontinuity), the occluded object's mask
    is over-eroded along that specific boundary. This models the network's
    tendency to avoid the uncertain boundary region where the two objects meet.

    An 'occlusion edge' pixel of object i is a boundary pixel of mask_i that
    is adjacent to a pixel occupied by a different, closer object j
    (depth_j < depth_i at that pixel).

    These occlusion-edge pixels are eroded with a higher probability than
    normal boundary pixels.

    loss_px : number of pixels to erode along occlusion edges.
    """
    result = {g: m.copy() for g, m in masks.items()}
    ids    = list(masks.keys())

    for id_i in ids:
        boundary = _boundary_map(masks[id_i], width=1)
        brows, bcols = np.where(boundary)
        if not len(brows):
            continue

        # Find neighbouring pixels in a 3×3 window
        H, W = masks[id_i].shape
        occ_rows, occ_cols = [], []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr = np.clip(brows + dr, 0, H - 1)
                nc = np.clip(bcols + dc, 0, W - 1)
                nb_geom  = geom_img[nr, nc]
                nb_depth = depth_img[nr, nc]
                self_depth = depth_img[brows, bcols]
                # Neighbour is a different, closer (occluding) object
                is_occluder = (
                    (nb_geom >= 0) &
                    (nb_geom != id_i) &
                    np.isfinite(nb_depth) &
                    np.isfinite(self_depth) &
                    (nb_depth < self_depth - 0.001)
                )
                occ_rows.append(brows[is_occluder])
                occ_cols.append(bcols[is_occluder])

        if not any(len(r) for r in occ_rows):
            continue

        occ_rows = np.concatenate(occ_rows)
        occ_cols = np.concatenate(occ_cols)
        if not len(occ_rows):
            continue

        # Probabilistically remove these occlusion-edge pixels
        p_remove = rng.uniform(0.5, 0.9, len(occ_rows))
        remove   = rng.random(len(occ_rows)) < p_remove
        result[id_i][occ_rows[remove], occ_cols[remove]] = False

    return result


def _apply_boundary_noise(
    mask:       np.ndarray,
    noise_px:   float,
    rng:        np.random.Generator,
) -> np.ndarray:
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
    # Signed distance transform: positive inside mask, negative outside
    dist_in  = distance_transform_edt(mask).astype(np.float32)
    dist_out = distance_transform_edt(~mask).astype(np.float32)
    sdt      = dist_in - dist_out

    # Spatially correlated Gaussian noise
    white     = rng.standard_normal(mask.shape).astype(np.float32)
    corr      = gaussian_filter(white, sigma=noise_px)
    corr     /= (corr.std() + 1e-12)
    amplitude = noise_px / 3.0
    return (sdt + amplitude * corr) > 0


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def _disk_struct(radius: int) -> np.ndarray:
    """Circular binary structuring element of given radius."""
    r  = max(1, int(radius))
    sz = 2 * r + 1
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x**2 + y**2) <= r**2


def _local_depth_gap(
    depth_img: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    mask_i: np.ndarray,
    mask_j: np.ndarray,
) -> np.ndarray:
    """
    Estimate the depth gap between objects i and j at each (row, col) position
    in the shared boundary zone.

    Uses a 5×5 window mean (via uniform_filter) of each object's depth to
    estimate local depth.  Mean vs. median: within a smooth depth surface the
    5×5 spread is negligible, so the approximation is valid and avoids
    per-pixel Python loops.
    """
    d = np.where(np.isfinite(depth_img), depth_img, 0.0)
    wi = mask_i.astype(np.float64)
    wj = mask_j.astype(np.float64)
    # Weighted sums over 5×5 neighbourhood
    sum_di = uniform_filter(d * wi, size=5, mode="constant")
    cnt_i  = uniform_filter(wi,     size=5, mode="constant")
    sum_dj = uniform_filter(d * wj, size=5, mode="constant")
    cnt_j  = uniform_filter(wj,     size=5, mode="constant")
    mean_i = np.where(cnt_i > 1e-6, sum_di / (cnt_i + 1e-12), 0.0)
    mean_j = np.where(cnt_j > 1e-6, sum_dj / (cnt_j + 1e-12), 0.0)
    return (mean_i[rows, cols] - mean_j[rows, cols]).astype(np.float32)


def _resolve_conflicts(masks: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """
    After all perturbations some pixels may belong to multiple masks (from
    dilation and confusion). Resolve by assigning each contested pixel to
    the mask with the shortest distance-to-boundary (i.e. the one for which
    this pixel is most 'interior').

    Also removes any mask that lost all its pixels.
    """
    ids     = [g for g, m in masks.items() if m.any()]
    if not ids:
        return {}

    H, W   = next(iter(masks.values())).shape

    # For each pixel, track which instance has the largest interior distance
    best_id   = np.full((H, W), -1,  dtype=np.int32)
    best_dist = np.full((H, W), -1.0, dtype=np.float32)

    for g in ids:
        m    = masks[g]
        dist = distance_transform_edt(m).astype(np.float32)
        better = m & (dist > best_dist)
        best_id[better]   = g
        best_dist[better] = dist[better]

    return {g: (best_id == g) for g in ids}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Main API
# ══════════════════════════════════════════════════════════════════════════════

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
    seed:                int   = 0,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Build per-instance perturbed binary masks in image space.

    Applies the five structured error models to the canonical geom_id image
    from scene_render, producing masks that replicate the boundary behaviour
    of a real 2D instance segmentation network.

    Parameters
    ----------
    render                 : dict from scene_render()
    erosion_px             : target erosion radius (pixels). Typical: 2-5 px.
    dilation_px            : dilation into background radius (pixels). Typical: 1-3 px.
    confusion_depth_sigma  : depth gap scale for inter-object confusion (metres).
                             Objects within this depth of each other have maximum
                             boundary confusion. Typical: 10-30 mm.
    confusion_boundary_px  : width of the shared boundary zone considered for
                             inter-object confusion (pixels). Typical: 3-6 px.
    occlusion_loss_px      : erosion strength at occlusion edges. Typical: 1-3 px.
    boundary_noise_px      : spatial correlation length of boundary threshold noise
                             (pixels). Matches network receptive field. Typical: 4-12 px.
    apply_*                : toggle each error model independently.
    resolve_conflicts      : if True, run _resolve_conflicts() so every pixel
                             belongs to at most one instance (single-label output).
                             Default False — masks may overlap, matching the
                             per-instance binary-mask output of a real network.
    seed                   : RNG seed for reproducibility.

    Returns
    -------
    masks   : {geom_id: (H, W) bool} — perturbed instance masks in image space.
              When resolve_conflicts=False (default), masks may overlap at
              shared boundary zones.
    geom_img: (H, W) int32 — canonical geom_id image (before perturbation).
    """
    rng      = np.random.default_rng(seed)
    geom_img = build_geom_id_image(render)
    depth_img = build_depth_image(render)
    masks    = _instance_masks(geom_img)

    if not masks:
        return {}, geom_img

    # ── Model 1: Erosion bias ─────────────────────────────────────────────
    if apply_erosion:
        masks = {
            g: _apply_erosion_bias(m, erosion_px, rng)
            for g, m in masks.items()
        }

    # ── Model 2: Dilation into background ────────────────────────────────
    if apply_dilation:
        masks = {
            g: _apply_dilation_into_background(m, geom_img, dilation_px, rng)
            for g, m in masks.items()
        }

    # ── Model 3: Inter-object confusion ──────────────────────────────────
    if apply_confusion and len(masks) > 1:
        masks = _apply_interobject_confusion(
            masks, depth_img, confusion_depth_sigma, confusion_boundary_px, rng
        )

    # ── Model 4: Occlusion edge loss ──────────────────────────────────────
    if apply_occlusion_loss:
        masks = _apply_occlusion_edge_loss(
            masks, depth_img, geom_img, occlusion_loss_px, rng
        )

    # ── Model 5: Boundary threshold noise ────────────────────────────────
    if apply_boundary_noise:
        masks = {
            g: _apply_boundary_noise(m, boundary_noise_px, rng)
            for g, m in masks.items()
        }

    # Optionally resolve conflicts (single-label); default keeps overlaps so that
    # each instance mask is independent — matching real segmentation network output.
    if resolve_conflicts:
        masks = _resolve_conflicts(masks)
    masks = {g: m for g, m in masks.items() if m.any()}

    return masks, geom_img


def segment_point_cloud(
    render:    dict,
    points:    np.ndarray,
    pixel_idx: np.ndarray,
    verbose: bool = False,
    **kwargs,
) -> dict[int, np.ndarray]:
    """
    Assign per-instance boolean masks to a point cloud using perturbed 2D masks.

    This is the main entry point. It calls build_perturbed_masks() and
    back-projects each per-instance mask to 3D via pixel_idx.

    A real 2D segmentation network outputs one independent binary mask per
    instance — masks CAN overlap at shared boundary pixels (a pixel predicted
    as belonging to two adjacent instances appears in both masks). This function
    replicates that behaviour by default (resolve_conflicts=False in kwargs).

    Parameters
    ----------
    render    : dict from scene_render()
    points    : (N, 3) point cloud (after noise/dropout pipeline)
    pixel_idx : (N,) flat pixel index for each point.
                  For canonical points: render["pixel_idx"][keep_mask]
                  For injected points (flying pixels, outliers): use -1.
                  Injected points (pixel_idx == -1) are False in all masks.
    **kwargs  : forwarded to build_perturbed_masks()
                  (erosion_px, dilation_px, confusion_depth_sigma,
                   resolve_conflicts, etc.)

    Returns
    -------
    instance_masks : dict[int, np.ndarray]
        {geom_id: (N,) bool} — one boolean mask per instance.
        A point may be True in multiple masks (overlap at boundaries).
        Points with pixel_idx == -1 are False in every mask.

    Notes
    -----
    To extract points for a single instance:
        inst_pts = points[instance_masks[inst_id]]

    To get the set of active instance ids:
        active_ids = list(instance_masks.keys())
    """
    start = time.time()
    masks, _ = build_perturbed_masks(render, **kwargs)

    N     = len(points)
    pidx  = np.asarray(pixel_idx)
    valid = pidx >= 0

    instance_masks: dict[int, np.ndarray] = {}
    for g, m in masks.items():
        label_flat = m.ravel()
        arr        = np.zeros(N, dtype=bool)
        arr[valid] = label_flat[pidx[valid]]
        instance_masks[g] = arr

    if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")
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
        # Which points appear in at least one mask
        any_assigned = np.zeros(N, bool)
        for mask in labels_perturbed.values():
            any_assigned |= mask

        for g in inst_ids:
            canon = labels_canonical == g
            pert  = labels_perturbed.get(g, np.zeros(N, bool))

            inter = (canon & pert).sum()
            union = (canon | pert).sum()
            ious.append(inter / (union + 1e-12))

            # Boundary loss: canonical points for this instance absent from all masks
            bl = (canon & ~any_assigned).sum()
            bl_fracs.append(bl / (canon.sum() + 1e-12))

            # Confusion: canonical points of this instance present in a different mask
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

            # Boundary loss: canonical points for this instance now unassigned
            bl = (canon & (labels_perturbed == -1)).sum()
            bl_fracs.append(bl / (canon.sum() + 1e-12))

            # Confusion: canonical points assigned to a different instance
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