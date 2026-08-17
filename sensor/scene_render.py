"""
scene_render.py — Synthetic structured-light depth sensor simulation pipeline.

scene_render() is the single entry point for geometry. It performs a pure
canonical raycast and returns a render dict. Every noise, dropout, and
outlier function in this file operates on that dict downstream.

Typical call sequence
─────────────────────
    render = scene_render(meshes, T_cam, look_at, fov, W, H)
    if render is None:
        return

    # 1. Dropout — remove physically unreturnable points
    keep = compute_dropout_mask(
        render, roughness=0.4,
        albedo_per_geom_id={2: 0.04},  # black rubber part
        density_cos_ref=0.7,           # oblique density thinning
    )

    # 2. Image-space reconstruction artifacts
    render = add_image_space_effects(render, keep,
                                     smooth_sigma_px=0.5,
                                     sigma_fringe_corr=0.0001)

    # 3. Edge artifacts — displace surviving pts + inject flying pixels
    pts, nrm = add_edge_artifacts(render, keep)

    # 4. Structured outliers
    r = subset_render(render, keep)
    mp, mn = add_multipath_outliers(r)
    pp, pn = add_pepper_noise(r)
    pts = np.vstack([pts, mp, pp])
    nrm = np.vstack([nrm, mn, pn])

    # Build per-point metadata for the full combined cloud.
    n_kept    = keep.sum()
    n_outlier = len(pts) - n_kept
    pix_all   = np.concatenate([render["pixel_idx"][keep],
                                 np.full(n_outlier, -1, np.int64)])
    cproj_all = np.concatenate([render["cos_proj"][keep],
                                 np.ones(n_outlier)])

    # 5. Scan-line banding
    pts = add_scan_line_banding(pts, nrm, pix_all, render["res"],
                                render["sensor_origin"])

    # 6. Sensor electronics noise
    pts = add_sensor_noise(pts, nrm, render["sensor_origin"],
                           pixel_idx=pix_all, res=render["res"],
                           cos_proj=cproj_all)

    # 7. Surface microgeometry
    pts = add_surface_noise(pts, nrm)

render dict fields
──────────────────
  Geometry  (N = projector-illuminated, shadow-free points)
    points           (N,3)   world-space hit positions
    normals          (N,3)   estimated surface normals
    geom_ids         (N,)    mesh index per point
    t_hit            (N,)    ray travel distance

  Per-point ray/sensor geometry
    ray_origins      (N,3)
    ray_dirs         (N,3)   unit camera ray direction
    proj_dirs        (N,3)   unit projector→point direction
    proj_dist        (N,)    projector–point distance
    cos_cam          (N,)    |n · v_cam|
    cos_proj         (N,)    |n · v_proj|
    snr_proxy        (N,)    cos_cam·cos_proj/proj_dist²  normalised [0,1]
    sensor_origin    (3,)    camera position (world)
    proj_origin      (3,)    projector position (world)

  Full image buffers
    ray_origins_img  (H*W,3)
    ray_dirs_img     (H*W,3)
    depth_img        (H,W)   t_hit for every pixel hit; NaN=miss.
                             Includes shadow-occluded hits.
    hit_mask         (H*W,)  bool
    pixel_idx        (N,)    flat pixel index of each visible point
    res              (H,W)   image resolution tuple
"""

import copy
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter, label as _ndimage_label
from geometry.geom_utils import estimate_normals
import time
import open3d as o3d
import sys
from rich import print as rp
# import cupy as cp

# Per-point keys in the render dict: arrays that must be indexed by a keep mask.
# All other keys (scalars, image buffers, metadata) are passed through unchanged.
_PER_POINT_KEYS = frozenset({
    "points", "normals", "geom_ids", "t_hit",
    "ray_origins", "ray_dirs", "proj_dirs", "proj_dist",
    "cos_cam", "cos_proj", "snr_proxy", "pixel_idx",
})

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Canonical renderer
# ══════════════════════════════════════════════════════════════════════════════

def scene_render(meshes:dict, T_cam, look_at, fov, res_width, res_height,
                 baseline=0.27, normal_radius=0.005, normal_max_nn=50,
                 verbose=False):
    """
    Pure canonical raycast. Applies only binary projector shadow — a geometric
    fact, not a stochastic model.

    FIX (was absolute 1 mm): shadow tolerance is now RELATIVE to range —
    max(1 mm, 0.05% of projector distance). This prevents false occlusion at
    close range and missed shadows at long range.

    Also computes snr_proxy once and stores it in the render dict, avoiding
    redundant recomputation in every downstream caller.
    """
    start = time.time()
    scene = o3d.t.geometry.RaycastingScene()
    for _, mesh_data in meshes.items():
        # Deep-copy the mesh before transforming so mesh_data.geom is never mutated.
        # Without this, repeated scene_render calls accumulate T_gt on the same object.
        mesh_copy = copy.deepcopy(mesh_data.geom)
        mesh_copy.transform(mesh_data.T_gt)
        mesh_data.id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_copy))

    rays    = scene.create_rays_pinhole(
        fov_deg=fov, center=look_at, eye=T_cam[:3, 3],
        up=T_cam[:3, 1], width_px=res_width, height_px=res_height,
    )
    # rays_np         = rays.numpy().reshape(-1, 6)
    rays_np         = np.asarray(rays.numpy()).reshape(-1, 6)
    ray_origins_img = rays_np[:, :3]
    ray_dirs_img    = rays_np[:, 3:]

    ans          = scene.cast_rays(rays)
    t_hit_all    = np.asarray(ans["t_hit"].numpy()).reshape( -1)
    # t_hit_all    = ans["t_hit"].numpy().reshape(-1)
    # geom_ids_all = ans["geometry_ids"].numpy().reshape(-1)
    geom_ids_all = np.asarray(ans["geometry_ids"].numpy()).reshape(-1)

    hit_mask = np.isfinite(t_hit_all)
    if not np.any(hit_mask):
        print("no hit point in scene")
        return None

    depth_flat           = np.full(res_width * res_height, np.nan, np.float32)
    depth_flat[hit_mask] = t_hit_all[hit_mask].astype(np.float32)
    depth_img            = depth_flat.reshape(res_height, res_width)

    t_hit    = t_hit_all[hit_mask]
    geom_ids = geom_ids_all[hit_mask]
    origins  = ray_origins_img[hit_mask]
    dirs     = ray_dirs_img[hit_mask]
    points   = origins + t_hit[:, None] * dirs
    normals  = estimate_normals(points, T_cam[:3, 3], normal_radius, normal_max_nn)

    proj_origin = T_cam[:3, 3] + baseline * T_cam[:3, 0]
    proj_vecs   = np.asarray(points) - np.asarray(proj_origin)
    proj_dist   = np.linalg.norm(proj_vecs, axis=1)
    proj_dirs   = proj_vecs / (proj_dist[:, None] + 1e-12)

    proj_rays = o3d.core.Tensor(
        np.hstack([np.broadcast_to((proj_origin[None]), (len(points), 3)), proj_dirs]).astype(np.float32),
                                    dtype=o3d.core.Dtype.Float32)
    # proj_t = scene.cast_rays(proj_rays)["t_hit"].numpy()
    proj_t = np.asarray(scene.cast_rays(proj_rays)["t_hit"].numpy())

    # Relative tolerance: 0.05% of range, floor 1 mm.
    shadow_tol  = np.maximum(1e-3, proj_dist * 5e-4)
    illuminated = proj_t >= proj_dist - shadow_tol

    v_cam    = -dirs
    v_proj   = -proj_dirs
    cos_cam  = np.abs(np.einsum("ij,ij->i", v_cam,  normals))
    cos_proj = np.abs(np.einsum("ij,ij->i", v_proj, normals))

    # SNR proxy precomputed once; avoids 3× redundant recomputation downstream.
    snr_raw   = cos_cam * cos_proj / (proj_dist**2 + 1e-12)
    snr_proxy = (snr_raw - snr_raw.min()) / (snr_raw.max() - snr_raw.min() + 1e-12)

    vis       = illuminated
    hit_idx   = np.where(hit_mask)[0]
    pixel_idx = hit_idx[vis]
    if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")

    return dict(
        points          =   points[vis], 
        normals         =   normals[vis], 
        geom_ids        =   geom_ids[vis],
        t_hit           =   t_hit[vis], 
        ray_origins     =   origins[vis], 
        ray_dirs        =   dirs[vis],
        proj_dirs       =   proj_dirs[vis], 
        proj_dist       =   proj_dist[vis],
        cos_cam         =   cos_cam[vis], 
        cos_proj        =   cos_proj[vis], 
        snr_proxy       =   snr_proxy[vis],
        sensor_origin   =   T_cam[:3, 3].copy(), 
        proj_origin     =   proj_origin,
        ray_origins_img =   ray_origins_img, 
        ray_dirs_img    =   ray_dirs_img,
        depth_img       =   depth_img, 
        hit_mask        =   hit_mask, 
        pixel_idx       =   pixel_idx,
        res             =   (res_height, res_width),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Depth image helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_depth_image(render):
    """(H,W) float32 from visible points only. NaN for missing pixels."""
    H, W = render["res"]
    img  = np.full(H * W, np.nan, np.float32)
    img[render["pixel_idx"]] = render["t_hit"].astype(np.float32)
    return img.reshape(H, W)


def compute_edge_strength(depth_img):
    """Sobel magnitude, NaN→far, normalised [0,1] by 99th pct."""
    gx, gy = compute_depth_gradient(depth_img)
    mag = np.sqrt(gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2).astype(np.float32)
    hi  = np.percentile(mag[mag > 0], 99) if np.any(mag > 0) else 1.0
    return (mag / (hi + 1e-12)).clip(0, 1).astype(np.float32)


def compute_depth_gradient(depth_img):
    """Sobel (gx, gy) each (H,W) float32. NaN→far before differencing."""
    d = depth_img.copy().astype(np.float64)
    far = float(np.nanmax(d)) * 10.0 if not np.all(np.isnan(d)) else 1e6
    d[np.isnan(d)] = far
    gx = np.zeros_like(d, np.float32); gy = np.zeros_like(d, np.float32)
    gx[1:-1,1:-1] = ((d[:-2,2:]-d[:-2,:-2]) + 2*(d[1:-1,2:]-d[1:-1,:-2]) + (d[2:,2:]-d[2:,:-2])).astype(np.float32) / 8.
    gy[1:-1,1:-1] = ((d[2:,:-2]-d[:-2,:-2]) + 2*(d[2:,1:-1]-d[:-2,1:-1]) + (d[2:,2:]-d[:-2,2:])).astype(np.float32) / 8.
    return gx, gy


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Dropout
# ══════════════════════════════════════════════════════════════════════════════

def _specular_keep(normals, ray_dirs, proj_dirs, roughness):
    """
    Bidirectional GGX specular keep-mask  (N,) bool.

    FIX 1 — Wrong light path (was reflecting camera ray, should reflect projector):
    For structured light the projector emits; the physical path is
      projector → surface → camera.
    We reflect v_proj (surface→projector) about n and check alignment with v_cam.

    FIX 2 — No diffuse floor (Lambertian surfaces were incorrectly dropped):
    lobe = (1 − r) · specular_term  +  r   (r = roughness acts as diffuse floor)
    At roughness=1 the floor equals 1.0, so Lambertian surfaces always pass.
    At roughness=0 it is pure specular — mirrors drop unless perfectly aligned.
    """
    v_cam  = -ray_dirs
    v_proj = -proj_dirs
    ndotl  = np.einsum("ij,ij->i", normals, v_proj).clip(0., 1.)
    refl   = 2. * ndotl[:, None] * normals - v_proj
    refl  /= np.linalg.norm(refl, axis=1, keepdims=True) + 1e-12
    cos_r  = np.einsum("ij,ij->i", refl, v_cam).clip(0., 1.)
    r      = np.broadcast_to(np.asarray(roughness, np.float64), len(normals)).copy()
    alpha  = r**2 + 1e-6
    lobe   = (1. - r) * cos_r ** (1. / alpha**2) + r   # diffuse floor = r
    return lobe > 0.01


def _specular_keep_anisotropic(normals, ray_dirs, proj_dirs,
                               alpha_t, alpha_b, brush_dir, anisotropy, roughness):
    """
    Anisotropic specular keep-mask using Ward BRDF  (N,) bool.

    Models brushed / milled / rolled surfaces with directional micro-grooves:
      alpha_t = roughness along brush direction  (tight, small — e.g. 0.10)
      alpha_b = roughness across brush direction (loose, large — e.g. 0.40)

    anisotropy blends Ward anisotropic with Ward isotropic:
      0.0 → delegates to existing _specular_keep() (no regression)
      1.0 → full Ward anisotropic

    Physical model
    ──────────────
    Ward spherical-Gaussian in the half-angle H domain:
        lobe_aniso = exp(-tan²θ_h · (cos²φ_h/α_t² + sin²φ_h/α_b²))
    where θ_h = elevation of H above n, φ_h = azimuth of H in the (t, b)
    surface tangent frame.  Normalisation constants are dropped — only the
    relative per-point keep probability matters.
    """
    if anisotropy == 0.:
        return _specular_keep(normals, ray_dirs, proj_dirs, roughness)

    N      = len(normals)
    v_cam  = -ray_dirs
    v_proj = -proj_dirs

    H = v_cam + v_proj
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-12

    n_dot_h  = np.einsum("ij,ij->i", normals, H).clip(0., 1.)
    tan_h_sq = np.maximum(0., 1. - n_dot_h ** 2) / (n_dot_h ** 2 + 1e-6)

    # ── Tangent frame ──────────────────────────────────────────────────────────
    if brush_dir is None:
        brush_world = np.broadcast_to(np.array([1., 0., 0.]), (N, 3)).copy()
    else:
        brush_world = np.broadcast_to(np.asarray(brush_dir, np.float64), (N, 3)).copy()

    dot_nb = np.einsum("ij,ij->i", normals, brush_world)
    t      = brush_world - dot_nb[:, None] * normals
    t_norm = np.linalg.norm(t, axis=1, keepdims=True)
    # Fallback: brush direction is parallel to normal → use world Y
    fallback = np.cross(normals, np.broadcast_to([0., 1., 0.], (N, 3)))
    fallback /= np.linalg.norm(fallback, axis=1, keepdims=True) + 1e-12
    t = np.where(t_norm > 1e-6, t / (t_norm + 1e-12), fallback)
    b = np.cross(normals, t)
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12

    h_t = np.einsum("ij,ij->i", H, t)
    h_b = np.einsum("ij,ij->i", H, b)

    # Ward anisotropic lobe
    exp_aniso  = -tan_h_sq * (h_t ** 2 / (alpha_t ** 2 + 1e-12)
                              + h_b ** 2 / (alpha_b ** 2 + 1e-12))
    lobe_aniso = np.exp(exp_aniso)

    # Ward isotropic baseline (for blending)
    alpha_iso = (alpha_t + alpha_b) / 2.
    lobe_iso  = np.exp(-tan_h_sq / (alpha_iso ** 2 + 1e-12))

    lobe = (1. - anisotropy) * lobe_iso + anisotropy * lobe_aniso
    r    = np.broadcast_to(np.asarray(roughness, np.float64), N).copy()
    return (1. - r) * lobe + r > 0.01     # same diffuse floor as _specular_keep


def _grazing_keep(cos_cam, cos_proj, cam_thr, proj_thr, steepness, rng):
    """Soft sigmoid grazing dropout. p_keep = sigmoid((cos−thr)/steepness)."""
    def _s(c, t): return rng.random(len(c)) < 1. / (1. + np.exp(-(c-t) / (steepness+1e-12)))
    return _s(cos_cam, cam_thr) & _s(cos_proj, proj_thr)


def _albedo_pepper_keep(snr, albedo, base_rate, rng):
    """
    Albedo- and SNR-weighted pepper dropout  (N,) bool.

    NEW — Dark surfaces were not modelled. A surface with albedo 0.03 (black
    rubber) absorbs ~97 % of the projected light regardless of geometry,
    driving effective SNR near zero and causing near-total dropout.

        effective_snr = snr_proxy * albedo      (both in [0, 1])
        p_drop = base_rate * (1 − effective_snr)

    Key: we do NOT renormalise effective_snr. Renormalising would divide out the
    albedo factor and make all surfaces equally likely to drop — defeating the
    purpose entirely. albedo=0.03 drives effective_snr ≈ 0 → p_drop ≈ base_rate
    for all points on that surface, regardless of geometry.
    """
    eff = snr * np.asarray(albedo, np.float64)   # in [0, 1]
    return rng.random(len(snr)) > base_rate * (1. - eff)


def _density_keep(cos_proj, cos_ref, rng):
    """
    Oblique-surface point density thinning  (N,) bool.

    NEW — At projector incidence angle θ the fringe pattern stretches by
    1/cos_proj across the surface, reducing the effective number of fringe
    cycles per camera pixel and hence spatial point density.

        p_keep = min(1, cos_proj / cos_ref)
    """
    return rng.random(len(cos_proj)) < np.minimum(1., cos_proj / (cos_ref + 1e-12))


def compute_dropout_mask(render, roughness=0.4,
                         cam_grazing_thresh=0.25, proj_grazing_thresh=0.25,
                         grazing_steepness=0.10, pepper_base_rate=0.04,
                         albedo_per_geom_id=None, default_albedo=0.7,
                         density_cos_ref=None,
                         anisotropy=0.0, alpha_t=0.10, alpha_b=0.40,
                         brush_dir=None,
                         seed=0, verbose=False):
    """
    Removes physically unreturnable points
    Boolean keep-mask  (N,)  for render["points"].

    Four effects in order:
      1. Specular    — bidirectional GGX: reflect v_proj off n, check v_cam.
                       Diffuse floor (= roughness) prevents Lambertian dropout.
                       When anisotropy > 0, Ward anisotropic BRDF is used instead,
                       modelling brushed / milled / rolled metal micro-grooves.
      2. Grazing     — sigmoid on both cos_cam and cos_proj.
      3. Albedo/SNR  — dark materials absorb projected light → high dropout.
      4. Density     — oblique surfaces get probabilistically thinned (optional).

    Parameters
    ----------
    roughness           : scalar or (N,) in [0,1]. 0=mirror, 1=Lambertian.
    albedo_per_geom_id  : {geom_id: albedo} material map.
                          white plastic ~0.85, bare steel ~0.60,
                          anodized Al ~0.15, black rubber ~0.03.
    default_albedo      : fallback for unspecified geom_ids.
    density_cos_ref     : enable density thinning; typical 0.7 (≈45° half-angle).
    anisotropy          : 0 = isotropic GGX (default, no change), 1 = full Ward
                          anisotropic. Set > 0 for brushed/milled metal parts.
    alpha_t             : Ward roughness along brush direction. Typical 0.05–0.20.
    alpha_b             : Ward roughness across brush direction. Typical 0.30–0.60.
    brush_dir           : (3,) world-space brush direction. None → horizontal [1,0,0].
    """
    start = time.time()
    rng    = np.random.default_rng(seed)
    albedo = np.full(len(render["geom_ids"]), default_albedo, np.float64)
    if albedo_per_geom_id:
        for gid, a in albedo_per_geom_id.items():
            albedo[render["geom_ids"] == gid] = float(a)

    snr = render.get("snr_proxy")
    if snr is None:
        raw = render["cos_cam"] * render["cos_proj"] / (render["proj_dist"]**2 + 1e-12)
        snr = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)

    if anisotropy > 0.:
        specular_mask = _specular_keep_anisotropic(
            render["normals"], render["ray_dirs"], render["proj_dirs"],
            alpha_t, alpha_b, brush_dir, anisotropy, roughness)
    else:
        specular_mask = _specular_keep(render["normals"], render["ray_dirs"],
                                       render["proj_dirs"], roughness)

    mask = (
        specular_mask
        & _grazing_keep(render["cos_cam"], render["cos_proj"],
                        cam_grazing_thresh, proj_grazing_thresh,
                        grazing_steepness, rng)
        & _albedo_pepper_keep(snr, albedo, pepper_base_rate, rng)
    )
    if density_cos_ref is not None:
        mask &= _density_keep(render["cos_proj"], density_cos_ref, rng)

    if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")

    return mask


def add_projector_nonuniformity(render, proj_fpn_sigma=0.05, proj_fpn_scale=20.0,
                                seed=0, verbose=False):
    """
    Projector illumination non-uniformity (projector-side FPN).

    DMD / LCD projector pixels have individual gain variation (~2–8 % for typical
    industrial projectors), causing spatially non-uniform fringe contrast across
    the field. This is DISTINCT from camera FPN (sensor-side, per camera pixel):
    - Camera FPN: same sensor pixel always reads the same bias, regardless of
      what scene point it sees.
    - Projector FPN: same projector pixel always illuminates with the same gain,
      regardless of which camera pixel captures the return. The same scene point
      viewed from two different camera poses has the same projector gain but
      different camera FPN.

    Model
    ─────
    3D Perlin noise evaluated at the unit projector direction vector
    (proj_dirs) parameterises the projector's angular image space. The gain
    G ∈ [1 − σ, 1 + σ] is multiplied into snr_proxy before dropout. At σ = 0.05
    this produces a ±5 % intensity variation, typical for an industrial DMD.

    Returns a shallow copy of render with snr_proxy modulated.

    Parameters
    ----------
    proj_fpn_sigma : 1σ gain variation (fractional). Typical: 0.03–0.08.
    proj_fpn_scale : Perlin scale in projector direction space. Higher → finer
                     non-uniformity pattern. Typical: 10–30.
    """
    start = time.time()
    # proj_dirs are unit vectors from projector to each surface point.
    # They parameterise projector image space: each unique direction maps to one
    # projector pixel. Scale by proj_fpn_scale to control spatial frequency.
    field = _Perlin3D(seed=seed + 2000)(render["proj_dirs"], scale=proj_fpn_scale)
    # Normalise to [-1, 1] then apply gain variation
    field  = field / (np.abs(field).max() + 1e-12)
    gain   = 1.0 + proj_fpn_sigma * field          # gain ∈ [1-σ, 1+σ]
    out    = dict(render)
    out["snr_proxy"] = np.clip(render["snr_proxy"] * gain, 0., 1.)
    if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")
    return out


def add_specular_patch_missing(render, keep, roughness=0.25,
                               specular_threshold=0.05, min_patch_area_px=20,
                               patch_dropout_rate=0.1, seed=0, verbose=False):
    """
    Coherent specular patch missing.

    Real structured-light sensors exhibit contiguous missing regions — not
    scattered per-point dropout — when the specular lobe of a smooth surface
    sweeps away from the camera. The GGX NDF D(H) is low across the whole
    connected patch simultaneously, so the entire region falls below the
    sensor's detection threshold together.

    This is DISTINCT from the per-point specular dropout in compute_dropout_mask:
    - compute_dropout_mask: stochastic per-point dropout scaled by lobe alignment.
    - add_specular_patch_missing: finds connected image-space regions where D(H)
      is uniformly low and drops the entire patch at once (coherent failure).

    Algorithm
    ─────────
    1. For each visible point: compute GGX Trowbridge-Reitz NDF D(H) where
       H = normalize(v_cam + v_proj).  Low D → specular lobe points away.
    2. Project D values to image space; threshold to find dark candidate pixels.
    3. scipy.ndimage.label to find connected dark regions.
    4. For each region with area ≥ min_patch_area_px: drop with patch_dropout_rate.

    Default parameters sit midway between matte plastic (roughness ≈ 0.4) and
    brushed metal (roughness ≈ 0.15) for a generic industrial SL camera.

    Parameters
    ----------
    roughness           : GGX roughness parameter (α). Lower → sharper lobe →
                          more coherent dark patches.  Default 0.25.
    specular_threshold  : Normalised NDF threshold below which a pixel is
                          classified as 'specular dark'. Default 0.15.
    min_patch_area_px   : Minimum connected-component area (pixels) to trigger
                          dropout.  Smaller patches are ignored. Default 20.
    patch_dropout_rate  : Probability that a qualifying dark patch is fully
                          dropped.  Default 0.6.
    """
    start = time.time()
    rng   = np.random.default_rng(seed)

    # ── GGX NDF D(H) for ALL visible points (dense image → coherent patches) ─────
    # Using ALL visible points (not just the post-dropout kept subset) is essential:
    # kept points are sparse (~5% image fill), so their connected components are
    # single pixels.  The specular dark patch is a property of surface geometry,
    # not of which points survived previous dropout stages.
    v_cam_all  = -render["ray_dirs"]
    v_proj_all = -render["proj_dirs"]
    nrm_all    = render["normals"]
    pidx_all   = render["pixel_idx"]

    H_all  = v_cam_all + v_proj_all
    H_norm = np.linalg.norm(H_all, axis=1, keepdims=True)
    H_all  = np.where(H_norm > 1e-12, H_all / (H_norm + 1e-12), nrm_all)
    ndoth  = np.abs(np.einsum("ij,ij->i", nrm_all, H_all)).clip(0., 1.)

    # GGX Trowbridge-Reitz NDF, normalised by its own peak at ndoth=1.
    # D_norm = alpha^4 / (ndoth^2*(alpha^2-1) + 1)^2  where alpha = roughness (GGX α).
    # At ndoth=1 (face-on): D_norm=1.0 → always lit, never dark.
    # At ndoth=0.707 (45° tilt), roughness=0.25: D_norm≈0.014 << 0.15 → dark.
    # Note: alpha = roughness directly (NOT roughness²).  Squaring twice would
    # give α=0.0625 for roughness=0.25, making D_norm≈0.10 even face-on — wrong.
    alpha     = roughness + 1e-6                         # GGX α = roughness ∈ (0,1]
    denom     = ndoth ** 2 * (alpha ** 2 - 1.0) + 1.0
    D_norm    = alpha ** 4 / (denom ** 2 + 1e-12)       # ∈ (0, 1]

    # ── Project D_norm to image space (dense) ─────────────────────────────────
    H_img, W_img = render["res"]
    D_flat           = np.zeros(H_img * W_img, np.float32)
    D_flat[pidx_all] = D_norm.astype(np.float32)
    D_img            = D_flat.reshape(H_img, W_img)

    lit_flat           = np.zeros(H_img * W_img, bool)
    lit_flat[pidx_all] = True
    lit_img            = lit_flat.reshape(H_img, W_img)

    # Pixels where the GGX lobe is weak (D_norm below threshold) are candidates
    # for coherent patch dropout.  specular_threshold=0.15 means the lobe has
    # dropped to <15 % of peak — empirically the point where SL decoders fail.
    dark_img = lit_img & (D_img < specular_threshold)

    if not dark_img.any():
        if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")
        return keep.copy()

    # ── Connected components → patch dropout ──────────────────────────────────
    labeled, n_comp = _ndimage_label(dark_img)

    # pixel → global index in render["points"] (all visible, not just kept)
    pix2global = np.full(H_img * W_img, -1, np.int32)
    pix2global[render["pixel_idx"]] = np.arange(len(render["points"]), dtype=np.int32)

    updated_keep = keep.copy()
    for comp_id in range(1, n_comp + 1):
        comp_flat = (labeled.ravel() == comp_id)
        if comp_flat.sum() < min_patch_area_px:
            continue
        if rng.random() >= patch_dropout_rate:
            continue
        gi = pix2global[comp_flat]
        gi = gi[gi >= 0]           # filter background pixels (should be none here)
        updated_keep[gi] = False

    if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")
    return updated_keep


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Image-space reconstruction artifacts
# ══════════════════════════════════════════════════════════════════════════════

def add_image_space_effects(render, keep_mask, smooth_sigma_px=0., z_ref=2., 
                            fringe_period_px=8., sigma_fringe_corr=0., seed=0,
                            verbose=False):
    """
    Image-space reconstruction artifacts
    Two depth-image-space effects that must precede edge-artifact computation.
    Returns a shallow copy of render with updated 'points' and 't_hit'.

    Effect 1 — Range-dependent spatial smoothing  (smooth_sigma_px > 0)
    ─────────────────────────────────────────────
    NEW. At long range the projected fringe pitch covers more world area per
    pixel. Surface detail finer than ~one fringe width cannot be resolved —
    a low-pass effect on the depth map.

    σ(z) = smooth_sigma_px · (z̄/z_ref)²   where z̄ = mean depth of kept pts.
    Density-normalised Gaussian blur preserves absolute depth at NaN boundaries.

    Effect 2 — Fringe phase correlation noise  (sigma_fringe_corr > 0)
    ──────────────────────────────────────────
    NEW. Adjacent pixels within one fringe period share the same captured
    phase images → their noise is spatially correlated within ~one fringe width.
    Produces the characteristic low-frequency ripple visible in real SL scans.

    Modelled as Gaussian-filtered (σ = fringe_period_px/2) white noise scaled
    by sigma_fringe_corr, applied as a depth offset along each point's LOS.

    Parameters
    ----------
    smooth_sigma_px    : blur σ at z_ref (pixels). Typical 0.3–1.0 px.
    fringe_period_px   : fringe period (pixels). Sets correlation length.
    sigma_fringe_corr  : 1σ correlated noise amplitude (metres).
                         Typical: 0.3–0.8 × sigma_z_ref.
    """
    start = time.time()
    H, W      = render["res"]
    pixel_idx = render["pixel_idx"]
    t_hit     = render["t_hit"].copy().astype(np.float64)
    points    = render["points"].copy()

    # Effect 1 — depth smoothing
    if smooth_sigma_px > 0.:
        vis_depth = make_depth_image(render)
        z_mean    = float(t_hit[keep_mask].mean()) if keep_mask.any() else z_ref
        sigma     = smooth_sigma_px * (z_mean / z_ref) ** 2
        if sigma >= 0.25:
            valid    = np.isfinite(vis_depth).astype(np.float32)
            d_filled = np.where(np.isfinite(vis_depth), vis_depth, 0.).astype(np.float32)
            d_smooth = gaussian_filter(d_filled, sigma) / (gaussian_filter(valid, sigma) + 1e-9)
            d_smooth = np.where(gaussian_filter(valid, sigma) > 0.05, d_smooth, np.nan)
            rows, cols = pixel_idx // W, pixel_idx % W
            t_new = d_smooth[rows, cols]
            ok = np.isfinite(t_new)
            t_hit[ok] = t_new[ok]
            points[ok] = render["ray_origins"][ok] + t_hit[ok, None] * render["ray_dirs"][ok]

    # Effect 2 — fringe phase correlation
    if sigma_fringe_corr > 0.:
        rng   = np.random.default_rng(seed)
        white = rng.standard_normal((H, W)).astype(np.float32)
        corr  = gaussian_filter(white, sigma=fringe_period_px / 2.)
        corr /= np.std(corr) + 1e-12
        rows, cols = pixel_idx // W, pixel_idx % W
        depth_off  = corr[rows, cols].astype(np.float64) * sigma_fringe_corr
        los        = points - render["sensor_origin"]
        los       /= np.linalg.norm(los, axis=1, keepdims=True) + 1e-12
        points    += depth_off[:, None] * los
        t_hit     += depth_off

    out           = dict(render)
    out["points"] = points
    out["t_hit"]  = t_hit
    if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Edge artifacts  (edge bleeding + flying pixels)
# ══════════════════════════════════════════════════════════════════════════════

def _build_tree(pts): return cKDTree(pts)

def _knn_indices(pts, k, tree=None):
    if tree is None: tree = _build_tree(pts)
    _, idx = tree.query(pts, k=k+1, workers=-1)
    return idx[:, 1:].astype(np.int32)

def _smooth_falloff_3d(s, edge_width, pts, k, tree=None):
    idx  = _knn_indices(pts, k, tree)
    d2   = ((pts[idx] - pts[:, None, :])**2).sum(2)
    w    = np.exp(-d2 / (2*edge_width**2))
    w   /= w.sum(1, keepdims=True) + 1e-12
    s2   = (w * s[idx]).sum(1).clip(0,1)
    return s2*s2 / (s2*s2 + (1-s2)**2 + 1e-12)

def _bleed_dirs_image(ray_dirs, gx_img, gy_img, pixel_idx, res):
    H, W = res
    gx   = gx_img[pixel_idx//W, pixel_idx%W].astype(np.float64)
    gy   = gy_img[pixel_idx//W, pixel_idx%W].astype(np.float64)
    up   = np.where(np.abs(ray_dirs[:,1:2]) < .9,
                    np.broadcast_to([0.,1.,0.], ray_dirs.shape),
                    np.broadcast_to([1.,0.,0.], ray_dirs.shape))
    tx   = np.cross(ray_dirs, up);  tx /= np.linalg.norm(tx,1,keepdims=True)+1e-12
    ty   = np.cross(tx, ray_dirs);  ty /= np.linalg.norm(ty,1,keepdims=True)+1e-12
    b    = gx[:,None]*tx + gy[:,None]*ty
    n    = np.linalg.norm(b, axis=1, keepdims=True)
    return np.where(n > 1e-8, b/(n+1e-12), 0.)

def add_edge_artifacts(render, keep_mask=None, max_bleed=0.006,
                       edge_width=0.004, k=20, n_per_edge_pixel=1.5,
                       depth_gap_fraction=0.95, edge_thresh=0.15, seed=0,
                       verbose=False):
    """
    Edge bleeding + flying pixel injection (fully vectorised).

    Edge bleeding
    ─────────────
    Foreground points near depth discontinuities are displaced toward the
    background along the image-space Sobel depth gradient projected into 3D.
    Δ = max_bleed · Gaussian-smoothed-sigmoid(edge_strength).

    Flying pixels
    ─────────────
    A pixel straddling a depth edge integrates fringe from both surfaces.
    The SL decoder places the output point between them.

    FIX — depth distribution: was Uniform(fg, bg). Now Beta(1.5, 3.0),
    mean ≈ 0.33. A pixel crossing an edge has the nearer surface covering
    more than half its area → decoded depth biased toward foreground.

    FIX — background depth: was Python for-loop + 7×7 patch per pixel.
    Now maximum_filter(full_depth, 7), computed once — O(H×W).
    """
    start = time.time()
    rng = np.random.default_rng(seed)
    if keep_mask is not None:
        pts  = render["points"][keep_mask];  nrm  = render["normals"][keep_mask]
        rdir = render["ray_dirs"][keep_mask]; pidx = render["pixel_idx"][keep_mask]
    else:
        pts  = render["points"];  nrm  = render["normals"]
        rdir = render["ray_dirs"]; pidx = render["pixel_idx"]

    H, W       = render["res"]
    vis_depth  = make_depth_image(render)
    # ALL hits, including shadow-occluded ones.
    full_depth = render["depth_img"].copy()
    edge_img   = compute_edge_strength(vis_depth)
    gx, gy     = compute_depth_gradient(vis_depth)

    # ── Edge bleeding ──────────────────────────────────────────────────────
    strength = edge_img[pidx//W, pidx%W].astype(np.float64)
    tree     = _build_tree(pts)
    falloff  = _smooth_falloff_3d(strength, edge_width, pts, k, tree)
    bleed    = _bleed_dirs_image(rdir, gx, gy, pidx, render["res"])
    bled_pts = pts + max_bleed * falloff[:, None] * bleed

    # ── Flying pixels — vectorised ─────────────────────────────────────────
    full_filled = np.where(np.isfinite(full_depth), full_depth, -np.inf)
    bg_depth    = maximum_filter(full_filled, size=7).astype(np.float32)
    bg_depth    = np.where(bg_depth > -1e30, bg_depth, np.nan)

    cand = np.where((edge_img > edge_thresh) & np.isfinite(full_depth))
    cand_flat = (cand[0] * W + cand[1]).astype(np.int64)

    if len(cand_flat):
        fg_d = full_depth.ravel()[cand_flat]
        bg_d = bg_depth.ravel()[cand_flat]
        gaps = bg_d - fg_d
        ok   = np.isfinite(bg_d) & (gaps > 1e-4)
        cand_flat, fg_d, gaps = cand_flat[ok], fg_d[ok], gaps[ok]
        es   = edge_img.ravel()[cand_flat]
        n_sp = rng.poisson(n_per_edge_pixel * es.astype(np.float64))
        tot  = int(n_sp.sum())
        if tot > 0:
            rep     = np.repeat(np.arange(len(cand_flat)), n_sp)
            # Beta(1.5, 3.0): mean≈0.33, foreground-biased mixed-pixel distribution
            u       = rng.beta(1.5, 3.0, tot)
            ts      = fg_d[rep] + u * depth_gap_fraction * gaps[rep]
            o_rep   = render["ray_origins_img"][cand_flat[rep]]
            d_rep   = render["ray_dirs_img"][cand_flat[rep]]
            fly_pts = o_rep + ts[:, None] * d_rep
            pix2pt  = np.full(H*W, -1, np.int32)
            pix2pt[pidx] = np.arange(len(pts), dtype=np.int32)
            pi      = pix2pt[cand_flat[rep]]
            si      = np.where(pi >= 0, pi, 0)
            fly_nrm = np.where(pi[:, None] >= 0, nrm[si], -d_rep)
            return (np.vstack([bled_pts, fly_pts]),
                    np.vstack([nrm,      fly_nrm]))

    if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")
    return bled_pts, nrm


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Structured outliers  (multipath + pepper + scan-line banding)
# ══════════════════════════════════════════════════════════════════════════════

def _snr_proxy(render):
    if "snr_proxy" in render:
        return render["snr_proxy"]
    raw = render["cos_cam"] * render["cos_proj"] / (render["proj_dist"]**2 + 1e-12)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)


def subset_render(render, keep_mask):
    """Shallow copy of a render dict with every per-point array filtered by keep_mask.

    Image buffers, res and origin scalars pass through unchanged.
    """
    return {k: (v[keep_mask] if k in _PER_POINT_KEYS else v) for k, v in render.items()}


def add_multipath_outliers(render, fringe_period=0.003, rate=0.01,
                           max_order=2, concavity_thresh=0.10,
                           phase_sigma_rel=0.07, seed=0,
                           verbose=False):
    """
    Structured outliers
    Ghost-surface points from multi-path interference in concave regions.

    Physical model
    ──────────────
    A bounced projector ray arrives at the surface with a longer optical path
    → positive phase offset → the ghost is almost always BEHIND (farther from
    sensor).

    FIX 1 — Sign bias was 50/50. Now 75 % positive (ghost farther from sensor).

    FIX 2 — Displacement was exactly ±k·λ (purely quantised). Real multipath
    adds phase noise from the indirect-path BRDF uncertainty.
    Now:  t_ghost = t_true + sign·k·λ + N(0, λ·phase_sigma_rel)

    Parameters
    ----------
    fringe_period    : SL fringe period in scene units (e.g. 0.003 m at 2 m range).
    rate             : base fraction of concave pixels that spawn a ghost.
    max_order        : maximum fringe-order displacement.
    concavity_thresh : normalised Laplacian threshold to identify concavities.
    phase_sigma_rel  : Gaussian spread around each integer order, as fraction
                       of fringe_period. Typical 0.05–0.10.
    """
    start = time.time()
    rng  = np.random.default_rng(seed)
    H, W = render["res"]
    depth = make_depth_image(render)
    d     = np.where(np.isfinite(depth), depth, 0.).astype(np.float64)
    lap   = np.roll(d,1,0)+np.roll(d,-1,0)+np.roll(d,1,1)+np.roll(d,-1,1)-4*d
    fv    = lap[np.isfinite(depth)]
    if not len(fv) or fv.max() <= 0:
        return np.empty((0,3)), np.empty((0,3))
    lap_n   = np.clip(lap / (fv.max()+1e-12), 0, 1)
    concave = (lap_n > concavity_thresh) & np.isfinite(depth)
    pix2pt  = np.full(H*W, -1, np.int32)
    pix2pt[render["pixel_idx"]] = np.arange(len(render["points"]), dtype=np.int32)
    concave_rc = np.argwhere(concave)
    if not len(concave_rc):
        return np.empty((0, 3)), np.empty((0, 3))
    rows_c, cols_c = concave_rc.T
    pt_c  = pix2pt[rows_c * W + cols_c]
    valid = pt_c >= 0
    rows_c, cols_c, pt_c = rows_c[valid], cols_c[valid], pt_c[valid]
    p_spawn = rate * lap_n[rows_c, cols_c]
    spawn   = rng.random(len(pt_c)) < p_spawn
    pt_c    = pt_c[spawn]
    if not len(pt_c):
        return np.empty((0, 3)), np.empty((0, 3))
    k    = rng.integers(1, max_order + 1, len(pt_c))
    sign = np.where(rng.random(len(pt_c)) < 0.75, 1, -1)   # 75 % behind surface
    eps  = rng.normal(0., fringe_period * phase_sigma_rel, len(pt_c))
    t_new = render["t_hit"][pt_c] + sign * k * fringe_period + eps
    ok = t_new > 0
    if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")
    return (render["ray_origins"][pt_c[ok]] + t_new[ok, None] * render["ray_dirs"][pt_c[ok]],
            render["normals"][pt_c[ok]].copy())


def add_pepper_noise(render, rate=0.005, depth_sigma_rel=0.05, seed=0,
                     verbose=False):
    """
    Isolated wrong-depth points from single-pixel decoder failures.
    Rate ∝ (1 − SNR); depth ~ N(z_true, z_true·depth_sigma_rel).
    """
    start = time.time()
    rng  = np.random.default_rng(seed)
    snr  = _snr_proxy(render)
    inv  = 1. - snr
    p    = np.clip(inv / (inv.mean()+1e-12) * rate, 0, 1)
    mask = rng.random(len(p)) < p
    if not mask.any():
        return np.empty((0,3)), np.empty((0,3))
    t    = render["t_hit"][mask]
    t_n  = rng.normal(t, t*depth_sigma_rel).clip(1e-3)
    if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")
    return (render["ray_origins"][mask] + t_n[:,None] * render["ray_dirs"][mask],
            render["normals"][mask].copy())


def add_scan_line_banding(points, normals, pixel_idx, res, sensor_origin,
                          band_amplitude=0.0003, fringe_period_px=8, seed=0,
                         verbose=False):
    """
    Scan-line banding
    Periodic depth banding from phase-unwrapping miscounts.

    NEW effect.

    Physical model
    ──────────────
    Multi-frequency SL phase unwrapping resolves fringe-order ambiguity by
    comparing decoded phases across frequencies. At fringe-period boundaries,
    thermal drift and projector non-uniformity cause the unwrapper to miscount
    by ±1 order. This shifts all pixels within the same fringe band by the same
    shared offset, producing the 'staircase' banding visible in real SL scans.

    All pixels in the same horizontal band (height = fringe_period_px rows)
    receive an independent Gaussian depth offset with σ = band_amplitude.
    Points without a pixel index (outliers, injected flying pixels, pixel_idx=-1)
    are left unchanged.

    Parameters
    ----------
    band_amplitude    : 1σ depth shift per band (metres). Typical 0.1–0.5 mm.
    fringe_period_px  : band height in pixels. Match your projector fringe pitch.
    """
    start = time.time()
    H, W     = res
    rng      = np.random.default_rng(seed)
    n_bands  = H // fringe_period_px + 2
    offsets  = rng.normal(0., band_amplitude, n_bands)
    has_pix  = np.asarray(pixel_idx) >= 0
    rows     = np.where(has_pix, np.asarray(pixel_idx) // W, 0)
    z_off    = np.where(has_pix, offsets[rows // fringe_period_px], 0.)
    los      = points - sensor_origin
    los     /= np.linalg.norm(los, axis=1, keepdims=True) + 1e-12
    if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")
    return points + z_off[:, None] * los


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — Sensor electronics noise
# ══════════════════════════════════════════════════════════════════════════════

def _axial_depth(pts, origin): return np.linalg.norm(pts - origin, axis=1)

def _sensor_ray(pts, origin):
    r = origin - pts; return r / (np.linalg.norm(r, axis=1, keepdims=True)+1e-12)

def _depth_scale(z, z_ref, model):
    if model == "quadratic": return (z/z_ref)**2
    if model == "linear":    return  z/z_ref
    raise ValueError(f"depth_model must be 'quadratic' or 'linear', got {model!r}")


class _Perlin3D:
    """Vectorised 3D gradient Perlin noise. Shared by all noise stages."""
    _G = np.array([[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
                   [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
                   [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]], np.float32)
    def __init__(self, seed=0, size=256):
        rng=np.random.default_rng(seed); p=np.arange(size,dtype=np.int32); rng.shuffle(p)
        self._p=np.tile(p,2); self._sz=size
    @staticmethod
    def _fade(t): return t*t*t*(t*(t*6-15)+10)
    def _g(self,h,x,y,z):
        g=self._G[h%12]; return g[:,0]*x+g[:,1]*y+g[:,2]*z
    def __call__(self, pts, scale=1.):
        xyz=pts*scale
        Xi=np.floor(xyz[:,0]).astype(np.int32)&(self._sz-1)
        Yi=np.floor(xyz[:,1]).astype(np.int32)&(self._sz-1)
        Zi=np.floor(xyz[:,2]).astype(np.int32)&(self._sz-1)
        xf=xyz[:,0]-np.floor(xyz[:,0]); yf=xyz[:,1]-np.floor(xyz[:,1]); zf=xyz[:,2]-np.floor(xyz[:,2])
        u,v,w=self._fade(xf),self._fade(yf),self._fade(zf); p=self._p
        A=p[Xi]+Yi; AA=p[A]+Zi; AB=p[A+1]+Zi
        B=p[Xi+1]+Yi; BA=p[B]+Zi; BB=p[B+1]+Zi
        L=lambda a,b,t: a+t*(b-a)
        x1=L(self._g(p[AA],  xf,  yf,  zf), self._g(p[BA],  xf-1,yf,  zf), u)
        x2=L(self._g(p[AB],  xf,  yf-1,zf), self._g(p[BB],  xf-1,yf-1,zf), u)
        x3=L(self._g(p[AA+1],xf,  yf,  zf-1),self._g(p[BA+1],xf-1,yf,  zf-1),u)
        x4=L(self._g(p[AB+1],xf,  yf-1,zf-1),self._g(p[BB+1],xf-1,yf-1,zf-1),u)
        return L(L(x1,x2,v),L(x3,x4,v),w)


def _systematic_bias(pts, origin, accuracy_ref, z_ref, model, bias_scale, seed):
    """Smooth deterministic warp (VDI/VDE accuracy spec). Fixed across frames."""
    field = _Perlin3D(seed=seed)(pts, scale=bias_scale)
    z     = _axial_depth(pts, origin)
    amp   = accuracy_ref * _depth_scale(z, z_ref, model)
    return pts + (amp * field)[:, None] * _sensor_ray(pts, origin)


def _fixed_pattern_noise(pts, origin, fpn_sigma, fpn_scale, seed,
                          pixel_idx, res):
    """
    Per-pixel persistent depth bias from sensor non-uniformity.

    FIX — Was evaluated in world-space 3D. FPN is a SENSOR PIXEL property:
    the same pixel always reads the same bias regardless of what it images.
    The same scene point scanned from a different pose gets different FPN.

    Correct domain: image-space (u, v) ∈ [0, fpn_scale]² for canonical points
    (pixel_idx >= 0).  Outlier/injected points (pixel_idx = -1) fall back to
    world-space Perlin with a seed offset to decorrelate from systematic bias.
    """
    ray = _sensor_ray(pts, origin)
    if pixel_idx is not None and res is not None:
        H, W    = res
        pidx    = np.asarray(pixel_idx)
        has_pix = pidx >= 0
        uv      = np.zeros((len(pts), 3), np.float64)
        if has_pix.any():
            r_ = pidx[has_pix] // W; c_ = pidx[has_pix] % W
            uv[has_pix, 0] = c_ / W * fpn_scale
            uv[has_pix, 1] = r_ / H * fpn_scale
        field = _Perlin3D(seed=seed+1000)(uv, scale=1.)
        if (~has_pix).any():
            field[~has_pix] = _Perlin3D(seed=seed+1000)(pts[~has_pix], scale=fpn_scale)
    else:
        field = _Perlin3D(seed=seed+1000)(pts, scale=fpn_scale)
    return pts + (fpn_sigma * field)[:, None] * ray


def _axial_noise(pts, origin, sigma_z_ref, z_ref, model, rng, cos_proj):
    """
    Gaussian noise along the sensor ray (Z-repeatability spec).

    FIX — Was angle-independent. For SL, fringe stretching at oblique
    projector incidence (angle θ from normal, cos θ = cos_proj) reduces the
    number of fringe cycles per pixel → lower phase-measurement SNR →
    higher depth noise.

        σ_eff = σ_ref · depth_scale(z) / cos_proj

    cos_proj clipped to [0.2, 1] to avoid divergence (heavily oblique points
    should be dropped by compute_dropout_mask before reaching here).
    Pass cos_proj=None to disable (isotropic behaviour).
    """
    z     = _axial_depth(pts, origin)
    sigma = sigma_z_ref * _depth_scale(z, z_ref, model)
    if cos_proj is not None:
        sigma = sigma / np.clip(np.asarray(cos_proj, np.float64), 0.2, 1.)
    return pts + rng.normal(0., sigma)[:, None] * _sensor_ray(pts, origin)


def _quantisation_noise(pts, origin, depth_res_ref, z_ref, model, rng):
    """Uniform ±½ LSB depth snap. LSB scales with depth same as axial noise."""
    z   = _axial_depth(pts, origin)
    lsb = depth_res_ref * _depth_scale(z, z_ref, model)
    q   = rng.uniform(-.5, .5, len(pts)) * lsb
    return pts + q[:, None] * _sensor_ray(pts, origin)


def _ray_jitter(pts, origin, sigma_px, sigma_global, rng):
    """
    Lateral XY displacement from angular ray uncertainty.

    Per-pixel (sigma_px): independent per pixel (lens PSF, aberrations).
    Global (sigma_global): one draw per frame, same for all points (vibration).
    Lateral error = z · δθ (small-angle approximation).
    """
    z   = _axial_depth(pts, origin)
    ray = _sensor_ray(pts, origin)
    ref = np.where(np.abs(ray[:,1:2]) < .9,
                   np.broadcast_to([0.,1.,0.], ray.shape),
                   np.broadcast_to([1.,0.,0.], ray.shape))
    t1 = np.cross(ray, ref); t1 /= np.linalg.norm(t1,1,keepdims=True)+1e-12
    t2 = np.cross(ray, t1);  t2 /= np.linalg.norm(t2,1,keepdims=True)+1e-12
    N  = len(pts)
    lat  = z[:,None] * (rng.normal(0,sigma_px,N)[:,None]*t1
                       +rng.normal(0,sigma_px,N)[:,None]*t2)
    lat += z[:,None] * (rng.normal(0,sigma_global)*t1
                       +rng.normal(0,sigma_global)*t2)
    return pts + lat


def add_sensor_noise(points, normals, sensor_origin,
                     sigma_z_ref=0.0002, accuracy_ref=0.0002, z_ref=2.0,
                     depth_model="quadratic", fpn_sigma=0.00003, fpn_scale=20.,
                     bias_scale=3., depth_res_ref=0.0001,
                     sigma_angle_px=1e-4, sigma_angle_global=2e-5,
                     pixel_idx=None, res=None, cos_proj=None, seed=0,
                     verbose=False):
    """
    Sensor electronics noise
    Full sensor electronics noise chain in physical signal order.

    Stages
    ──────
      1. Systematic bias     — deterministic smooth warp; identical every frame.
      2. Fixed pattern noise — per-pixel persistent bias evaluated in IMAGE
                               SPACE (pixel coords) for canonical points.
                               [FIX] was world-space Perlin — wrong domain.
      3. Axial Gaussian      — Z-repeatability; amplified by 1/cos_proj at
                               oblique projector incidence.
                               [FIX] was angle-independent.
      4. Quantisation        — uniform ±½ LSB depth snap.
      5. Ray angular jitter  — lateral XY: per-pixel independent + global
                               correlated rigid frame shift.

    New parameters vs previous version
    ────────────────────────────────────
    pixel_idx  : (N,) int64. -1 for injected outlier/flying-pixel points.
                 Enables image-space FPN for canonical points.
    res        : (H,W). Required with pixel_idx.
    cos_proj   : (N,) projector incidence cosine for axial noise amplification.
                 Use 1.0 for injected points (no amplification).
    """
    start = time.time()
    rng = np.random.default_rng(seed)
    pts = _systematic_bias(points, sensor_origin, accuracy_ref, z_ref,
                           depth_model, bias_scale, seed)
    pts = _fixed_pattern_noise(pts, sensor_origin, fpn_sigma, fpn_scale,
                               seed, pixel_idx, res)
    pts = _axial_noise(pts, sensor_origin, sigma_z_ref, z_ref,
                       depth_model, rng, cos_proj)
    pts = _quantisation_noise(pts, sensor_origin, depth_res_ref,
                              z_ref, depth_model, rng)
    pts = _ray_jitter(pts, sensor_origin, sigma_angle_px, sigma_angle_global, rng)
    if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")
    return pts


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — Surface microgeometry  (fBm / Perlin normal displacement)
# ══════════════════════════════════════════════════════════════════════════════

class _FBM:
    """Fractal Brownian Motion. H→1 = smooth; H→0 = rough."""
    def __init__(self, H=0.75, octaves=6, lacunarity=2., seed=0):
        gain=lacunarity**(-H); amps=gain**np.arange(octaves)
        self._amps=amps/amps.sum()
        self._ps=[_Perlin3D(seed=seed+i) for i in range(octaves)]
        self._lac=lacunarity
    def __call__(self, pts, scale=1.):
        n=np.zeros(len(pts),np.float64); f=scale
        for a,p in zip(self._amps,self._ps):
            n+=a*p(pts,scale=f); f*=self._lac
        return n


def add_surface_noise(points, normals, mode="fbm", amplitude=0.004,
                      scale=8., H=0.75, octaves=6, lacunarity=2., seed=0,
                      verbose=False):
    """
    Surface microgeometry
    Displace each point along its surface normal by an analytical noise field.
    Models machined tooling marks, casting texture, or paint grain not in CAD.
    Applied last — on top of all electronics and outlier effects.

    Parameters
    ----------
    mode      : "fbm" (natural spectrum, recommended) | "perlin" (single octave).
    amplitude : max normal-direction displacement (metres).
    scale     : spatial frequency; higher → finer detail.
    H         : [fBm] Hurst exponent. 0.75 = natural rough surface.
    octaves   : [fBm] number of frequency octaves.
    lacunarity: [fBm] frequency multiplier per octave.
    """
    start = time.time()
    if mode == "perlin":
        fn = _Perlin3D(seed=seed)
    elif mode == "fbm":
        fn = _FBM(H=H, octaves=octaves, lacunarity=lacunarity, seed=seed)
    else:
        raise ValueError(f"mode must be 'perlin' or 'fbm', got {mode!r}")
    if verbose: rp(f"{sys._getframe().f_code.co_name} took {np.round(time.time() - start, 6)}s")
    return points + amplitude * fn(points, scale=scale)[:, None] * normals
