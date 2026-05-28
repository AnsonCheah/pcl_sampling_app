# sensor/ change log

---

## 2026-04-16 — Noise stage additions + multi-label segmentation

### 1. `add_projector_nonuniformity()` — new stage (scene_render.py, Section 3)

**Physical motivation**  
DMD/LCD projector pixels have individual gain variation (~2–8 % for typical industrial projectors). This causes non-uniform fringe contrast across the field — a distinct effect from camera FPN (which is sensor-side). The same scene point viewed from two camera positions has the same projector gain but different camera FPN.

**Model**  
3D Perlin noise evaluated in projector direction space (`proj_dirs`). Gain G ∈ [1 − σ, 1 + σ] is multiplied into `snr_proxy` before dropout.

**Default parameters**  
| Parameter | Value | Physical meaning |
|---|---|---|
| `proj_fpn_sigma` | 0.05 | ±5% gain variation — typical DMD projector |
| `proj_fpn_scale` | 20.0 | Spatial frequency in projector direction space |

**Call position**: before `compute_dropout_mask()` in both `synthetic_stage.py` and `mujoco_bin_scene.py`.

**Test**: `test_projector_nonuniformity_modulates_snr`, `test_projector_nonuniformity_is_deterministic` — **PASS**

---

### 2. `add_specular_patch_missing()` — new stage (scene_render.py, Section 3)

**Physical motivation**  
Real SL sensors exhibit *contiguous* missing regions when the specular lobe of a smooth surface sweeps away from the camera. The existing per-point `_specular_keep` drops points individually (stochastic). This new stage finds connected image-space regions where GGX NDF D(H) is uniformly low and drops the entire patch together (coherent failure).

**Model**  
GGX Trowbridge-Reitz NDF D(H) where H = normalize(v_cam + v_proj). Threshold → connected components (`scipy.ndimage.label`) → stochastic patch dropout.

**Default parameters** (midpoint: matte plastic roughness ≈ 0.4, brushed metal ≈ 0.15)  
| Parameter | Value | Physical meaning |
|---|---|---|
| `roughness` | 0.25 | GGX α parameter; lower = sharper lobe |
| `specular_threshold` | 0.15 | Normalised NDF below which pixel is "specular dark" |
| `min_patch_area_px` | 20 | Min connected-component area to trigger dropout |
| `patch_dropout_rate` | 0.6 | Probability a qualifying patch is dropped |

**Call position**: after `compute_dropout_mask()`, before `add_image_space_effects()`.

**Test**: `test_specular_patch_missing_reduces_keep`, `test_specular_patch_missing_lambertian_unchanged` — **PASS**

---

### 3. `_specular_keep_anisotropic()` + `anisotropy` param in `compute_dropout_mask()` (scene_render.py)

**Physical motivation**  
Brushed/milled/rolled metal surfaces have directional micro-grooves (roughness α_t ≠ α_b in tangent plane). The specular lobe becomes an elongated ellipse, producing stripe-shaped rather than round missing regions.

**Model**  
Ward anisotropic BRDF. Tangent frame (t, b) built from world-space brush direction projected into the local tangent plane. Blended with isotropic Ward via `anisotropy` ∈ [0, 1].

**Default parameters**  
| Parameter | Value | Physical meaning |
|---|---|---|
| `anisotropy` | 0.0 | 0 = isotropic GGX (no change to existing); 1 = full Ward |
| `alpha_t` | 0.10 | Roughness along brush direction |
| `alpha_b` | 0.40 | Roughness across brush direction |
| `brush_dir` | None | None → horizontal [1,0,0] |

**Regression safety**: `anisotropy=0.0` delegates to existing `_specular_keep()` unchanged.

**Test**: `test_anisotropic_roughness_zero_matches_isotropic`, `test_anisotropic_roughness_elongates_dropout` — **PASS**

---

### 4. Multi-label segmentation (segment_instances.py)

**Physical motivation**  
Real 2D segmentation networks output one independent binary mask per instance — masks CAN overlap at shared boundary pixels. The previous implementation called `_resolve_conflicts()` to collapse each pixel to a single winner, discarding overlap that a real network produces.

**Changes**  
- `build_perturbed_masks()`: added `resolve_conflicts=False` parameter. Default `False` keeps overlapping masks. Pass `True` to restore the old single-label behaviour.
- `segment_point_cloud()`: return type changed from `(N,) int32` to `dict[int, np.ndarray]` (`{geom_id: (N,) bool}`). Injected points (pixel_idx == -1) are `False` in all masks.

**Call site updates**  
- `stages/synthetic_stage.py`: `labels` → `label_masks`, `np.unique(labels[labels>=0])` → `list(label_masks.keys())`, `pts[labels==inst_id]` → `pts[label_masks[inst_id]]`
- `physics/mujoco_bin_scene.py __main__`: same changes

**Test**: `test_segment_returns_dict`, `test_no_resolve_conflicts_allows_overlap`, `test_resolve_conflicts_flag_eliminates_overlap`, `test_injected_points_are_false_in_all_masks`, `test_single_instance_no_overlap` — **PASS**

---

## Test files added

| File | Coverage |
|---|---|
| `sensor/tests/_fixtures.py` | Shared synthetic scene builders (no file I/O) |
| `sensor/tests/test_scene_render.py` | 20 tests: all existing stages + 3 new stages |
| `sensor/tests/test_segment_instances.py` | 13 tests: existing + 5 multi-label tests |

Run from project root:
```
python sensor/tests/test_scene_render.py
python sensor/tests/test_segment_instances.py
```

---

## Implementation notes (issues found and fixed during test runs)

### GGX α convention in `add_specular_patch_missing`
The GGX NDF takes `alpha = roughness` directly (not `roughness²`).  Using `alpha = roughness²` for `roughness=0.25` gives `alpha=0.0625`, which makes `D_norm ≈ 0.10` even for a face-on surface — incorrectly classifying it as dark.  Fixed: `alpha = roughness + 1e-6`.

### Normal estimation at reduced test resolution
`estimate_normals` uses `radius=0.005 m` (5 mm).  At 120×160 resolution and 1.5 m range the point spacing is ~6.7 mm, so each point has zero neighbours and PCA returns the default `[0,0,−1]` regardless of surface tilt.  Fixed: `_fixtures.py` passes `normal_radius=0.020 m` (3× point spacing) to `scene_render`.  Production runs at higher resolution where 5 mm radius is appropriate.

### Test assertion for `test_specular_patch_missing_reduces_keep`
Uses `patch_dropout_rate=1.0, min_patch_area_px=1` to guarantee deterministic dropout whenever a qualifying dark region exists, removing sensitivity to the RNG seed.
