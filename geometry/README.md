# geometry

Base layer for 3D math, file I/O, and the shared `O3DSceneObject` dataclass. No local imports -- all other packages depend on this one.

## Contents

| File | Description |
|------|-------------|
| `geom_utils.py` | `O3DSceneObject`, transform/quaternion helpers, normals, diameter and spacing, Open3D<->trimesh conversion |
| `file_utils.py` | PLY export with the comment protocol, scene/sample listing, file dialogs |
| `math_utils.py` | `find_cdf_knee` -- CDF knee split, used for adaptive downsampling |
| `ambiguity.py` | Ambiguity-axis search, fold fitting, per-point heat map |
| `mesh_repair.py` | Mesh diagnosis, debris removal, unit normalisation |
| `tray_utils.py` | Footprint polygons, tray collision frames and conforming height fields |
| `convex_decomp.py` | VHACD convex decomposition wrapper |
| `visualize_ambiguity.py` | Standalone viewer for an ambiguity profile |

## `O3DSceneObject`

Universal geometry carrier passed between all packages:

```python
@dataclass
class O3DSceneObject:
    geom: o3d.geometry.Geometry3D
    ref_geom: Optional[o3d.geometry.Geometry3D] = None
    material: Optional[rendering.MaterialRecord] = None
    id: Optional[int] = None       # assigned by scene_render() as a side effect
    T_gt: Optional[np.ndarray] = None
    xyz0: Optional[np.ndarray] = None
    xyz1: Optional[np.ndarray] = None
    overlap: Optional[float] = None
```

**`id` is `None` until `sensor.scene_render()` assigns it.** Do not read `.id` before calling `scene_render`.

## PLY comment keys

`file_utils.py` writes ground-truth pose data as PLY header comments. These keys are parsed by downstream loaders -- do not rename them:

```
geocenter_x, geocenter_y, geocenter_z
geocenter_qx, geocenter_qy, geocenter_qz, geocenter_qw
gt_x, gt_y, gt_z
gt_qx, gt_qy, gt_qz, gt_w
```

Note the ground-truth scalar term is `gt_w`, **not** `gt_qw` -- that is what `render_stage.py` writes and what `MM_Optimizer/optimizer_utils.read_gt_pose_from_ply` requires.

The `geocenter_*` keys carry the model frame expressed in the pre-recentre frame, i.e. where
the geocenter origin sat before `recenter_mesh_pcd` moved everything. They read as identity on
a bundle that was never recentred, which is honest: nothing was applied.

## Key helpers

```python
# Transforms -- the single definitions; do not re-inline these
make_transform(rotation=None, translation=None) -> np.ndarray      # 4x4
pose_to_matrix([x, y, z, qw, qx, qy, qz])       -> np.ndarray      # 4x4, scalar-first
mat_to_wxyz(T)                                  -> np.ndarray      # (w, x, y, z)

# Model frame. Returns the transform to APPLY; translation is in column 3.
pcd_geocenter(pcd, axis=None) -> np.ndarray                        # 4x4

# Scale references. model_diameter is THE diameter definition (see below).
model_diameter(obj) -> float
median_spacing(obj) -> float

# Camera extrinsic from position + look-at
camera_view_matrix(eye, center, up) -> np.ndarray                  # 4x4

# CDF knee split, used for adaptive downsampling and edge extraction
find_cdf_knee(values) -> tuple[float, int, int]   # (threshold, percentile, index)

# Distinct colour per index, golden-ratio hue walk; caller picks the tone
golden_hue_color(i, saturation, value) -> tuple[float, float, float]

# Open3D display wrapper (blocking)
o3d_display(geometries: list) -> o3d.visualization.Visualizer
```

## Constraints

- **One-way imports**: this package must never import `sensor`, `physics`, `registration`, or `stages`.
- **`pcd_geocenter()` returns a 4x4 transform, not a point** -- the matrix you apply to the cloud to put it in the frame. Translation is in **column 3** (`[:3, 3]`), row 3 is `[0, 0, 0, 1]`. (This note previously claimed row 3; it was wrong, and `SaveStage` read row 3 for years, writing `geocenter_x/y/z = 0.0` into every exported PLY.)
- **`reference_frames_agree(a, b)`** -- returns `None` when two reference clouds are the same export in the same model frame, else a reason. The model frame is reproducible only while the mesh *and* the sampling settings are unchanged, so any code pairing a reference cloud with generated scenes should check.

## Ambiguity threshold calibration

The thresholds in `AmbiguityConfig` are calibrated, not chosen. The evidence is recorded
here so they can be retuned deliberately; the dataclass itself just states what each knob
does.

**Verification tolerance** (`epsilon_spacing_factor = 3.0`). Set by the cloud's own
resolution, not the part size: a rotated point lands *between* samples, so anything below
the sampling pitch reports missing symmetry that is really missing resolution. The factor
is the sample's covering radius, not its nearest-neighbour distance. Calibrated on exact
symmetries of cylinder / torus / box / sphere, which reach 0.998-0.999 at 3.0, while
non-symmetry rotations of the same parts stay at 0.20-0.31.

**Global classification** (`global_area_frac = 0.95`, `global_view_frac = 0.98`,
`global_view_area_floor = 0.80`). Surface coverage alone is too strict for manufactured
parts. Validated against BOP's published symmetry annotations for the 30 T-LESS objects:
10 parts BOP calls symmetric were missed on coverage alone, their best `area_fraction`
spanning 0.81-0.95 -- a symmetric body with a small boss, hole or chamfer breaking exact
agreement at the 5-20% level. All 10 had `view_fraction == 1.00` and the correct fold, so
the axis was recovered perfectly and only the label was wrong. Hence the `view_fraction`
clause: an axis that makes every viewpoint ambiguous will flip the matcher from anywhere.

That clause alone is too permissive, so it carries a coverage floor. Without one, the
three BOP-asymmetric T-LESS objects all get promoted to global -- they too have an axis
ambiguous from every viewpoint, explaining 0.71-0.80 of the surface. They are *nearly*
symmetric; BOP calls them asymmetric because with the whole model in hand the poses are
separable.

**The separation is thin.** Across the 30 objects, symmetric parts bottom out at
`area_fraction` 0.809 and asymmetric ones top out at 0.803. A 0.006 gap is not a natural
boundary -- it is a continuum of "how nearly symmetric", and the threshold is calibrated on
30 samples either side of it. It gets 29/30. Do not read it as a law: where a published
annotation exists, prefer it. `registration/ppf/bench/metrics.py` already does, falling
back to this classification only for parts BOP has never seen.

**Per-view survival** (`f_tau = 0.40`). This asks "would the matcher plausibly land on
this pose", *not* BOP-Distrib's "is this pose provably indistinguishable". Their tau (~28
points, well under 1% of the model) answers the second question, and using it here rejects
every real case: ambiguity on a manufactured part is partial -- a large chunk of the visible
patch aligns elsewhere and the rest does not, which is exactly the pose a matcher scoring
by inlier fraction will return. Calibrated on 25333MB000: 300 random rotations reach a
best-per-view explained fraction of p50 0.033, p99 0.273, max 0.315, and none reach 0.60;
the genuine ambiguity axes reach 0.709-0.735. Requiring 60% explained sits in that gap.

**Fold acceptance** (`fold_pass_fraction = 0.5`). A candidate fold order is accepted on a
*strict majority* of its wanted angles passing the explains-probe, not all of them. An axis
does not have to be a global symmetry to be worth searching: where 3 of 4 quadrants around a
feature carry an identical flat plane and the 4th does not, the rotations that skip the odd
quadrant are genuinely ambiguous while the rest are not. Measured on 25333MB000's disc axis
(`area_fraction` 0.497, bar 0.440): 90deg explains 0.489 and 270deg explains 0.481, but
180deg only 0.317. Under the old all-pass rule that one failure knocked out every order down
to C2 and the axis was reported C1 -- `angleStep` 360, which is MechVision's *off* (it sweeps
`minAngle..maxAngle` in `angleStep` increments), so a real ambiguity went unsearched. Majority
is a strict relaxation of all-pass, so every globally symmetric part is unaffected; the
regression suite pins cylinder/sphere/cone at 0, box at C2, and the 3/4/6-prisms at their
exact folds. Continuity (fold 0) deliberately stays on all-pass: calling a partially
ambiguous axis continuous would disable orientation scoring on it entirely, which is worse
than under-reporting the fold.

**Ranking exponent** (`rank_area_exponent = 2.0`). `score = view_fraction *
area_fraction ** k` estimates P(the matcher returns this wrong pose): `view_fraction` is
how often the ambiguity is geometrically available, `area_fraction` how much of the whole
model still coincides. `k` models the second term's shape -- `k=0` means verification
ignores the unseen model (equivalent to `onlyConsiderVisibleSurfaceOfModel=True`), `k=1` a
linear fit score, `k>1` reflects verification being a *threshold*, which collapses
acceptance faster than linearly once overlap drops below it. A logistic centred on the
effective threshold would be the honest form; this is a one-knob approximation. Calibrated
on 25333MB000, where only the off-centroid disc axis produces flips in real scenes: it
leads for `k > 1.40`, and by 32% at `k = 2.0`. Ranking is a pure function of the stored
per-axis metrics, so `rank_axes` can retune this without re-running the analysis.

**Frame stability.** `pcd_geocenter`'s in-plane X direction is well conditioned in
practice: dropping 0.1% / 1% / 5% of 25333MB000's points rotates it by 0.045 / 0.35 / 0.21
degrees, its in-plane PCA eigenvalue ratio being 1.20 (the direction only becomes arbitrary
as that ratio approaches 1.0). What is *not* stable is which ambiguity axis wins -- see the
determinism note in `geometry/CLAUDE.md`.

## Axis ranking: score decides, fold only breaks ties

`rank_axes` orders by score. The fold preference is a **tie-break only, among global axes**:
of two axes equally likely to be returned, prefer the one that costs more to get wrong. A
hex prism's C2 and C6 axes are both global and score within a thousandth of each other, and
reporting C2/180deg for a part that needs C6/60deg leaves two thirds of the ambiguity
unmitigated. A cylinder's continuous axis beats its perpendicular C2s the same way.

The tie-break must **not** reach view-dependent axes. Their scores are two orders of
magnitude smaller (0.01-0.03, not ~1.0), so an absolute `round(score, 2)` window puts every
one of them in a single bucket and silently promotes fold to the primary sort key. Measured
on 25333MB000: six axes scoring 0.0017-0.0172, the highest being the off-centroid disc axis
at fold 1 -- demoted to rank 2 behind two C2 axes scoring *less*, because 0.0172 and 0.0170
both round to 0.02 and fold 2 > fold 1. (That measurement predates `fold_pass_fraction`;
the same axis now fits C4 and leads on score by a wide margin. The tie-break reasoning is
unchanged -- an absolute window would still collapse view-dependent scores into one bucket.) That is exactly the ranking `rank_area_exponent` was
calibrated to avoid. Hence the window is a fraction of the leading score, not an absolute
step, so it means the same thing whether scores sit near 1.0 or near 0.01.

## Why `analyse_ambiguity` recentres before analysing

The result is *supposed* to be translation-invariant and is not. A part left in assembly
coordinates (25333MB000's STL sits 0.85 m out) yields a different answer from the same cloud
centred: centred gives 4 axes with dominant area 0.496 (the disc axis); at 0.85 m it gives 5
axes with dominant area 0.359, the disc axis split into 0.415 + 0.461 fragments that
individually lose to a C2 axis. The frame is then built around the wrong axis.

Every path through the app centres the mesh (`start_express_sampling`, `run_headless`,
batch), which is why this stayed hidden -- only `bench/generate_scenes.py` skipped it. Rather
than rely on callers remembering, the function recentres internally and restores the axes
afterwards. Only axis *points* are translation-dependent; directions, angles, and per-view
and per-point scores are unaffected.

## Debris-removal thresholds

A cluster is dropped only if it is negligible in area **and** spatially detached **and**
materially inflating the bounding box. All three are needed. Measured against every STL in
the repo:

| mesh | outcome |
|---|---|
| `96330MB000.STL` | 1 candidate (2 tris, area frac 3.8e-11, gap 0.98 m), 75.8% shrink -> drop |
| `96330MB100.STL` | 11 clusters, 0 candidates (nested shells, all inside the main AABB) -> keep |
| `40mm.STL` | 208 clusters, 135 candidates totalling 42% of triangles, 0.0% shrink -> refuse |

The last row is why `DEBRIS_MIN_SHRINK` and `DEBRIS_MAX_TRI_FRAC` both exist: a multi-body
assembly trips the area-and-gap test on nearly every body, and dropping them would silently
delete most of the model.
