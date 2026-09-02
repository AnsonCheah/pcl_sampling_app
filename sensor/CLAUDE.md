# sensor/

Models a structured-light depth sensor: canonical raycast -> physical noise chain -> instance segmentation errors.

## Non-Obvious Constraints

**Call order is load-bearing.** Each noise function builds on the previous state. The sequence in `RenderStage.worker()` is intentional -- do not reorder:
`compute_dropout_mask` -> `add_image_space_effects` -> `add_edge_artifacts` -> multipath/pepper -> `add_scan_line_banding` -> `add_sensor_noise` -> `add_surface_noise`

**`scene_render()` must stay pure.** No app state, no side effects except assigning `O3DSceneObject.id` on the meshes passed in. It is called by both `RaycastStage` (reference cloud generation) and `RenderStage` (synthetic scene) -- any statefulness would corrupt the reference pass.

**Noise functions must stay independently callable.** Phase 2 will fit each noise function's parameters against real data residuals individually. If they are internally chained, per-function residual fitting becomes impossible.

**`segment_instances.py` operates in image space, not 3D.** Erosion, dilation, and confusion are applied to the 2D pixel grid from the render dict, then back-projected. 3D-space erosion would not replicate the errors a real segmentation network produces.

**GGX alpha is `roughness`, not `roughness^2`.** In `add_specular_patch_missing`, squaring it gives
`alpha = 0.0625` at `roughness = 0.25`, which drives `D_norm ~ 0.10` even for a face-on surface and
misclassifies it as dark. Use `alpha = roughness + 1e-6`.

**`estimate_normals` radius must exceed the point spacing.** The production default is
`radius = 0.005 m`. At reduced test resolution (120x160 at 1.5 m range) the spacing is ~6.7 mm, so
every point has zero neighbours and PCA returns the default `[0, 0, -1]` regardless of true surface
tilt -- silently, with no error. Keep radius >= 3x point spacing; `sensor/tests/_fixtures.py` passes
`normal_radius = 0.020 m` for this reason.

**Parametric parameters are physical, not tuning knobs.** `roughness`, `albedo`, `sigma_fringe_corr` etc. must map to physically measurable quantities. If a parameter has no physical interpretation, it belongs in the learned residual (Phase 2), not here.
