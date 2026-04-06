# sensor/

Models a structured-light depth sensor: canonical raycast → physical noise chain → instance segmentation errors.

## Non-Obvious Constraints

**Call order is load-bearing.** Each noise function builds on the previous state. The sequence in `SyntheticStage.worker()` is intentional — do not reorder:
`compute_dropout_mask` → `add_image_space_effects` → `add_edge_artifacts` → multipath/pepper → `add_scan_line_banding` → `add_sensor_noise` → `add_surface_noise`

**`scene_render()` must stay pure.** No app state, no side effects except assigning `O3DSceneObject.id` on the meshes passed in. It is called by both `RaycastStage` (reference cloud generation) and `SyntheticStage` (synthetic scene) — any statefulness would corrupt the reference pass.

**Noise functions must stay independently callable.** Phase 2 will fit each noise function's parameters against real data residuals individually. If they are internally chained, per-function residual fitting becomes impossible.

**`segment_instances.py` operates in image space, not 3D.** Erosion, dilation, and confusion are applied to the 2D pixel grid from the render dict, then back-projected. 3D-space erosion would not replicate the errors a real segmentation network produces.

**Parametric parameters are physical, not tuning knobs.** `roughness`, `albedo`, `sigma_fringe_corr` etc. must map to physically measurable quantities. If a parameter has no physical interpretation, it belongs in the learned residual (Phase 2), not here.
