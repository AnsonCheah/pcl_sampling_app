# registration

Geometry-derived pose estimation parameter auto-tuning. All formulas are closed-form and interpretable — no lookup tables, no fitted coefficients.

## Contents

| File | Description |
|------|-------------|
| `heuristic_engine.py` | `MeshAnalyser`, `PartProfile`, `SceneAnalyser`, `SceneProfile`, `ParameterResolver` |
| `coarse_match.py` | PPF + Hough voting using OpenCV `ppf_match_3d`, with SO(3) projection fix |
| `ppf_helpers.py` | PPF parameter derivation from CAD geometry |

## Architecture

```
MeshAnalyser(mesh) → PartProfile
                          ↓
SceneAnalyser(scene) → SceneProfile
                          ↓
                   ParameterResolver → registration parameters
```

`MeshAnalyser` extracts intrinsic geometry properties (diameter, surface area, curvature statistics). `ParameterResolver` derives registration parameters entirely from those properties using closed-form geometric reasoning.

## `coarse_match.py`

Wraps OpenCV `ppf_match_3d` for PPF feature extraction and Hough-space voting:

```python
from registration.coarse_match import coarse_match

poses = coarse_match(
    model_pcd,    # o3d.geometry.PointCloud — reference
    scene_pcd,    # o3d.geometry.PointCloud — observed scene
    params        # dict of PPF parameters
)
```

**SVD projection fix**: OpenCV's `ppf_match_3d` returns non-unit-length quaternions for recovered poses. The code projects them onto SO(3) via SVD before returning. This is a known OpenCV bug.

## `ppf_helpers.py`

Derives PPF sampling parameters directly from CAD geometry:

```python
from registration.ppf_helpers import compute_ppf_params

params = compute_ppf_params(mesh)
# returns: voxel_size, ppf_distance_step, ppf_normal_step, ...
```

All distance thresholds are normalised by `L_max` — the longest bounding box dimension. This is the single normalisation anchor; introducing a second anchor breaks scaling consistency across part sizes.

## Integration status

`coarse_match.py` and `heuristic_engine.py` are implemented but not yet wired into the app pipeline. Integration is planned for Phase 1 completion.

## Constraints

- **No local imports**: this package does not import `geometry`, `sensor`, `physics`, or `stages`. It is standalone.
- **All formulas are closed-form**: fitted coefficients, lookup tables, and empirically tuned constants belong in the Phase 2 learned residual — not here.
- **`L_max` is the single normalisation anchor**: all spatial thresholds must scale from the longest bounding box dimension.
