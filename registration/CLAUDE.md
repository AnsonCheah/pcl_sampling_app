# registration/

Geometry-derived parameter resolution for pose estimation + PPF coarse matcher. Standalone — no local imports.

## Non-Obvious Constraints

**Do not remove the SVD projection fix in `coarse_match.py`.** OpenCV `ppf_match_3d` returns quaternions that are not guaranteed unit-length. The fix projects the raw rotation onto SO(3) via SVD. OpenCV upstream has not patched this.

**All parameter formulas must stay closed-form.** `heuristic_engine.py` derives registration parameters from geometry (bounding box, surface area, curvature). If a formula needs a lookup table or a fitted coefficient, it belongs in Phase 2 (learned residual), not here. The interpretability of closed-form formulas is what makes per-parameter residual fitting possible later.

**`L_max` is the single normalisation anchor.** All distance thresholds scale from the longest bounding box dimension. Do not introduce a second normalisation anchor — it breaks scaling consistency across parts ranging from 20 mm bolts to 400 mm brackets.

**Not yet wired into app stages.** `coarse_match.py` and `heuristic_engine.py` are not called by any stage. Integration is planned for Phase 1 completion.
