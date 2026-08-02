# registration/

From-scratch PPF + Hough voting matcher (`ppf/`). Imports `geometry` only.

## Non-Obvious Constraints

**The dependency rule changed: this package may now import `geometry`, and nothing else.** It
was previously "standalone — no local imports". That existed because the package was a
bolt-on; it now needs the shared geometry helpers and consumes `AmbiguityProfile` directly. It
must still not import `sensor`, `physics`, or `stages`.

**`coarse_match.py`, `heuristic_engine.py` and `ppf_helpers.py` are gone.** They wrapped
OpenCV's `ppf_match_3d` (never installed, never declared in `environment.yaml`, imported by
nothing) and resolved parameters for an FPFH/RANSAC/ICP pipeline that does not exist — no
file in this repo calls `open3d.pipelines.registration` at all. What was worth keeping moved:
`SensorProfile` → `ppf/config.py`; diameter/area/SO(3) helpers → `geometry/geom_utils.py`;
curvature and feature-size estimation → `geometry/curvature.py`.

**Do not build on OpenCV or PCL PPF.** OpenCV `ppf_match_3d` has an open heap-corruption bug
from 2015 (opencv_contrib #170), a wrong-pose bug in its own official sample (#2034), and an
unnormalised quaternion in `clusterPoses` (#3223). PCL's `PPFRegistration` is missing an
`acos` in the feature itself (#1171) and its maintainers advise against using it. Misc3D is
worth reading but destroys its accumulator inside an OpenMP loop, so vote diagnostics never
reach Python, and it offers no way to weight a vote.

**Parameters are derived, not tuned.** `PPFConfig.derive()` takes a model cloud and a
`SensorProfile` and produces everything. There are no per-part tunables — that is the point,
because a knob that needs tuning needs tuning for all 5000 parts. Adding a parameter with a
hand-picked default is the failure mode to watch for; if a value cannot be derived from part
geometry or a one-time sensor calibration, it belongs in the small set of explicit
*application policies* (`model_target_points`, `accept_score`) and must be justified there.

**The sensor floor is not a second normalisation anchor.** The old `L_max`-only rule is
relaxed: absolute floors derived from depth noise are allowed, for the same reason
`AmbiguityConfig` already allows them — *"the absolute floors exist for sensor-physics
reasons, not as size rules."* What is still forbidden is a second *size* anchor.

**`tau` is bisected on the real downsampled point count, never from a surface-area formula.**
`tau = sqrt(SA/M)` with `SA ~ N*s^2` is wrong by ~4.5x on randomly sampled clouds (a Poisson
process has median NN distance `0.4697/sqrt(density)`, not `1/sqrt(density)`). Model point
count goes as `tau^-2` and work as its square, so a 2x error in `tau` is a ~20x error in
runtime. This bit the module it replaced.

**Vote deduplication does not subsume the per-bucket cap.** Dedup runs *after* the table
lookup is expanded, so it corrects planar vote *bias* but not *cost*. A 100x30x20 box has six
distinct normals; its feature bins reach 12 000 entries and one instance expands to ~1.8e9
votes without `PPFConfig.max_bucket_entries`. Both mechanisms are needed and they solve
different problems.

**Bucket entries are capped by striding, never truncation.** Pair enumeration is ordered by
model point, so keeping the first N entries would retain one contiguous patch of the part and
bias every pose voted from that bin toward it.

**Pose clustering is SE(3), not translation-only.** Translation-only NMS merges two different
orientations of the same part at the same location — exactly the symmetry-flip case the
ablation exists to measure.

**Rotations must be re-projected onto SO(3) after averaging** (`geometry.project_to_so3`). The
mean of several rotation matrices is not one, and an unprojected mean scales and shears the
model just enough to look like a near-miss at verification.

**Dedup uses sort-then-diff, not `np.unique`.** NumPy 2.x routes `unique` through a hash table
that is pathologically slow on wide int64 keys: 0.411 s vs 0.020 s on 1.4 M values, and it is
the hottest call in the matcher.
