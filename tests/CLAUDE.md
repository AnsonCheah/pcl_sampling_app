# tests/

## Non-Obvious Constraints

**`test_headless.py` is not a test suite.** It is a manual smoke script written to quickly exercise the headless app path. It has no assertions. Do not treat it as ground truth for correctness.

**The test agent is read-only.** It runs tests and reports results. It never modifies production code. Invoke it after changes in any other package.

**Unit tests must not instantiate `MeshSamplingApp`.** Use synthetic numpy arrays as fixtures. Loading the full app in unit tests makes them slow, fragile, and dependent on STL files.

**How to invoke the test agent:**
After making changes, specify which domain changed and what to verify:
```
"I changed sensor/scene_render.py — run tests/unit/test_scene_render.py and report pass/fail"
"Run tests/test_headless.py and report any exceptions"
```
