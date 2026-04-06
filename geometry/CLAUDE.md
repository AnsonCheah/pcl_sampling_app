# geometry/

Shared 3D math, I/O, and the `O3DSceneObject` dataclass. Base layer — imports nothing local.

## Non-Obvious Constraints

**If you want to import from sensor/physics/registration inside here, the dependency graph is wrong.** Refactor the caller instead.

**PLY comment format is a load-bearing protocol.** Key names (`geocenter_x`, `gt_qw`, etc.) are parsed by downstream loaders. Do not rename or reformat them without updating all readers.

**`O3DSceneObject.id` is unset at construction.** It is assigned as a side effect of `scene_render()` adding geometry to the O3D raycast scene. Any code reading `.id` before a `scene_render()` call will get `None`.

**`pcd_geocenter()` returns a 4×4 transform, not a point.** Row 3 (index `[3, :]`) holds the translation. This is non-standard — most transforms store translation in column 3. Check before using.
