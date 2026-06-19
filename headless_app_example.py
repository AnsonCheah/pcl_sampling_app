from app_v2 import MeshSamplingApp
from enums import Stage
# from synthetic_pcl_utils import scene_render, fibonacci_sphere, projector_from_camera
import numpy as np
import open3d as o3d
import time
from geometry.geom_utils import o3d_display
from rich import print as rp

app = MeshSamplingApp(headless=True)
app.stages[Stage.IMPORT_MESH]._run_worker()

# app.stages[Stage.IMPORT_MESH].decompose_thread.join()
# vis = o3d_display(app.convex_meshes)
# vis.run()
# vis.destroy_window()

process_start = time.time()
app.stages[Stage.DOWNSAMPLE].use_adaptive = True
app._express_sampling_worker()
print(f"Sampling took {time.time() - process_start}s")

print(f"mean point count = {app.stages[Stage.RAYCAST].point_count_mean}")
print(f"point count range = {app.stages[Stage.RAYCAST].point_count_range}")
print(f"Process took {time.time() - process_start}s")
# o3d.visualization.draw_geometries([app.down_pcd], width=1080, height=720, zoom=1.0)


# Default: auto-size part count to ~60% volumetric fill of the bin (scales with part size).
app.stages[Stage.SYNTHETIC].fill_rate = 0.6
# To force an exact count instead, uncomment:
# app.stages[Stage.SYNTHETIC].generate_mode = "count"
# app.stages[Stage.SYNTHETIC].num_targets = 6
app.stages[Stage.SYNTHETIC].rendering_flag = True
app.stages[Stage.SYNTHETIC]._run_worker()
rp([obj.geom for obj in app.stages[Stage.SYNTHETIC].o3d_scene.values()])
vis = o3d_display([obj.geom for obj in app.stages[Stage.SYNTHETIC].o3d_scene.values()])
vis.run()
vis.destroy_window()