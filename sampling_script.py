from sampling_app import MeshSamplingApp

if __name__== "__main__":
    app = MeshSamplingApp(headless=True)
    app.import_mesh()
    app._raycasting_worker()
    app.down_pcd=app.raw_pcd
    app._downsample_worker()
    app.save_pcd()