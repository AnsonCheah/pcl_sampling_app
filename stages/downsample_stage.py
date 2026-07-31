import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from enums import Stage
from stages.stage_base import BaseStage
from geometry.geom_utils import mask_point_cloud, normalize_normals, pcd_geocenter, extract_edge_points
from geometry.math_utils import find_cdf_knee
from geometry.ambiguity import AmbiguityConfig, ambiguity_geometries, analyse_ambiguity
import copy

class DownsampleStage(BaseStage):
    downstream = {
        "down_pcd": lambda: None,
        "down_pcd_surface": lambda: None,
        "down_pcd_edge": lambda: None,
        "feature_pcd": lambda: None,
        "pcd_flat": lambda: None,
        "geocenter": lambda: np.eye(4),
        "ambiguity_profile": lambda: None,
    }

    def __init__(self, app):
        self.name = Stage.DOWNSAMPLE.name
        self.use_adaptive = False
        self.coarse_factor = 2.0
        self.curvature_k_neighbors = 5
        self.cloud_radio_idx = 0  # 0=surface, 1=edge
        # Ambiguity analysis costs ~10-60s depending on cloud size, so it is switchable;
        # with it off the model frame falls back to the historical PCA-canonical one.
        self.run_ambiguity = True
        self.show_ambiguity = False
        self.preview_max_axes = 6
        super().__init__(app)

    def on_clear(self):
        """Drop the preview when this stage's products are cleared, so a stale heat map
        cannot outlive the profile it was drawn from."""
        self.show_ambiguity = False
        if not self.app.headless and getattr(self, "btn_preview_ambiguity", None) is not None:
            self.btn_preview_ambiguity.text = "Preview Ambiguity Axes"

    def analyse_ambiguity_profile(self):
        """Detect view-dependent and global ambiguity for the downsampled cloud.

        Runs before the geocenter is decided, because the dominant axis is what the model
        frame gets built around: MechVision's symmetry search can only rotate about a
        geocenter frame axis, so an axis that is not a frame axis cannot be mitigated at
        any angleStep.
        """
        if not self.run_ambiguity or self.app.target_mesh is None:
            return None
        pcd = self.app.down_pcd_surface
        if pcd is None or len(pcd.points) < 16:
            return None

        if not self.app.headless:
            self.app.show_progress("Analysing pose ambiguity...")
        profile = analyse_ambiguity(
            self.app.target_mesh, pcd, AmbiguityConfig(),
            progress_cb=None if self.app.headless else self.app.update_progress)

        dom = profile.dominant
        if dom is None:
            print("[INFO] Ambiguity: none found — pose should be uniquely determined")
        else:
            fold = "continuous" if dom.fold == 0 else f"C{dom.fold}"
            step = "-" if dom.fold == 0 else f"{dom.angle_step_deg():.0f}deg"
            print(f"[INFO] Ambiguity: {len(profile.axes)} axes "
                  f"({profile.n_significant_axes} significant), "
                  f"discriminative={profile.discriminative_fraction:.3f}")
            print(f"[INFO]   dominant {fold} angleStep={step} "
                  f"dir={np.round(dom.direction, 3)} "
                  f"views={dom.view_fraction:.2f} global={dom.is_global}")
        return profile

    def build_panel(self):
        if self.app.headless:
            return
        v = gui.Vert(4)

        self.radio_adaptive = self.register_widget(gui.RadioButton(gui.RadioButton.HORIZ))
        self.radio_adaptive.set_items(["Uniform", "Adaptive"])
        self.radio_adaptive.selected_index = 1 if self.use_adaptive else 0
        self.btn_downsample = self.register_widget(gui.Button("Downsample"))
        self.btn_downsample.set_on_clicked(self.start)
        self.btn_recenter = self.register_widget(gui.Button("Recenter Point Cloud"), lambda: self.app.down_pcd is not None)
        self.btn_recenter.set_on_clicked(self.recenter_mesh_pcd)

        self.btn_reset = self.register_widget(gui.Button("Restart Downsample"), lambda: self.app.down_pcd is not None)
        self.btn_reset.set_on_clicked(self.reset)

        self.radio_cloud = self.register_widget(
            gui.RadioButton(gui.RadioButton.HORIZ),
            lambda: self.app.down_pcd_surface is not None
        )
        self.radio_cloud.set_items(["Surface", "Edge"])
        self.radio_cloud.selected_index = 0
        self.radio_cloud.set_on_selection_changed(self._on_cloud_radio_changed)

        self.chk_ambiguity = self.register_widget(gui.Checkbox("Analyse pose ambiguity"))
        self.chk_ambiguity.checked = self.run_ambiguity
        self.chk_ambiguity.set_on_checked(self._on_ambiguity_toggled)

        self.btn_preview_ambiguity = self.register_widget(
            gui.Button("Preview Ambiguity Axes"),
            lambda: self.app.ambiguity_profile is not None
        )
        self.btn_preview_ambiguity.set_on_clicked(self._toggle_ambiguity_preview)

        v.add_child(gui.Label("Downsampling"))
        v.add_child(gui.Label(""))
        v.add_child(self.radio_adaptive)
        v.add_child(self.chk_ambiguity)
        v.add_child(self.btn_downsample)
        v.add_child(self.btn_preview_ambiguity)
        v.add_child(self.btn_recenter)
        v.add_child(self.btn_reset)
        v.add_child(gui.Label(""))
        v.add_child(gui.Label("Show Cloud:"))
        v.add_child(self.radio_cloud)
        print("loaded downsample panel")

        return v

    def next_enabled(self) -> bool:
        return self.app.down_pcd is not None

    def _on_ambiguity_toggled(self, checked):
        self.run_ambiguity = bool(checked)

    def _toggle_ambiguity_preview(self):
        self.show_ambiguity = not self.show_ambiguity
        if not self.app.headless:
            self.btn_preview_ambiguity.text = (
                "Hide Ambiguity Axes" if self.show_ambiguity else "Preview Ambiguity Axes")
        self._refresh_ui()

    def _ambiguity_materials(self):
        """Unlit for points, lit for the rods.

        The heat map encodes a score in the colour, so shading the cloud would modulate
        every point by its surface orientation and make two points with the same score
        render differently. The rods stay lit so they read as solid objects.
        """
        pts = rendering.MaterialRecord()
        pts.shader = "defaultUnlit"
        pts.point_size = 4.0
        solid = rendering.MaterialRecord()
        solid.shader = "defaultLit"
        return pts, solid

    def _draw_ambiguity_preview(self):
        """Overlay the discriminative heat map and the ranked ambiguity axes.

        Hot points are explained by no ambiguity transform and are what actually pins the
        pose down; cool points are interchangeable with somewhere else on the model. The
        rods are the ranked axes (hottest = rank 0), each with a knob where the axis sits
        and reference spheres at the centroid and AABB centre.
        """
        profile = self.app.ambiguity_profile
        pcd = self.app.down_pcd_surface
        if profile is None or pcd is None:
            return
        geoms, legend = ambiguity_geometries(pcd, profile, max_axes=self.preview_max_axes)
        pts_mat, solid_mat = self._ambiguity_materials()
        for i, g in enumerate(geoms):
            mat = pts_mat if isinstance(g, o3d.geometry.PointCloud) else solid_mat
            self.app.scene.scene.add_geometry(f"ambiguity_{i}", g, mat)

        if not legend:
            print("[INFO] Ambiguity preview: no axes — pose should be uniquely determined")
            return
        print(f"[INFO] Ambiguity preview: {len(legend)} axes "
              f"(hot = rank 0 = most viewpoints affected)")
        for row in legend:
            step = "-" if row["angle_step_deg"] == 0 else f"{row['angle_step_deg']:.0f}deg"
            print(f"[INFO]   rank {row['rank']}: {row['fold']:<10} step={step:<7} "
                  f"views={row['view_fraction']:.2f} "
                  f"off-centroid={row['offset_from_centroid_mm']:.2f}mm")

    def _on_cloud_radio_changed(self, idx):
        self.cloud_radio_idx = idx
        self.app._clear_scene()
        if idx == 0 and self.app.down_pcd_surface is not None:
            self.app.scene.scene.add_geometry("surface_pcd", self.app.down_pcd_surface, self.app.default_point_material)
        elif idx == 1 and self.app.down_pcd_edge is not None:
            self.app.scene.scene.add_geometry("edge_pcd", self.app.down_pcd_edge, self.app.default_point_material)
        self.app._reframe()   # content swap -> reframe (posts a redraw too)

    def _refresh_ui(self):
        if self.app.headless:
            return

        if self.show_ambiguity and self.app.ambiguity_profile is not None \
                and self.app.down_pcd_surface is not None:
            self.app.main_thread(lambda: self.app._clear_scene())
            self.app.main_thread(self._draw_ambiguity_preview)
        elif self.app.down_pcd_surface is not None:
            self.app.main_thread(lambda: self.app._clear_scene())
            if self.cloud_radio_idx == 0:
                self.app.main_thread(lambda: self.app.scene.scene.add_geometry("surface_pcd", self.app.down_pcd_surface, self.app.default_point_material))
            elif self.cloud_radio_idx == 1 and self.app.down_pcd_edge is not None:
                self.app.main_thread(lambda: self.app.scene.scene.add_geometry("edge_pcd", self.app.down_pcd_edge, self.app.default_point_material))
        elif self.app.cropped_pcd is not None:
            self.app.main_thread(lambda: self.app._clear_scene())
            self.app.main_thread(lambda: self.app.scene.scene.add_geometry("cropped_pcd", self.app.cropped_pcd, self.app.default_point_material))
        self.app.main_thread(self.app._reframe)   # content swap -> reframe (posts a redraw too)
        self.app.main_thread(self.enable_widgets)


    def worker(self):
        if not self.app.headless:
            self.use_adaptive = self.radio_adaptive.selected_index == 1
            self.app.show_progress("Downsampling point cloud...")
        if self.app.cropped_pcd is None:
            print("cropped pcd is none")
            return
        self.app.clear_state_from(self.stage_key)

        bbox = self.app.cropped_pcd.get_minimal_oriented_bounding_box()
        self.voxel_size = np.round(np.clip((np.asarray(bbox.volume()) / 3), 0.001, 0.005), 4)
        self.adaptive_voxel_downsample() if self.use_adaptive else self.uniform_voxel_downsample()

        # Edge extraction on the downsampled surface cloud
        if not self.app.headless:
            self.app.show_progress("Extracting edge points...")
        edge_mask = extract_edge_points(self.app.down_pcd_surface, self.voxel_size)
        pcd_edge_raw = mask_point_cloud(self.app.down_pcd_surface, edge_mask)
        self.app.down_pcd_edge = normalize_normals(pcd_edge_raw.voxel_down_sample(self.voxel_size))

        self.app.ambiguity_profile = self.analyse_ambiguity_profile()
        self.app.geocenter = np.round(self._geocenter_for(self.app.down_pcd), decimals=5)
        print(f"[INFO] Downsampled {self.voxel_size * 1000:.2f}mm from {len(self.app.cropped_pcd.points)} to {len(self.app.down_pcd.points)} points")
        print(f"[INFO] Edge cloud: {len(self.app.down_pcd_edge.points)} points ({edge_mask.sum()} before voxel pass)")
        self._refresh_ui()

    def _geocenter_for(self, pcd):
        """Model frame for ``pcd``: ambiguity-aligned when an axis was found, else PCA.

        Records ``frame_changed`` on the profile so the caller can tell whether the
        exported bundle differs from the historical PCA frame — the parts where it does
        are exactly the ones whose scenes must be regenerated alongside it.
        """
        profile = self.app.ambiguity_profile
        dominant = profile.dominant if profile is not None else None
        if dominant is None:
            return pcd_geocenter(pcd)
        tf, changed = pcd_geocenter(pcd, axis=dominant)
        profile.frame_changed = bool(changed)
        if changed:
            print("[INFO] Model frame rebuilt around the dominant ambiguity axis "
                  "(now frame Z) — regenerate this part's scenes alongside the bundle")
        else:
            print("[INFO] PCA frame already agrees with the ambiguity axis — frame unchanged")
        return tf

    def uniform_voxel_downsample(self):
        self.app.down_pcd = normalize_normals(self.app.cropped_pcd.voxel_down_sample(self.voxel_size))
        self.app.down_pcd_surface = self.app.down_pcd

    def adaptive_voxel_downsample(self):
        variation = self.compute_curvature(self.app.cropped_pcd)
        threshold, percentile, _ = find_cdf_knee(variation)
        feature_mask = variation >= threshold

        pcd_feature = mask_point_cloud(self.app.cropped_pcd, feature_mask)
        pcd_flat = mask_point_cloud(self.app.cropped_pcd, ~feature_mask)
        pcd_feature = pcd_feature.voxel_down_sample(self.voxel_size)
        pcd_flat = pcd_flat.voxel_down_sample(self.voxel_size * self.coarse_factor)
        self.app.feature_pcd = pcd_feature
        self.app.pcd_flat = pcd_flat
        self.app.down_pcd = pcd_feature + pcd_flat
        normalize_normals(self.app.down_pcd)
        self.app.down_pcd_surface = copy.deepcopy(self.app.down_pcd)

    def compute_curvature(self, pcd):
        pts = np.asarray(pcd.points)
        n_points = len(pts)
        tree = o3d.geometry.KDTreeFlann(pcd)
        curv = np.zeros(n_points, dtype=np.float64)
        batch_size = max(1, n_points // 100)  # Update progress every 1%

        for i in range(n_points):
            _, idx, _ = tree.search_knn_vector_3d(pts[i], self.curvature_k_neighbors)
            nbrs = np.asarray(pts[idx])
            centered = nbrs - nbrs.mean(axis=0)
            C = (centered.T @ centered) / (len(nbrs) - 1)
            eigvals = np.linalg.eigvalsh(np.asarray(C))
            eigval_sum = eigvals.sum()
            curv[i] = eigvals[0] / eigval_sum if eigval_sum > 1e-12 else 0.0
            if not self.app.headless and (i + 1) % batch_size == 0:
                self.app.update_progress((i + 1) / n_points)

        if not self.app.headless:
            self.app.update_progress(1.0)

        return curv

    def recenter_mesh_pcd(self):
        if self.app.cropped_pcd is None:
            return
        T = self._geocenter_for(self.app.down_pcd_surface)

        self.app.down_pcd_surface.transform(T)
        # In uniform mode down_pcd is the same object as down_pcd_surface — skip to avoid double-transform
        if self.app.down_pcd is not self.app.down_pcd_surface:
            self.app.down_pcd.transform(T)
        if self.app.down_pcd_edge is not None:
            self.app.down_pcd_edge.transform(T)
        if self.app.feature_pcd is not None:
            self.app.feature_pcd.transform(T)
        if self.app.pcd_flat is not None:
            self.app.pcd_flat.transform(T)
        self.app.raw_pcd.transform(T)
        self.app.cropped_pcd.transform(T)
        self.app.target_mesh.transform(T)
        for mesh in self.app.convex_meshes:
            mesh.transform(T)

        # The profile was computed in the pre-recentre frame; move it with everything
        # else so the exported sidecar describes the axis in the frame the exported cloud
        # actually lives in. Left untransformed it would aim MechVision's rotation search
        # at the wrong line.
        if self.app.ambiguity_profile is not None:
            self.app.ambiguity_profile = self.app.ambiguity_profile.transformed(T)

        # The cloud has just been placed into its own model frame, so the geocenter
        # relative to it is identity by construction — which is what geo_center.json
        # already asserts. Recomputing a PCA frame here would disagree with that file for
        # any part whose frame is ambiguity-aligned rather than PCA-aligned.
        self.app.geocenter = np.eye(4)
        print("recentered mesh and pointcloud")
        self._refresh_ui()   # swaps the geometry, then reframes — do not reframe before this
