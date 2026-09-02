"""Tests for DownsampleStage: uniform vs adaptive output shape and the early return when
there is no cropped cloud. Uses a directly-sampled box cloud so no slow raycast is needed."""

import numpy as np

from enums import Stage


def _seed_cropped(app, box_mesh, n=5000):
    app.target_mesh = box_mesh
    app.cropped_pcd = box_mesh.sample_points_uniformly(n)


def test_uniform_downsample(headless_app, box_mesh):
    app = headless_app
    _seed_cropped(app, box_mesh)
    stage = app.stages[Stage.DOWNSAMPLE]
    stage.use_adaptive = False

    stage.worker()

    assert app.down_pcd is not None
    assert len(app.down_pcd.points) <= len(app.cropped_pcd.points)
    # uniform mode aliases the surface cloud to down_pcd; the recenter logic relies on this.
    assert app.down_pcd_surface is app.down_pcd
    assert app.geocenter.shape == (4, 4)
    assert app.down_pcd_edge is not None


def test_adaptive_downsample(headless_app, box_mesh):
    app = headless_app
    _seed_cropped(app, box_mesh)
    stage = app.stages[Stage.DOWNSAMPLE]
    stage.use_adaptive = True

    stage.worker()

    assert app.feature_pcd is not None
    assert app.pcd_flat is not None
    # adaptive mode keeps surface as an independent deepcopy of down_pcd.
    assert app.down_pcd_surface is not app.down_pcd
    assert app.down_pcd_edge is not None


def test_no_cropped_cloud_early_return(headless_app):
    app = headless_app
    app.cropped_pcd = None
    app.stages[Stage.DOWNSAMPLE].worker()   # must not raise
    assert app.down_pcd is None


def test_worker_does_not_move_the_geometry(headless_app, box_mesh):
    """Downsampling computes the profile but applies no frame.

    In the stage-by-stage GUI flow the operator must see the cloud sit still until they ask
    for the recentre, and `app.geocenter` records what has been APPLIED -- so it stays
    identity here. It used to hold a *pending* transform, which made "has this been
    recentred?" unanswerable from the attribute alone.
    """
    app = headless_app
    _seed_cropped(app, box_mesh)
    stage = app.stages[Stage.DOWNSAMPLE]
    stage.use_adaptive = False
    stage.run_ambiguity = False       # keep it quick; the frame question is orthogonal

    stage.worker()

    assert np.allclose(app.geocenter, np.eye(4))
    mesh_centre = np.asarray(box_mesh.get_axis_aligned_bounding_box().get_center())
    assert np.allclose(mesh_centre, np.asarray(app.target_mesh
                                               .get_axis_aligned_bounding_box().get_center()))


def _downsampled(app, mesh):
    _seed_cropped(app, mesh)
    stage = app.stages[Stage.DOWNSAMPLE]
    stage.use_adaptive = False
    stage.run_ambiguity = False       # keep it quick; the frame question is orthogonal
    stage.worker()
    return stage


def test_recenter_moves_everything_by_the_transform_it_records(headless_app, make_box_mesh):
    """Mesh and clouds must move together, by exactly the transform `app.geocenter` records.

    RenderStage pairs `down_pcd` with a `T_gt` derived from `target_mesh`, so if the two ever
    diverge every exported pose is silently offset by the difference.
    """
    app = headless_app
    stage = _downsampled(app, make_box_mesh(extents=(0.10, 0.03, 0.02)))

    # Push everything off-origin so the recentre has real work to do.
    shift = np.eye(4)
    shift[:3, 3] = [0.021, -0.013, 0.008]
    for geom in (app.target_mesh, app.raw_pcd, app.cropped_pcd, app.down_pcd_surface,
                 app.down_pcd_edge):
        if geom is not None:
            geom.transform(shift)
    if app.down_pcd is not None and app.down_pcd is not app.down_pcd_surface:
        app.down_pcd.transform(shift)

    before_mesh = np.asarray(app.target_mesh.vertices).copy()
    before_cloud = np.asarray(app.down_pcd_surface.points).copy()

    stage.recenter_mesh_pcd()

    T = app.geocenter
    assert not np.allclose(T, np.eye(4)), "the applied transform must be recorded"
    assert np.allclose(np.asarray(app.target_mesh.vertices),
                       before_mesh @ T[:3, :3].T + T[:3, 3], atol=1e-9)
    assert np.allclose(np.asarray(app.down_pcd_surface.points),
                       before_cloud @ T[:3, :3].T + T[:3, 3], atol=1e-9)
    # PCA frame -> the cloud mean lands on the origin.
    assert np.allclose(np.asarray(app.down_pcd_surface.points).mean(axis=0), 0.0, atol=1e-9)


def test_recenter_is_idempotent(headless_app, make_box_mesh):
    """Why `recenter_mesh_pcd` needs no "already applied" guard: re-deriving the frame from an
    already-recentred cloud gives (numerically) the identity, so a second press is a no-op.

    The floor is `np.round(rot, decimals=6)` inside `pcd_geocenter._pack`, which for a part
    this size leaves sub-micron residue. Uses an elongated box rather than the default cube:
    a cube's covariance is isotropic, so its PCA axes are genuinely arbitrary and the test
    would be measuring that degeneracy instead of the property it is after.
    """
    app = headless_app
    stage = _downsampled(app, make_box_mesh(extents=(0.10, 0.03, 0.02)))

    stage.recenter_mesh_pcd()
    once = np.asarray(app.down_pcd_surface.points).copy()
    geocenter_once = app.geocenter.copy()

    stage.recenter_mesh_pcd()
    twice = np.asarray(app.down_pcd_surface.points)

    assert np.abs(twice - once).max() < 1e-7
    assert np.allclose(app.geocenter, geocenter_once, atol=1e-6)


def test_frame_mode_follows_the_ambiguity_checkbox(headless_app, box_mesh):
    """Unticking "Analyse pose ambiguity" after a run must fall back to PCA rather than
    silently reusing a profile the operator has just said they do not want."""
    from geometry.ambiguity import AmbiguityAxis, AmbiguityProfile

    app = headless_app
    stage = app.stages[Stage.DOWNSAMPLE]
    ax = AmbiguityAxis(direction=np.array([0.0, 0.0, 1.0]), point=np.array([0.004, 0.0, 0.0]),
                       fold=2, angles_deg=[180.0], is_global=True,
                       view_fraction=1.0, area_fraction=0.99)
    app.ambiguity_profile = AmbiguityProfile(axes=[ax], dominant=ax)

    stage.run_ambiguity = True
    assert stage._use_ambiguity_frame() is True
    assert stage._recenter_mode_label() == "Recenter to Ambiguity Axis"

    stage.run_ambiguity = False
    assert stage._use_ambiguity_frame() is False
    assert stage._recenter_mode_label() == "Recenter to PCA Frame"

    # No profile at all -> PCA, whatever the checkbox says.
    stage.run_ambiguity = True
    app.ambiguity_profile = None
    assert stage._use_ambiguity_frame() is False


def test_owned_state_exists_before_any_panel_is_built(headless_app):
    """Every `downstream` attribute is seeded before the stages are constructed.

    `build_panel` reads app state (e.g. the recentre button's label needs
    `app.ambiguity_profile`), so seeding has to precede construction. It used not to, and
    each reader defended itself with `getattr(app, "x", None)`; this pins the ordering that
    made those guards unnecessary. Headless returns from `build_panel` early, so a
    regression here would surface only in the GUI, at startup.
    """
    from app import STAGE_CLASSES
    app = headless_app
    for cls in STAGE_CLASSES.values():
        for attr in cls.downstream:
            assert hasattr(app, attr), f"{cls.__name__} owns {attr!r} but it was never seeded"
