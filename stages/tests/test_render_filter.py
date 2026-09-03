"""Fast unit tests for RenderStage's 2D candidate filter (_passes_2d_filter).

The filter mirrors the real Mask2Former post-filter: an instance is kept only if its
2D-mask elongation and image-area fraction both fall inside the auto-derived per-part
ranges. No scene/sim needed -- pure function, so this stays out of the `slow` suite."""

from stages.render_stage import _passes_2d_filter


def test_passes_when_in_range():
    assert _passes_2d_filter(2.0, 0.05, (1.0, 4.0), (0.01, 0.1)) is True


def test_rejects_aspect_out_of_range():
    assert _passes_2d_filter(5.0, 0.05, (1.0, 4.0), (0.01, 0.1)) is False  # too elongated
    assert _passes_2d_filter(0.9, 0.05, (1.0, 4.0), (0.01, 0.1)) is False  # below min


def test_rejects_area_out_of_range():
    assert _passes_2d_filter(2.0, 0.5, (1.0, 4.0), (0.01, 0.1)) is False   # too large
    assert _passes_2d_filter(2.0, 0.001, (1.0, 4.0), (0.01, 0.1)) is False  # too small


def test_boundaries_inclusive():
    assert _passes_2d_filter(1.0, 0.01, (1.0, 4.0), (0.01, 0.1)) is True
    assert _passes_2d_filter(4.0, 0.1, (1.0, 4.0), (0.01, 0.1)) is True


def test_missing_range_skips_that_dimension():
    # area range present, aspect range None -> only area is gated.
    assert _passes_2d_filter(99.0, 0.05, None, (0.01, 0.1)) is True
    assert _passes_2d_filter(99.0, 0.5, None, (0.01, 0.1)) is False
    # both None -> always passes.
    assert _passes_2d_filter(99.0, 99.0, None, None) is True


def test_distance_correction_effect():
    # Scene measured area_ratio rescaled to reference distance: ratio * (d_scene/d_ref)^2.
    # A scene part at 2x the reference distance projects to 1/4 the area; correcting by
    # (2.0/1.0)^2 = 4 brings it back into the reference range.
    area_range = (0.04, 0.16)
    scene_area_ratio = 0.025          # measured small because part is far
    corrected = scene_area_ratio * (2.0 / 1.0) ** 2  # = 0.1, in range
    assert _passes_2d_filter(2.0, corrected, (1.0, 4.0), area_range) is True
    assert _passes_2d_filter(2.0, scene_area_ratio, (1.0, 4.0), area_range) is False
