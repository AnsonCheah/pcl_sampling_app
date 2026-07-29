"""Fast tests for the merged scene-mode combo -> (arrangement, structure_type) mapping.

The Scene stage collapsed the old Random/Structured combo + None/Partition/Tray radio into a
single 4-item combobox. These tests pin the mapping and the derived helper predicates without
running any physics. Headless build_panel() creates no widgets, so a tiny fake combo stands in
for the real Combobox.

Run:  python -m pytest stages/tests/test_scene_mode_mapping.py -q
"""
import pytest

from enums import Stage
from stages.scene_stage import SceneStage


class _FakeCombo:
    def __init__(self, text):
        self.selected_text = text


EXPECTED = {
    "Cluttered": ("random", "none"),
    "Arranged":  ("structured", "none"),
    "Partition": ("structured", "partition"),
    "Tray":      ("structured", "tray"),
}


def test_scene_modes_dict_matches_spec():
    assert SceneStage._SCENE_MODES == EXPECTED


@pytest.mark.parametrize("label,arrangement,structure",
                         [(k, v[0], v[1]) for k, v in EXPECTED.items()])
def test_helpers_derive_from_combo(headless_app, label, arrangement, structure):
    scene = headless_app.stages[Stage.SCENE]
    scene.arrangement_combo = _FakeCombo(label)          # headless has no real widget
    assert scene._is_random() == (arrangement == "random")
    assert scene._structure_type() == structure
