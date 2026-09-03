"""Tests for search_config -- the angleStep ladder MechVision's symmetry search draws from.

Run from project root:
    python -m pytest MM_Optimizer/tests/test_search_config.py -q
    python MM_Optimizer/tests/test_search_config.py
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import MM_Optimizer.search_config as SC


# -----------------------------------------------------------------------------
# angleStep ladder
# -----------------------------------------------------------------------------

def test_angle_step_ladder_divides_360():
    """Every step must tile the circle exactly.

    MechVision sweeps minAngle..maxAngle in angleStep increments. A step that does not
    divide 360 leaves a remainder arc: the seeds are unevenly spaced and one gap is wider
    than the rest, so the search is weakest at an arbitrary orientation.
    """
    for step in SC.ANGLE_STEP_LADDER:
        assert 360 % step == 0, f"angleStep {step} does not divide 360"


def test_angle_step_ladder_respects_floor():
    """The 5 degree floor is what stops a trial from seeding ~73 registrations per pose."""
    assert min(SC.ANGLE_STEP_LADDER) == SC.ANGLE_STEP_FLOOR
    assert SC.ANGLE_STEP_FLOOR == 5
    assert all(s >= SC.ANGLE_STEP_FLOOR for s in SC.ANGLE_STEP_LADDER)


def test_angle_step_ladder_is_sorted_unique_and_ends_at_360():
    """Ascending and unique so the suggested INDEX carries ordinal meaning for TPE/GP.

    Encoding an ordered quantity as an unordered categorical is the classic search-space
    mistake: samplers that use distance between values lose the fact that 90 is nearer to
    72 than to 5.
    """
    assert SC.ANGLE_STEP_LADDER == sorted(SC.ANGLE_STEP_LADDER)
    assert len(set(SC.ANGLE_STEP_LADDER)) == len(SC.ANGLE_STEP_LADDER)
    assert SC.ANGLE_STEP_LADDER[-1] == SC.ANGLE_STEP_NONE == 360
    assert all(isinstance(s, int) for s in SC.ANGLE_STEP_LADDER)


def test_rotation_strategy_z_matches_mechvision_encoding():
    """rotationStrategy is 0=X, 1=Y, 2=Z. Recentring puts the ambiguity axis on frame Z.

    mm_adapter defaults this to 1.0 (Y), so leaving it unset sweeps the wrong axis.
    """
    assert SC.ROTATION_STRATEGY_Z == 2.0


if __name__ == "__main__":
    test_angle_step_ladder_divides_360()
    test_angle_step_ladder_respects_floor()
    test_angle_step_ladder_is_sorted_unique_and_ends_at_360()
    test_rotation_strategy_z_matches_mechvision_encoding()
    print("All tests PASSED")
