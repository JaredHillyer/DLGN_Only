# Guards the AdultDataset hours-per-week bucket assignment bug fix.
# Original uci_datasets.py used `==` (comparison) instead of `=` (assignment)
# in the hours-per-week if/elif chain, so attr_v was never set and the bucket
# was always whatever Python's name resolution happened to find (usually a NameError
# or a stale value from a previous iteration).
#
# This test verifies that the fixed code assigns the correct bucket string.
import pytest
from pathlib import Path


def _get_hours_bucket(hours_value: int) -> str:
    """Replicate the fixed hours-per-week bucket logic in isolation."""
    if hours_value == 0:
        attr_v = 'no-hours'
    elif hours_value <= 12:
        attr_v = 'mini-hours'
    elif hours_value <= 25:
        attr_v = 'half-hours'
    elif hours_value <= 40:
        attr_v = 'full-hours'
    elif hours_value < 60:
        attr_v = 'more-hours'
    else:
        attr_v = 'most-hours'
    return attr_v


@pytest.mark.parametrize('hours,expected_bucket', [
    (0,   'no-hours'),
    (1,   'mini-hours'),
    (12,  'mini-hours'),
    (13,  'half-hours'),
    (25,  'half-hours'),
    (26,  'full-hours'),
    (40,  'full-hours'),
    (41,  'more-hours'),
    (59,  'more-hours'),
    (60,  'most-hours'),
    (80,  'most-hours'),
])
def test_hours_bucket_logic(hours, expected_bucket):
    assert _get_hours_bucket(hours) == expected_bucket


def test_dlgn_uci_hours_bucket_matches_fixed():
    """Verify that dlgn/data/uci.py uses assignment in the hours bucket block.

    Read the source file directly instead of importing dlgn.data.uci so this
    regression test still runs in environments without a healthy Torch runtime.
    """
    src = (
        Path(__file__).resolve().parents[1]
        / 'dlgn'
        / 'data'
        / 'uci.py'
    ).read_text()

    # The bug: all 6 branches originally had `==` on the left of string literals.
    # After the fix, each line assigns with `=`.
    # Check that none of the hours-per-week bucket lines use `==` for assignment.
    lines = src.splitlines()
    in_hours_block = False
    for line in lines:
        stripped = line.strip()
        if 'hours-per-week' in stripped and 'attr_type ==' in stripped:
            in_hours_block = True
        if in_hours_block:
            # These lines should use `attr_v =` (assignment), not `attr_v ==` (comparison)
            if stripped.startswith('attr_v'):
                assert '==' not in stripped, (
                    f'Bug not fixed: found `==` in hours-per-week assignment line: {stripped!r}'
                )
            if stripped.startswith('else:') or stripped.startswith('raise'):
                break
