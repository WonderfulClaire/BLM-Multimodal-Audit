import pytest
from data_flywheel.synthetic_preferences import preference_case


def test_verified_negatives_and_split_boundary():
    row = {"id": "x", "split": "train", "ground_truth": ["C1"],
           "case": {"sections": {"mobility": ["vehicle travels at 55 km/h"]}}}
    result = preference_case(row)
    assert len(result['preference_candidates']) == 7
    assert all('C1' in x and len(x) == 2 for x in result['preference_candidates'])
    with pytest.raises(ValueError, match='training'):
        preference_case(dict(row, split='test'))
    with pytest.raises(ValueError, match='verification'):
        preference_case(dict(row, ground_truth=['C2']))
