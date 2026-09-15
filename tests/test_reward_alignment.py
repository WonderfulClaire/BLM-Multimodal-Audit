import pytest
from data_flywheel.rl_feedback import route_group


def test_reward_quality_rank_inversion_is_not_rl_ready():
    result = route_group([.9, .1], [0, 1])
    assert result['route'] == 'audit_reward_quality_conflict'
    assert result['reward_quality_inversions'] == 1
    assert route_group([0, .9, .8], [0, .5, 1])['route'] == 'audit_reward_quality_conflict'
    assert route_group([.5, .5, .9], [0, .5, 1])['route'] == 'rl_ready'
    with pytest.raises(ValueError):
        route_group([0, 1], [0, 1], epsilon=-1)
