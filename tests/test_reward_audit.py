from grpo_post_training.reward_audit import audit_reward_alignment


def test_reward_audit_detects_rank_inversions():
    audit = audit_reward_alignment(
        rewards=[0.9, 0.8, 0.1],
        eval_scores=[0.0, 1.0, 0.5],
        success_threshold=1.0,
    )
    assert audit.n == 3
    assert audit.comparable_pairs == 3
    assert audit.inversions == 2
    assert audit.inversion_rate == 2 / 3
    assert audit.top_reward_failure_rate == 1.0
    assert audit.best_reward_index == 0
    assert audit.best_eval_index == 1


def test_reward_audit_accepts_aligned_ordering_and_ties():
    aligned = audit_reward_alignment([0.1, 0.5, 0.9], [0.0, 0.5, 1.0])
    assert aligned.inversions == 0
    assert aligned.inversion_rate == 0.0
    assert aligned.top_reward_failure_rate == 0.0

    tied = audit_reward_alignment([0.5, 0.5], [0.0, 1.0])
    assert tied.comparable_pairs == 0
    assert tied.ties == 1


def test_reward_audit_validates_inputs():
    import pytest

    with pytest.raises(ValueError):
        audit_reward_alignment([1.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        audit_reward_alignment([1.0], [1.0], success_threshold=-1)
