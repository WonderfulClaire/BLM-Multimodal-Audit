from grpo_post_training.flywheel import filter_candidates


def test_review_gate_and_test_split_exclusion():
    c = {
        "candidate_id": "a",
        "source_id": "x",
        "split": "train",
        "teacher_model": "teacher-v1",
        "rule_version": "v1",
        "ground_truth": {"risk_category": "normal"},
        "image": "x.png",
        "prompt": "audit",
    }
    d = {
        "candidate_id": "a",
        "approved": True,
        "reviewer": "human",
        "reason": "checked image and rule",
    }
    assert not filter_candidates([c], [])[0]
    assert not filter_candidates([c], [d], {"x"})[0]
    assert len(filter_candidates([c, c], [d])[0]) == 1
