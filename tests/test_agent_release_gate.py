import pytest
from data_flywheel.agent_evaluation import compare_agent_predictions


def examples(split="test"):
    cases = [{"id": str(i), "group_id": str(i), "split": split,
              "ground_truth": ["C1"]} for i in range(20)]
    old = [{"scenario_id": str(i), "ground_truth": ["C1"],
            "predicted_root_causes": [], "tool_failure_count": 0} for i in range(20)]
    new = [dict(r, predicted_root_causes=["C1"]) for r in old]
    return cases, old, new


def test_development_score_cannot_pass_release_gate():
    result = compare_agent_predictions(*examples("synthetic_dev"))
    assert result["statistical_gate_passed"]
    assert not result["accept"] and not result["test_split_only"]
    assert compare_agent_predictions(*examples())["accept"]


def test_protocol_regression_and_changed_labels_are_rejected():
    cases, old, new = examples()
    new[0]["tool_failure_count"] = 1
    assert not compare_agent_predictions(cases, old, new)["accept"]
    new[0]["ground_truth"] = ["C2"]
    with pytest.raises(ValueError, match="independent labels"):
        compare_agent_predictions(cases, old, new)
    with pytest.raises(ValueError, match="Training data"):
        compare_agent_predictions(*examples("train"))
