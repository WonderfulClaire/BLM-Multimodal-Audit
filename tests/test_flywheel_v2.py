import pytest
from data_flywheel.feedback import feedback_to_pairs
from data_flywheel.curation import candidate_digest, review_candidate, HoldoutIndex
from data_flywheel.selection import select_failures
from data_flywheel.release import build_release
from data_flywheel.evaluation import compare_predictions


def candidate():
    return {
        "id": "aug1",
        "source_id": "case1",
        "group_id": "video1",
        "split": "train",
        "image": "a.png",
        "image_sha256": "a" * 64,
        "prompt": "audit",
        "ground_truth": {
            "risk_category": "normal",
            "risk_level": "low",
            "reason": "no risk visible",
        },
        "teacher_model": "teacher-v1",
        "rule_version": "rule-v1",
        "error_type": "false_positive",
    }


def review(c, decision="accept", reviewer="independent-judge"):
    return {
        "candidate_id": c["id"],
        "candidate_digest": candidate_digest(c),
        "decision": decision,
        "reviewer": reviewer,
        "reason": "Checked against image and rules",
    }


def test_unselected_is_not_automatically_negative():
    f = {
        "id": "f",
        "source_id": "s",
        "image": "x.png",
        "corrected_reason": "true reason",
        "reviewer": "h",
        "rule_version": "v1",
        "suggestions": [
            {"text": "false reason", "verdict": "incorrect"},
            {"text": "also plausible", "verdict": "unselected"},
        ],
    }
    result = feedback_to_pairs(f)
    assert [x["rejected"] for x in result] == ["false reason"]
    assert result[0]["chosen"] == "true reason"


def test_changed_candidate_invalidates_old_review():
    c = candidate()
    r = review(c)
    c["ground_truth"]["risk_category"] = "risk"
    assert review_candidate(c, r, HoldoutIndex())["status"] == "quarantine"


def test_renamed_test_image_and_video_group_cannot_enter_training():
    c = candidate()
    for holdout in [
        HoldoutIndex(image_hashes={c["image_sha256"]}),
        HoldoutIndex(group_ids={"video1"}),
    ]:
        assert review_candidate(c, review(c), holdout)["status"] == "reject"


def test_teacher_cannot_judge_own_output_and_conflict_needs_review():
    c = candidate()
    assert (
        review_candidate(c, review(c, reviewer="teacher-v1"), HoldoutIndex())["status"]
        == "quarantine"
    )
    assert (
        review_candidate(c, review(c), HoldoutIndex(), rule_pass=False)["status"]
        == "quarantine"
    )


def test_error_selection_has_cluster_quota_and_skips_evaluation():
    rows = [
        {
            "id": str(i),
            "split": "train",
            "group_id": "same",
            "error_type": "false_negative",
            "severity": 1,
            "uncertainty": 0.2,
            "disagreement": 0.5,
            "cost": 1,
        }
        for i in range(10)
    ]
    rows += [
        {
            "id": "other",
            "split": "train",
            "group_id": "other",
            "error_type": "false_positive",
            "severity": 0.5,
            "uncertainty": 0.2,
            "disagreement": 0.2,
            "cost": 1,
        },
        dict(rows[0], id="test", split="test"),
    ]
    selected = select_failures(rows, budget=3, max_per_group=1)
    assert {r["id"] for r in selected} == {"0", "other"}


def test_release_retains_replay_and_is_immutable(tmp_path):
    c = candidate()
    approved = {**c, "curation": review_candidate(c, review(c), HoldoutIndex())}
    base = [
        dict(candidate(), id=f"b{i}", image_sha256=str(i) * 64, group_id=f"base{i}")
        for i in range(4)
    ]
    manifest = build_release(base, [approved], tmp_path / "r1", max_new_fraction=0.25)
    assert manifest["base_count"] == 4 and manifest["new_count"] == 1
    with pytest.raises(FileExistsError):
        build_release(base, [approved], tmp_path / "r1")


def test_reward_rise_without_accuracy_gain_cannot_pass():
    baseline = [
        {
            "id": str(i),
            "group_id": str(i),
            "expected": "A",
            "prediction": "B",
            "reward": 0.2,
        }
        for i in range(20)
    ]
    after = [dict(x, reward=0.9) for x in baseline]
    result = compare_predictions(baseline, after)
    assert not result["accept"] and result["accuracy_delta"] == 0


def test_unpaired_evaluation_is_rejected():
    with pytest.raises(ValueError):
        compare_predictions(
            [{"id": "x", "expected": "A", "prediction": "A"}],
            [{"id": "y", "expected": "A", "prediction": "A"}],
        )


def test_complete_round_rejects_leak_and_stale_review(tmp_path):
    from scripts.run_flywheel_demo import generate
    from data_flywheel.round import run_round

    config = generate(tmp_path / "inputs")
    report = run_round(config, tmp_path / "release")
    assert report["approved"] == 1 and report["new_count"] == 1
    assert report["rejected"] == 1 and report["quarantined"] == 1
    assert report["preference_pairs"] == 1


def test_rl_group_routing_does_not_manufacture_signal():
    from data_flywheel.rl_feedback import route_group

    assert route_group([1.0, 1.0], [1.0, 1.0])["route"] == "mastered_replay"
    assert route_group([0.2, 0.2], [0.0, 0.0])["route"] == "teacher_or_sft_repair"
    assert route_group([0.2, 0.9], [0.0, 1.0])["route"] == "rl_ready"
    assert route_group([0.2, 0.9], [0.0, 0.0])["route"] == "audit_reward_only_variance"
    assert route_group([0.2, 0.2], [0.0, 1.0])["route"] == "repair_reward_resolution"


def test_agent_cases_use_content_hash_without_requiring_an_image():
    from data_flywheel.curation import case_digest

    c = {
        "id": "agent-a",
        "source_id": "train-source",
        "group_id": "drive-session",
        "split": "train",
        "task_type": "agent",
        "prompt": "Diagnose throughput",
        "symptom": "Low throughput",
        "case": {"sections": {"mobility": ["Speed 60 km/h"]}},
        "ground_truth": ["C1"],
        "teacher_model": "teacher",
        "rule_version": "v1",
    }
    c["case_sha256"] = case_digest(c["case"])
    r = review(c)
    assert review_candidate(c, r, HoldoutIndex())["status"] == "accept"
    assert (
        review_candidate(c, r, HoldoutIndex(case_hashes={c["case_sha256"]}))["status"]
        == "reject"
    )
    c["case"]["sections"]["mobility"] = ["Speed 20 km/h"]
    assert review_candidate(c, r, HoldoutIndex())["status"] == "reject"
