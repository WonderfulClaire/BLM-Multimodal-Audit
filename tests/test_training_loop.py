import json
import torch
from scripts.make_fixture import generate
from grpo_post_training.train import run_training
from grpo_post_training.reward import RewardModel
from grpo_post_training.policy import AuditOutput
from grpo_post_training.reward import AuditSample


def test_real_rollout_rewards_and_parameter_update(tmp_path):
    torch.set_num_threads(1)
    generate(tmp_path / "data")
    report = run_training(
        tmp_path / "data/audit.jsonl", tmp_path / "out", steps=3, group_size=4
    )
    assert report["parameter_delta_l2"] > 0
    rows = [
        json.loads(x)
        for x in (tmp_path / "out/trajectories.jsonl").read_text().splitlines()
    ]
    data = {
        r["id"]: r
        for r in map(
            json.loads, (tmp_path / "data/audit.jsonl").read_text().splitlines()
        )
    }
    scorer = RewardModel()
    assert len(rows) == 12
    for row in rows:
        src = data[row["id"]]
        score = scorer.score(
            AuditSample(
                src["id"],
                src["prompt"],
                AuditOutput(**src["ground_truth"]),
                AuditOutput(**row["prediction"]).to_json(),
            )
        )
        expected = sum(
            getattr(scorer, "w_" + k) * getattr(score, k)
            for k in ("recall", "category_hit", "consistency", "format", "instruction")
        )
        assert abs(expected - row["reward"]) < 1e-6
    checkpoint = torch.load(tmp_path / "out/checkpoint.pt", weights_only=False)
    assert checkpoint["optimizer"]["state"]
