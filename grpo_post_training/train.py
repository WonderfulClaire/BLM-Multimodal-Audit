"""Executable compact-model GRPO with on-policy structured answer generation.

Three semantic response tokens encode category, level and reason. This finite
output vocabulary makes CPU integration tests meaningful; it is not a natural
language Qwen training benchmark. Rewards always score the generated answer.
"""

import argparse, copy, json
from pathlib import Path
from dataclasses import asdict, dataclass
import torch
from PIL import Image
from .policy import SimpleMLLMPolicy, AuditOutput
from .reward import AuditSample, RewardModel
from .grpo import GRPOLoss, GRPOConfig
from visual_pretrain.train_manifest import tokenize, image_tensor
from visual_pretrain.dynamic_resolution import SmartResizeConfig


@dataclass
class TrainConfig:
    lr: float = 1e-4
    group_size: int = 4
    max_new_tokens: int = 3
    device: str = "cpu"


class AnswerVocabulary:
    def __init__(self, rows):
        self.values = [
            sorted({row["ground_truth"][field] for row in rows})
            for field in ("risk_category", "risk_level", "reason")
        ]
        self.ids = []
        offset = 300
        for choices in self.values:
            self.ids.append(list(range(offset, offset + len(choices))))
            offset += len(choices)
        if offset > 512:
            raise ValueError("Compact model supports at most 212 answer symbols")

    def decode(self, ids):
        values = [
            choices[allowed.index(int(index))]
            for choices, allowed, index in zip(self.values, self.ids, ids)
        ]
        return AuditOutput(*values)

    def constrain(self, logits, step):
        allowed = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
        allowed[self.ids[step]] = True
        return logits.masked_fill(~allowed, -1e4)


def score_sequence(model, images, sequences, prompt_len, vocab):
    logits = model(images, sequences[:, :-1])
    return torch.stack(
        [vocab.constrain(logits[:, prompt_len - 1 + j], j) for j in range(3)], 1
    )


@torch.no_grad()
def rollout(model, images, prompts, vocab):
    seq = prompts
    for j in range(3):
        logits = vocab.constrain(model(images, seq)[:, -1], j)
        seq = torch.cat([seq, torch.multinomial(logits.softmax(-1), 1)], 1)
    return seq


def run_training(manifest, output, steps=3, group_size=4, seed=42, device="cpu"):
    if group_size < 2:
        raise ValueError("GRPO requires at least two samples per prompt")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    manifest = Path(manifest)
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    if not rows:
        raise ValueError("Empty training set")
    if any(row.get("split", "train") != "train" for row in rows):
        raise ValueError("Do not train on held-out rows")
    vocab = AnswerVocabulary(rows)
    policy = SimpleMLLMPolicy(
        vocab_size=512,
        img_embed_dim=48,
        text_embed_dim=48,
        text_num_layers=1,
        text_num_heads=6,
        max_text_len=64,
    ).to(device)
    # Disable dropout for consistent rollout/new-policy probability ratios.
    policy.eval()
    reference = copy.deepcopy(policy).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    loss_fn = GRPOLoss(GRPOConfig(group_size=group_size))
    scorer = RewardModel()
    resize = SmartResizeConfig(min_pixels=28 * 28, max_pixels=56 * 56)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    traces = []
    failures = []
    initial = torch.cat([p.detach().flatten().cpu() for p in policy.parameters()])
    # Explicit ordering allows reasoning then general rows in one optimizer session.
    rows = sorted(rows, key=lambda x: 0 if x.get("stage") == "reasoning" else 1)
    for step in range(steps):
        row = rows[step % len(rows)]
        image = (
            image_tensor(Image.open(manifest.parent / row["image"]), resize)
            .unsqueeze(0)
            .to(device)
        )
        images = image.repeat(group_size, 1, 1, 1)
        prompts = (
            tokenize(row["prompt"], length=24)
            .unsqueeze(0)
            .repeat(group_size, 1)
            .to(device)
        )
        old = copy.deepcopy(policy).requires_grad_(False).eval()
        seq = rollout(old, images, prompts, vocab)
        actions = seq[:, 24:]
        outputs = [vocab.decode(ids.tolist()) for ids in actions]
        reward_values = []
        for i, pred in enumerate(outputs):
            sample = AuditSample(
                row["id"],
                row["prompt"],
                AuditOutput(**row["ground_truth"]),
                pred.to_json(),
            )
            breakdown = scorer.score(sample)
            reward = sum(
                getattr(scorer, "w_" + key) * getattr(breakdown, key)
                for key in (
                    "recall",
                    "category_hit",
                    "consistency",
                    "format",
                    "instruction",
                )
            )
            reward_values.append(reward)
            trace = {
                "step": step,
                "id": row["id"],
                "stage": row.get("stage", "general"),
                "sample": i,
                "response_token_ids": actions[i].tolist(),
                "prediction": asdict(pred),
                "reward": reward,
                "components": asdict(breakdown),
            }
            traces.append(trace)
            if (
                pred.risk_category != row["ground_truth"]["risk_category"]
                or reward < 0.8
            ):
                failures.append(
                    {
                        **trace,
                        "rule_version": row.get("rule_version", "v1"),
                        "status": "needs_review",
                        "split": "train",
                    }
                )
        rewards = torch.tensor(reward_values, device=device)
        with torch.no_grad():
            old_logits = score_sequence(old, images, seq, 24, vocab)
            ref_logits = score_sequence(reference, images, seq, 24, vocab)
        for _ in range(2):
            logits = score_sequence(policy, images, seq, 24, vocab)
            loss, parts = loss_fn(
                logits,
                old_logits,
                ref_logits,
                actions,
                rewards,
                torch.ones_like(actions, dtype=torch.float),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        history.append(
            {
                "step": step,
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std().item(),
                "loss": loss.item(),
                **parts,
            }
        )
    final = torch.cat([p.detach().flatten().cpu() for p in policy.parameters()])
    summary = {
        "seed": seed,
        "steps": steps,
        "group_size": group_size,
        "parameter_delta_l2": float((final - initial).norm()),
        "history": history,
        "device": str(device),
        "scope": "compact structured-output training; not business benchmark",
    }
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vocabulary": vocab.values,
            "config": summary,
        },
        output / "checkpoint.pt",
    )
    for name, data in [
        ("trajectories.jsonl", traces),
        ("review_queue.jsonl", failures),
    ]:
        (output / name).write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data)
        )
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--output", default="runs/grpo")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    print(
        json.dumps(
            run_training(a.manifest, a.output, a.steps, a.group_size, device=a.device)
        )
    )


if __name__ == "__main__":
    main()
