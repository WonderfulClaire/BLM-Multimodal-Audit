"""Run actual compact-policy rollouts and GRPO updates on generated fixtures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_fixture import generate
from grpo_post_training.train import run_training

if __name__ == "__main__":
    generate(Path("runs/fixtures"))
    print(run_training("runs/fixtures/audit.jsonl", "runs/grpo", steps=6))
