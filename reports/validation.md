# Verification — 2026-09-15

Environment: Python 3.12, PyTorch 2.14.0, local CPU, OMP_NUM_THREADS=1.

- `python -m pytest tests -q`: 42 passed, 3.74 s.
- Actual Gloo P2P communication with 3 and 4 processes, unequal batches including an empty rank: gathered values and backward gradients match centralized computation.
- Packing rejects oversized samples instead of silently dropping them; video truncation keeps frame paths and counts consistent.
- Causal masking, spatial rotary transformation, cross-video/padding isolation, verified data gating and train-only candidate filtering passed.
- Three-step rendered-image training: total loss 7.18553 → 1.99536 → 1.59713. This is in-sample integration evidence, not generalization.
- GRPO integration test sampled 12 actual structured responses, independently recomputed every reward, checked parameter updates and optimizer checkpoint state.

No pretrained-model benchmark, GPU throughput measurement, real moderation dataset evaluation or Ascend validation was completed. CLI demonstrations write detailed artifacts to `runs/`; no model weights or private data are committed.

## GPU and flywheel follow-up

A six-step compact-model GRPO run executed on CUDA with PyTorch 2.6.0+cu124, producing a parameter delta L2 of 0.259926. Raw metrics are in `gpu_smoke_20260915.json`; its legacy `scope` string incorrectly says CPU, while the executed command explicitly selected `--device cuda`. The code now records the device separately. This remains a synthetic integration test, not business quality.

The expanded flywheel passed review-binding, held-out image/case identity, RL group routing, replay quota and full-round tests. A real Agent trace-driven round selected 8 synthetic compound cases, validated them, and released 32 replay + 8 new cases. Downstream retraining did not pass the old-capability gate; see the linked 5G experiment report.
