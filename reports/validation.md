# Verification — 2026-09-15

Environment: Python 3.12, PyTorch 2.14.0, local CPU, OMP_NUM_THREADS=1.

- `python -m pytest tests -q`: 31 passed, 5.75 s.
- Actual Gloo P2P communication with 3 and 4 processes, unequal batches including an empty rank: gathered values and backward gradients match centralized computation.
- Packing rejects oversized samples instead of silently dropping them; video truncation keeps frame paths and counts consistent.
- Causal masking, spatial rotary transformation, cross-video/padding isolation, verified data gating and train-only candidate filtering passed.
- Three-step rendered-image training: total loss 7.18553 → 1.99536 → 1.59713. This is in-sample integration evidence, not generalization.
- GRPO integration test sampled 12 actual structured responses, independently recomputed every reward, checked parameter updates and optimizer checkpoint state.

No pretrained-model benchmark, GPU throughput measurement, real moderation dataset evaluation or Ascend validation was completed. CLI demonstrations write detailed artifacts to `runs/`; no model weights or private data are committed.
