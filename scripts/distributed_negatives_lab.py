"""torchrun lab: genuine DDP gradient equivalence and measured gather timings.

Run with --backend gloo locally or nccl on assigned GPUs. Native all_gather
selects its own communication algorithm; it is not a forced single-ring baseline.
"""

import argparse
import copy
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from visual_pretrain.dist_sync import CLIPContrastiveLoss, gather_features, sync_dynamic_batch_size


def native_gather(x):
    sizes = sync_dynamic_batch_size(len(x), device=x.device)
    padded = F.pad(x, (0, 0, 0, max(sizes) - len(x))).contiguous()
    return torch.cat([block[:n] for block, n in zip(dist_nn.all_gather(padded), sizes)]), sizes


class Twin(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Linear(8, 4, bias=False)
        self.text = nn.Linear(8, 4, bias=False)
        self.objective = CLIPContrastiveLoss(temperature=1)

    def forward(self, image, text, mode="bidir"):
        i, t = F.normalize(self.image(image), dim=-1), F.normalize(self.text(text), dim=-1)
        if mode == "central":
            return self.objective(i, t, i, t, rank=0, batch_sizes=[len(i)])
        gather = (lambda x: gather_features(x, "all_reduce")) if mode == "bidir_allreduce" else (
            gather_features if mode == "bidir" else native_gather)
        gi, sizes = gather(i)
        gt, _ = gather(t)
        return self.objective(i, t, gi, gt, batch_sizes=sizes)


def correctness(device, world, rank):
    scenarios = [[2] * world, list(range(world)), [0] * (world-1) + [4]]
    results = []
    for sizes in scenarios:
        for mode in ("bidir", "bidir_allreduce", "native"):
            torch.manual_seed(117)
            base = Twin().to(device=device, dtype=torch.float64)
            reference = copy.deepcopy(base)
            ddp = DDP(base, device_ids=[device.index] if device.type == "cuda" else None)
            torch.manual_seed(211)
            images = torch.randn(sum(sizes), 8, device=device, dtype=torch.float64)
            texts = torch.randn_like(images)
            loss_ref = reference(images, texts, "central")
            loss_ref.backward()
            start, count = sum(sizes[:rank]), sizes[rank]
            loss = ddp(images[start:start+count], texts[start:start+count], mode)
            loss.backward()
            average = loss.detach().clone()
            dist.all_reduce(average)
            average /= world
            torch.testing.assert_close(average, loss_ref.detach(), atol=1e-8, rtol=1e-7)
            errors = {}
            for (name, actual), (_, expected) in zip(base.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, atol=1e-8, rtol=1e-7)
                error = (actual.grad - expected.grad).abs().max()
                dist.all_reduce(error, op=dist.ReduceOp.MAX)
                errors[name] = float(error)
            results.append({"mode": mode, "sizes": sizes, "global_mean_loss": float(average),
                            "max_gradient_error_by_parameter": errors})
            del ddp, base, reference
    try:
        gather_features(torch.empty(0, 4, device=device))
        raise AssertionError("All-empty batches must be rejected")
    except ValueError as exc:
        assert "Global batch is empty" in str(exc)
    return results


def benchmark(device, world, repeats):
    results = []
    for batch in (32, 512):
        for mode, gather in (("bidir", gather_features),
                             ("bidir_allreduce", lambda x: gather_features(x, "all_reduce")),
                             ("native", native_gather)):
            durations = []
            for step in range(repeats + 3):
                x = torch.randn(batch, 512, device=device, requires_grad=True)
                dist.barrier()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                full, _ = gather(x)
                full.square().mean().backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                duration = torch.tensor([(time.perf_counter()-start)*1000], device=device)
                dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                if step >= 3:
                    durations.append(float(duration))
            results.append({"mode": mode, "batch_per_rank": batch, "dimension": 512,
                            "dtype": "float32", "repeats": repeats,
                            "max_rank_median_ms": statistics.median(durations),
                            "max_rank_p95_ms": sorted(durations)[int(.95*(len(durations)-1))],
                            "samples_ms": durations,
                            "scope": "feature gather plus scalar objective backward, including length exchange; not full CLIP training"})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank) if args.backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group(args.backend, timeout=timedelta(seconds=120))
    world, rank = dist.get_world_size(), dist.get_rank()
    checks = correctness(device, world, rank)
    timing = benchmark(device, world, args.repeats)
    if rank == 0:
        report = {"world_size": world, "backend": args.backend, "torch": torch.__version__,
                  "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                  "all_empty_rejected": True, "correctness": checks, "timings": timing}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as out:
            out.write(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
