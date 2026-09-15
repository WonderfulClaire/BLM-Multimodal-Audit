"""Real multi-process Gloo communication, including an empty rank."""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from visual_pretrain.dist_sync import gather_features, CLIPContrastiveLoss


def worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    sizes = list(range(world))
    full = torch.arange(sum(sizes) * 3, dtype=torch.float32).reshape(-1, 3) / 10
    offset = sum(sizes[:rank])
    local = full[offset : offset + sizes[rank]].clone().requires_grad_()
    gathered, actual_sizes = gather_features(local)
    torch.testing.assert_close(gathered, full)
    assert actual_sizes == sizes
    (gathered.square().sum() * (rank + 1)).backward()
    torch.testing.assert_close(
        local.grad, 2 * local.detach() * sum(range(1, world + 1))
    )
    # Candidate-side and anchor-side gradients must match a centralized objective.
    images = full.clone().requires_grad_()
    texts = (full.flip(1) + 0.2).clone().requires_grad_()
    baseline = CLIPContrastiveLoss(temperature=1, learnable_temp=False)(images, texts)
    baseline.backward()
    li = full[offset : offset + sizes[rank]].clone().requires_grad_()
    lt = (full.flip(1) + 0.2)[offset : offset + sizes[rank]].clone().requires_grad_()
    loss = CLIPContrastiveLoss(temperature=1, learnable_temp=False)(li, lt)
    loss.backward()
    torch.testing.assert_close(
        li.grad / world,
        images.grad[offset : offset + sizes[rank]],
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        lt.grad / world, texts.grad[offset : offset + sizes[rank]], atol=1e-6, rtol=1e-5
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [3, 4])
def test_uneven_ring_matches_centralized_gradients(world):
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            worker, args=(world, str(Path(tmp) / "rendezvous")), nprocs=world, join=True
        )
