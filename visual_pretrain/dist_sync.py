"""Differentiable uneven-batch feature exchange and symmetric CLIP loss.

DDP synchronizes parameter gradients, not contrastive candidates. Forward uses
bidirectional ring exchanges; backward sums the candidate gradients from every
rank. Communication rounds alone do not establish throughput improvement.
"""

from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def sync_dynamic_batch_size(local_bs, group=None, device=None):
    if not dist.is_initialized():
        return [local_bs]
    if device is None:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend(group) == "nccl"
            else torch.device("cpu")
        )
    t = torch.tensor([local_bs], device=device, dtype=torch.long)
    outputs = [torch.zeros_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(outputs, t, group=group)
    return [int(x.item()) for x in outputs]


@dataclass
class DynamicBatchMeta:
    local_batch_size: int


class RingAllGather:
    def __init__(self, world_size):
        self.world_size = world_size
        self.comm_rounds = world_size - 1

    def simulate(self, rank, local_data):
        raise RuntimeError("A single tensor cannot simulate remote data; run torchrun")


class BidirAllGather:
    def __init__(self, world_size):
        if world_size < 1:
            raise ValueError("world_size must be positive")
        self.world_size = world_size
        self.comm_rounds = world_size // 2

    def gather(self, tensor):
        return gather_features(tensor)

    def simulate(self, rank, local_data):
        raise RuntimeError("Use gather under torchrun for real communication")


class _BidirGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sizes):
        world = dist.get_world_size()
        rank = dist.get_rank()
        maximum = max(sizes)
        padded = F.pad(x, (0, 0, 0, maximum - x.shape[0])).contiguous()
        blocks = [torch.empty_like(padded) for _ in range(world)]
        blocks[rank] = padded
        right = (rank + 1) % world
        left = (rank - 1) % world
        clockwise = padded
        counter = padded
        for hop in range(1, world // 2 + 1):
            recv_left = torch.empty_like(padded)
            ops = [
                dist.P2POp(dist.isend, clockwise, right, tag=2 * hop),
                dist.P2POp(dist.irecv, recv_left, left, tag=2 * hop),
            ]
            dual = (rank - hop) % world != (rank + hop) % world
            if dual:
                recv_right = torch.empty_like(padded)
                ops += [
                    dist.P2POp(dist.isend, counter, left, tag=2 * hop + 1),
                    dist.P2POp(dist.irecv, recv_right, right, tag=2 * hop + 1),
                ]
            for request in dist.batch_isend_irecv(ops):
                request.wait()
            blocks[(rank - hop) % world] = recv_left
            clockwise = recv_left
            if dual:
                blocks[(rank + hop) % world] = recv_right
                counter = recv_right
        ctx.sizes = sizes
        ctx.rank = rank
        ctx.maximum = maximum
        return torch.cat([b[:n] for b, n in zip(blocks, sizes)], 0)

    @staticmethod
    def backward(ctx, grad):
        chunks = grad.split(ctx.sizes)
        padded = torch.stack(
            [F.pad(x, (0, 0, 0, ctx.maximum - x.shape[0])) for x in chunks]
        ).contiguous()
        dist.all_reduce(padded)
        return padded[ctx.rank, : ctx.sizes[ctx.rank]], None


def gather_features(x):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x, [x.shape[0]]
    sizes = sync_dynamic_batch_size(x.shape[0], device=x.device)
    if max(sizes) == 0:
        raise ValueError("Global batch is empty")
    return _BidirGather.apply(x, sizes), sizes


class CLIPContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, learnable_temp=True):
        super().__init__()
        value = torch.tensor(1 / temperature).log()
        if learnable_temp:
            self.logit_scale = nn.Parameter(value)
        else:
            self.register_buffer("logit_scale", value)

    def forward(
        self,
        image_embeds,
        text_embeds,
        all_image_embeds=None,
        all_text_embeds=None,
        *,
        rank=None,
        batch_sizes=None,
    ):
        rank = (
            (dist.get_rank() if dist.is_initialized() else 0) if rank is None else rank
        )
        if image_embeds.shape != text_embeds.shape:
            raise ValueError("Local pairs must align")
        if all_image_embeds is None:
            all_image_embeds, batch_sizes = gather_features(image_embeds)
            all_text_embeds, _ = gather_features(text_embeds)
        if batch_sizes is None:
            if all_image_embeds.shape[0] != image_embeds.shape[0]:
                raise ValueError("Provide batch_sizes for pre-gathered candidates")
            batch_sizes = [len(image_embeds)]
        if sum(batch_sizes) != len(all_image_embeds) or len(all_image_embeds) != len(
            all_text_embeds
        ):
            raise ValueError("Candidate size mismatch")
        b = len(image_embeds)
        total = sum(batch_sizes)
        if total == 0:
            raise ValueError("Empty batch")
        labels = torch.arange(b, device=image_embeds.device) + sum(batch_sizes[:rank])
        scale = self.logit_scale.exp().clamp(max=100)
        i2t = scale * image_embeds @ all_text_embeds.T
        t2i = scale * text_embeds @ all_image_embeds.T
        if b == 0:
            return (i2t.sum() + t2i.sum()) * 0
        # DDP averages ranks; weight local sums so the result matches a global mean.
        return (
            (
                F.cross_entropy(i2t, labels, reduction="sum")
                + F.cross_entropy(t2i, labels, reduction="sum")
            )
            * len(batch_sizes)
            / (2 * total)
        )
