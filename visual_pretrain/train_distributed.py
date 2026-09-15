"""torchrun entry point for real image/region/hard-negative DDP training.

Compact randomly initialized towers; full-manifest batches, no large-scale claim.
Each rank must own at least one image. Region counts may differ or be zero.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from .model import BLMCLIP
from .fg_clip import FGClipLoss
from .distributed_fg import DistributedFGClipLoss
from .dynamic_resolution import SmartResizeConfig
from .train_manifest import image_tensor, tokenize


def load_examples(manifest):
    cfg = SmartResizeConfig(min_pixels=28*28, max_pixels=112*112)
    examples = []
    for line in manifest.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        provenance = row.get('provenance', {})
        image_path = manifest.parent / row['image']
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != provenance.get('image_sha256'):
            raise ValueError('Pipeline image hash does not match current image')
        if provenance.get('negative_verification') != 'explicit_review_or_verifier':
            raise ValueError('Need pipeline negative-verification provenance')
        with Image.open(image_path) as handle:
            image = handle.convert('RGB')
        regions = []
        for r in row['regions']:
            x1, y1, x2, y2 = r['bbox']
            if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                raise ValueError('Invalid region bounds')
            if not r['negatives'] or r['negatives'][0] == r['positive']:
                raise ValueError('Need an explicitly verified, distinct negative in the pipeline manifest')
            regions.append((image_tensor(image.crop(r['bbox']), cfg), tokenize(r['positive']),
                            tokenize(r['negatives'][0])))
        examples.append((image_tensor(image, cfg), tokenize(row['caption']), regions))
    return examples


class TrainingModel(nn.Module):
    def __init__(self, distributed=True):
        super().__init__()
        self.encoder = BLMCLIP(embed_dim=32, img_embed_dim=48, text_embed_dim=48,
            img_backbone_kwargs={'depth': 2, 'num_heads': 6, 'window_size': 4},
            text_backbone_kwargs={'num_layers': 1, 'num_heads': 6})
        self.objective = DistributedFGClipLoss() if distributed else FGClipLoss()

    def forward(self, examples):
        device = next(self.parameters()).device
        gi = torch.cat([self.encoder.encode_image(x[0][None].to(device)) for x in examples])
        gt = self.encoder.encode_text(torch.stack([x[1] for x in examples]).to(device))
        regions = [r for x in examples for r in x[2]]
        if regions:
            ri = torch.cat([self.encoder.encode_image(r[0][None].to(device)) for r in regions])
            rt = self.encoder.encode_text(torch.stack([r[1] for r in regions]).to(device))
            hn = self.encoder.encode_text(torch.stack([r[2] for r in regions]).to(device))[:, None]
        else:
            ri, rt, hn = gi[:0], gt[:0], gt[:0, None]
        return self.objective(gi, gt, ri, rt, hn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=['gloo', 'nccl'], default='gloo')
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verify-central', action='store_true')
    a = p.parse_args()
    if a.steps < 1: raise ValueError('steps must be positive')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if a.backend == 'nccl' else torch.device('cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group(a.backend, timeout=timedelta(seconds=180))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if a.output.exists(): raise ValueError('Use a new output directory')
        dist.barrier()
        examples = load_examples(a.manifest)
        if len(examples) < world: raise ValueError('Need at least one image per rank')
        # Contiguous partitions preserve the centralized contrastive pair order.
        sizes = [len(examples)//world + (r < len(examples)%world) for r in range(world)]
        start = sum(sizes[:rank]);local = examples[start:start+sizes[rank]]
        torch.manual_seed(a.seed)
        # Disable dropout for deterministic global-reference equivalence. Gradients remain enabled.
        model = TrainingModel().to(device).eval()
        ddp = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ref = None
        if a.verify_central:
            ref = TrainingModel(False).to(device).eval()
            ref.load_state_dict(model.state_dict())
        config = {'seed': a.seed, 'steps': a.steps, 'backend': a.backend, 'world_size': world,
                  'images_per_rank': sizes, 'regions_per_rank': [sum(len(x[2]) for x in examples[sum(sizes[:r]):sum(sizes[:r+1])]) for r in range(world)],
                  'manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                  'torch': str(torch.__version__), 'dropout_enabled': False,
                  'scope': 'compact synthetic or user supplied manifest; full batch; no business accuracy claim'}
        if rank == 0:
            a.output.mkdir(parents=True)
            (a.output/'config.json').write_text(json.dumps(config, indent=2))
        history = []
        for step in range(a.steps):
            dist.barrier()
            if device.type == 'cuda': torch.cuda.synchronize(device)
            begin = time.perf_counter()
            optim.zero_grad()
            loss, parts = ddp(local)
            loss.backward()
            metrics = torch.tensor([loss.detach().item(), *parts.values()], device=device)
            dist.all_reduce(metrics);metrics /= world
            check = None
            if step == 0 and ref is not None:
                expected, _ = ref(examples);expected.backward()
                torch.testing.assert_close(metrics[0], expected.detach(), atol=1e-4, rtol=1e-4)
                max_error = 0.
                for (name, actual), (_, target) in zip(model.named_parameters(), ref.named_parameters()):
                    if actual.grad is None or target.grad is None:
                        raise AssertionError(f'Missing gradient: {name}')
                    torch.testing.assert_close(actual.grad, target.grad, atol=2e-4, rtol=2e-3, msg=name)
                    max_error = max(max_error, float((actual.grad-target.grad).abs().max()))
                error = torch.tensor(max_error, device=device);dist.all_reduce(error, op=dist.ReduceOp.MAX)
                check = float(error);del ref;ref = None
            if not torch.isfinite(metrics).all(): raise RuntimeError('Nonfinite training loss')
            optim.step()
            if device.type == 'cuda': torch.cuda.synchronize(device)
            duration = torch.tensor(time.perf_counter()-begin, device=device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            row = dict(zip(['loss', *parts], metrics.cpu().tolist()))
            row.update(step=step, max_rank_seconds=float(duration), central_max_gradient_error=check)
            history.append(row)
            if rank == 0:
                with (a.output/'metrics.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
        if rank == 0:
            torch.save({'model': model.state_dict(), 'optimizer': optim.state_dict(), 'config': config}, a.output/'checkpoint.pt')
        dist.barrier()
    finally:
        dist.destroy_process_group()

if __name__ == '__main__': main()
