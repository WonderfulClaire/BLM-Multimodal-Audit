"""The full three-term objective must equal a centralized optimizer update."""
import copy
import tempfile
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from visual_pretrain.fg_clip import FGClipLoss

class EncodedObjective(nn.Module):
    def __init__(self, distributed=False):
        super().__init__()
        from visual_pretrain.distributed_fg import DistributedFGClipLoss
        self.encoder = nn.Linear(5, 4, bias=False)
        self.loss = DistributedFGClipLoss() if distributed else FGClipLoss()
    def forward(self, values):
        return self.loss(*(torch.nn.functional.normalize(self.encoder(v), dim=-1) for v in values))[0]

def worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=3)
    try:
        torch.manual_seed(6)
        ref = EncodedObjective().double()
        actual = EncodedObjective(True).double()
        actual.load_state_dict(ref.state_dict())
        ddp = DDP(actual)
        for region_sizes in ([0, 1, 3], [0, 0, 0]):
            sizes = [1, 2, 3]
            values = [torch.randn(6, 5, dtype=torch.double) for _ in range(2)]
            values += [torch.randn(sum(region_sizes), 5, dtype=torch.double) for _ in range(2)]
            values += [torch.randn(sum(region_sizes), 2, 5, dtype=torch.double)]
            local = []
            for i, v in enumerate(values):
                ns = sizes if i < 2 else region_sizes
                start = sum(ns[:rank]);local.append(v[start:start + ns[rank]])
            ref.zero_grad();actual.zero_grad()
            expected = ref(values);expected.backward()
            got = ddp(local);got.backward()
            avg = got.detach().clone();dist.all_reduce(avg);avg /= 3
            torch.testing.assert_close(avg, expected, atol=1e-9, rtol=1e-7)
            for p, q in zip(actual.parameters(), ref.parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=1e-8, rtol=1e-7)
    finally:
        dist.destroy_process_group()

def test_full_objective_uneven_and_empty_regions():
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker, args=(str(Path(tmp)/'rdzv'),), nprocs=3, join=True)

def test_distributed_manifest_rejects_changed_image(tmp_path):
    import json
    import pytest
    from PIL import Image
    from scripts.make_fixture import generate
    from visual_pretrain.train_distributed import load_examples
    generate(tmp_path)
    row = json.loads((tmp_path/'raw.jsonl').read_text().splitlines()[0])
    row['regions'] = []
    row['provenance'] = {'negative_verification': 'explicit_review_or_verifier', 'image_sha256': '0'*64}
    manifest = tmp_path/'manifest.jsonl';manifest.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='image hash'):
        load_examples(manifest)
