"""Distributed three-term loss; DDP rank means reproduce global sample means."""
import torch
import torch.distributed as dist
from torch.nn import functional as F
from .fg_clip import FGClipLoss
from .dist_sync import gather_features, sync_dynamic_batch_size

class DistributedFGClipLoss(FGClipLoss):
    def _paired(self, image, text, sizes):
        if image.shape != text.shape:
            raise ValueError('Image/text pairs must align')
        if sum(sizes) == 0:
            return (image.sum() + text.sum() + self.logit_scale) * 0
        images, _ = gather_features(image)
        texts, _ = gather_features(text)
        rank = dist.get_rank() if dist.is_initialized() else 0
        labels = torch.arange(len(image), device=image.device) + sum(sizes[:rank])
        scale = self.logit_scale.exp().clamp(max=100)
        it = scale * image @ texts.T
        ti = scale * text @ images.T
        if not len(image):
            return (it.sum() + ti.sum()) * 0
        return (F.cross_entropy(it, labels, reduction='sum') +
                F.cross_entropy(ti, labels, reduction='sum')) * len(sizes) / (2 * sum(sizes))

    def forward(self, gi, gt, ri, rt, hn=None):
        image_sizes = sync_dynamic_batch_size(len(gi), device=gi.device)
        region_sizes = sync_dynamic_batch_size(len(ri), device=gi.device)
        if not sum(image_sizes):
            raise ValueError('Global image batch is empty')
        global_loss = self._paired(gi, gt, image_sizes)
        region_loss = self._paired(ri, rt, region_sizes)
        hard_loss = (ri.sum() + rt.sum() + self.logit_scale) * 0
        if hn is not None:
            if hn.ndim != 3 or hn.shape[0] != len(ri) or hn.shape[2] != ri.shape[1]:
                raise ValueError('Hard negatives must have shape [regions, negatives, dimension]')
            hard_loss = hard_loss + hn.sum() * 0
            if len(ri):
                logits = torch.cat(((ri * rt).sum(-1, keepdim=True),
                                    (ri[:, None] * hn).sum(-1)), -1)
                logits = self.logit_scale.exp().clamp(max=100) * logits
                hard_loss = F.cross_entropy(logits, torch.zeros(len(ri), dtype=torch.long,
                                                device=ri.device), reduction='sum')
                hard_loss = hard_loss * len(region_sizes) / sum(region_sizes)
        total = self.global_weight * global_loss + self.region_weight * region_loss + self.hard_neg_weight * hard_loss
        return total, {'loss_global': float(global_loss.detach()),
                       'loss_region': float(region_loss.detach()),
                       'loss_hard_neg': float(hard_loss.detach())}
