import torch
from visual_pretrain.fg_clip import HardNegativeBuilder
from visual_pretrain.dynamic_frames import FrameAttention, VideoEncoder
from visual_pretrain.dynamic_resolution import QwenViT, MRotaryEmbedding2D
from visual_pretrain.dist_sync import CLIPContrastiveLoss
from grpo_post_training.policy import SimpleMLLMPolicy


def test_negative_changes_one_attribute():
    assert HardNegativeBuilder().build("红色瓶子") == "蓝色瓶子"


def test_video_segments_do_not_leak():
    torch.manual_seed(1)
    layer = FrameAttention(24, 3).eval()
    x = torch.randn(1, 4, 24)
    ids = torch.tensor([[0, 0, 1, 1]])
    y = layer(x, ids, use_full=True)
    changed = x.clone()
    changed[:, 2:] += 20
    assert torch.allclose(
        y[:, :2], layer(changed, ids, use_full=True)[:, :2], atol=1e-6
    )


def test_padding_does_not_change_video_embedding():
    torch.manual_seed(2)
    m = VideoEncoder(24, 3).eval()
    x = torch.randn(1, 5, 24)
    y = m(x, torch.tensor([3]))
    x[:, 3:] += 30
    assert torch.allclose(y, m(x, torch.tensor([3])), atol=1e-6)


def test_causal_policy_cannot_read_future():
    torch.manual_seed(3)
    m = SimpleMLLMPolicy(
        vocab_size=32, img_embed_dim=48, text_embed_dim=48, text_num_layers=1
    ).eval()
    image = torch.randn(1, 3, 56, 56)
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 8, 9]])
    assert torch.allclose(m(image, a)[:, :2], m(image, b)[:, :2], atol=1e-6)


def test_rope_preserves_vector_norm():
    rope = MRotaryEmbedding2D(8)
    x = torch.randn(1, 6, 8)
    assert torch.allclose(x.norm(dim=-1), rope(x, 2, 3).norm(dim=-1), atol=1e-6)


def test_vision_applies_rope():
    m = QwenViT(embed_dim=48, depth=1, num_heads=6, window_size=2)
    calls = []
    hooks = [
        mod.register_forward_hook(lambda *args: calls.append(1))
        for mod in m.modules()
        if isinstance(mod, MRotaryEmbedding2D)
    ]
    m(torch.randn(1, 3, 56, 84))
    for h in hooks:
        h.remove()
    assert calls, "position encoder must participate in forward"


def test_distributed_labels_use_rank_offset():
    all_emb = torch.eye(3)
    local = all_emb[1:]
    fn = CLIPContrastiveLoss(temperature=0.1, learnable_temp=False)
    loss = fn(local, local, all_emb, all_emb, rank=1, batch_sizes=[1, 2])
    assert loss < 0.001
