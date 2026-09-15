"""
tests/smoke_test.py
===================
5 个 smoke test，确保所有模块能 import + 关键 forward 能跑通。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from visual_pretrain import (
    BidirAllGather,
    BLMCLIP,
    FGClipLoss,
    QwenViT,
    RingAllGather,
    SmartResizeConfig,
    offline_binpack,
)

from grpo_post_training import (
    AuditOutput,
    AuditSample,
    GRPOConfig,
    GRPOLoss,
    RewardModel,
    SimpleMLLMPolicy,
    parse_audit_output,
)


def test_visual_pretrain_imports():
    """Test 1: 视觉预训练模块全部 import。"""
    print("Test 1: visual_pretrain imports ... ", end="")
    # 上面 from ... import 已经验证了
    print("✓")


def test_grpo_imports():
    """Test 2: GRPO 模块全部 import。"""
    print("Test 2: grpo_post_training imports ... ", end="")
    print("✓")


def test_smart_resize():
    """Test 3: Smart Resize 函数。"""
    print("Test 3: SmartResize ... ", end="")
    cfg = SmartResizeConfig(patch_size=28)
    h, w = cfg.smart_resize(1024, 768)
    assert h % 28 == 0 and w % 28 == 0
    print(f"✓ ({h}x{w})")


def test_qwen_vit_forward():
    """Test 4: QwenViT forward。"""
    print("Test 4: QwenViT forward ... ", end="")
    vit = QwenViT(embed_dim=384, depth=2, num_heads=6, window_size=4)
    img = torch.randn(2, 3, 224, 224)
    emb = vit(img)
    assert emb.shape == (2, 384)
    print(f"✓ ({emb.shape})")


def test_blm_clip_forward():
    """Test 5: BLMCLIP forward。"""
    print("Test 5: BLMCLIP forward ... ", end="")
    model = BLMCLIP(embed_dim=128, img_embed_dim=384, text_embed_dim=384)
    img = torch.randn(2, 3, 224, 224)
    txt = torch.randint(0, 1000, (2, 32))
    img_emb, txt_emb = model(img, txt)
    assert img_emb.shape == (2, 128)
    assert txt_emb.shape == (2, 128)
    # 验证 L2 normalize
    assert torch.allclose(img_emb.norm(dim=-1), torch.ones(2), atol=1e-5)
    print(f"✓ ({img_emb.shape}, {txt_emb.shape})")


def test_binpack():
    """Test 6: Binpack offline + online。"""
    print("Test 6: Binpack offline + online ... ", end="")
    import numpy as np

    np.random.seed(0)
    samples = [
        type("S", (), {"image_id": f"img_{i}", "meta": (200, 200), "token_count": 50})()
        for i in range(20)
    ]
    from visual_pretrain.binpack import ImageSample as IS

    samples = [
        IS(
            image_id=f"img_{i}",
            meta=(np.random.randint(28, 500), np.random.randint(28, 500)),
        )
        for i in range(20)
    ]
    bins = offline_binpack(samples)
    assert len(bins) > 0
    assert all(b.total_tokens <= 4096 for b in bins)
    print(f"✓ ({len(bins)} bins)")


def test_fg_clip_loss():
    """Test 7: FG-CLIP 三损失。"""
    print("Test 7: FG-CLIP 三损失 ... ", end="")
    loss_fn = FGClipLoss()
    B, R, D, K = 4, 3, 64, 2
    gi = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    gt = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    ri = torch.nn.functional.normalize(torch.randn(R, D), dim=-1)
    rt = torch.nn.functional.normalize(torch.randn(R, D), dim=-1)
    hn = torch.nn.functional.normalize(torch.randn(R, K, D), dim=-1)
    total, comp = loss_fn(gi, gt, ri, rt, hn)
    assert "loss_global" in comp and "loss_region" in comp and "loss_hard_neg" in comp
    print(f"✓ (total={total.item():.4f})")


def test_dist_sync_comparison():
    """Test 8: Ring vs Bidir 通信轮次对比。"""
    print("Test 8: Ring vs Bidir 通信轮次 ... ", end="")
    for ws in [2, 4, 8]:
        ring = RingAllGather(ws)
        bidir = BidirAllGather(ws)
        # Bidir 应该 <= Ring（ws=2 时相等，ws>=4 时严格 <）
        assert bidir.comm_rounds <= ring.comm_rounds
    print("✓")


def test_reward_5_dim():
    """Test 9: Reward 5 维评分。"""
    print("Test 9: Reward Model 5 维 ... ", end="")
    rm = RewardModel()
    sample = AuditSample(
        image_id="v1",
        prompt="审核视频",
        ground_truth=AuditOutput("暴力", "高危", "画面血腥"),
        model_output='{"risk_category": "暴力", "risk_level": "高危", "reason": "画面血腥"}',
    )
    rb = rm.score(sample)
    assert rb.total > 0.5
    print(f"✓ (total={rb.total:.2f})")


def test_grpo_loss():
    """Test 10: GRPO Loss forward。"""
    print("Test 10: GRPO Loss forward ... ", end="")
    B, G, T, V = 2, 4, 8, 100
    loss_fn = GRPOLoss(GRPOConfig(group_size=G))
    new_logits = torch.randn(B * G, T, V)
    old_logits = torch.randn(B * G, T, V)
    ref_logits = torch.randn(B * G, T, V)
    actions = torch.randint(0, V, (B * G, T))
    rewards = torch.randn(B * G)
    mask = torch.ones(B * G, T)
    loss, comp = loss_fn(new_logits, old_logits, ref_logits, actions, rewards, mask)
    assert "policy_loss" in comp
    print(f"✓ (loss={loss.item():.4f})")


def test_mllm_policy():
    """Test 11: SimpleMLLMPolicy forward + generate。"""
    print("Test 11: MLLM Policy forward + generate ... ", end="")
    model = SimpleMLLMPolicy(vocab_size=1000, img_embed_dim=384, text_embed_dim=384)
    img = torch.randn(2, 3, 224, 224)
    txt = torch.randint(0, 1000, (2, 32))
    logits = model(img, txt)
    assert logits.shape == (2, 32, 1000)
    prompt = torch.randint(0, 1000, (1, 8))
    generated = model.generate(img[:1], prompt, max_new_tokens=4)
    assert generated.shape == (1, 12)
    print(f"✓ (logits={logits.shape}, gen={generated.shape})")


def test_parse_audit():
    """Test 12: 解析模型输出。"""
    print("Test 12: ParseAudit ... ", end="")
    text = '{"risk_category": "色情", "risk_level": "中危", "reason": "擦边"}'
    parsed = parse_audit_output(text)
    assert parsed is not None
    assert parsed.risk_category == "色情"
    print(f"✓ ({parsed.to_json()})")


if __name__ == "__main__":
    print("=" * 60)
    print("HuaweiBLM · Smoke Test")
    print("=" * 60)
    test_visual_pretrain_imports()
    test_grpo_imports()
    test_smart_resize()
    test_qwen_vit_forward()
    test_blm_clip_forward()
    test_binpack()
    test_fg_clip_loss()
    test_dist_sync_comparison()
    test_reward_5_dim()
    test_grpo_loss()
    test_mllm_policy()
    test_parse_audit()
    print("\n" + "=" * 60)
    print("✓ 12/12 smoke test 全部通过")
    print("=" * 60)
