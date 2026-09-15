# 真实视觉语言模型后训练

`grpo_post_training.vlm` 将实际图片经处理器送入 Qwen2.5-VL，生成自然语言 JSON 审核回答；支持 LoRA SFT、同策略组采样 GRPO 及独立评测。它补充原有三个语义 token 的紧凑训练器，后者仍保留用于快速单元测试。

真实VLM入口采用官方处理器和预训练视觉主干。自实现的SmartResize、窗口注意力与分布式双塔损失仍在visual_pretrain入口中验证，没有把紧凑视觉主干冒充或替换成完整Qwen视觉主干。

## 固定的小型验证协议

模型为 Qwen/Qwen2.5-VL-3B-Instruct，revision `66285546d2b821cf421d4f5eb2576359d3770cd3`。原始权重遵循模型提供方许可，不随本仓库发布。当前运行环境为 torch2.6.0、transformers4.51.3、peft0.15.2、accelerate1.6.0。

在任何模型评测前生成 train/dev/test，各12张图片、4个布局组。组内只改变方块颜色，问题和其他画面元素保持一致。画面均有红色圆形干扰物，测试规则明确它不算风险，只有红色方块算风险。不同split的图像哈希和布局组不重叠。该规则完全是人工集成测试，不是实际内容审核政策。

固定seed42，SFT24次更新、lr1e-4，随后GRPO6组、每组4条、lr1e-5、beta0.01，最多生成192个token。LoRA rank8、alpha16、dropout0，作用于语言层q/v投影；视觉主干和原始模型权重冻结。只训练train；三个预先确定的阶段均报告开发与测试结果，不依据结果追加参数扫描。

## 概率、参考模型与奖励

每个生成回答按照原始图片、问题和实际响应token重新计算概率。视觉token和问题token不作为预测目标。单步GRPO使用冻结的采样旧概率与裁剪目标；参考分布通过禁用新增policy adapter计算。

如果使用SFT初始化，先将SFT adapter合并进冻结基础模型，再加新的policy adapter。因此参考分布包含SFT能力；重载GRPO模型时也必须先加载相同SFT adapter。SFT对照评测采用同样的合并过程，避免把合并差异算作RL变化。

奖励仅为严格JSON格式下类别和等级同时正确时的1，否则0。它不使用理由关键词奖励；理由文本会生成和记录，但没有独立语义评分，不能声称推理理由已被验证。全组奖励相同跳过更新；全错进入补示范方向，全对进入回放，不强造有效梯度。

GRPO训练错误另外写入review_queue.jsonl，包含图片内容哈希、来源组、原始回答与规则版本。图片路径相对该队列所在目录，可送入data_flywheel的预算选择和教师提议阶段；记录状态始终是needs_independent_review，不自带批准或新的标准答案。评测模式不生成回流队列，辅助函数也拒绝开发/测试split。

## 运行

安装基础依赖及 `requirements-vlm.txt`，准备固定版本模型目录，并设置 `RCA_VLM_MODEL` 为该目录。

```bash
python -m scripts.make_vlm_fixture runs/vlm-fixture
python -m grpo_post_training.vlm eval runs/vlm-fixture/dev.jsonl --model "$RCA_VLM_MODEL" --output runs/vlm-base-dev
python -m grpo_post_training.vlm sft runs/vlm-fixture/train.jsonl --model "$RCA_VLM_MODEL" --output runs/vlm-sft --steps 24 --lr 1e-4
python -m grpo_post_training.vlm eval runs/vlm-fixture/dev.jsonl --model "$RCA_VLM_MODEL" --sft-adapter runs/vlm-sft/adapter --output runs/vlm-sft-dev
python -m grpo_post_training.vlm grpo runs/vlm-fixture/train.jsonl --model "$RCA_VLM_MODEL" --sft-adapter runs/vlm-sft/adapter --output runs/vlm-grpo --steps 6 --group-size 4
python -m grpo_post_training.vlm eval runs/vlm-fixture/dev.jsonl --model "$RCA_VLM_MODEL" --sft-adapter runs/vlm-sft/adapter --adapter runs/vlm-grpo/adapter --output runs/vlm-grpo-dev
```

测试时将dev.jsonl替换为test.jsonl，使用新的输出目录。所有输出目录均为只创建模式。配置记录manifest、逐图哈希和训练设置；输出逐例预测、原始生成、奖励、梯度、KL、参数变化与adapter。当前按单样本处理，优先可追溯性，不主张训练吞吐领先。

该验证的验收对象是图片条件生成、回答掩码、实际参数更新、参考模型重载及结果可追溯；真实业务效果仍需要授权数据与对应评测协议。
