# 看懂实际训练界面，再回答框架与指标

TensorBoard展示日志，训练脚本负责更新参数。这里展示2026-09-15实际模型实验的日志回放，不是在线GPU负载界面，也没有补造未记录的曲线。

## 我们实际用什么

| 层次 | 本次实现 | 不要混淆 |
|---|---|---|
| 模型 | BLM用Qwen2.5-VL-3B-Instruct；Agent用Qwen3-1.7B | 模型名不是训练框架 |
| 底层计算 | PyTorch，自动求导与AdamW | 负责前向、反向、优化器更新 |
| 模型接口 | Hugging Face Transformers | 加载模型、处理图片或文本、生成回答 |
| 参数高效训练 | PEFT LoRA | 当前只更新adapter，非全参数训练 |
| 训练循环 | 自定义SFT/GRPO循环 | 本次实际运行入口没有调用TRL Trainer或veRL分布式训练器 |
| 多卡视觉训练 | torchrun、DDP、NCCL及可微特征同步 | 这一入口与单卡VLM/Agent后训练分开验证 |
| 监控 | JSONL原始日志，导入TensorBoard | TensorBoard不是RLHF，也不负责执行训练 |

这些是独立复现的事实，不能替代历史实习团队的真实技术栈。Agent仓库保留上游Agent-R1/veRL来源，但仓库含有某框架不等于本次执行了它的训练路径。

## 第一遍怎么点

1. 打开Scalars，Runs只勾选BLM/SFT_24与BLM/SFT_96；暂不把不同任务或不同目标的loss放在一起比较。
2. Horizontal Axis选Step；Smoothing调0，取消Ignore outliers in chart scaling，先看原始波动。平滑只改变显示，不改变训练。
3. Filter tags输入`train/loss`。这里SFT的step从0开始；24次更新的最后横坐标是23，不是漏掉一步。
4. 改成`eval_dev`，看`joint_correct`、`class_风险_joint_accuracy`和`all_correct_group_fraction`。
5. 点Text，展开examples_dev：对照原始回答与标签。只看一个总分，发现不了“全部判正常”。
6. 再选Agent/GRPO_verified，看`train/group_reward_std`、`train/optimizer_updated`、`train/rewards_mean`和`train/kl`。

SFT的loss是回答token平均交叉熵，目标含JSON格式与理由文本。大量容易预测的格式token变准，也会降低loss。它不等于风险召回率。GRPO的policy loss又是另一种目标，组内优势均值为0时，损失数值接近0仍可能有非零梯度；不能拿SFT与GRPO的loss高度判断谁训练得更好。

当前图中评测只有实际执行过的点。Wall time是日志导入时间；没有原始训练时间戳，所以不应从该轴推算耗时。SFT的response_tokens是示范回答长度，不是自由生成的思考长度。当前Agent关闭thinking，不能声称监控过独立think长度。

## 指标分成三个问题

| 要回答的问题 | 主要指标 | 看到异常先检查什么 |
|---|---|---|
| 训练有没有正常学习 | SFT loss、梯度范数、实际更新标记 | mask是否正确、标签是否对齐、梯度是否有限、参数是否真的更新 |
| RL是否拿到了有用反馈 | 奖励均值与分项、组内标准差、全组同分比例 | 全对还是全错、奖励漏洞、生成是否多样，而不是盲目加步数 |
| 策略是否偏离过快 | KL、clip fraction、生成行为 | 结合学习率、更新次数与参考策略；没有通用正确阈值 |
| 任务是否变好 | BLM类别/等级联合准确率、风险类召回、分组全对率；Agent根因集合exact match、set F1、漏诊/多报 | 类别退化、格式投机、固定集泄漏、工具证据是否真实有用 |
| 调用是否有效 | 工具失败率、调用次数、成功率下的成本 | 减少调用是否同时损害成功率；计数是否包含submit |
| 运行是否高效 | GPU显存、利用率、生成token/s、rollout/更新耗时 | 记录设备、batch、token长度、并行度；瞬间GPU截图不能证明整体提速 |

本次TensorBoard确有loss、梯度、KL、奖励、更新标记和部分评测；clip fraction来自Agent日志。entropy、持续GPU历史、token/s等没有被补造为已监控曲线。真实图像审核还需独立理由评分与真实风险测试集，本次未完成这些业务评测。

## 一次实际调参是怎么做的

SFT24在开发集正确8/12，但所有图片都判为正常；风险类0/4。首先看逐例回答，避免把66.7%当成有效识别。

随后做两项定位：SFT24连训练集也只有8/12；未微调模型在只问颜色的训练图探针中，能正确识别红/蓝/绿方块。这提示应先检查规则学习和训练是否充分，而非立即增加图片分辨率或改检测器。

只做一个训练覆盖对照：相同12条训练图、seed42、学习率1e-4和LoRA配置，从头训练96步。每次处理1条样本，24步约2遍，96步约8遍。SFT96训练集/开发集均12/12；随后冻结新布局的24张确认图，SFT24为16/24、风险0/8，SFT96为24/24、风险8/8。仅能说明此合成规则的布局迁移改善，不能说真实业务审核满分。

这次改的是训练预算，不是同时改学习率、奖励、模型大小后再挑最好结果。它是一次证据支持的诊断，不说明“所有欠佳结果都加训练步数”。

## 参数究竟在哪里改

实际入口是`grpo_post_training/vlm.py`。示例中的模型目录由使用者设置，输出目录必须不存在。

```bash
python -m grpo_post_training.vlm sft runs/vlm-fixture/train.jsonl \
  --model "$RCA_VLM_MODEL" --output runs/sft96-new \
  --steps 96 --lr 1e-4 --seed 42

python -m grpo_post_training.vlm grpo runs/vlm-fixture/train.jsonl \
  --model "$RCA_VLM_MODEL" --sft-adapter runs/sft/adapter \
  --output runs/grpo-new --steps 6 --group-size 4 --lr 1e-5 --beta .01 --max-tokens 192
```

- `--lr`控制每次参数更新尺度。梯度不稳定先排数据和数值错误，再考虑降低；不是只凭一处尖峰就改。
- `--steps`控制训练循环次数。GRPO每个step是一个采样组，全组同分可跳过优化，因此不等于实际更新次数。
- `--group-size`是同一问题的回答数；增加可能带来更多比较信息，也提高采样成本，不保证能解决全组同分。
- `--beta`是KL系数，限制策略偏离参考分布的力度；增大约束更强，但也可能压制有效学习。
- `--max-tokens`限制生成长度。判断截断要核对是否达到上限与终止token；调低会省计算，也可能截掉答案。
- LoRA rank8/alpha16、clip0.2、梯度裁剪1.0在代码中固定，当前不是命令行参数。梯度日志记录裁剪前范数，所以数值大于1不等于裁剪失效。

修改配置后用新目录运行，训练与开发集用于排错选型；确认测试一旦用于选型，就不再是未见测试。不要因为某个固定测试集掉分反复修到满分，再称作独立泛化。

## 面试口述

> 在我后续独立复现里，用的是PyTorch、Transformers和PEFT，SFT/GRPO是自定义训练循环，使用LoRA更新参数。模型自己采样，环境或规则返回奖励，再计算组内优势和策略损失；原始日志保存为JSONL，用TensorBoard看曲线。视觉对比学习的多卡入口另外使用DDP和NCCL。

> 指标我分训练状态、任务效果和运行效率。SFT看loss、梯度；RL看奖励、组内方差、KL、有效更新比例，同时看原始生成。业务评测再看风险召回或者根因集合正确率，不拿reward替代独立评测。这次就出现loss下降、总体正确率66.7%，但风险类全漏的情况。我通过逐例输出、训练集表现和颜色探针定位后，只增加SFT训练步数做对照，并用新布局确认结果。

历史实习没负责完整RL部署时，先讲清参与范围，再切换到这次实际跑过的复现。看过界面只是开始，能把曲线异常关联到代码、样本和下一步实验，才是调参经验。

来源：[本次VLM原始记录](../reports/vlm/20260915/REPORT.md)、训练源代码、[TensorBoard官方入门](https://www.tensorflow.org/tensorboard/get_started)。


## 从仓库证据重建界面

```bash
pip install -r requirements-monitoring.txt
python -m scripts.export_tensorboard reports/vlm/20260915/tensorboard-spec.json --output runs/tensorboard-review
tensorboard --logdir runs/tensorboard-review --host 127.0.0.1 --port 6006
```

打开http://127.0.0.1:6006。输出目录必须不存在；这是可重建的日志回放。本机带读还导入了另一个Agent仓库的日志；公开spec仅引用本仓库的BLM证据，无私人绝对路径。
