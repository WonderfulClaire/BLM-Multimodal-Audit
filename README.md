# BLM Multimodal Audit

面向图像与视频审核的多模态训练项目：从细粒度数据生产、视觉表征学习，到结构化审核输出的 GRPO 后训练与错例回流。

## 核心模块

| 目标 | 实现 | 验证 |
|---|---|---|
| 保留细节、控制视觉 token 预算 | Smart Resize、真实窗口划分、全局注意力层、Q/K 二维 RoPE | 尺寸约束、RoPE 范数及前向调用测试 |
| 扩大对比学习负例池 | 不等长 batch 双向 ring 特征同步、可微反向聚合、双向 CLIP | 3/4 进程及空 rank，数值与集中式梯度对齐 |
| 学会区分局部细节 | Caption → 表达式 → 检测框 → 经审核的难负例；全局/区域/难负例三项损失 | 真实渲染图像训练、非法框与假负例拒绝测试 |
| 在视频预算内保留时序信息 | 覆盖与变化量抽帧、分段 mask、局部及全局帧注意力、binpack | 帧数预算、跨视频及 padding 隔离测试 |
| 用反馈改善审核策略 | 自回归采样、可解释奖励、GRPO、冻结参考策略、错例队列与教师候选审核入口 | 奖励逐条重算、参数更新及 checkpoint 测试 |

## 快速运行

Python 3.12；无需下载模型即可验证核心训练链路。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
OMP_NUM_THREADS=1 python -m pytest tests -q
python -m scripts.run_visual_pretrain_demo
python -m scripts.run_grpo_demo
```

演示图片由程序绘制，框和属性来自渲染过程；红色标为“风险”只是人工定义的集成测试规则。输出保存在 `runs/`，包含各损失、生成轨迹、奖励分解、错例和模型权重。

## 数据生产与训练

```bash
python -m scripts.make_fixture runs/fixtures
python -m visual_pretrain.pipeline runs/fixtures/raw.jsonl --output runs/fixtures/train.jsonl
python -m visual_pretrain.train_manifest runs/fixtures/train.jsonl --output runs/visual --steps 10
python -m grpo_post_training.train runs/fixtures/audit.jsonl --output runs/grpo --steps 6
```

`pipeline` 支持缓存的模型输出，也提供 Qwen2.5-VL、spaCy、YOLO-World 后端。使用模型后端时安装 `requirements-models.txt`，自行准备权重和 spaCy 模型，通过 `--caption-model`、`--spacy-model`、`--yolo-weights` 指定。难负例必须经显式审核或传入 verifier；缺失模型或检测框会报错，不会生成随机框补位。

错例回流入口 `grpo_post_training.flywheel` 要求候选记录、独立审核决定及禁止混入训练的评测 ID 列表；输出仅包含通过审核、有教师/规则版本、无重复的训练样本。教师服务由使用方接入，审核通过不代表真实业务正确率提升。

## 实验范围

当前可运行训练器使用从头初始化的小型双塔与结构化生成模型。GRPO 的三个输出 token 分别表示类别、等级、理由，适合核验采样、奖励、梯度和存盘链路；完整自然语言多模态模型训练仍需接入真实权重与训练后端。

二维视觉 RoPE 已实现；完整多模态语言模型的时间/高度/宽度 MRoPE 不包含在该紧凑模型中。视频抽帧采用确定性启发式，不能保证短事件召回。视觉训练入口逐图编码动态分辨率；分布式同步和视频表征模块有独立测试，尚未接入统一的大规模多机训练入口。

查看 [验证记录](reports/validation.md) 与 [实验协议](docs/EXPERIMENT_PROTOCOL.md)。CPU 集成测试不代表业务准确率、GPU 吞吐或 Ascend 实验结果。原始业务数据、内部模型、个人简历均不包含在仓库中。

## 方法与许可

项目使用通用的 CLIP、GRPO、窗口注意力与旋转位置编码思想，并为审核任务实现数据检查、训练链路和测试。方法名称不表示原创算法声明。代码采用 MIT；可选预训练模型和第三方后端遵循各自许可，尤其请按 Ultralytics 的许可选择检测部署方式。
