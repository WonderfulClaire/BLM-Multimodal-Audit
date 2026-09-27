# Reward Audit：不要把训练 Reward 当作业务正确率

多维 reward 可以比单一 reward 更稳定，但它仍然只是优化目标。这个仓库已经把 reward 分成 recall、category、consistency、format、instruction 五个维度；下一步需要继续检查：reward 的排序是否真的和独立 evaluator 的排序一致。

## 为什么需要独立 Reward Audit

如果两个候选输出：

    A: training reward = 0.90, independent eval = 0
    B: training reward = 0.80, independent eval = 1

训练目标会偏向 A，但真实评价更喜欢 B。这叫 reward-quality inversion。

如果这种 inversion 很多，继续加大 GRPO 训练只会更强地优化一个错位目标。

## 新增指标

grpo_post_training.reward_audit.audit_reward_alignment 计算：

- inversion_rate：reward 与独立 evaluator 排序相反的 pair 比例；
- top_reward_failure_rate：最高 reward 样本中，独立 evaluator 判失败的比例；
- best_reward_index vs best_eval_index：最高 reward 与最高真实评价是否同一个候选；
- ties：reward / evaluator 无法提供排序信号的 pair。

示例：

    from grpo_post_training.reward_audit import audit_reward_alignment

    audit = audit_reward_alignment(
        rewards=[0.9, 0.8, 0.1],
        eval_scores=[0.0, 1.0, 0.5],
        success_threshold=1.0,
    )
    print(audit.to_dict())

## GRPO 训练中怎么用

每个 prompt 的 rollout group 同时保留：

    candidate
    training_reward
    independent_eval
    reward_breakdown
    length

训练仍然可以使用 training reward，但 release gate 应额外检查：

    reward up
    secure / independent eval up
    inversion rate down or stable
    top-reward failure rate down or stable

如果只看到 reward 上涨，不能说能力提升。

## 推荐实验表

| Model | Train reward | Independent score | Inversion rate | Top-reward failure |
| --- | ---: | ---: | ---: | ---: |
| SFT | TBD | TBD | TBD | TBD |
| SFT + GRPO | TBD | TBD | TBD | TBD |
| Reward-v2 GRPO | TBD | TBD | TBD | TBD |

此外继续分风险类别报告漏检/误检，不要让总体平均掩盖高危类别回退。

## 和 Data Flywheel 的关系

已有 flywheel 会把 reward-quality conflict 路由到审查分支。这个 audit 模块提供一个更直接的 group-level 定量指标：

    rollout group
        ↓
    training reward + independent evaluation
        ↓
    pairwise reward audit
        ↓
    aligned → RL candidate
    conflict → reward / data audit

这使“reward 是否值得继续优化”本身成为可测试对象，而不是凭直觉判断。
