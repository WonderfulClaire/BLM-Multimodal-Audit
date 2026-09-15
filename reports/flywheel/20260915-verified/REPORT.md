# 数值审核与真实Agent轨迹回流

结论：本轮成功识别2个有学习信号的训练案例，独立观测审核通过，但两条均已存在于512条基础训练集，最终新增0条。没有将重复案例包装为新数据，也未据此重复启动训练。

输入来自5G Diagnostic Agent的修复编号泄漏协议：1.7B混合课程SFT，2048步后执行8组、每组4条的GRPO采样。6组全部正确且奖励相同，进入回放；2组存在正确性差异且奖励排序一致，进入RL。实际训练器和离线飞轮对8组的分类完全一致。

`numeric_audit.py`仅解析工具观测，不读取教师ground_truth或audit_measurements。它独立提取速度、下倾角、距离/RSRP、载频干扰、PCI余数、切换次数、A3阈值及资源块；缺失、不支持或相互矛盾的记录拒绝审核。512条数值课程全部通过观测标签一致性核验，见../../numeric-validator.json。

这是针对numeric-fixture-v1固定语法的合成任务规则，不是通用电信专家、人工复核或真实业务验证。该规则与任务生成器采用同一人工诊断定义，只在实现和输入路径上独立。通过审核不能证明因果诊断适用于真实网络。

回流记录包含来源ID、内容摘要、规则版本及独立审核决定；冻结测试仅用于建立禁止混入训练的索引，没有用于错误挖掘。原始决策、数据版本报告和输入/源码哈希保留于本目录。

可执行入口：

```bash
python -m scripts.run_agent_flywheel INPUT_DIRECTORY --output NEW_ROUND_DIRECTORY --validator measured
```

输入目录沿用已有四个文件：train.jsonl、compound_train.jsonl、trajectories.jsonl、eval12.jsonl。最后一个名称为历史兼容，内容是全部需要隔离的评测案例，不限制12条。只有rl_ready和teacher_or_sft_repair组产生监督候选；奖励或效率冲突组保留在路由记录中，不自动转成训练数据。

本轮不宣称飞轮回训提升。混合课程SFT的首次冻结测试收益记录在Agent仓库，不能归因于此后运行的回流审核或两次GRPO参数更新。
