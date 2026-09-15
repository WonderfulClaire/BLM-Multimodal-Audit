# 跨卡负例：梯度等价与反向通信优化

2026-09-15。结论：在本次真实DDP测试中，双向特征交换的参数梯度与集中式目标一致。NCCL反向从完整AllReduce改为SUM ReduceScatter后，大消息微基准改善，小消息未改善；不声称通用或端到端加速。

## 正确性

实际四进程DDP双线性编码器、L2归一化、可学习logit scale。独立参考不调用分布式gather，而直接计算全局对称CLIP目标。比较rank平均后的loss，以及DDP已经平均的图像编码器、文本编码器和温度参数梯度。

batch分布为`[2,2,2,2]`、`[0,1,2,3]`和`[0,0,0,4]`；全局全空一致拒绝。CPU/Gloo与四卡NCCL通过。FP64下NCCL最大参数梯度绝对误差为约`1.11e-16`。这是有限场景的正确性检验，不包含BF16收敛、业务表征质量、多节点或异常进程恢复。

## 反向改动

AllGather让每个特征被所有rank消费，其反向需把各消费者对同一特征块的梯度求和后归还所有者。旧版AllReduce全部块再取本卡块；新版NCCL直接ReduceScatter，Gloo使用原回退。输入仍构造完整padding张量，不能据此宣称峰值显存降低。

保留`backward_mode="all_reduce"`用于同次实验中的旧版对照；默认`auto`在NCCL使用ReduceScatter。原生对照为PyTorch可求导AllGather，具体算法由库决定，不是强制单向Ring。

## 通信微基准

CUDA/PyTorch2.6.0，四张服务器报告的RTX4090设备；所用GPU4–7处于同一NUMA节点，`nvidia-smi topo -m`显示NODE路径，没有NVLink。原始设备名称不等于已验证消费版显存规格。

维度512、FP32、预热3次、测30次。每次测量取各rank最大墙钟耗时，表中为其中位数。包括长度交换、特征汇集、平方均值目标反向和GPU同步，不含完整视觉训练。固定方法顺序，未进行多次随机交错复测。

| 每卡特征数 | Bidir + ReduceScatter | Bidir + AllReduce | 原生可求导AllGather |
|---|---:|---:|---:|
| 32 | 0.665918ms | 0.640750ms | 0.769325ms |
| 512 | 0.807974ms | 0.955660ms | 0.867642ms |

大块新版相对旧版中位数低约15.5%；小块新版约高3.9%。单次微基准只能支持当前条件下的观察，不能写成“训练速度提升15.5%”或“通信成本必降50%”。原始逐次耗时和p95见JSON；AllReduce大块尾部有抖动。

前一轮20次基准也保留：当时bidir仍使用AllReduce反向，32特征时0.674ms对原生0.805ms，512特征时0.965ms对原生0.886ms。它促成了检查反向通信的假设，不与新版混成一个未经说明的平均值。

## 复现

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --master-addr=127.0.0.1 --master-port=29672 --nnodes=1 --nproc-per-node=4 \
  --module scripts.distributed_negatives_lab --backend nccl --repeats 30 \
  --output runs/distributed-new.json
```

CPU用`--backend gloo`。先确认所选设备空闲。输出只创建不覆盖；同一端口不能与其他任务冲突。Mac的自动hostname rendezvous曾解析失败，显式localhost启动成功。原生兼容对照使用的`torch.distributed.nn.functional`在本机2.14提示弃用，服务器2.6仍可用；本报告不据此承诺未来版本API兼容。

前后源码SHA-256、Gloo结果、两轮NCCL结果均在本目录。下一步如研究系统收益，应固定全局batch/数据目标，加入动态batch倾斜、端到端step、显存、拓扑和固定质量评测。
# 完整图像训练补充验证

新增 `visual_pretrain.train_distributed`：动态分辨率图像和实际裁剪区域经过模型编码，全局与区域分别形成跨卡负例池，难负例按区域全局数量校正 DDP 平均权重。完整模型首步梯度与集中式原始 FGClipLoss 独立对照。

- CPU/Gloo 两进程，3 张合成图按 2/1 切分：FP32 参数梯度最大误差 4.77e-7，完成 3 步。
- GPU/NCCL 两卡，同一图像批次：最大误差 7.75e-7，完成 10 步；47 个参数张量均变化，47 份优化器状态，checkpoint 通过 weights_only 重载。
- 独立三进程测试额外覆盖区域数 0/1/3 与全局无区域，核对编码器及温度参数梯度。

原始记录位于 integrated-gloo、integrated-nccl、integrated-nccl-v2。首版保存的 TorchVersion 对象阻止默认安全加载；改为字符串后重新执行 NCCL 实验并验证加载，保留历史记录。源码哈希见 integrated-source.json。

这证明了分布式训练目标和存盘链路的正确性。图像仅为 3 张程序绘制的颜色方块，训练 loss 下降不是业务能力提升。完整批次及逐图编码尚未优化数据吞吐，其耗时不可与纯通信微基准直接比较，首步还包含集中式参考计算。入口要求每个 rank 至少一张图；图像完全空 rank 仅在底层同步测试中覆盖。
