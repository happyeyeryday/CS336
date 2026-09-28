# 作业三：分布式并行

`dp/` 包含逐参数 DDP、分桶 DDP 和优化器状态分片；`tpandpp/tppp.py` 包含张量并行与双阶段流水线并行。这里对应本地课程的第三次作业，不等同于 Stanford 原版第三次作业。本地没有第三次作业题目和原测评入口。

## 无 GPU 也能检查数值

在本目录使用 Python 3.11，CPU/Gloo 以两个进程执行小规模参考对照：

```bash
uv sync --locked --extra cpu --python 3.11
uv run --locked --extra cpu python examples/smoke.py --mode ddp
uv run --locked --extra cpu python examples/smoke.py --mode tp
uv run --locked --extra cpu python examples/smoke.py --mode pp
```

DDP 检查两种封装的梯度平均、分桶梯度累积 `no_sync()`，以及分片 AdamW 连续三步与非分片参考模型的参数一致性。TP/PP 检查输出、输入梯度及全部参数梯度，参考计算采用单进程 PyTorch 函数。

结果位于 `outputs/smoke/`，包含各模式 JSON，以及 TP/PP 的 `.npz`。同模式重跑会覆盖输出，可通过 `--output-dir` 指定新目录。随机种子固定为 2025。

## 双 GPU 验证

```bash
uv sync --locked --extra cu124 --python 3.11
uv run --locked --extra cu124 python examples/smoke.py --mode ddp --device cuda --nproc-per-node 2
uv run --locked --extra cu124 python examples/smoke.py --mode tp --device cuda --nproc-per-node 2
uv run --locked --extra cu124 python examples/smoke.py --mode pp --device cuda --nproc-per-node 2
```

这三条 CUDA/NCCL 命令需两张可见 GPU，本次整理未实测。可使用 `CUDA_VISIBLE_DEVICES` 选择卡；入口采用 `torch.multiprocessing.spawn`，**不要再用 torchrun 包一层**。

## 参数和边界

- `--device cpu|cuda` 选择 Gloo/NCCL；`--port 0` 默认自动选本地空闲端口，冲突时可指定 `--port 29531`。
- `--nproc-per-node` 指定进程数。小样例 DDP 支持 1–4，TP 支持 1、2、4，PP 固定为 2；TP 检查头数及 FFN 维度的整除条件。
- PP 保留原来的 8 层网络和 `out.sum() * 1000` 损失缩放，微批数默认 4；只实现两阶段通信。它的调度仍需进一步评估，不宣称实现理想的计算通信重叠。
- `ShardedOptimizer` 适合本示例中的同 dtype、同设备参数；没有完整的跨 rank checkpoint 汇总/恢复功能。
- CPU 检查用于算法与通信正确性，不能推导 GPU 的吞吐量、通信重叠或显存效果。

具体检查结果见 [复现状态](../docs/REPRODUCIBILITY.md)。
