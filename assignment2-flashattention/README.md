# 作业二：FlashAttention

核心实现为 `flashattention2.py`，题目见 [PDF](cs336_spring2025_assignment2_systems.pdf)。本目录只收录现有 attention 核心代码，不声称覆盖 systems 作业全部题目。

## 环境与运行

需要 Linux、可用 NVIDIA GPU 及兼容 PyTorch 2.6 的驱动。环境锁定 Triton 3.2.0；先确认 `nvidia-smi` 能正常显示设备。

```bash
uv sync --locked --python 3.11
uv run --locked python examples/smoke.py --device cuda:0
```

示例固定种子 2025，使用 FP32，测试长度 64、65、128 的因果/非因果 attention，共 6 组；逐项比较输出、dQ、dK、dV 与稠密 PyTorch 参考实现。容差为 `atol=rtol=2e-4`，结果写入 `outputs/smoke/result.json`。同名输出会覆盖，可通过 `--output-dir` 修改。

没有 CUDA 时入口会明确报错，不能作为验证通过。仅查看用法可以执行 `python examples/smoke.py --help`，不需要导入 Triton。

## 实现边界

- 输入为形状相同的 `[heads, sequence, head_dim]` Q/K/V；`head_dim` 至少 16 且为 2 的幂。示例验证维度为 32；未覆盖的 dtype/形状不作正确性承诺。
- 前向使用 Triton 分块及 online softmax。为对照 FP32 参考计算，点积显式使用 IEEE 精度；后续测速应记录精度设置。
- 反向是 `torch.compile` 包装的 PyTorch 计算，会重建完整的 attention 矩阵，因此不是完整的低显存 FlashAttention-2 反向 kernel。
- 本次整理环境没有 GPU，**GPU 数值测试尚未执行**，也未提供吞吐量或显存指标。

欢迎带上 GPU 型号、驱动、PyTorch/Triton 版本及 `result.json` 交流；记录位置见 [复现状态](../docs/REPRODUCIBILITY.md)。
