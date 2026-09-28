# CS336 课程作业与复现交流

整理个人课程中的四次作业，方便同学阅读代码、运行小实验和讨论实现。内容参考 Stanford CS336 Spring 2025，并包含本地课程的并行计算作业。**这是学习仓库，部分官方接口尚未实现，不代表四份完整标准答案。**

## 从哪里开始

| 作业 | 内容 | 最小运行条件 | 当前边界 |
| --- | --- | --- | --- |
| [1 · Basics](assignment1-basics/README.md) | 字符/词级 BPE、Transformer、训练与推理 | CPU 即可运行微型示例 | 原官方 adapters 未接入；BPE 不是官方要求的 byte-level tokenizer |
| [2 · FlashAttention](assignment2-flashattention/README.md) | Triton 前向、PyTorch 编译反向 | Linux、NVIDIA GPU、CUDA/Triton | GPU 数值验证待有 GPU 的环境执行；反向会构造完整 attention 矩阵 |
| [3 · Parallelism](assignment3-parallelism/README.md) | DDP、优化器分片、TP、PP | CPU/Gloo 可检查正确性；GPU/NCCL 验证需相应显卡 | PP 仅支持两个进程；无原课程测评脚本 |
| [4 · Data](assignment4-data/README.md) | 数据清洗、过滤、去重模板 | Python 3.11；训练另需 GPU/数据 | 11 个数据处理接口待完成 |

验证记录与限制见 [复现状态](docs/REPRODUCIBILITY.md)。首次阅读建议按 1 → 2 → 3 → 4 的顺序。

## 五分钟开始：CPU 微型训练

先按 [uv 官方说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv。依赖锁文件由 uv 0.12.19 生成，复现建议使用同版本。

```bash
git clone https://github.com/happyeyeryday/CS336.git
cd CS336/assignment1-basics
uv sync --locked --extra cpu --python 3.11
uv run --locked --extra cpu python examples/smoke.py
```

成功后终端输出 `PASS`，结果位于作业目录下的 `outputs/smoke/`。首次安装需要下载 Python/PyTorch 等依赖；“五分钟”指依赖就绪后的微型示例，不是完整模型训练。

四份作业**分别安装环境**，不要在根目录运行一个全局 `pytest`。作业一和作业四使用不同版本的 `cs336_basics`，合并环境会产生包名冲突。Linux 是主要复现平台；CUDA 示例建议使用 Linux，其他系统尚未验证。

## 为什么仓库很小

发布内容仅保留代码、依赖锁文件、说明、题目和测试必需的样本/小模型，约 17 MiB。原始目录约 7.86 GiB，主要是虚拟环境和 CUDA 库，安装时可重新获取。

运行生成的模型、数据、日志写入忽略目录；无需 Git LFS，也无需下载整个原始 ZIP。外部数据、未收录的 Paloma 文件、课件与参考材料见 [资源说明](docs/RESOURCES.md)。

## 一起交流

复现出错或有改进思路可在 [Issues](https://github.com/happyeyeryday/CS336/issues) 讨论。请附作业编号、运行命令、Python/PyTorch 版本、CPU/GPU 信息、随机种子及错误输出；不要附大数据或整个虚拟环境。贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。

上游代码的 Stanford 版权和许可证保留在作业一、作业四目录；来源及本次整理的改动见 [来源说明](docs/PROVENANCE.md)。
