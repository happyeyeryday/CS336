# 作业一：BPE 与 Transformer

实现位于 `cs336_basics/`；`transformer_main.py` 是原有训练/推理入口。题目见 [PDF](cs336_spring2025_assignment1_basics.pdf)，官方原始说明保留为 [README.upstream.md](README.upstream.md)。

## 最小复现

在本目录执行，使用独立 Python 3.11 环境：

```bash
uv sync --locked --extra cpu --python 3.11
uv run --locked --extra cpu python examples/smoke.py
```

示例使用仓库内的小语料，固定种子 2025，训练 20 次 BPE 合并，再训练一个两层、宽度 32 的 Transformer 共 5 步；检查词表保存/加载、训练梯度、参数更新、模型和优化器恢复、生成 4 个 token。生成文本仅用于检查流程，没有语言质量保证。

可调整训练步数、语料与输出目录：

```bash
uv run --locked --extra cpu python examples/smoke.py --steps 10 --output-dir outputs/demo
```

输出包含 `bpe_vocab.pkl`、`bpe_merges.pkl`、`tiny.pt` 和 `result.json`。已有同名输出会被此次微型示例覆盖；需要保留时指定不同目录。

单 NVIDIA GPU 可使用独立的 CUDA 12.4 PyTorch 变体：

```bash
uv sync --locked --extra cu124 --python 3.11
uv run --locked --extra cu124 python examples/smoke.py --device cuda:0
```

`cpu` 与 `cu124` 互斥。切换后运行命令也保留对应 `--extra`，避免 uv 重新切换依赖。CUDA 路径尚未在本次整理中实测。

## 自己的数据与原训练入口

先用有限数量的词训练 tokenizer；默认只读取前 100,000 个词，避免无意启动大型任务：

```bash
uv run --locked --extra cpu python examples/train_tokenizer.py \
  --dataset examples/tiny_corpus.txt --merges 20 --output-prefix outputs/tokenizer/bpe
```

保存和加载统一采用文件前缀：`save("outputs/tokenizer/bpe")` 对应 `load("outputs/tokenizer/bpe")`。传入完整文件名时使用 `load_files(vocab_path, merge_path)`。训练入口里的 `--vocab`、`--merge` 分别接收两个完整文件路径；缺少文件会报错，不会生成虚假词表。

准备好自己的 UTF-8 训练集及对应 tokenizer 后，可运行原训练入口，例如：

```bash
uv run --locked --extra cu124 python transformer_main.py --train \
  --dataset data/owt_train.txt \
  --vocab outputs/tokenizer/bpe_vocab.pkl --merge outputs/tokenizer/bpe_merges.pkl \
  --device cuda --gpu_id 0 --batch_size 2 --max_steps 100 --ckpt_dir outputs/train
```

上例展示参数用法，实际训练前应针对目标语料重新训练 tokenizer；其默认模型远大于微型示例。参数详见 `--help` 和 `config.py`。原训练入口自动读取指定 checkpoint 目录的 `latest.pt` 续训；开始新实验使用新目录。原训练与 PPL 流程未进行完整实验验证，不能据此宣称复现原课程指标。

## 已知限制与官方测试

- `tests/adapters.py` 的 21 个官方接口仍为 `NotImplementedError`。`uv run --locked --extra cpu pytest tests` 会失败；保留这些测试是为了后续接入，新增 smoke 通过不代表官方测试通过。
- 当前 BPE 是按词拆分的字符级实现，使用 `</w>`；空白会归一化，未登录符号可能被丢弃，合并策略仍需进一步核对。它不具备官方 byte-level tokenizer 的完整语义。
- 根目录的 `bpe.py` 是旧实现，作为历史参考保留；复现入口统一使用 `cs336_basics/bpe.py`。
- 原有 `tests/fixtures/` 和 `tests/_snapshots/` 包含测试必需的小模型及快照，应保留。

当前验证结果见 [复现状态](../docs/REPRODUCIBILITY.md)。
