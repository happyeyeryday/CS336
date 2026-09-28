# 作业四：数据处理模板（待完成）

题目见 [PDF](cs336_spring2025_assignment4_data.pdf)，原始说明保留为 [README.upstream.md](README.upstream.md)。目前 `cs336_data/` 只有包初始化代码，`tests/adapters.py` 中 11 个处理接口全部未实现。保留模板供同学一起补全。

## 独立环境

```bash
uv sync --locked --extra cpu --python 3.11
uv run --locked --extra cpu pytest --collect-only -q tests
```

Linux 安装 fasttext 等包可能需要 C++ 编译器及开发工具。上面只收集测试，不执行，也不代表功能通过。实际运行下面命令会因未实现接口而失败：

```bash
uv run --locked --extra cpu pytest tests
```

作业四使用配套的 PyTorch 2.7 和 `cs336-basics/`，不能替换为作业一的个人实现，也不要将两份包安装到同一环境。运行配套训练前需单独准备数据和 GPU，并关闭或配置 WandB；本次未验证完整训练。

## 实现清单

| 部分 | 接口数量 | 状态 |
| --- | --- | --- |
| HTML 正文提取 | 1 | 待实现 |
| 语言识别 | 1 | 待实现，需对应分类器 |
| 邮箱、电话、IP 脱敏 | 3 | 待实现 |
| NSFW、仇恨言论、质量分类 | 3 | 待实现，需对应分类器/训练流程 |
| Gopher 质量过滤 | 1 | 待实现 |
| 精确行去重、MinHash 去重 | 2 | 待实现 |

建议先完成无需外部模型的文本提取、脱敏、规则过滤和去重，再处理分类器及语料流水线。将实现放入 `cs336_data/`，通过 `tests/adapters.py` 接入测试。

## 外部资源

```bash
bash get_assets.sh
```

该脚本只下载原模板提供的 **两个 Dolma 分类器**，默认写入本目录的 `cs336_data/assets/`，支持 `ASSETS_DIR` 指定位置。它不会自动提供语言识别、质量模型、训练语料或 Paloma 验证数据，也不会补全任何接口。

配套训练的原始 YAML 中仍有课程服务器路径，使用前须按自己的数据位置覆盖。未收录的资源及原本地 Paloma 文件信息见 [资源说明](../docs/RESOURCES.md)。
