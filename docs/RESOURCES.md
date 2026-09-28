# 数据与外部资源

最小 CPU 示例使用仓库内微型语料和随机张量，不需要下载训练集或预训练模型。环境依赖在安装时下载，**不随 Git 仓库分发**。

## 作业一

官方模板给出的数据来源为 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 与 [CS336 OWT sample](https://huggingface.co/datasets/stanford-cs336/owt-sample)。下载命令见 [上游 README](../assignment1-basics/README.upstream.md)。下载前确认可用磁盘空间；全文训练不是最小示例的一部分。

将自备语料放到 `assignment1-basics/data/`，通过 `--dataset` 显式指定。个人 tokenizer 的生成命令见 [作业一说明](../assignment1-basics/README.md)。已有课程训练权重未在原目录中找到，不提供虚构的预训练模型下载地址。

## 作业四

`get_assets.sh` 保留上游提供的 NSFW 与仇恨言论分类器链接。目标目录为 `assignment4-data/cs336_data/assets/`，支持 `ASSETS_DIR` 覆盖。语言识别、质量分类器及语料处理仍需实现者根据题目准备，脚本不会下载它们。

原目录有一个未纳入 Git 的本地验证文件：

| 项目 | 信息 |
| --- | --- |
| 文件名 | `paloma_c4_100_domain.bin` |
| 大小 | 19,522,804 bytes，约 18.62 MiB |
| SHA-256 | `f251207286a007eb1dcbe813b08fc9ae077f821caa15041e9c2bf030fa83319f` |
| 来源 | 原本地课程目录；尚未确认公开下载地址 |
| 建议放置位置 | `assignment4-data/data/paloma_c4_100_domain.bin` |

上游配置期望的是 `tokenized_paloma_c4_100_domains_validation.bin`。**不能仅凭名称相似认定上述本地文件就是同一份数据**；使用前应确认 tokenizer、编码 dtype 和来源，再覆盖训练配置中的验证路径。在来源确认前它不是复现依赖。

配套训练入口及配置在 `cs336-basics/`；原配置的服务器绝对路径是上游示例，需要换成实际路径。作业四当前仅分享模板，不宣称数据处理或训练实验可完整复现。

## 未收录的材料

原目录中的 MOE/post-training 课件、参考论文和评分文件未加入公开仓库；它们不是当前最小示例的运行依赖。三份已有作业题目 PDF 各保留一份。第三次作业的题目和评分脚本目前缺失，首页已明确说明。

以后若需共享大型训练结果，优先在 Release 或原数据平台提供独立附件，并记录版本、大小、SHA-256 和放置位置，不将大文件写入 Git 历史。参见 [GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
