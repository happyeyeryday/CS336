# 来源与整理记录

## 上游

- 作业一的脚手架、测试、题目和许可证来自 [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)。个人实现原目录为 `llm源代码/ass1手搓bpe+llm/assignment1-basics`。
- FlashAttention 实现来自本地 `llm源代码/flashattention核心代码`，题目来自本地保存的 Stanford assignment 2 PDF。
- 并行实现来自本地 `llm源代码/3p`，对应本地课程作业；未找到题目或测评代码，不自行冒充 Stanford assignment 3。
- 数据作业来自 [stanford-cs336/assignment4-data](https://github.com/stanford-cs336/assignment4-data)，原本地上游 HEAD 为 `5a5f890`。其中 `cs336-basics/` 是配套训练实现。

保留作业一、四目录下的 Stanford 许可证及上游 README/CHANGELOG。个人实现、示例和文档由仓库维护者整理，不扩大第三方材料的授权范围。

## 本次整理的行为改动

- 作业一：统一 tokenizer 文件前缀加载，保留训练初始字符表，移除虚假词表生成；默认 GPU 改为 0，训练数据使用相对路径，补充设备/步数/批大小参数及训练结束时保存。
- 作业二：明确支持的输入形状，FP32 点积显式采用 IEEE 精度以供正确性对照；没有进行 GPU 性能优化或实测。
- 作业三：补充小规模启动与参考验证入口、CPU/Gloo 路径、端口和输出配置、维度限制；修复 TP 输入梯度重复求和和分片优化器只清理本地分片梯度的问题。
- 作业四：保留未实现接口；整理资源脚本及说明，不修改配套模型训练算法。

没有导入旧环境、旧 Git 历史或历史训练结果。微型示例的成功仅说明其覆盖的路径，不代表完成原课程全部要求。
