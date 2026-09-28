import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，并使用别名 nn
import numpy as np  # 导入 NumPy 库，并使用别名 np

class Embedding(nn.Module):  # 定义一个名为 Embedding 的类，它继承自 nn.Module，用于将 token ID 转换为词嵌入向量
    def __init__(self, num_embeddings: int, embedding_dim: int,
        device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        参数:
            num_embeddings: int 词汇表的大小
            embedding_dim: int 嵌入向量的维度
            device: torch.device | None = None 参数所在的设备
            dtype: torch.dtype | None = None 参数的数据类型
        """
        super().__init__()  # 调用父类 nn.Module 的构造函数，这是自定义 Module 的标准做法
        factory_kwargs = {"device": device, "dtype": dtype}  # 创建一个字典，用于后续创建张量时传递设备和数据类型参数

        self.num_embeddings = num_embeddings  # 将词汇表大小保存为实例属性
        self.embedding_dim = embedding_dim  # 将嵌入维度保存为实例属性
        self.device = device  # 将设备信息保存为实例属性
        self.dtype = dtype  # 将数据类型信息保存为实例属性

        # 调用 initialize_Weights 方法来初始化嵌入矩阵
        w_init = self.initialize_Weights(num_embeddings, embedding_dim, factory_kwargs)
        # nn.Parameter: 将一个张量包装成模型的参数。
        # 被 nn.Parameter 包装后的张量会自动被认为是模型的可训练参数，
        # 这意味着在调用 model.parameters() 时它会被返回，并且在反向传播时会自动计算其梯度。
        self.weight = nn.Parameter(w_init)

    def initialize_Weights(self, vocab_size: int, d_model: int, factory_kwargs: dict) -> torch.Tensor:
        """
        使用截断正态分布方法初始化权重 W
        """
        # torch.empty: 创建一个指定形状、设备和数据类型的张量，但其内部数据是未初始化的（即内存中的任意值）。
        # 这里创建了一个形状为 (vocab_size, d_model) 的矩阵，作为嵌入查找表。
        W = torch.empty(vocab_size, d_model, **factory_kwargs)
        mean = 0  # 设置正态分布的均值
        std = 1   # 设置正态分布的标准差

        # nn.init.trunc_normal_: 这是一个in-place（就地）操作（由末尾的下划线 `_` 表示），
        # 它用从截断正态分布中抽取的值来填充输入的张量 W。
        # 截断意味着如果抽样值超出了某个范围（这里是 [-3, 3]），就会被丢弃并重新抽样，直到值在范围内为止。
        # 这有助于防止初始化时出现极端权重值。
        nn.init.trunc_normal_(W, mean, std, -3, 3)
        return W  # 返回初始化后的权重张量

    # def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    #     """
    #     查找给定 token ID 对应的嵌入向量。
    #     参数:
    #         token_ids: (batch_size, sequence_length) 形状的张量，包含 token 的 ID。
    #     返回:
    #         output: (batch_size, sequence_length, embedding_dim) 形状的张量，包含对应的嵌入向量。
    #     """
    #     # 从输入张量的形状中获取批量大小和序列长度
    #     batch_size, sequence_length = token_ids.shape
    #     # 创建一个用于存放输出结果的空张量
    #     output = torch.empty(batch_size, sequence_length, self.embedding_dim, device=self.device, dtype=self.dtype)

    #     # 遍历输入中的每个序列
    #     for i, seq in enumerate(token_ids):
    #         # 遍历序列中的每个 token ID
    #         for j, token_id in enumerate(seq):
    #             # 从嵌入矩阵 self.weight 中通过索引 token_id 查找对应的嵌入向量，
    #             # 并将其赋值给输出张量的相应位置。
    #             # self.weight 是一个 (num_embeddings, embedding_dim) 的矩阵，
    #             # token_id 是一个整数，用作行索引。
    #             output[i][j] = self.weight[token_id]

    #     # 注意：虽然这种循环实现是正确的，但在 PyTorch 中效率很低。
    #     # 更高效、更符合 PyTorch 风格的实现是直接使用高级索引：
    #     # return self.weight[token_ids]
    #     # 这行代码可以替代上面的整个循环，并利用 PyTorch 的底层优化来快速完成查找。

    #     return output  # 返回填充了嵌入向量的输出张量
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
