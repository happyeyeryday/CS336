import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，并使用别名 nn
from .RMSNorm import RMSNorm  # 从同级目录导入 RMSNorm 模块
from .Attention import MultiheadSelfAttention, softmax  # 从同级目录导入多头自注意力和 softmax 函数
from .PositionwiseFeedForward import PositionwiseFeedForward  # 从同级目录导入逐位置前馈网络模块
from .Embedding import Embedding  # 从同级目录导入 Embedding 模块
from .Linear import Linear  # 从同级目录导入 Linear 模块
from .Loss import cross_entropy_loss # 导入损失函数

class Transformer(nn.Module):  # 定义一个 Transformer 模块（通常称为 Transformer Block 或 Layer）
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None):
        """
        参数:
            d_model (int): Transformer 模块输入的维度。
            num_heads (int): 多头自注意力中使用的头数。
            d_ff (int): 逐位置前馈网络内层的维度。
            max_seq_len (int): 预缓存的最大序列长度。
            theta (float): RoPE 参数。
            device: 目标设备
        """
        super().__init__()  # 调用父类 nn.Module 的构造函数
        self.d_model = d_model  # 将模型维度保存为实例属性
        self.num_heads = num_heads  # 将头数保存为实例属性
        self.d_ff = d_ff  # 将前馈网络内层维度保存为实例属性

        self.rms_norm1 = RMSNorm(d_model=d_model, device=device)
        self.rms_norm2 = RMSNorm(d_model=d_model, device=device)
        self.attn = MultiheadSelfAttention(  # 多头自注意力模块
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True,  # 明确使用旋转位置编码
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        返回:
            Float[Tensor, "batch sequence_length d_model"] 在输入特征上运行 Transformer 模块并使用 RoPE 后的输出张量。
        """
        # 第一个子层：多头自注意力（Pre-Norm 结构）
        # 将 token_positions 显式传递给注意力模块
        y = x + self.attn(self.rms_norm1(x), token_positions=token_positions)

        # 第二个子层：前馈网络（Pre-Norm 结构）
        # 将第一个子层的输出 y 进行层归一化，然后通过前馈网络，最后将结果与 y 相加（残差连接）。
        output = y + self.ff(self.rms_norm2(y))
        return output

class TransformerLM(nn.Module):  # 定义一个完整的 Transformer 语言模型
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int,
                num_heads: int, d_ff: int, rope_theta: float, device=None):
        """
        参数:
            vocab_size (int): 要预测的输出词汇表中的唯一项数。
            context_length (int): 一次处理的最大 token 数。
            d_model (int): 模型嵌入和子层输出的维度。
            num_layers (int): 要使用的 Transformer 层数。
            num_heads (int): 多头注意力中使用的头数。`d_model` 必须能被 `num_heads` 整除。
            d_ff (int): 前馈网络内层的维度（参考论文 3.3 节）。
            rope_theta (float): RoPE 的 $Theta$ 参数。
            device: 目标设备
        """
        super().__init__()  # 调用父类 nn.Module 的构造函数
        self.vocab_size = vocab_size  # 将词汇表大小保存为实例属性
        self.context_length = context_length  # 将上下文长度保存为实例属性
        self.num_layers = num_layers  # 将 Transformer 层数保存为实例属性

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        # nn.ModuleList: 一个可以持有子模块的列表。
        # 传递给 ModuleList 的模块会被正确注册，并且模型的方法（如 .parameters() 或 .to()）会作用于它们。
        # 这里用它来堆叠多个 Transformer 模块。
        self.layers = nn.ModuleList(
            [Transformer(  # 创建一个 Transformer 模块实例
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device
            )
            for _ in range(num_layers)]  # 根据 num_layers 循环创建多个层
        )
        self.rms_norm = RMSNorm(d_model=d_model, device=device)
        self.output_embeddings = Linear(in_features=d_model, out_features=vocab_size, device=device)

    def forward(self, in_indices: torch.Tensor, targets: torch.Tensor | None = None):
        """
        参数:
            in_indices (Int[Tensor, "batch_size sequence_length"]) 用于运行语言模型的输入索引张量。
            targets (Int[Tensor, "batch_size sequence_length"] | None): 训练时的目标索引。

        返回:
            如果提供了 targets，返回 (logits, loss)；否则返回 (logits, None)。
        """
        # 在顶层模型中创建 token_positions
        batch_size, seq_len = in_indices.shape
        token_positions = torch.arange(seq_len, device=in_indices.device).unsqueeze(0).expand(batch_size, -1)

        # 1. 将输入的 token ID 转换为词嵌入向量
        x = self.token_embeddings(in_indices)

        # 2. 依次通过堆叠的 Transformer 模块，并传递位置信息
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        # 3. 在所有 Transformer 模块之后进行最终的归一化
        x_norm = self.rms_norm(x)
        # 4. 通过输出线性层将归一化后的向量投影到词汇表空间，得到 logits
        logits = self.output_embeddings(x_norm)

        loss = None
        if targets is not None:
            # 如果提供了目标，则计算损失
            # view(-1) 将张量展平成一维
            loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
