import torch
import torch.nn as nn
from .Attention import MultiheadSelfAttention
from .PositionwiseFeedForward import PositionwiseFeedForward
from .RMSNorm import RMSNorm

class TransformerBlock(nn.Module):
    """
    定义一个标准的 Transformer 模块。
    它遵循 Pre-Norm 结构: Norm -> Attention -> Residual -> Norm -> FFN -> Residual
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, use_rope: bool = False, max_seq_len: int | None = None, theta: float | None = None):
        """
        参数:
            d_model (int): 模型的维度
            num_heads (int): 多头注意力的头数
            d_ff (int): 前馈网络中间层的维度
            use_rope (bool): 是否在注意力中使用旋转位置编码
            max_seq_len (int): RoPE 的最大序列长度
            theta (float): RoPE 的 theta 参数
        """
        super().__init__()
        # 注意力块之前的 RMSNorm
        self.attn_norm = RMSNorm(d_model)

        # 多头自注意力模块
        self.attention = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            theta=theta
        )

        # 前馈网络块之前的 RMSNorm
        self.ffn_norm = RMSNorm(d_model)

        # 逐位置前馈网络 (SwiGLU)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer 模块的前向传播。

        参数:
            x (Float[Tensor, "batch_size seq_len d_model"]): 输入张量
        返回:
            Float[Tensor, "batch_size seq_len d_model"]: 输出张量
        """
        # 1. 注意力块：先归一化，然后通过注意力模块，最后加上残差连接
        x = x + self.attention(self.attn_norm(x))

        # 2. 前馈网络块：同样先归一化，然后通过前馈网络，最后加上残差连接
        x = x + self.ffn(self.ffn_norm(x))

        return x
