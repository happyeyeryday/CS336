import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，并使用别名 nn
import numpy as np  # 导入 NumPy 库
from einops import rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        # 预计算 theta 的幂次方，形状为 (d_k/2,)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        # 预计算位置 m，形状为 (max_seq_len,)
        m = torch.arange(max_seq_len, device=device)

        # 计算 m * freqs^T，得到形状为 (max_seq_len, d_k/2) 的角度矩阵
        freqs = torch.einsum("i,j->ij", m, freqs)

        # 注册为 buffer，这样它会随模型移动到不同设备，但不是模型参数
        # 注意：我们不再将 freqs 复制拼接
        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x 的形状: (batch, num_heads, seq_len, d_k)
        # token_positions 的形状: (batch, seq_len)

        # 从预计算的 buffer 中根据位置索引出 cos 和 sin 值
        # 形状: (batch, seq_len, d_k/2)
        cos = self.freqs_cos[token_positions]
        sin = self.freqs_sin[token_positions]

        # 增加 num_heads 维度以进行广播
        # 形状: (batch, 1, seq_len, d_k/2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # 将 x 的最后一个维度两两一组进行旋转
        # 1. 将 x 变形为 (..., seq_len, d_k/2, 2)
        x_pairs = rearrange(x, '... (d j) -> ... d j', j=2)

        # 2. 获取 x_even (x0, x2, ...) 和 x_odd (x1, x3, ...)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        # 3. 应用旋转公式
        # x' = x * cos - rotate_half(x) * sin
        # rotate_half(x) 的实现：将 x_odd 取反，然后和 x_even 拼接
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # 将旋转后的偶数和奇数部分重新组合
        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1)

        # 将形状恢复为 (..., seq_len, d_k)
        return rearrange(x_rotated, '... d j -> ... (d j)')
