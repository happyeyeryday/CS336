import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，并使用别名 nn
import math  # 导入数学库
from einops import einsum, rearrange  # 从 einops 库导入 einsum 和 rearrange 函数，用于高效、可读的张量操作
from .Linear import Linear  # 从同级目录导入自定义的 Linear 模块
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding  # 从同级目录导入自定义的 RoPE 模块

def softmax(x: torch.Tensor, dim: int):
    """
    给定一个输入张量，返回对指定 `dim` 进行 softmax 计算后的输出。

    参数:
        x (Float[Tensor, "..."]): 要进行 softmax 的输入特征。形状是任意的。
        dim (int): 在 `x` 的哪个维度上应用 softmax。

    返回:
        Float[Tensor, "..."]: 与 `x` 形状相同，但在指定 `dim` 上被归一化的张量。
    """
    # torch.max: 找到指定维度上的最大值。keepdim=True 保持了维度，便于后续广播。
    # .values 获取最大值本身。这是为了数值稳定性，防止 exp 操作溢出。
    x_max = torch.max(x, dim, keepdim=True).values
    # 从原始输入中减去最大值
    x_stable = x - x_max
    # torch.exp: 计算张量中每个元素的指数。
    x_exp = torch.exp(x_stable)
    # torch.sum: 计算指定维度上所有元素的和，作为 softmax 的分母。
    output = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return output

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    给定键（K）、查询（Q）和值（V）张量，返回缩放点积注意力的输出。

    参数:
        Q (Float[Tensor, " ... queries d_k"]): 查询张量
        K (Float[Tensor, " ... keys d_k"]): 键张量
        V (Float[Tensor, " ... values d_v"]): 值张量
        mask (Float[Tensor, " ... queries keys"] | None): 掩码张量
    返回:
        Float[Tensor, " ... queries d_v"]: 缩放点积注意力的输出
    """
    d_k = K.shape[-1]  # 获取 K 向量的最后一个维度的大小
    # einsum: einops 库中的函数，用于执行爱因斯坦求和约定。
    # "... queries d_k, ... keys d_k -> ... queries keys" 表示将 Q 和 K 进行批量矩阵乘法。
    # 它将 Q 的 queries 维度和 K 的 keys 维度进行点积，得到注意力分数矩阵。
    # 这等价于 Q @ K.transpose(-2, -1)。
    attention_score = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)

    if mask is not None:  # 如果提供了掩码
        # masked_fill_: 一个就地操作，将掩码中为 True 的位置用指定值填充。
        # ~mask 是对布尔掩码取反。所以，这里是将原始掩码为 False 的位置填充为负无穷。
        # 这样在 softmax 后，这些位置的注意力权重会变为 0。
        attention_score = attention_score.masked_fill(~mask, float('-inf'))

    # 对注意力分数在最后一个维度（keys维度）上应用 softmax，得到注意力权重。
    # 然后将权重与 V 矩阵相乘（@ 是矩阵乘法），得到最终的输出。
    output = softmax(attention_score, -1) @ V
    return output

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_rope: bool = False, max_seq_len: int | None = None,
                 theta: float | None = None):
        """
        参数:
            d_model (int): 前馈网络输入和输出的维度。
            num_heads (int): 多头注意力中使用的头数。
            use_rope (bool): 是否使用 RoPE
            max_seq_len (int): 如果你的实现预先缓存了，最大序列长度。
            theta (float): RoPE 参数。
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len) if use_rope else None
        self.q_proj = Linear(d_model, d_model)  # 用于生成 Query 的线性投影层
        self.k_proj = Linear(d_model, d_model)  # 用于生成 Key 的线性投影层
        self.v_proj = Linear(d_model, d_model)  # 用于生成 Value 的线性投影层
        self.o_proj = Linear(d_model, d_model)  # 用于最终输出的线性投影层


    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor | None = None):
        """
        参数:
            in_features (Float[Tensor, "... sequence_length d_in"]): 输入张量。
            token_positions (Int[Tensor, " ... sequence_length"] | None): 可选的张量，表示令牌的位置

        返回:
            Float[Tensor, " ... sequence_length d_out"]: 经过优化的、批处理的多头注意力实现的输出张量。
        """
        seq_len = in_features.shape[-2]  # 获取序列长度
        # torch.cat: 将多个张量沿指定维度拼接。这里将 Q, K, V 的投影权重拼接在一起。
        # 这是一个优化技巧，可以用一次大的矩阵乘法计算出 Q, K, V。
        qkv_proj = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
        # 执行一次矩阵乘法得到拼接后的 QKV
        qkv = in_features @ qkv_proj.T
        # torch.chunk: 将张量沿指定维度分割成多个块。这里将拼接的 QKV 分割回独立的 Q, K, V。
        q, k, v = qkv.chunk(3, -1)

        # rearrange: einops 库中的函数，用于以一种可读的方式重排张量维度。
        # "... seq_len (h d_head) -> ... h seq_len d_head" 的意思是：
        # 将最后一个维度 (h * d_head) 分割成 h (头数) 和 d_head (每个头的维度) 两个维度。
        q = rearrange(
            q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        k = rearrange(
            k, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        v = rearrange(
            v, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )

        if self.use_rope:  # 如果使用旋转位置编码
            q = self.rope(q, token_positions)  # 对 Q 应用 RoPE
            k = self.rope(k, token_positions)  # 对 K 应用 RoPE

        # torch.triu: 获取矩阵的上三角部分。diagonal=1 表示不包括主对角线。
        # .bool() 转换为布尔类型。这创建了一个未来位置的掩码（值为 True）。
        # --- 核心修改 ---
        # 在创建时就将 casual_mask 放到正确的设备上
        casual_mask = torch.triu(torch.ones(seq_len, seq_len, device=in_features.device), diagonal=1).bool()
        # 扩展掩码维度以匹配注意力分数的形状 (batch, heads, seq_len, seq_len)
        casual_mask = casual_mask[None, None, :, :]
        # ~casual_mask 取反，使得当前和过去的位置为 True，未来的位置为 False，符合 SDPA 函数的掩码要求。
        output = scaled_dot_product_attention(q, k, v, ~casual_mask)
        # 再次使用 rearrange 将多头维度的输出合并回 d_model 维度。
        output = rearrange(
            output, "... h seq_len d_head ->  ... seq_len (h d_head)"
        )
        # 通过最后的线性层进行输出投影
        return self.o_proj(output)
