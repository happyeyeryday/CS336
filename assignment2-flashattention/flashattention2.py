
import torch
import math
import triton
import triton.language as tl
from math import ceil
from einops import rearrange, einsum
from typing import Tuple, Optional
from torch.autograd import Function
# 开启 TF32 加速 matmul
torch.set_float32_matmul_precision('high')

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_q, N_k, scale,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # 1. 设置身份
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    q_start_row = query_tile_index * Q_TILE_SIZE

    # 2. 创建块指针
    Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb, (N_q, D), (stride_qq, stride_qd), (q_start_row, 0), (Q_TILE_SIZE, D), (1, 0))
    KT_block_ptr = tl.make_block_ptr(K_ptr + batch_index * stride_kb, (D, N_k), (stride_kd, stride_kk), (0, 0), (D, K_TILE_SIZE), (0, 1))
    V_block_ptr = tl.make_block_ptr(V_ptr + batch_index * stride_vb, (N_k, D), (stride_vk, stride_vd), (0, 0), (K_TILE_SIZE, D), (1, 0))

    # 3. 初始化累加器
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)

    # 4. 加载 Q 块并预处理
    Q_i = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    Q_i_fp32 = (Q_i * scale).to(tl.float32)

    # 5. 核心循环
    T_k = tl.cdiv(N_k, K_TILE_SIZE)
    for j in range(T_k):
        k_start_col = j * K_TILE_SIZE

        K_j = tl.load(KT_block_ptr, boundary_check=(1,), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        S_ij = tl.dot(Q_i_fp32, K_j.to(tl.float32), input_precision="ieee")

        # 精确的边界处理
        q_rows_offsets = q_start_row + tl.arange(0, Q_TILE_SIZE)
        k_cols_offsets = k_start_col + tl.arange(0, K_TILE_SIZE)
        q_mask = q_rows_offsets[:, None] < N_q
        k_mask = k_cols_offsets[None, :] < N_k
        S_ij = tl.where(q_mask & k_mask, S_ij, -float('inf'))

        if is_causal:
            causal_mask = q_rows_offsets[:, None] >= k_cols_offsets[None, :]
            S_ij = tl.where(causal_mask, S_ij, -float('inf'))

        # 经过验证的、数值稳定的 Online Softmax
        m_i_new = tl.maximum(m_i, tl.max(S_ij, axis=1))
        P_ij = tl.exp(S_ij - m_i_new[:, None])
        exp_m_diff = tl.exp(m_i - m_i_new)

        l_i = l_i * exp_m_diff + tl.sum(P_ij, axis=1)

        o_i = o_i * exp_m_diff[:, None]

        # 核心对齐修改：执行 float32 @ float32 的 dot 运算
        o_i += tl.dot(P_ij, V_j.to(tl.float32), input_precision="ieee")

        m_i = m_i_new

        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        KT_block_ptr = KT_block_ptr.advance((0, K_TILE_SIZE))

    # 6. 最终归一化并写回
    # 允许在 padding 行上产生 0/0=nan，因为 store 时会忽略
    o_i = o_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    O_block_ptr = tl.make_block_ptr(O_ptr + batch_index * stride_ob, (N_q, D), (stride_oq, stride_od), (q_start_row, 0), (Q_TILE_SIZE, D), (1, 0))
    L_block_ptr = tl.make_block_ptr(L_ptr + batch_index * stride_lb, (N_q,), (stride_lq,), (q_start_row,), (Q_TILE_SIZE,), (0,))

    tl.store(O_block_ptr, o_i.to(Q_ptr.type.element_ty), boundary_check=(0,))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


@torch.compile(fullgraph=True)
def _flash_attention_backward_compiled(
    Q, K, V, O, L, grad_O, is_causal,
    batch_size, N_q, N_k, d_head,
    B_q, B_k
):
    """基于前向传播保存的L重建attention，确保数值一致性"""
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    scale = d_head ** -0.5

    for b in range(batch_size):
        Q_b = Q[b].float()
        K_b = K[b].float()
        V_b = V[b].float()
        dO_b = grad_O[b].float()
        L_b = L[b]  # 使用前向传播保存的L (float32)

        # 关键：重新计算scores但使用前向传播的L来构建attention
        scores = (Q_b @ K_b.T) * scale  # [seq_len, seq_len]

        if is_causal:
            seq_len = scores.size(0)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1)
            scores = scores.masked_fill(mask.bool(), -float('inf'))

        # 关键改进：使用前向传播的L来重建正确的attention权重
        # 而不是重新计算softmax
        attn = torch.exp(scores - L_b.unsqueeze(1))  # 使用保存的L值

        # 标准backward计算
        dV[b] = (attn.T @ dO_b).to(V.dtype)
        grad_attn = dO_b @ V_b.T
        S = (grad_attn * attn).sum(dim=-1, keepdim=True)
        grad_scores = (grad_attn - S) * attn

        dQ[b] = ((grad_scores * scale) @ K_b).to(Q.dtype)
        dK[b] = ((grad_scores * scale).T @ Q_b).to(K.dtype)

    return dQ, dK, dV

# -------------------------------
# Autograd Function for FlashAttention
# -------------------------------
class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        if not (Q.is_cuda and K.is_cuda and V.is_cuda):
            raise ValueError("FlashAttention2 requires CUDA tensors")
        if Q.shape != K.shape or Q.shape != V.shape or Q.ndim != 3:
            raise ValueError("This study implementation supports matching [heads, sequence, head_dim] tensors")
        if Q.shape[-1] < 16 or Q.shape[-1] & (Q.shape[-1] - 1):
            raise ValueError("head_dim must be a power of two and at least 16")
        if Q.dtype != K.dtype or Q.dtype != V.dtype or Q.device != K.device or Q.device != V.device:
            raise ValueError("Q/K/V must share dtype and device")
        # 输入验证 - 现在期望3D张量 [n_heads, seq_len, d_head]
        assert Q.dim() == 3, f"Q must be 3D [n_heads, seq_len, d_head], got {Q.dim()}D"
        assert K.dim() == 3, f"K must be 3D [n_heads, seq_len, d_head], got {K.dim()}D"
        assert V.dim() == 3, f"V must be 3D [n_heads, seq_len, d_head], got {V.dim()}D"

        n_heads, N_q, d_head = Q.shape
        _, N_k, _ = K.shape
        scale = d_head ** -0.5

        O = torch.empty_like(Q)
        L = torch.empty((n_heads, N_q), device=Q.device, dtype=torch.float32)

        Q_TILE_SIZE, K_TILE_SIZE = 64, 64
        T_q = math.ceil(N_q / Q_TILE_SIZE)
        grid = (T_q, n_heads)  # 改为 n_heads

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k, scale,
            D=d_head, Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_O):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        n_heads, N_q, d_head = Q.shape  # 改为 n_heads
        _, N_k, _ = K.shape

        # B_q, B_k = 64, 64
        B_q, B_k = 64,64
        dQ, dK, dV = _flash_attention_backward_compiled(
            Q, K, V, O, L, grad_O, is_causal,
            n_heads, N_q, N_k, d_head,  # 改为 n_heads
            B_q, B_k
        )

        return dQ, dK, dV, None

#11.19
