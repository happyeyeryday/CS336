import os
import math
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
from datetime import timedelta

# Default output label; override through config["result_prefix"].
your_name = "study"

# ==========================================
# 1. Manual Math Operations
# ==========================================
@torch.jit.script
def manual_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    # weight: (out, in)
    out = torch.matmul(x, weight.t())
    if bias is not None:
        out = out + bias
    return out

@torch.jit.script
def manual_erf(x: torch.Tensor) -> torch.Tensor:
    """
    手动实现高精度 erf (Abramowitz and Stegun 7.1.26)
    最大误差: 1.5e-7，足以满足 fp32 精度要求
    只使用了: abs, exp, sign 和基础加减乘除
    """
    # 1. 保存符号，因为 erf 是奇函数: erf(-x) = -erf(x)
    sign = torch.sign(x)
    x_abs = torch.abs(x)

    # 2. 常数 (A&S 7.1.26)
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # 3. 计算多项式部分
    t = 1.0 / (1.0 + p * x_abs)
    # Horner's method 计算多项式以提高数值稳定性
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * torch.exp(-x_abs * x_abs)

    return sign * y

@torch.jit.script
def manual_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + manual_erf(x * 0.7071067811865475))
    # return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

@torch.jit.script
def manual_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 1. 数值稳定性：减去最大值，防止 exp 溢出
    # keepdim=True 是为了后续广播
    max_val = x.amax(dim=dim, keepdim=True)
    x_safe = x - max_val

    # 2. 计算 exp
    exp_x = torch.exp(x_safe)

    # 3. 计算 sum
    sum_exp = exp_x.sum(dim=dim, keepdim=True)

    # 4. 归一化
    return exp_x / sum_exp

# @torch.jit.script
# def manual_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
#     mean = x.mean(dim=-1, keepdim=True)
#     var = x.var(dim=-1, keepdim=True, unbiased=False)
#     out = (x - mean) / torch.sqrt(var + eps)
#     if weight is not None:
#         out = out * weight
#     if bias is not None:
#         out = out + bias
#     return out

@torch.jit.script
def manual_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # 获取最后一个维度的元素个数 (D)
    D = x.shape[-1]

    # 1. 手动计算均值: sum(x) / D
    # 使用 sum 是绝对安全的，它是最基础算子
    mean = x.sum(dim=-1, keepdim=True) / D

    # 2. 手动计算方差: sum((x - mean)^2) / D
    # 注意：LayerNorm 通常使用有偏估计 (unbiased=False)，即除以 D 而不是 D-1
    diff = x - mean
    var = (diff * diff).sum(dim=-1, keepdim=True) / D

    # 3. 归一化
    out = diff / torch.sqrt(var + eps)

    # 4. 仿射变换 (Affine Transform)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias

    return out

# ==========================================
# 2. TP Communication Primitives (关键修复)
# ==========================================

class TP_Copy(torch.autograd.Function):
    """
    Forward: Pass through (Identity) - 复制输入到各卡
    Backward: All-Reduce (Sum) - 梯度求和
    用于 Column Parallel 的输入处
    """
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.contiguous().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad

class TP_Reduce(torch.autograd.Function):
    """
    Forward: All-Reduce (Sum) - 汇总各卡输出
    Backward: Pass through (Identity) - 梯度分发
    用于 Row Parallel 的输出处
    """
    @staticmethod
    def forward(ctx, input):
        dist.all_reduce(input, op=dist.ReduceOp.SUM)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# ==========================================
# 3. Helper Functions
# ==========================================

def setup(rank, world_size, port, device="cuda"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if device == "cuda":
        if torch.cuda.device_count() < world_size:
            raise ValueError(f"Need {world_size} visible CUDA devices")
        torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if device == "cuda" else "gloo",
                                rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=120))


# 新增一个根据 UID 生成端口的辅助函数
def get_user_base_port():
    try:
        uid = os.getuid()
    except AttributeError:
        uid = 0 # Windows fallback
    # 端口范围 20000 ~ 60000，根据 UID 偏移，避免冲突
    return 20000 + (uid % 2000) * 10

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(rank, device="cuda"):
    return f"cuda:{rank}" if device == "cuda" else "cpu"


def save_results(config, mode, results):
    directory = Path(config.get("output_dir", "outputs"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{config.get('result_prefix', your_name)}_{mode}.npz"
    np.savez(path, **results)
    print(f"[{mode.upper()}] Saved {path}")


def load_weight(name, input_dict, shape, device, dtype=torch.float32):
    if input_dict and name in input_dict:
        w = input_dict[name].to(device=device, dtype=dtype).detach().clone()
    else:
        # 简单初始化
        w = torch.randn(shape, device=device, dtype=dtype) / math.sqrt(shape[-1])
    w.requires_grad_(True)
    return w

def get_sharded_param(name, input_dict, full_shape, rank, world_size, split_dim, device):
    full_param = load_weight(name, input_dict, full_shape, device)
    if split_dim is None:
        return full_param
    with torch.no_grad():
        chunks = full_param.chunk(world_size, dim=split_dim)
        local_param = chunks[rank].clone().detach()
    local_param.requires_grad_(True)
    return local_param

# ==========================================
# 4. Tensor Parallelism
# ==========================================

def tensor_parallelism_main(rank: int, world_size: int, input: dict, config: dict):
    # 修改端口：使用动态端口替代硬编码的 29501
    base_port = get_user_base_port()
    device = get_device(rank, config.get("device", "cuda"))

    # 统一种子，保证 LayerNorm 权重一致
    torch.manual_seed(2025)

    if input:
        B, T, D = input['x'].shape
        n_heads = config.get("n_heads", 16)
        d_ff = config.get("d_ff", 4096)
    else:
        B = config.get('B', 8)
        T = config.get('T', 2048)
        D = config.get('d_model', 4096)
        n_heads = config.get('n_heads', 32)
        d_ff = config.get('d_ff', D * 4)

    if n_heads < 1 or D % n_heads or n_heads % world_size or d_ff % world_size:
        raise ValueError("TP requires d_model % n_heads == 0 and heads/d_ff divisible by world_size")
    setup(rank, world_size, config.get("port", base_port + 1), config.get("device", "cuda"))
    head_dim = D // n_heads
    local_n_heads = n_heads // world_size

    # --- Load Weights ---
    if input:
        x_global = input['x'].to(device=device, dtype=torch.float32).detach().clone()
    else:
        x_global = torch.randn(B, T, D, device=device, dtype=torch.float32)
    x_global.requires_grad_(True)

    params = {}
    params['ln1.weight'] = load_weight('ln1.weight', input, (D,), device)
    params['ln1.bias']   = load_weight('ln1.bias', input, (D,), device)
    params['ln2.weight'] = load_weight('ln2.weight', input, (D,), device)
    params['ln2.bias']   = load_weight('ln2.bias', input, (D,), device)

    params['attn.q.w'] = get_sharded_param('attn.q_proj.weight', input, (D, D), rank, world_size, 0, device)
    params['attn.k.w'] = get_sharded_param('attn.k_proj.weight', input, (D, D), rank, world_size, 0, device)
    params['attn.v.w'] = get_sharded_param('attn.v_proj.weight', input, (D, D), rank, world_size, 0, device)
    params['attn.o.w'] = get_sharded_param('attn.out_proj.weight', input, (D, D), rank, world_size, 1, device)

    params['ffn.1.w'] = get_sharded_param('ffn.fc1.weight', input, (d_ff, D), rank, world_size, 0, device)
    params['ffn.2.w'] = get_sharded_param('ffn.fc2.weight', input, (D, d_ff), rank, world_size, 1, device)

    # --- Forward ---
    h = manual_layer_norm(x_global, params['ln1.weight'], params['ln1.bias'])

    # [TP] Column Parallel 输入前，必须建立 Backward 的 AllReduce 机制
    h_tp = TP_Copy.apply(h)

    q = manual_linear(h_tp, params['attn.q.w'])
    k = manual_linear(h_tp, params['attn.k.w'])
    v = manual_linear(h_tp, params['attn.v.w'])

    def split_heads(t):
        return t.view(B, T, local_n_heads, head_dim).transpose(1, 2)

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores - scores.amax(dim=-1, keepdim=True)
    # attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = manual_softmax(scores, dim=-1)
    y_heads = torch.matmul(attn_weights, v)

    y_local = y_heads.transpose(1, 2).contiguous().view(B, T, -1)
    h_attn_out = manual_linear(y_local, params['attn.o.w'])

    # [TP] Row Parallel 输出后，必须使用 Forward 的 AllReduce 机制
    h_attn_out = TP_Reduce.apply(h_attn_out)

    x = x_global + h_attn_out
    h2 = manual_layer_norm(x, params['ln2.weight'], params['ln2.bias'])

    # [TP] Column Parallel 输入前
    h2_tp = TP_Copy.apply(h2)

    ffn_mid = manual_linear(h2_tp, params['ffn.1.w'])
    ffn_mid = manual_gelu(ffn_mid)
    ffn_out = manual_linear(ffn_mid, params['ffn.2.w'])

    # [TP] Row Parallel 输出后
    ffn_out = TP_Reduce.apply(ffn_out)

    final_out = x + ffn_out

    # --- Backward ---
    loss = final_out.sum()
    loss.backward()

    # --- Save Results ---
    if input:
        # TP_Copy already sums the branch gradients. The residual path is also
        # replicated: reducing x.grad again would multiply it by world_size.
        results = {}
        if rank == 0:
            results['x'] = x_global.detach().cpu().numpy()
            if x_global.grad is not None:
                results['x_grad'] = x_global.grad.detach().cpu().numpy()
            results['out'] = final_out.detach().cpu().numpy()

        def gather_and_save(local_tensor, name, split_dim=None):
            val = local_tensor.detach()
            grad = local_tensor.grad.detach() if local_tensor.grad is not None else torch.zeros_like(val)

            if split_dim is None:
                # ================= 修复开始 =================
                # 错误代码：dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                # 原因：TP_Copy 已经处理了梯度聚合，本地梯度即为全局完整梯度。
                # 再次 Sum 会导致数值翻倍。直接取 Rank 0 的即可。

                if rank == 0:
                    results[name] = val.cpu().numpy()
                    results[f"{name}_grad"] = grad.cpu().numpy()
                # ================= 修复结束 =================
            else:
                # 切分参数的逻辑保持不变
                gathered_vals = [torch.zeros_like(val) for _ in range(world_size)]
                gathered_grads = [torch.zeros_like(grad) for _ in range(world_size)]
                dist.all_gather(gathered_vals, val)
                dist.all_gather(gathered_grads, grad)

                if rank == 0:
                    full_val = torch.cat(gathered_vals, dim=split_dim)
                    full_grad = torch.cat(gathered_grads, dim=split_dim)
                    results[name] = full_val.cpu().numpy()
                    results[f"{name}_grad"] = full_grad.cpu().numpy()

        gather_and_save(params['ln1.weight'], 'ln1.weight')
        gather_and_save(params['ln1.bias'], 'ln1.bias')
        gather_and_save(params['ln2.weight'], 'ln2.weight')
        gather_and_save(params['ln2.bias'], 'ln2.bias')
        gather_and_save(params['attn.q.w'], 'attn.q_proj.weight', 0)
        gather_and_save(params['attn.k.w'], 'attn.k_proj.weight', 0)
        gather_and_save(params['attn.v.w'], 'attn.v_proj.weight', 0)
        gather_and_save(params['attn.o.w'], 'attn.out_proj.weight', 1)
        gather_and_save(params['ffn.1.w'], 'ffn.fc1.weight', 0)
        gather_and_save(params['ffn.2.w'], 'ffn.fc2.weight', 1)

        if rank == 0:
            save_results(config, "tp", results)

    cleanup()

# ==========================================
# 5. Pipeline Parallelism (Semi-Async 1F1B)
# ==========================================

def pipeline_parallelism_main(rank: int, world_size: int, input: dict, config: dict, num_micro_batches: int = 4):
    # 修改端口：使用动态端口替代硬编码的 29502
    base_port = get_user_base_port()
    device = get_device(rank, config.get("device", "cuda"))
    torch.manual_seed(2025)

    if input:
        B, D = input['x'].shape
        num_layers = 8
    else:
        B = config.get('B', 16384)
        D = config.get('D', 16384)
        num_layers = 8

    if world_size != 2:
        raise ValueError("This pipeline implementation requires exactly two processes")
    if num_micro_batches < 1 or B % num_micro_batches:
        raise ValueError("batch size must be divisible by the positive micro-batch count")
    setup(rank, world_size, config.get("port", base_port + 2), config.get("device", "cuda"))
    layers_per_rank = num_layers // world_size
    my_layer_indices = list(range(rank * layers_per_rank, (rank + 1) * layers_per_rank))

    params = []
    for i in my_layer_indices:
        w_name = f"linears.{i}.weight"
        params.append(load_weight(w_name, input, (D, D), device))

    # Data Split
    if rank == 0:
        if input:
            data = input['x'].to(device=device, dtype=torch.float32).detach().clone()
        else:
            data = torch.randn(B, D, device=device, dtype=torch.float32)
        data.requires_grad_(True)
        micro_batches = data.chunk(num_micro_batches, dim=0)
    else:
        micro_batches = [None] * num_micro_batches

    rank1_outputs_list = []

    # Pre-allocate Buffers
    mb_size = B // num_micro_batches
    if rank == 1:
        recv_buf = torch.empty(mb_size, D, device=device, dtype=torch.float32)
    if rank == 0:
        grad_buf = torch.empty(mb_size, D, device=device, dtype=torch.float32)

    # 辅助 JIT 执行函数
    def run_layers(x_in):
        curr = x_in
        for w in params:
            curr = manual_linear(curr, w)
            curr = manual_gelu(curr)
        return curr

    # ================= 优化后的调度逻辑 =================

    if rank == 0:
        # [Rank 0] 策略:
        # 1. 预先计算第一个 batch 并 "发射" (isend)
        # 2. 在等待当前梯度时，并行计算下一个 batch (Overlap)

        # --- 启动阶段 ---
        curr_x = micro_batches[0]
        curr_act = run_layers(curr_x)
        send_req = dist.isend(curr_act.data, dst=1)

        # --- 流水线循环 ---
        for i in range(num_micro_batches - 1):
            # ✅ 1. 先接收梯度（优先级最高，避免阻塞 Rank 1）
            dist.recv(grad_buf, src=1)

            # ✅ 2. 确保发送完成
            send_req.wait()

            # ✅ 3. 反向传播（释放显存）
            curr_act.backward(grad_buf)

            # ✅ 4. 计算下一个（此时 Rank 1 已经在处理了）
            next_x = micro_batches[i+1]
            next_act = run_layers(next_x)

            # ✅ 5. 发送下一个
            send_req = dist.isend(next_act.data, dst=1)

            curr_act = next_act

        # --- 收尾阶段 ---
        dist.recv(grad_buf, src=1)
        send_req.wait()
        curr_act.backward(grad_buf)

    elif rank == 1:
        # [Rank 1] 策略:
        # 简单的 Recv -> Fwd -> Bwd -> Isend
        # 使用 Isend 发送梯度可以让 Rank 1 立即进入下一轮接收，减少等待

        send_reqs = []  # 保存所有请求
        for i in range(num_micro_batches):
            # 1. 阻塞接收
            dist.recv(recv_buf, src=0)

            # 2. 建立计算图
            curr_x = recv_buf.clone().detach()
            curr_x.requires_grad_(True)

            # 3. Forward
            curr_out = run_layers(curr_x)

            if input:
                rank1_outputs_list.append(curr_out.detach())

            # 4. Backward
            loss_mb = curr_out.sum() * 1e3
            loss_mb.backward()

            # 5. 发送梯度 (非阻塞)
            send_req = dist.isend(curr_x.grad, dst=0)
            send_reqs.append((send_req, curr_x.grad))

        # 6. 等待所有发送完成
        for req, grad_buffer in send_reqs:
            req.wait()

    # ===================================================

    # --- Save Results ---
    if input:
        if rank == 1:
            full_out = torch.cat(rank1_outputs_list, dim=0)
            dist.send(full_out, dst=0)
            for p in params:
                dist.send(p.detach(), dst=0)
                dist.send(p.grad.detach(), dst=0)
        elif rank == 0:
            results = {}
            results['x'] = data.detach().cpu().numpy()
            if data.grad is not None:
                results['x_grad'] = data.grad.detach().cpu().numpy()

            out_buf = torch.empty(B, D, device=device, dtype=torch.float32)
            dist.recv(out_buf, src=1)
            results['out'] = out_buf.cpu().numpy()

            for idx, p in enumerate(params):
                layer_id = my_layer_indices[idx]
                results[f"linears.{layer_id}.weight"] = p.detach().cpu().numpy()
                results[f"linears.{layer_id}.weight_grad"] = p.grad.detach().cpu().numpy()

            start_remote = 4
            for i in range(start_remote, 8):
                w_buf = torch.empty(D, D, device=device, dtype=torch.float32)
                g_buf = torch.empty(D, D, device=device, dtype=torch.float32)
                dist.recv(w_buf, src=1)
                dist.recv(g_buf, src=1)
                results[f"linears.{i}.weight"] = w_buf.cpu().numpy()
                results[f"linears.{i}.weight_grad"] = g_buf.cpu().numpy()

            save_results(config, "pp", results)

    cleanup()
