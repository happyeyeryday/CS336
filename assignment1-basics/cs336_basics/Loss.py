import torch  # 导入 PyTorch 库
import numpy as np  # 导入 NumPy 库

def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    给定一个输入的 logits 张量和一个目标的类别索引张量，计算平均交叉熵损失。

    参数:
        inputs (Float[Tensor, "batch_size vocab_size"]): 输入张量。inputs[i][j] 是第 i 个样本属于第 j 个类别的未归一化 logit 值。
        targets (Int[Tensor, "batch_size"]): 目标张量，形状为 (batch_size,)，包含正确类别的索引。
            每个值必须在 0 和 `num_classes - 1` 之间。

    返回:
        Float[Tensor, ""]: 所有样本的平均交叉熵损失，一个标量张量。
    """
    # 提取与目标类别相对应的 logits
    # targets.unsqueeze(-1): 将 targets 从 (batch_size,) 变形为 (batch_size, 1)，以匹配 gather 函数的索引要求。
    # inputs.gather(dim=-1, index=...): 沿着最后一个维度（dim=-1，即 vocab_size 维度），
    # 根据 targets 中提供的索引，从 inputs 中收集元素。
    # 结果 target_logits 的形状是 (batch_size, 1)，包含了每个样本正确类别的 logit 值。
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1))

    # 使用 log-sum-exp 技巧来保证数值稳定性。
    # torch.logsumexp(inputs, -1, keepdim=True): 这是一个数值稳定的计算 log(sum(exp(inputs))) 的方法。
    # 它在内部通过减去最大值来避免浮点数上溢。
    # dim=-1 表示沿着最后一个维度（类别维度）进行操作。
    # keepdim=True 保持了被缩减的维度，所以输出形状为 (batch_size, 1)。
    # 这个结果对应于交叉熵损失公式中 log 分母项。
    logsumexp = torch.logsumexp(inputs, -1, keepdim=True)

    # 计算每个样本的损失。交叉熵损失公式为 -log(softmax(x))_i = -x_i + log(sum(exp(x_j)))
    # 这里的 -target_logits 对应 -x_i，logsumexp 对应 log(sum(exp(x_j)))。
    # 这样计算可以避免直接计算 softmax 时可能出现的数值问题。
    loss_matrix = -target_logits + logsumexp

    # 计算所有样本的平均损失
    # torch.mean(loss_matrix): 计算 loss_matrix 张量中所有元素的平均值，得到一个标量。
    loss = torch.mean(loss_matrix)
    return loss  # 返回最终的平均损失值
