import torch
from torch.optim import Optimizer
import torch.distributed as dist
from typing import Type, Any

class ShardedOptimizer(Optimizer):
    # ShardedOptimizer 继承自 Optimizer，用于实现优化器状态的分片存储（类似 ZeRO-1/2 的思想）
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        # 将输入的参数生成列表保存，这是所有的模型参数
        self.all_params = list(params)

        # Create a mapping from parameter ID to its global index
        # 创建一个从参数对象 ID 到其在参数列表中全局索引的映射，用于后续确定参数归属
        self.param_to_global_idx = {id(p): i for i, p in enumerate(self.all_params)}

        # Get distributed information
        # 获取分布式环境信息（rank 和 world_size）
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            # 如果未初始化分布式环境，默认为单进程模式
            self.rank = 0
            self.world_size = 1

        # Each rank is responsible for a shard of parameters
        # 每个 rank 只负责参数列表中的一部分（分片）
        # 使用切片操作，步长为 world_size，起始位置为 rank，实现简单的轮询分配
        self.param_shard = self.all_params[self.rank::self.world_size]

        # Local optimizer only manages the shard of parameters
        # 初始化本地优化器，它只管理当前 rank 负责的那部分参数分片
        # optimizer_cls 是传入的优化器类（如 SGD, Adam 等），kwargs 是优化器参数（如 lr, momentum）
        self.local_optimizer = optimizer_cls(self.param_shard, **kwargs)

        # Initialize the parent class with the local optimizer's param_groups
        # 使用本地优化器的参数组和默认值初始化父类 Optimizer
        super().__init__(self.local_optimizer.param_groups, self.local_optimizer.defaults)

        # [新增] 预先按 Owner Rank 分组参数，避免在 step() 中重复计算
        # self.params_by_owner[i] 存储所有归属于 rank i 的参数
        self.params_by_owner = [[] for _ in range(self.world_size)]
        for param in self.all_params:
            idx = self.param_to_global_idx[id(param)]
            owner = idx % self.world_size
            self.params_by_owner[owner].append(param)

    @torch.no_grad()
    def step(self, closure=None):
        # 1. Local optimizer performs a step, updating only the local shard of parameters
        # 1. 本地优化器执行一步更新，仅更新当前 rank 负责的那部分参数
        # 因为每个 rank 只持有部分参数的优化器状态，所以内存占用减少了
        loss = self.local_optimizer.step(closure)

        if self.world_size > 1:
            # 2. Synchronize updated parameters across all processes
            # [修改] 批量广播优化：按 Owner 分组 -> 展平 -> 广播 -> 解包
            # 原来的写法是逐个参数广播，会导致巨大的 Kernel Launch 开销

            for owner_rank in range(self.world_size):
                # 获取属于该 owner 的所有参数
                params_to_sync = self.params_by_owner[owner_rank]

                if not params_to_sync:
                    continue

                # A. 将参数展平为一个连续的大张量 (Flatten)
                # 这将通信次数从 N_params 降低到 N_ranks
                flat_params = torch._utils._flatten_dense_tensors(params_to_sync)

                # B. 广播这个大张量
                dist.broadcast(flat_params, src=owner_rank)

                # C. 非 Owner 节点接收数据后，需要解包并更新本地参数
                if self.rank != owner_rank:
                    # _unflatten_dense_tensors 返回的是新张量列表
                    synced_params = torch._utils._unflatten_dense_tensors(flat_params, params_to_sync)

                    # 将同步后的数据复制回原参数
                    for p, synced_p in zip(params_to_sync, synced_params):
                        p.copy_(synced_p)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        # Every rank computes gradients for the whole model, not just its shard.
        for param in self.all_params:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    # Bucketed DDP may install a view into a flat gradient buffer.
                    param.grad = param.grad.detach()
                    param.grad.zero_()
