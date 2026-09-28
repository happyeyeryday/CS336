import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import contextmanager # [新增] 导入上下文管理器工具

class DDPBase(nn.Module):
    """Common utilities shared by DDP wrappers."""
    # DDP 包装器的通用工具基类
    def __init__(self, module: nn.Module):
        super().__init__()
        # 保存被包装的模型
        self.module = module
        # 默认值；如果分布式环境已初始化，将被覆盖
        self.rank = 0
        self.world_size = 1
        # 初始化分布式环境并广播参数
        self._init_dist_and_broadcast()

    def __getattr__(self, name):
        # 将缺失的属性委托给被包装的模块，以便方便调用
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def _init_dist_and_broadcast(self):
        """Initialize rank/world_size and broadcast parameters from rank 0."""
        # 初始化 rank/world_size 并从 rank 0 广播参数
        if dist.is_initialized():
            # 获取当前进程的 rank
            self.rank = dist.get_rank()
            # 获取总进程数 (world size)
            self.world_size = dist.get_world_size()
            # 在不计算梯度的上下文中执行
            with torch.no_grad():
                # 遍历模型的所有参数
                for p in self.module.parameters():
                    # 将参数数据从 rank 0 广播到所有其他进程，确保初始权重一致
                    dist.broadcast(p.data, src=0)

class DDPModel(DDPBase):
    def __init__(self, module: nn.Module):
        super().__init__(module)
        # 用于存储异步通信句柄的列表
        self.handles = []
        # 为模型的每个参数注册 hook
        for p in self.module.parameters():
            if p.requires_grad:
                # 注册在梯度累积后触发的 hook
                p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, p):
        # 如果参数没有梯度或者不是分布式环境，直接返回
        if p.grad is None or self.world_size <= 1:
            return

        # 对梯度进行异步的 All-Reduce 操作（求和）
        handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
        # 保存句柄以便后续等待完成
        self.handles.append(handle)

    def forward(self, *inputs, **kwargs):
        # 前向传播直接调用被包装的模块
        return self.module(*inputs, **kwargs)


    def finish_gradient_synchronization(self):
        # 如果不是分布式环境，无需同步
        if self.world_size <= 1:
            return

        # 等待所有异步 All-Reduce 操作完成
        for handle in self.handles:
            handle.wait()

        # 对所有参数的梯度进行平均（除以 world_size）
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad /= self.world_size

        # 清空句柄列表，为下一次迭代做准备
        self.handles.clear()



class DDPBucketed(DDPBase):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__(module)

        # [新增] 控制是否进行梯度同步的标志，默认为 True
        # 当使用梯度累积时，在前 N-1 步将其设为 False
        self.sync_gradients = True

        # 存储桶的列表，每个桶包含一组参数
        self.buckets = []
        # 参数 ID 到桶索引的映射
        self.param_to_bucket = {}


        # 用于记录已处理参数的集合
        visited = set()

        # 反向遍历参数（因为反向传播是从后往前的，这样可以尽早触发通信）
        params_in_reverse = list(self.module.parameters())[::-1]

        # 当前正在构建的桶
        curr_bucket = []
        # 当前桶的大小（字节）
        curr_size = 0
        # 将桶大小限制从 MB 转换为字节
        bucket_size_mb = bucket_size_mb * 1024 * 1024  # Convert to bytes


        # 遍历反向参数列表进行分桶
        for p in params_in_reverse:
            # 只处理需要梯度的参数且未被访问过的
            if p.requires_grad and id(p) not in visited:
                # 计算参数占用的字节数
                param_size = p.numel() * p.element_size()

                # 如果当前桶加上新参数超过了限制，且当前桶不为空，则封桶
                if curr_size + param_size > bucket_size_mb and curr_bucket:
                    self.buckets.append(curr_bucket)
                    curr_bucket = []
                    curr_size = 0

                # 将参数加入当前桶
                curr_bucket.append(p)
                curr_size += param_size
                # 标记参数为已访问
                visited.add(id(p))


        # 将最后一个桶（如果有）加入列表
        if curr_bucket:
            self.buckets.append(curr_bucket)


        # 建立参数 ID 到桶索引的映射
        for i, bucket in enumerate(self.buckets):
            for param in bucket:
                self.param_to_bucket[id(param)] = i

        # 用于记录每个桶中已有多少个参数完成了梯度计算
        self.bucket_grad_ready_counts = [0] * len(self.buckets)
        # 存储通信句柄
        self.handles = []

        # 为所有需要梯度的参数注册 hook
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    # [新增] no_sync 上下文管理器
    @contextmanager
    def no_sync(self):
        """
        上下文管理器：在此上下文中禁用梯度同步。
        用于梯度累积的前 N-1 步，避免不必要的通信。
        用法:
            with model.no_sync():
                loss.backward()
        """
        old_sync = self.sync_gradients
        self.sync_gradients = False
        try:
            yield
        finally:
            self.sync_gradients = old_sync

    def _grad_hook(self, param):
        # [修改] 增加 self.sync_gradients 的检查
        # 如果参数无梯度、非分布式环境、或者处于 no_sync 上下文中，直接返回
        # 在 no_sync 模式下，PyTorch 会自动累积梯度到 .grad，但我们不触发通信
        if param.grad is None or self.world_size <= 1 or not self.sync_gradients:
            return

        # 获取参数所属的桶索引
        bucket_index = self.param_to_bucket[id(param)]
        bucket = self.buckets[bucket_index]

        # 增加该桶已就绪梯度的计数
        self.bucket_grad_ready_counts[bucket_index] += 1

        # 如果该桶内所有参数的梯度都已就绪
        if self.bucket_grad_ready_counts[bucket_index] == len(bucket):
            # 收集桶内所有参数的梯度
            grads_to_reduce = [p.grad for p in bucket]
            # 将梯度展平为一个连续的张量，以提高通信效率
            flat_grads = torch._utils._flatten_dense_tensors(grads_to_reduce)

            # 对展平后的梯度进行异步 All-Reduce 求和
            handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
            # 保存句柄、展平的梯度张量和对应的桶，以便后续处理
            self.handles.append((handle, flat_grads, bucket))

    def forward(self, *args, **kwargs):
        # 前向传播直接调用被包装模块
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # 非分布式环境直接返回
        if self.world_size <= 1:
            return

        # 遍历所有通信句柄
        for handle, flat_grads, bucket in self.handles:
            # 等待通信完成
            handle.wait()
            # 计算平均梯度
            flat_grads /= self.world_size

            # 将展平的梯度还原回各个参数的形状
            unflattened_grads = torch._utils._unflatten_dense_tensors(flat_grads, bucket)
            # 将还原后的梯度赋值回参数的 .grad 属性
            for param, grad in zip(bucket, unflattened_grads):
                param.grad = grad

        # 清空句柄列表
        self.handles.clear()
        # 重置桶计数器
        self.bucket_grad_ready_counts = [0] * len(self.buckets)
