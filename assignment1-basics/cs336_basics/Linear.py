import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，并使用别名 nn
import numpy as np  # 导入 NumPy 库，并使用别名 np

class Linear(nn.Module):  # 定义一个名为 Linear 的类，它继承自 torch.nn.Module，这是所有神经网络模块的基类
    def __init__(self, in_features: int, out_features: int,
        device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        参数:
            in_features: int 输入张量的最后一个维度的大小
            out_features: int 输出张量的最后一个维度的大小
            device: torch.device | None = None 参数所在的设备 (例如, 'cpu' 或 'cuda')
            dtype: torch.dtype | None = None 参数的数据类型 (例如, torch.float32)
        """
        super().__init__()  # 调用父类 nn.Module 的构造函数，这是必须的步骤
        factory_kwargs = {"device": device, "dtype": dtype}  # 创建一个字典，用于后续创建张量时传递设备和数据类型参数

        self.in_features = in_features  # 将输入特征数保存为实例属性
        self.out_features = out_features  # 将输出特征数保存为实例属性
        self.device = device  # 将设备信息保存为实例属性
        self.dtype = dtype  # 将数据类型信息保存为实例属性

        # 调用 initialize_Weights 方法来初始化权重矩阵
        w_init = self.initialize_Weights(out_features, in_features, factory_kwargs)
        # 将初始化的权重封装成 nn.Parameter，这样 PyTorch 才能追踪它的梯度
        self.weight = nn.Parameter(w_init)

    def initialize_Weights(self, out_dim: int, in_dim: int, factory_kwargs: dict) -> torch.Tensor:
        """
        使用截断正态分布方法初始化权重 W
        """
        # 创建一个形状为 (out_dim, in_dim) 的空张量，设备和数据类型由 factory_kwargs 指定
        W = torch.empty(out_dim, in_dim, **factory_kwargs)
        # 设置正态分布的均值
        mean = 0
        # 计算标准差，这是一种常见的权重初始化策略（Xavier/Glorot 初始化的一种变体）
        std = np.sqrt(2 / (in_dim + out_dim))

        # 使用截断正态分布就地（in-place）填充张量 W
        # 值的范围被限制在 [mean - 3*std, mean + 3*std] 内
        nn.init.trunc_normal_(W, mean, std, -3*std, 3*std)
        # 返回初始化后的权重张量
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 定义模型的前向传播逻辑
        # x 是输入张量，形状通常是 (..., in_features)
        # self.weight.T 是权重矩阵的转置，形状是 (in_features, out_features)
        # @ 符号是矩阵乘法操作
        output = x @ self.weight.T
        # 返回输出张量，形状是 (..., out_features)
        return output
