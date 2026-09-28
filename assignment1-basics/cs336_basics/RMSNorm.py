import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，并使用别名 nn

class RMSNorm(nn.Module):  # 定义一个名为 RMSNorm 的类，它继承自 nn.Module。RMSNorm 是一种层归一化的变体。
    def __init__(self, d_model: int, eps: float = 1e-5,
        device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        参数:
            d_model: int 模型的隐藏维度
            eps: float = 1e-5 用于数值稳定性的 epsilon 值
            device: torch.device | None = None 参数所在的设备
            dtype: torch.dtype | None = None 参数的数据类型
        """
        super().__init__()  # 调用父类 nn.Module 的构造函数
        factory_kwargs = {"device": device, "dtype": dtype}  # 创建一个字典，用于后续创建张量时传递设备和数据类型参数
        self.d_model = d_model  # 将模型的隐藏维度保存为实例属性
        self.eps = eps  # 将 epsilon 值保存为实例属性

        w_init = self.initialize_Weights(d_model, factory_kwargs)  # 调用权重初始化方法
        self.weight = nn.Parameter(w_init)  # 将权重封装成 nn.Parameter，使其成为可训练的参数

    def initialize_Weights(self, d_model: int, factory_kwargs: dict):
        """
        将权重 w 初始化为 1
        """
        # 创建一个长度为 d_model 的张量，所有元素都为 1
        w = torch.ones(d_model, **factory_kwargs)
        return w  # 返回初始化后的权重张量

    def RMS(self, x: torch.tensor, d_model: int, eps: float):
        """
        计算输入张量 x 的均方根 (Root Mean Square)
        """
        # 计算 x 中每个元素的平方，然后在最后一个维度上求均值
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        # 计算均方根，加上 eps 以防止除以零
        rms = torch.sqrt(ms + eps)
        return rms  # 返回计算得到的均方根值

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理一个形状为 (batch_size, sequence_length, d_model) 的输入张量，
        并返回一个相同形状的张量。
        """
        in_dtype = x.dtype  # 保存输入张量的原始数据类型
        x = x.to(torch.float32)  # 将输入张量转换为 float32 类型以进行高精度计算

        # 计算输入张量 x 的均方根
        rms = self.RMS(x, self.d_model, self.eps)
        # 将输入 x 除以其均方根进行归一化，然后乘以可学习的权重 self.weight
        result = (x / rms) * self.weight

        # 将结果转换回原始输入的数据类型并返回
        return result.to(in_dtype)
