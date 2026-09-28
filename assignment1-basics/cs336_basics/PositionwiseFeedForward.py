import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，并使用别名 nn
from .Linear import Linear  # 从同级目录导入自定义的 Linear 模块


def silu(x: torch.Tensor) -> torch.Tensor:
        # 定义 SiLU (Sigmoid-weighted Linear Unit) 激活函数，也称为 Swish。
        # torch.sigmoid: 对输入张量 x 的每个元素应用 Sigmoid 函数，公式为 1 / (1 + exp(-x))。
        # Sigmoid 函数将任意实数映射到 (0, 1) 区间。
        # SiLU 的计算方式是输入 x 与其 sigmoid 值的逐元素乘积。
        return x * torch.sigmoid(x)

class PositionwiseFeedForward(nn.Module):  # 定义逐位置前馈网络模块
    def __init__(self, d_model: int, d_ff: int):
        """
        参数:
            d_model (int): 前馈网络输入和输出的维度
            d_ff (int): 在 SwiGLU 内部发生的上投影（up-project）的维度
        """
        super().__init__()  # 调用父类 nn.Module 的构造函数
        self.d_model = d_model  # 将输入/输出维度保存为实例属性
        self.d_ff = d_ff  # 将中间层维度保存为实例属性

        # SwiGLU 结构需要三个线性层
        # w1 和 w3 用于将输入从 d_model 扩展到 d_ff
        # w2 用于将结果从 d_ff 压缩回 d_model
        self.w1 = Linear(d_model, d_ff)  # 第一个线性变换层
        self.w2 = Linear(d_ff, d_model)  # 第二个（输出）线性变换层
        self.w3 = Linear(d_model, d_ff)  # 第三个线性变换层，用作门（gate）


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        实现 SwiGLU 前馈网络。
        论文参考: "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
        公式为: SwiGLU(x) = (xW1 * sigmoid(xW1)) * W2  (论文中的公式)
        这里的实现是: SwiGLU(x, W, V, W2) = (silu(xW) * xV) * W2
        其中 W -> w1, V -> w3, W2 -> w2
        参数:
            x (Float[Tensor, "... d_model"]): 输入到前馈层的嵌入向量

        返回:
            Float[Tensor, "... d_model"]: 与输入嵌入形状相同的输出嵌入向量
        """
        # SwiGLU 的计算过程:
        # 1. self.w1(x): 对输入 x 进行第一次线性变换。
        # 2. silu(...): 对结果应用 SiLU 激活函数。
        # 3. self.w3(x): 对输入 x 进行第二次线性变换，作为门控。
        # 4. ... * ...: 将激活后的结果与门控结果进行逐元素相乘。
        # 5. self.w2(...): 将相乘后的结果通过输出线性层，映射回 d_model 维度。
        output = self.w2(silu(self.w1(x)) * self.w3(x))
        return output  # 返回最终结果
