import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm module.
        
        Args:
            dim (int): The dimension of the input tensor.
            eps (float): A small value to prevent division by zero.

        ---
        初始化 RMSNorm 模块。

        参数:
            dim (int): 输入张量的维度。
            eps (float): 一个很小的值，用于防止除以零。
        """
        super().__init__()
        self.eps = eps
        # The gamma parameter
        # gamma 参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        """
        Apply the RMSNorm normalization.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The normalized tensor.
        
        ---
        应用 RMSNorm 归一化。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the RMSNorm module.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        ---
        RMSNorm 模块的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 应用 RMSNorm 后的输出张量。
        """
        # In the paper, the output is g_i * x_i, where g_i is the learnable parameter.
        # The formula is:
        # (x / sqrt(1/n * sum(x_i^2) + eps)) * g
        #
        # In this implementation, we multiply the learnable parameter `weight` with the normalized tensor.
        #
        # ---
        # 在论文中，输出是 g_i * x_i，其中 g_i 是可学习的参数。
        # 公式为:
        # (x / sqrt(1/n * sum(x_i^2) + eps)) * g
        #
        # 在这个实现中，我们将可学习的参数 `weight` 与归一化后的张量相乘。
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
