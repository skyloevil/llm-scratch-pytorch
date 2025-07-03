import torch
import torch.nn as nn
import torch.nn.functional as F

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

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    """
    Precompute the theta and position frequencies for RoPE.
    
    Args:
        head_dim (int): The dimension of each attention head.
        seq_len (int): The length of the sequence.
        device (str): The device to store the tensors on.
        theta (float): The base value for the theta calculation.

    Returns:
        torch.Tensor: The precomputed cosine frequencies.
        torch.Tensor: The precomputed sine frequencies.

    ---
    为 RoPE 预计算 theta 和位置频率。

    参数:
        head_dim (int): 每个注意力头的维度。
        seq_len (int): 序列的长度。
        device (str): 存储张量的设备。
        theta (float): 用于计算 theta 的基值。

    返回:
        torch.Tensor: 预计算的余弦频率。
        torch.Tensor: 预计算的正弦频率。
    """
    # As per the paper, the dimension of the rotary features must be even.
    # 根据论文，旋转特征的维度必须是偶数。
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    # Build the theta parameter
    # theta_i = 10000^(-2(i-1)/d) for i = 1, 2, ..., d/2
    # 构建 theta 参数
    # theta_i = 10000^(-2(i-1)/d) for i = 1, 2, ..., d/2
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (head_dim / 2)

    # Construct the positions (the 'm' parameter)
    # 构建位置（'m' 参数）
    m = torch.arange(seq_len, device=device) # (seq_len)

    # Multiply each theta by each position
    # (seq_len) outer product (head_dim / 2) -> (seq_len, head_dim / 2)
    # 将每个 theta 与每个位置相乘
    freqs = torch.outer(m, theta).float()

    # We can compute complex numbers in the polar form c = R * exp(i * m * theta), where R = 1.
    # The complex numbers are then computed as c = cos(m * theta) + i * sin(m * theta)
    # 我们可以用极坐标形式计算复数 c = R * exp(i * m * theta)，其中 R = 1。
    # 然后复数计算为 c = cos(m * theta) + i * sin(m * theta)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs) # (seq_len, head_dim / 2)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    """
    Apply rotary positional embeddings to the input tensor.
    
    Args:
        x (torch.Tensor): The input tensor (q or k) of shape (B, Seq_Len, H, Head_Dim).
        freqs_complex (torch.Tensor): The precomputed complex frequencies.
        device (str): The device to store the tensors on.

    Returns:
        torch.Tensor: The tensor with rotary embeddings applied.

    ---
    将旋转位置嵌入应用于输入张量。

    参数:
        x (torch.Tensor): 输入张量 (q 或 k)，形状为 (B, Seq_Len, H, Head_Dim)。
        freqs_complex (torch.Tensor): 预计算的复数频率。
        device (str): 存储张量的设备。

    返回:
        torch.Tensor: 应用了旋转嵌入的张量。
    """
    # Reshape the x tensor to match the shape of the complex numbers
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2, 2)
    # 重塑 x 张量以匹配复数的形状
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor
    # (Seq_Len, Head_Dim/2) -> (1, Seq_Len, 1, Head_Dim/2)
    # 重塑 freqs_complex 张量以匹配 x_complex 张量的形状
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # Multiply the x_complex tensor by the freqs_complex tensor
    # This results in the rotation of the complex number
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    # 将 x_complex 张量与 freqs_complex 张量相乘
    # 这会导致复数的旋转
    x_rotated = x_complex * freqs_complex

    # Convert the complex number back to a real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    # 将复数转换回实数
    x_out = torch.view_as_real(x_rotated)

    # Reshape the output tensor to match the original shape
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    # 重塑输出张量以匹配原始形状
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        """
        Initialize the FeedForward module.
        
        Args:
            dim (int): The input dimension.
            hidden_dim (int): The hidden dimension of the FeedForward layer.

        ---
        初始化 FeedForward 模块。

        参数:
            dim (int): 输入维度。
            hidden_dim (int): FeedForward 层的隐藏维度。
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the FeedForward module.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor after passing through the FeedForward layer.

        ---
        FeedForward 模块的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 通过 FeedForward 层后的输出张量。
        """
        # SwiGLU activation function
        # swish(x) = x * sigmoid(x)
        # (B, Seq_Len, Dim) -> (B, Seq_Len, Hidden_Dim)
        # SwiGLU 激活函数
        swish = F.silu(self.w1(x))
        
        # (B, Seq_Len, Dim) -> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        
        # Element-wise multiplication
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) = (B, Seq_Len, Hidden_Dim)
        # 逐元素相乘
        x = swish * x_V
        
        # (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Dim)
        x = self.w2(x)
        
        return x