import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # Later set in the build method
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

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

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        """
        Initialize the SelfAttention module.
        
        Args:
            args (ModelArgs): The arguments for the model.

        ---
        初始化 SelfAttention 模块。

        参数:
            args (ModelArgs): 模型的参数。
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass for the SelfAttention module.
        
        Args:
            x (torch.Tensor): The input tensor.
            freqs_complex (torch.Tensor): The precomputed complex frequencies for RoPE.
            mask (torch.Tensor): The attention mask.
        
        Returns:
            torch.Tensor: The output tensor after self-attention.

        ---
        SelfAttention 模块的前向传播。

        参数:
            x (torch.Tensor): 输入���量。
            freqs_complex (torch.Tensor): 用于 RoPE 的预计算复数频率。
            mask (torch.Tensor): 注意力掩码。

        返回:
            torch.Tensor: 经过自注意力处理后的输出张量。
        """
        batch_size, seq_len, _ = x.shape

        # (B, Seq_Len, Dim) -> (B, Seq_Len, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)
        xv = self.wv(x)

        # (B, Seq_Len, H_Q * Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        # (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings
        # 应用旋转位置嵌入
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Grouped Query Attention
        # 如果 H_KV < H_Q, 我们需要重复 K 和 V 张量来匹配 Q 的头的数量
        if self.n_kv_heads < self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            xk = xk.repeat_interleave(n_rep, dim=2)
            xv = xv.repeat_interleave(n_rep, dim=2)

        # (B, Seq_Len, H, Head_Dim) -> (B, H, Seq_Len, Head_Dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # (B, H, Seq_Len, Head_Dim) @ (B, H, Head_Dim, Seq_Len) -> (B, H, Seq_Len, Seq_Len)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores.masked_fill_(mask == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1).type_as(xq)

        # (B, H, Seq_Len, Seq_Len) @ (B, H, Seq_Len, Head_Dim) -> (B, H, Seq_Len, Head_Dim)
        output = torch.matmul(scores, xv)

        # (B, H, Seq_Len, Head_Dim) -> (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, Dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        return self.wo(output)

class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        """
        Initialize the EncoderLayer module.
        
        Args:
            args (ModelArgs): The arguments for the model.

        ---
        初始化 EncoderLayer 模块。

        参数:
            args (ModelArgs): 模型的参数。
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        hidden_dim = 4 * self.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        # 将 hidden_dim 四舍五入到 multiple_of 参数的最近倍数
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(self.dim, hidden_dim)
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass for the EncoderLayer module.
        
        Args:
            x (torch.Tensor): The input tensor.
            freqs_complex (torch.Tensor): The precomputed complex frequencies for RoPE.
            mask (torch.Tensor): The attention mask.
        
        Returns:
            torch.Tensor: The output tensor after passing through the encoder layer.

        ---
        EncoderLayer 模块的前向传播。

        参数:
            x (torch.Tensor): 输入张量。
            freqs_complex (torch.Tensor): 用于 RoPE 的预计算复数频率。
            mask (torch.Tensor): 注意力掩码。

        返回:
            torch.Tensor: 通过编码器层后的输出张量。
        """
        # Pre-normalization and residual connection for attention
        # 注意力层的预归一化和残差连接
        h = x + self.attention(self.attention_norm(x), freqs_complex, mask)
        
        # Pre-normalization and residual connection for feed-forward
        # 前馈层的预归一化和残差连接
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        """
        Initialize the Transformer module.
        
        Args:
            args (ModelArgs): The arguments for the model.

        ---
        初始化 Transformer 模块。

        参数:
            args (ModelArgs): 模型的参数。
        """
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderLayer(args))
            
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Forward pass for the Transformer module.
        
        Args:
            tokens (torch.Tensor): The input token IDs.
            start_pos (int): The starting position for the sequence.
        
        Returns:
            torch.Tensor: The output logits.

        ---
        Transformer 模块的前向传播。

        参数:
            tokens (torch.Tensor): 输入的 token ID。
            start_pos (int): 序列的起始位置。

        返回:
            torch.Tensor: 输出的 logits。
        """
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"
        
        h = self.tok_embeddings(tokens)
        
        self.freqs_complex = self.freqs_complex.to(h.device)
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        mask = torch.full((seq_len, seq_len), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=1)
        
        for layer in self.layers:
            h = layer(h, freqs_complex, mask)
            
        h = self.norm(h)
        output = self.output(h).float()
        return output