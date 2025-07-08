# 从零开始实现 LLaMA 2

本项目旨在从零开始、逐步实现 LLaMA 2 模型。LLaMA 2 是由 Meta AI 开发的一系列先进的大语言模型，以其强大的性能和开放的研究许可而闻名。

## 与 GPT-2 的主要区别

相较于本项目中已实现的 GPT-2，LLaMA 2 引入了多项架构上的改进，使其在性能和效率上都有显著提升。关键区别包括：

1.  **RMS 归一化 (RMS Norm)**: LLaMA 2 使用 RMSNorm 作为其归一化层，取代了传统的 LayerNorm。RMSNorm 更计算高效，并且在一些大型模型上表现出了更好的性能。

2.  **旋转位置嵌入 (Rotary Positional Embeddings, RoPE)**: LLaMA 2 采用 RoPE 来编码词元在序列中的位置信息，而不是使用 GPT-2 中的绝对位置嵌入。RoPE 被证明在捕捉长距离依赖关系方面更有效。

3.  **SwiGLU 激活函数**: LLaMA 2 在其前馈网络（FFN）中使用了 SwiGLU 激活函数，这是一种门控线性单元（Gated Linear Unit）的变体，有助于提升模型的表达能力。

4.  **分组查询注意力 (Grouped-Query Attention, GQA)**: 为了优化推理效率，特别是在处理长序列时，LLaMA 2 的某些版本采用了 GQA。这是一种在多头注意力（MHA）��多查询注意力（MQA）之间的折中方案。

## 实现路线图

我们将遵循主 `README.md` 中定义的路线图，逐步实现这些核心组件，最终构建一个功能完整的 LLaMA 2 模型。
