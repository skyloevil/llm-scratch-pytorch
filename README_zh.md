[English](./README.md)

# llm-scratch-pytorch
llm-scratch-pytorch - 本项目旨在为初学者提供一个友好的环境，专注于理解 PyTorch 的基础知识，并从零开始、一步步地实现大语言模型（LLM）。

## 目录
- [llm-scratch-pytorch](#llm-scratch-pytorch)
  - [目录](#目录)
  - [安装](#安装)
  - [实现过程](#实现过程)
    - [Pytorch 基础](#pytorch-基础)
      - [梯度计算](#梯度计算)
    - [从零实现 GPT2](#从零实现-gpt2)
    - [从零实现 LLaMA2](#从零实现-llama2)
    - [从零实现 Flash Attention](#从零实现-flash-attention)
  - [参考资料](#参考资料)
  - [工具](#工具)
  - [致谢](#致谢)

## 安装

要安装所需的依赖，请运行：

```bash
pip install -r requirements.txt
```

如果你有支持 CUDA 的 GPU 并希望使用 CUDA 进行加速，请确保已安装相应的 CUDA 工具包和驱动程序。如果可用，PyTorch 将自动使用 CUDA。你可以使用以下 Python 代码验证 CUDA 是否可用：

```python
import torch
print(torch.cuda.is_available())
```

如果输出 `True`，则你的环境已准备好进行 GPU 加速。

**如果你没有本地 GPU 环境**，我们建议使用 **Runpod 云环境**进行安装：
Runpod 推荐链接: https://runpod.io?ref=4dzcggxy
通过此链接，你将获得：
- 注册并消费10美元后，可获得一次性5至500美元的随机信用奖励
- 即时访问 GPU 资源，立即开始

## 实现过程
### [Pytorch 基础](https://github.com/skyloevil/llm-scratch-pytorch/tree/main/pytorch_basis)
#### [梯度计算](https://github.com/skyloevil/llm-scratch-pytorch/tree/main/pytorch_basis/computation_gragh)
- [✅] 梯度基础
- [✅] 偏导数
- [✅] 计算图
- [✅] 前向和反向传播
- [✅] torch_variables_grad_inplace_operation（torch 变量、梯度和原地操作）
- [✅] retain_graph（保留计算图）

### [从零实现 GPT2](https://github.com/skyloevil/llm-scratch-pytorch/tree/main/pytorch_gp2_from_scratch)
- [✅] 探索 GPT-2 (124M) OpenAI 检查点
- [✅] 第一部分: 实现 GPT-2 nn.Module
- [✅] 加载 huggingface/GPT-2 参数
- [✅] 实现前向传播以获得 logits
- [✅] 采样初始化、前缀 token、分词
- [✅] 采样循环
- [✅] 采样，自动检测设备
- [✅] 开始训练：数据批次 (B,T) → logits (B,T,C)
- [✅] 交叉熵损失
- [✅] 优化循环：在单个批次上过拟合
- [✅] 轻量级数据加载器
- [✅] wte 和 lm_head 参数共享
- [✅] 模型初始化：std 0.02，残差初始化
- [✅] 第二部分: 让它变快。GPU、混合精度，1000ms
- [✅] Tensor Cores、代码计时、TF32 精度，333ms
- [✅] float16、梯度缩放器、bfloat16，300ms
- [✅] torch.compile、Python 开销、核融合，130ms
- [✅] flash attention，96ms
- [✅] 优美/丑陋的数字。词汇表大小 50257 → 50304，93ms
- [✅] 第三部分: 超参数、AdamW、梯度裁剪
- [✅] 学习率调度器：预热 + 余弦衰减
- [✅] 批次大小调度、权重衰减、FusedAdamW，90ms
- [✅] 梯度累积
- [❌] 分布式数据并行 (DDP)
- [❌] GPT-2、GPT-3 中使用的数据集，FineWeb (EDU)
- [❌] 验证数据分割、验证损失、采样恢复
- [❌] 评估：HellaSwag，开始运行
- [❌] 第四部分: 清晨见分晓！GPT-2、GPT-3 复现
- [❌] 特别推荐 llm.c，一个等效但更快的纯 C/CUDA 代码实现
- [❌] 总结，终于完成了，构建 nanogpt github 仓库

### [从零实现 LLaMA2](https://github.com/skyloevil/llm-scratch-pytorch/tree/main/pytorch_llama2_from_scratch)
- [❌] 介绍
- [❌] LLaMA 架构
- [❌] 嵌入
- [❌] 编写 Transformer 代码
- [❌] 旋转位置嵌入 (Rotary Positional Embedding)
- [❌] RMS 归一化
- [❌] 编码器层
- [❌] 带 KV 缓存的自注意力机制
- [❌] 分组查询注意力 (Grouped Query Attention)
- [❌] 编写自注意力代码
- [❌] 带 SwiGLU 的前馈层
- [❌] 模型权重加载
- [❌] 推理策略
- [❌] 贪心策略
- [❌] 波束搜索 (Beam Search)
- [❌] 温度 (Temperature)
- [❌] 随机采样
- [❌] Top K
- [❌] Top P
- [❌] 编写推理代码

### [从零实现 Flash Attention](https://github.com/skyloevil/llm-scratch-pytorch/tree/main/triton_flash_attention_scratch)
- [❌] 多头注意力
- [❌] 为什么需要 Flash Attention
- [❌] 安全的 Softmax
- [❌] 在线 Softmax
- [❌] 在线 Softmax (证明)
- [❌] 块状矩阵乘法
- [❌] Flash Attention 前向传播 (手动)
- [❌] Flash Attention 前向传播 (论文)
- [❌] CUDA 简介与示例
- [❌] 张量布局
- [❌] Triton 简介与示例
- [❌] Flash Attention 前向传播 (编码)
- [❌] Flash Attention 2 中的 LogSumExp 技巧
- [❌] 导数、梯度、雅可比矩阵
- [❌] 自动求导
- [❌] MatMul 操作的雅可比矩阵
- [❌] 通过 Softmax 的雅可比矩阵
- [❌] Flash Attention 反向传播 (论文)
- [❌] Flash Attention 反向传播 (编码)
- [❌] Triton 自动调优
- [❌] Triton 技巧：软件流水线
- [❌] 运行代码

## 参考资料

启发本项目的主要教育资源和实现：

- **[PyTorch Grad Tutorials](https://github.com/chunhuizhang/bilibili_vlogs/tree/master/learn_torch/grad)**: 演示 PyTorch 自动微分系统和梯度计算的实践示例。
- **[Computation Graph Visualization](https://github.com/chunhuizhang/bilibili_vlogs/blob/master/learn_torch/grad/03_computation_graph.ipynb)**: 解释 PyTorch 在反向传播过程中如何构建和遍历计算图的交互式 notebook。
- **[Forward & Backward Pass](https://github.com/chunhuizhang/bilibili_vlogs/blob/master/learn_torch/grad/04_backward_step.ipynb)**: 逐步讲解神经网络前向/反向操作与 PyTorch 内部机制。
- **[NanoGPT Implementation](https://github.com/karpathy/build-nanogpt)**: Andrej Karpathy 的极简 GPT 实现，清晰地展示了 transformer 架构的要点。
- **[PyTorch LLaMA](https://github.com/hkproj/pytorch-llama)**: 用于教育目的的纯 PyTorch 实现的 LLaMA 架构，代码清晰、易于修改。
- **[Triton & Cuda Flash Attention](https://github.com/hkproj/triton-flash-attention)**: 使用 Triton 和 CUDA 的 Flash Attention 参考实现，为大语言模型提供高效的注意力机制。

## 工具

我们推荐以下工具来帮助模型开发和优化：

- **[Tokenizer](https://tiktokenizer.vercel.app/)**: 一个交互式的分词器演练场，有助于可视化和理解文本是如何被分词的，对提示工程和调试很有用。
- **[KV Cache Size Calculator](https://lmcache.ai/kv_cache_calculator.html)**: 一个方便的计算器，用于估算 transformer 模型中键值缓存所需的 GPU 内存，对优化推理性能至关重要。

## 致谢  

我们衷心感谢以下个人和组织所做的贡献和启发：  

- **[PyTorch](https://github.com/pytorch)**: 构建了驱动本项目的深度学习基础框架。PyTorch 团队对开源创新的奉献精神是无价的。  
- **[chunhuizhang](https://github.com/chunhuizhang)**: 感谢您的技术见解和协作努力，帮助改进了本项目的关键部分。  
- **[Andrej Karpathy](https://github.com/karpathy)**: 您在人工智能领域的开创性研究和教育贡献深刻地影响了我们的方法。我们感谢您的领导。  
- **[yihong0618](https://github.com/yihong0618)**: 您的创意实现和开源精神激励我们不断突破。感谢您与世界分享您的工作。
- **[OpenAI](https://github.com/openai)**: 通过开创性的研究和工具推动了该领域的发展，为像我们这样的项目赋能。您对道德 AI 发展的承诺树立了重要榜样。
- **[Hugging Face](https://github.com/huggingface)**: 通过开放模型和库实现了 NLP 的民主化，加速了我们的开发。您包容的生态系统为该领域树立了标杆。  
- **[Umar Jamil](https://github.com/hkproj)**: 您深思熟虑的反馈和技术专长加强了本项目��感谢您的奉献和协作精神。

这项工作得益于开源贡献者的集体努力；我们旨在发扬他们的协作精神。
