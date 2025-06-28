# llm-scratch-pytorch
llm-scratch-pytorch - The code is designed to be beginner-friendly, with a focus on understanding the fundamentals of PyTorch and implementing LLMs from scratch,step by step.

## Table of Contents
- [llm-scratch-pytorch](#llm-scratch-pytorch)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Process](#process)
    - [Pytorch basis](#pytorch-basis)
      - [Grad](#grad)
    - [GPT2 scratch](#gpt2-scratch)
    - [LLaMA2 scratch](#llama2-scratch)
  - [Reference](#reference)
  - [Tools](#tools)
  - [Acknowledgments](#acknowledgments)

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

If you have a CUDA-capable GPU and want to use CUDA for acceleration, make sure you have the appropriate CUDA toolkit and drivers installed. PyTorch will automatically use CUDA if available. You can verify CUDA is available in Python with:

```python
import torch
print(torch.cuda.is_available())
```

If this prints `True`, your environment is ready for GPU acceleration.

## Process
### [Pytorch basis](https://github.com/skyloevil/llm-scratch-pytorch/tree/main/pytorch_basis)
#### [Grad](https://github.com/skyloevil/llm-scratch-pytorch/tree/main/pytorch_basis/computation_gragh)
- [✅] grad basis
- [✅] partial derivaties
- [✅] compute graph
- [✅] forward & backward
- [✅] torch_variables_grad_inplace_operation
- [✅] retain_graph

### [GPT2 scratch](https://github.com/skyloevil/llm-scratch-pytorch/tree/main/pytorch_gp2_from_scratch)
- [✅] exploring the GPT-2 (124M) OpenAI checkpoint
- [✅] SECTION 1: implementing the GPT-2 nn.Module
- [✅] loading the huggingface/GPT-2 parameters
- [✅] implementing the forward pass to get logits
- [✅] sampling init, prefix tokens, tokenization
- [✅] sampling loop
- [✅] sample, auto-detect the device
- [✅] let’s train: data batches (B,T) → logits (B,T,C)
- [✅] cross entropy loss
- [✅] optimization loop: overfit a single batch
- [✅] data loader lite
- [✅] parameter sharing wte and lm_head
- [✅] model initialization: std 0.02, residual init
- [✅] SECTION 2: Let’s make it fast. GPUs, mixed precision, 1000ms
- [✅] Tensor Cores, timing the code, TF32 precision, 333ms
- [✅] float16, gradient scalers, bfloat16, 300ms
- [✅] torch.compile, Python overhead, kernel fusion, 130ms
- [✅] flash attention, 96ms
- [✅] nice/ugly numbers. vocab size 50257 → 50304, 93ms
- [❌] SECTION 3: hyperpamaters, AdamW, gradient clipping
- [❌] learning rate scheduler: warmup + cosine decay
- [❌] batch size schedule, weight decay, FusedAdamW, 90ms
- [❌] gradient accumulation
- [❌] distributed data parallel (DDP)
- [❌] datasets used in GPT-2, GPT-3, FineWeb (EDU)
- [❌] validation data split, validation loss, sampling revive
- [❌] evaluation: HellaSwag, starting the run
- [❌] SECTION 4: results in the morning! GPT-2, GPT-3 repro
- [❌] shoutout to llm.c, equivalent but faster code in raw C/CUDA
- [❌] summary, phew, build-nanogpt github repo

### [LLaMA2 scratch](https://github.com/skyloevil/llm-scratch-pytorch/tree/main/pytorch_llama2_from_scratch)
- [❌] Introduction
- [❌] LLaMA Architecture
- [❌] Embeddings
- [❌] Coding the Transformer
- [❌] Rotary Positional Embedding
- [❌] RMS Normalization
- [❌] Encoder Layer
- [❌] Self Attention with KV Cache
- [❌] Grouped Query Attention
- [❌] Coding the Self Attention
- [❌] Feed Forward Layer with SwiGLU
- [❌] Model weights loading
- [❌] Inference strategies
- [❌] Greedy Strategy
- [❌] Beam Search
- [❌] Temperature
- [❌] Random Sampling
- [❌] Top K
- [❌] Top P
- [❌] Coding the Inference

## Reference

Key educational resources and implementations that inspired this work:

- **[PyTorch Grad Tutorials](https://github.com/chunhuizhang/bilibili_vlogs/tree/master/learn_torch/grad)**: Practical examples demonstrating PyTorch's automatic differentiation system and gradient computation.
- **[Computation Graph Visualization](https://github.com/chunhuizhang/bilibili_vlogs/blob/master/learn_torch/grad/03_computation_graph.ipynb)**: Interactive notebook explaining how PyTorch constructs and traverses computation graphs during backpropagation.
- **[Forward & Backward Pass](https://github.com/chunhuizhang/bilibili_vlogs/blob/master/learn_torch/grad/04_backward_step.ipynb)**: Step-by-step walkthrough of neural network forward/backward operations with PyTorch internals.
- **[NanoGPT Implementation](https://github.com/karpathy/build-nanogpt)**: Andrej Karpathy's minimal GPT implementation that clearly demonstrates transformer architecture essentials.
- **[PyTorch LLaMA](https://github.com/hkproj/pytorch-llama)**: Clean, hackable implementation of the LLaMA architecture in pure PyTorch for educational purposes.

## Tools

We recommend the following tools to help with model development and optimization:

- **[Tokenizer](https://tiktokenizer.vercel.app/)**: An interactive tokenizer playground that helps visualize and understand how text gets tokenized, useful for prompt engineering and debugging.
- **[KV Cache Size Calculator](https://lmcache.ai/kv_cache_calculator.html)**: A handy calculator for estimating GPU memory requirements of key-value caches in transformer models, crucial for optimizing inference performance.

## Acknowledgments  

We sincerely appreciate the following individuals and organizations for their contributions and inspiration:  

- **[PyTorch](https://github.com/pytorch)**: For building the foundational deep learning framework that powers this project. The PyTorch team’s dedication to open-source innovation has been invaluable.  
- **[chunhuizhang](https://github.com/chunhuizhang)**: Thank you for your technical insights and collaborative efforts, which helped improve critical components of this work.  
- **[Andrej Karpathy](https://github.com/karpathy)**: Your pioneering research and educational contributions in AI have deeply influenced our approach. We’re grateful for your leadership.  
- **[yihong0618](https://github.com/yihong0618)**: Your creative implementations and open-source spirit have motivated us to push boundaries. Thank you for sharing your work with the world.
- **[OpenAI](https://github.com/openai)**: For advancing the field with groundbreaking research and tools that empower projects like ours. Your commitment to ethical AI development sets a vital example.
- **[Hugging Face](https://github.com/huggingface)**: For democratizing NLP with open models and libraries that accelerated our development. Your inclusive ecosystem sets a benchmark for the field.  
- **[Umar Jamil](https://github.com/hkproj)**: Your thoughtful feedback and technical expertise have strengthened this project. Thank you for your dedication and collaborative spirit.

This work thrives on the collective effort of open-source contributors; we aim to honor their spirit of collaboration.  