# pytorch_practice_code
PyTorch Practice Code - A collection of PyTorch fundamentals including basic syntax, computation graphs, loss calculations, and model implementations. Ideal for beginners learning deep learning with PyTorch.

## code
- [grad dir](https://github.com/chunhuizhang/bilibili_vlogs/tree/master/learn_torch/grad)
- [computation_graph](https://github.com/chunhuizhang/bilibili_vlogs/blob/master/learn_torch/grad/03_computation_graph.ipynb)
- [forward & backward](https://github.com/chunhuizhang/bilibili_vlogs/blob/master/learn_torch/grad/04_backward_step.ipynb)
- [build-nanogpt](https://github.com/karpathy/build-nanogpt)

## process
### Pytorch basis
- [✅] grad
- [✅] partial derivaties
- [✅] compute graph
- [✅] forward & backward
- [✅] torch_variables_grad_inplace_operation
- [✅] retain_graph

### GPT2 scratch
- [✅] exploring the GPT-2 (124M) OpenAI checkpoint
- [✅] SECTION 1: implementing the GPT-2 nn.Module
- [✅] loading the huggingface/GPT-2 parameters
- [✅] implementing the forward pass to get logits
- [✅] sampling init, prefix tokens, tokenization
- [✅] sampling loop
- [❌] sample, auto-detect the device
- [❌] let’s train: data batches (B,T) → logits (B,T,C)
- [❌] cross entropy loss
- [❌] optimization loop: overfit a single batch
- [❌] data loader lite
- [❌] parameter sharing wte and lm_head
- [❌] model initialization: std 0.02, residual init
- [❌] SECTION 2: Let’s make it fast. GPUs, mixed precision, 1000ms
- [❌] Tensor Cores, timing the code, TF32 precision, 333ms
- [❌] float16, gradient scalers, bfloat16, 300ms
- [❌] torch.compile, Python overhead, kernel fusion, 130ms
- [❌] flash attention, 96ms
- [❌] nice/ugly numbers. vocab size 50257 → 50304, 93ms
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

## tools
- [tokenizer](https://tiktokenizer.vercel.app/)
- [kvcache size calculator](https://lmcache.ai/kv_cache_calculator.html)

## Acknowledgments
- [chunhuizhang](https://github.com/chunhuizhang)
- [Andrej Karpathy](https://github.com/karpathy)