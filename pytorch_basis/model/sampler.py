import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

logits =torch.tensor([[ 2.0, 1.0, 0.0, -1.0], [ 0.2, -0.1, 0.0, 1.0]])
temperatures = torch.tensor([0.0, 1.0])      # 样本0要贪心；样本1要随机采样

sampler = Sampler()
tokens = sampler(logits, temperatures).tolist()
print("tokens: ",tokens)

'''
(transformer) $ python sampler.py 
tokens:  [0, 3]
(transformer) $ python sampler.py 
tokens:  [0, 1]
(transformer) $ python sampler.py 
tokens:  [0, 0]
(transformer) $ python sampler.py 
tokens:  [0, 3]
(transformer) $ python sampler.py 
tokens:  [0, 0]
(transformer) $ python sampler.py 
tokens:  [0, 3]
(transformer) $ python sampler.py 
tokens:  [0, 0]
'''