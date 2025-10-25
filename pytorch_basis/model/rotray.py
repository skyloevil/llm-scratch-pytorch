import torch

rotary_dim = 128 
inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
print("inv_freq.shape:",inv_freq.shape)
print("inv_freq:",inv_freq)
t = torch.arange(40960,dtype=torch.float)
print("t.shape:",t.shape)
print("t:",t)
freqs = torch.einsum("i , j -> i j", t, inv_freq)
print("freqs.shape:",freqs.shape)
print("freqs:",freqs)
cos = freqs.cos()
sin = freqs.sin()
print("cos.shape:",cos.shape)
print("cos:",cos)
print("sin.shape:",sin.shape)
print("sin:",sin)

