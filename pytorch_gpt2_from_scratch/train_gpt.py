#Ref code from build-nanogpt:   https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
#Ref code from openai/gpt-2:    https://github.com/openai/gpt-2/blob/master/src/model.py

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel
import time

#--------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd,3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))

    
    def forward(self,x):
        B,T,C = x.size()

        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

        #att = (q@k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0,float('-inf'))
        #att = F.softmax(att,dim=-1) 
        #y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        
        #flash attention
        #good article: https://zhuanlan.zhihu.com/p/639228219 / https://zhuanlan.zhihu.com/p/669926191
        '''
        before y = F.scaled_dot_product_attention(q, k, v, is_causal=True):
        W0628 04:28:50.516000 5517 torch/_inductor/utils.py:1250] [0/0] Not enough SMs to use max_autotune_gemm mode
        step 0,loss: 10.928955078125,dt:26301.28ms, tokens/sec: 155.73
        step 1,loss: 9.525428771972656,dt:152.38ms, tokens/sec: 26880.97
        step 2,loss: 8.986692428588867,dt:152.37ms, tokens/sec: 26882.19
        step 3,loss: 8.701053619384766,dt:151.92ms, tokens/sec: 26962.05
        step 4,loss: 8.394303321838379,dt:151.58ms, tokens/sec: 27022.83
        
        after y = F.scaled_dot_product_attention(q, k, v, is_causal=True):
        W0628 10:22:49.362000 1640 torch/_inductor/utils.py:1250] [0/0] Not enough SMs to use max_autotune_gemm mode
        step 0,loss: 10.929035186767578,dt:20899.85ms, tokens/sec: 195.98
        step 1,loss: 9.525280952453613,dt:106.14ms, tokens/sec: 38590.84
        step 2,loss: 8.98654556274414,dt:105.91ms, tokens/sec: 38675.54
        step 3,loss: 8.700865745544434,dt:105.33ms, tokens/sec: 38887.49
        '''
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd), 
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        # weight sharing scheme 768*50257 / 124M = 0.3112691613 = 31.13%
        self.transformer.wte.weight = self.lm_head.weight

        #initialize weights
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            # initialize linear layers with a normal distribution
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                #every block has 2 residual connections, so we scale the std by 1/sqrt(2*n_layer)
                std *= (2 * self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # initialize embedding layers with a normal distribution
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None):
        # idx is of shape (B,T)
        B,T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = pos_emb+tok_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1)) #(B*T,vocab_size),(B*T)
        return logits,loss

        
    @classmethod
    def from_pretrained(cls,model_type):
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
        print("loading weights from pretrained gpt:%s" %model_type)

        config_args = {
            'gpt2':          dict(n_layer=12,n_head=12,n_embd=768),
            'gpt-medium':   dict(n_layer=24,n_head=16,n_embd=1024),
            'gpt-large':    dict(n_layer=36,n_head=20,n_embd=1280),
            'gpt-xl':       dict(n_layer=48,n_head=25,n_embd=1600),
        }[model_type]

        '''
        before config_args['vocab_size'] = 50257 change to 50304,funny trick,nice/ugly numbers. vocab size 50257 → 50304~
        W0628 10:22:49.362000 1640 torch/_inductor/utils.py:1250] [0/0] Not enough SMs to use max_autotune_gemm mode
        step 0,loss: 10.929035186767578,dt:20899.85ms, tokens/sec: 195.98
        step 1,loss: 9.525280952453613,dt:106.14ms, tokens/sec: 38590.84
        step 2,loss: 8.98654556274414,dt:105.91ms, tokens/sec: 38675.54
        step 3,loss: 8.700865745544434,dt:105.33ms, tokens/sec: 38887.49

        after config_args['vocab_size'] = 50257 change to 50304
        step 0,loss: 10.929035186767578,dt:1909.72ms, tokens/sec: 2144.82
        step 1,loss: 9.525178909301758,dt:105.93ms, tokens/sec: 38666.66
        step 2,loss: 8.98649787902832,dt:104.86ms, tokens/sec: 39063.45
        step 3,loss: 8.700571060180664,dt:105.11ms, tokens/sec: 38970.49
        '''
        config_args['vocab_size'] = 50304
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        #print("sd_keys: ",sd_keys)
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        #print("sd_keys_hf: ",sd_keys_hf)
        #hf weights don't have .attn.masked_bias key.
        #sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

#--------------------------------------------------------------------------------------
#data loader file 
#https://youtu.be/l8pRSuU81PU?si=HxnEZ2Qsj78jzP0p
import tiktoken
class DataLoaderLite:
    def __init__(self,B,T):
        self.B = B
        self.T = T
    
        with open('input.txt','r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        self.current_position = 0
    
    def next_batch(self):
        B,T = self.B,self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)

        self.current_position += B * T

        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x,y

#--------------------------------------------------------------------------------------
# load pretrain test code:
#print("Are U OK?")
#model = GPT.from_pretrained('gpt2')   
#print("I'm very OK!")
#--------------------------------------------------------------------------------------
# device detection code:
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"
print(f"using device_type: {device}")
#--------------------------------------------------------------------------------------
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#--------------------------------------------------------------------------------------
'''
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('input.txt','r') as f:
    text = f.read()
tokens = enc.encode(text)
B,T = 4,32
buf = torch.tensor(tokens[:B*T+1])
x = buf[:-1].view(B,T).to(device)
y = buf[1:].view(B,T).to(device)
'''
train_loader = DataLoaderLite(B=4,T=1024)
#get logits
model = GPT(GPTConfig())
model.to(device)
'''
RTX 4000 Ada
before model = torch.compile(model) :
step 0,loss: 10.928787231445312,dt:709.12ms, tokens/sec: 5776.17
step 1,loss: 9.524967193603516,dt:303.48ms, tokens/sec: 13496.70
step 2,loss: 8.98651123046875,dt:303.37ms, tokens/sec: 13501.68
step 3,loss: 8.700054168701172,dt:303.34ms, tokens/sec: 13502.80

after model = torch.compile(model):
W0628 04:28:50.516000 5517 torch/_inductor/utils.py:1250] [0/0] Not enough SMs to use max_autotune_gemm mode
step 0,loss: 10.928955078125,dt:26301.28ms, tokens/sec: 155.73
step 1,loss: 9.525428771972656,dt:152.38ms, tokens/sec: 26880.97
step 2,loss: 8.986692428588867,dt:152.37ms, tokens/sec: 26882.19
step 3,loss: 8.701053619384766,dt:151.92ms, tokens/sec: 26962.05
step 4,loss: 8.394303321838379,dt:151.58ms, tokens/sec: 27022.83
'''
model = torch.compile(model)  # compile the model for better performance
#logits,loss = model(x,y)
#print(loss)
#tensor(11.0549, device='mps:0', grad_fn=<NllLossBackward0>) = -ln(1/50257)
#https://www.youtube.com/watch?v=l8pRSuU81PU&t=3194s

'''
RTX 4000 Ada
before torch.set_float32_matmul_precision("high"): #TF32
step 0,loss: 10.90705680847168,dt:1243.05ms, tokens/sec: 6590.27
step 1,loss: 9.505132675170898,dt:963.93ms, tokens/sec: 8498.56
step 2,loss: 8.920276641845703,dt:961.54ms, tokens/sec: 8519.63

after torch.set_float32_matmul_precision("high"):
step 0,loss: 11.002205848693848,dt:971.97ms, tokens/sec: 8428.24
step 1,loss: 9.620403289794922,dt:708.99ms, tokens/sec: 11554.42
step 2,loss: 8.985677719116211,dt:709.15ms, tokens/sec: 11551.93
'''
torch.set_float32_matmul_precision("high")

#--------------------------------------------------------------------------------------
'''
step    0 | loss: 10.929035 | lr 6.000000e-05 |norm:31.1151 | dt:22156.18ms | tokens/sec: 184.87
step    1 | loss: 9.625555 | lr 1.200000e-04 |norm:9.9653 | dt:111.46ms | tokens/sec: 36750.11
step    2 | loss: 9.288238 | lr 1.800000e-04 |norm:8.5428 | dt:111.12ms | tokens/sec: 36861.53
step    3 | loss: 9.759915 | lr 2.400000e-04 |norm:7.5720 | dt:111.11ms | tokens/sec: 36863.19
step    4 | loss: 9.020742 | lr 3.000000e-04 |norm:4.5275 | dt:111.40ms | tokens/sec: 36769.15
step    5 | loss: 8.418500 | lr 3.600000e-04 |norm:4.1224 | dt:110.43ms | tokens/sec: 37091.85
step    6 | loss: 8.204685 | lr 4.200000e-04 |norm:2.3306 | dt:110.41ms | tokens/sec: 37097.22
step    7 | loss: 7.944987 | lr 4.800000e-04 |norm:2.9877 | dt:110.24ms | tokens/sec: 37156.19
step    8 | loss: 7.707809 | lr 5.400000e-04 |norm:1.7011 | dt:110.39ms | tokens/sec: 37104.91
step    9 | loss: 7.291522 | lr 6.000000e-04 |norm:2.2257 | dt:110.29ms | tokens/sec: 37137.47
step   10 | loss: 7.133273 | lr 6.000000e-04 |norm:1.7963 | dt:111.18ms | tokens/sec: 36840.43
step   11 | loss: 7.066081 | lr 5.991677e-04 |norm:1.1539 | dt:111.76ms | tokens/sec: 36649.21
step   12 | loss: 7.146005 | lr 5.966759e-04 |norm:1.5587 | dt:111.37ms | tokens/sec: 36777.57
step   13 | loss: 7.006813 | lr 5.925399e-04 |norm:1.2228 | dt:110.89ms | tokens/sec: 36935.87
step   14 | loss: 6.611938 | lr 5.867853e-04 |norm:1.3372 | dt:112.25ms | tokens/sec: 36489.17
step   15 | loss: 6.623370 | lr 5.794475e-04 |norm:0.9707 | dt:112.37ms | tokens/sec: 36449.92
step   16 | loss: 6.529044 | lr 5.705718e-04 |norm:1.6986 | dt:111.78ms | tokens/sec: 36644.37
step   17 | loss: 6.312853 | lr 5.602128e-04 |norm:1.1591 | dt:111.04ms | tokens/sec: 36888.68
step   18 | loss: 6.561875 | lr 5.484346e-04 |norm:2.4942 | dt:110.86ms | tokens/sec: 36948.42
step   19 | loss: 6.499338 | lr 5.353096e-04 |norm:1.0101 | dt:111.48ms | tokens/sec: 36741.31
step   20 | loss: 6.817120 | lr 5.209188e-04 |norm:2.7525 | dt:111.70ms | tokens/sec: 36668.61
step   21 | loss: 6.587343 | lr 5.053510e-04 |norm:1.7314 | dt:111.09ms | tokens/sec: 36872.45
step   22 | loss: 6.682228 | lr 4.887020e-04 |norm:1.2198 | dt:111.40ms | tokens/sec: 36768.83
step   23 | loss: 6.741624 | lr 4.710746e-04 |norm:0.9836 | dt:110.95ms | tokens/sec: 36918.96
step   24 | loss: 6.741573 | lr 4.525774e-04 |norm:0.8798 | dt:111.48ms | tokens/sec: 36743.20
step   25 | loss: 6.685856 | lr 4.333245e-04 |norm:0.9160 | dt:111.27ms | tokens/sec: 36810.98
step   26 | loss: 6.539587 | lr 4.134346e-04 |norm:1.1471 | dt:111.57ms | tokens/sec: 36711.63
step   27 | loss: 6.574798 | lr 3.930302e-04 |norm:1.0432 | dt:111.68ms | tokens/sec: 36675.89
step   28 | loss: 6.570073 | lr 3.722373e-04 |norm:1.0808 | dt:113.06ms | tokens/sec: 36227.25
step   29 | loss: 6.408570 | lr 3.511840e-04 |norm:0.8717 | dt:111.75ms | tokens/sec: 36652.50
step   30 | loss: 6.305978 | lr 3.300000e-04 |norm:0.9523 | dt:111.85ms | tokens/sec: 36621.40
step   31 | loss: 6.239697 | lr 3.088160e-04 |norm:1.0876 | dt:110.93ms | tokens/sec: 36922.77
step   32 | loss: 6.408067 | lr 2.877627e-04 |norm:0.9818 | dt:111.74ms | tokens/sec: 36655.00
step   33 | loss: 6.549937 | lr 2.669698e-04 |norm:0.9403 | dt:110.56ms | tokens/sec: 37047.78
step   34 | loss: 6.536313 | lr 2.465654e-04 |norm:1.0917 | dt:111.08ms | tokens/sec: 36872.69
step   35 | loss: 6.536059 | lr 2.266755e-04 |norm:1.1388 | dt:110.58ms | tokens/sec: 37042.19
step   36 | loss: 6.382875 | lr 2.074226e-04 |norm:0.9951 | dt:110.33ms | tokens/sec: 37123.67
step   37 | loss: 6.546589 | lr 1.889254e-04 |norm:0.9262 | dt:111.21ms | tokens/sec: 36830.87
step   38 | loss: 6.326313 | lr 1.712980e-04 |norm:0.8783 | dt:112.48ms | tokens/sec: 36414.69
step   39 | loss: 6.165541 | lr 1.546490e-04 |norm:0.9403 | dt:110.86ms | tokens/sec: 36947.70
step   40 | loss: 6.288169 | lr 1.390812e-04 |norm:0.9853 | dt:111.14ms | tokens/sec: 36855.60
step   41 | loss: 6.377492 | lr 1.246904e-04 |norm:1.0718 | dt:111.51ms | tokens/sec: 36730.55
step   42 | loss: 6.172297 | lr 1.115654e-04 |norm:1.0675 | dt:111.91ms | tokens/sec: 36600.34
step   43 | loss: 6.149868 | lr 9.978716e-05 |norm:0.8971 | dt:112.93ms | tokens/sec: 36271.39
step   44 | loss: 6.309943 | lr 8.942824e-05 |norm:0.8781 | dt:111.16ms | tokens/sec: 36848.88
step   45 | loss: 6.226062 | lr 8.055253e-05 |norm:0.7721 | dt:111.04ms | tokens/sec: 36886.06
step   46 | loss: 6.104761 | lr 7.321474e-05 |norm:0.7722 | dt:111.05ms | tokens/sec: 36885.91
step   47 | loss: 6.129297 | lr 6.746012e-05 |norm:1.0375 | dt:111.67ms | tokens/sec: 36678.24
step   48 | loss: 6.180681 | lr 6.332415e-05 |norm:0.8881 | dt:112.38ms | tokens/sec: 36448.69
step   49 | loss: 6.077621 | lr 6.083232e-05 |norm:0.9115 | dt:111.72ms | tokens/sec: 36662.98
'''
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    # use cosine decay down to min_lr
    decay_ratio = (it-warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1, f"decay_ratio {decay_ratio} out of bounds"
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # cos from 1 to -1 # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr) 
#--------------------------------------------------------------------------------------

#optimize!
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8)
for i in range(50):
    t0 = time.time()
    x,y = train_loader.next_batch()
    x,y = x.to(device),y.to(device)
    optimizer.zero_grad()
    '''
    RTX 4000 Ada
    before with torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    step 0,loss: 10.928506851196289,dt:665.98ms, tokens/sec: 6150.37
    step 1,loss: 9.525293350219727,dt:369.31ms, tokens/sec: 11090.90
    step 2,loss: 8.986087799072266,dt:367.56ms, tokens/sec: 11143.78
    step 3,loss: 8.699840545654297,dt:368.03ms, tokens/sec: 11129.46

    after with torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    step 0,loss: 10.928787231445312,dt:709.12ms, tokens/sec: 5776.17
    step 1,loss: 9.524967193603516,dt:303.48ms, tokens/sec: 13496.70
    step 2,loss: 8.98651123046875,dt:303.37ms, tokens/sec: 13501.68
    step 3,loss: 8.700054168701172,dt:303.34ms, tokens/sec: 13502.80
    '''
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits,loss = model(x,y)
        #import code; code.interact(local=locals())  # for debugging, you can inspect logits, loss, etc.
        '''
        model.transformer.h[0].attn.c_proj.weight.dtype
        model.transformer.wte.weight.dtype
        '''
    loss.backward()
    '''
    [CN]
    torch.nn.utils.clip_grad_norm_() 是一个原地操作（in-place operation，注意函数名末尾的下划线），它主要做两件事：
    计算所有参数梯度的L2范数：首先它会计算模型所有可训练参数的梯度的L2范数（即所有梯度平方和开根号）。
    按比例缩放梯度：如果这个范数超过了给定的阈值（这里是1.0），就会将所有梯度按比例缩放，使得它们的总范数等于这个阈值。
    参数说明：
    model.parameters()：获取模型中所有需要梯度更新的参数
    1.0：设定的最大范数阈值
    返回值norm是裁剪前的梯度范数（即原始梯度总范数）。
    为什么要用梯度裁剪？
    防止梯度爆炸（特别是在RNN中常见）
    使训练过程更稳定
    可以允许使用更大的学习率
    数学表达：
    如果原始梯度总范数为 total_norm，那么裁剪后的梯度为：
    gradient * (max_norm / max(total_norm, max_norm))
    这相当于当 total_norm > max_norm 时，所有梯度都会按比例缩小，使得新的总范数等于 max_norm。

    [EN]
    This code utilizes the gradient clipping functionality in PyTorch. Specifically:  
    `torch.nn.utils.clip_grad_norm_()` is an in-place operation (note the underscore at the end of the function name), which performs two main tasks:  
    1. **Compute the L2 norm of all parameter gradients**: First, it calculates the L2 norm (Euclidean norm) of the gradients across all trainable parameters (i.e., the square root of the sum of squared gradients).  
    2. **Scale gradients proportionally**: If this norm exceeds the given threshold (here, `1.0`), it scales down all gradients proportionally so that their total norm equals this threshold.  
    **Parameters**:  
    - `model.parameters()`: Retrieves all trainable parameters of the model.  
    - `1.0`: The maximum norm threshold.  
    **Return value (`norm`)**: The original gradient norm before clipping.  
    **Why use gradient clipping?**  
    - Prevents **gradient explosion** (common in RNNs).  
    - Makes training more stable.  
    - Allows the use of higher learning rates.  
    **Mathematical Formulation**:  
    If the original gradient norm is `total_norm`, the clipped gradients become:  
    `gradient * (max_norm / max(total_norm, max_norm))`  
    This means that if `total_norm > max_norm`, all gradients are scaled down proportionally so that the new total norm equals `max_norm`.
    '''
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0) # gradient clipping
    lr = get_lr(i)  # learning rate scheduling
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()  # wait for all kernels to finish
    t1 = time.time()
    dt = (t1 - t0) * 1000  # convert to milliseconds
    token_per_sec = (train_loader.B * train_loader.T) / (t1-t0)  # tokens per second
    print(f"step {i:4d} | loss: {loss.item():.6f} | lr {lr:4e} |norm:{norm:.4f} | dt:{dt:.2f}ms | tokens/sec: {token_per_sec:.2f}")
import sys; sys.exit(0)

#--------------------------------------------------------------------------------------
'''
[CN]
PyTorch 内部自动处理：在 model.eval() 模式下，nn.Dropout 会自动关闭，
但 PyTorch 的 nn.Dropout 在训练时已经进行了 1/(1-p) 的放大，
因此在测试时无需额外缩放（因为训练时的期望已经和测试时对齐）。
[EN]
PyTorch handles this automatically: in model.eval() mode, nn.Dropout is automatically disabled.
Moreover, nn.Dropout in PyTorch already applies a scaling factor of 1/(1-p) during training,
so no additional scaling is needed during evaluation (because the expected value during training is already aligned with evaluation).
'''

num_return_sequences = 5
max_length = 30

model = GPT(GPTConfig())
model.eval()
model.to(device)

# https://tiktokenizer.vercel.app/
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello,I'm a language model,")
tokens = torch.tensor(tokens,dtype=torch.long)
#print("tokens size: ",tokens.shape)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
#print("tokens unsqueeze size: ",tokens.shape)
x = tokens.to(device)
#print("x:",x," x.shape:",x.shape)
#--------------------------------------------------------------------------------------
#37:08
#generate! right now x is (B,T) where B=5,T=8 (Hello,I'm a language model,)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        '''
            model(x) shape:  torch.Size([5, 8, 50257])
            logits[:,-1,:] shape:  torch.Size([5, 50257])
            model(x) shape:  torch.Size([5, 9, 50257])
            logits[:,-1,:] shape:  torch.Size([5, 50257])
            ..........
            model(x) shape:  torch.Size([5, 28, 50257])
            logits[:,-1,:] shape:  torch.Size([5, 50257])
            model(x) shape:  torch.Size([5, 29, 50257])
            logits[:,-1,:] shape:  torch.Size([5, 50257])
        '''

        #All the token logit streaming,from first to the current tail,and continous adding
        logits = model(x) # (B, T) -> (B, T, vocab_size)
        #print("model(x) shape: ",logits.shape)
        #always GET the final one token logit
        logits = logits[:,-1,:] # (B, vocab_size)
        #print("logits[:,-1,:] shape: ",logits.shape)
        probs = F.softmax(logits,dim=-1)    #(B, vocab_size)
        topk_probs,tok_indices = torch.topk(probs,50,dim=-1)    #(B, 50),(B, 50)
        ix = torch.multinomial(topk_probs,1)    #(B, 1)
        '''
            Example
                If:
                tok_indices is [[5, 12, 8], [20, 15, 30]] (shape 2×3)
                ix is [[1], [0]] (shape 2×1)
                Then torch.gather(tok_indices, -1, ix) would:
                For first row: take index 1 → 12
                For second row: take index 0 → 20
                Result: [[12], [20]]
        '''
        xcol = torch.gather(tok_indices,-1,ix)  #(B, 1) get token id 
        x = torch.cat((x,xcol),dim=1) #(B, T) -> (B, T+1)

for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    #print("x type,tokens type: ",type(x),type(tokens))
    decoded = enc.decode(tokens)
    print(">",decoded)
