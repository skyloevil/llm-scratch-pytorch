#ref code:https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel

#--------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd,3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)

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

        att = (q@k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0,float('-inf'))
        att = F.softmax(att,dim=-1) 
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd)

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
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 384


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

    def forward(self,idx):
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
        return logits

        
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

        config_args['vocab_size'] = 50257
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
'''
PyTorch 内部自动处理：在 model.eval() 模式下，nn.Dropout 会自动关闭，
但 PyTorch 的 nn.Dropout 在训练时已经进行了 1/(1-p) 的放大，
因此在测试时无需额外缩放（因为训练时的期望已经和测试时对齐）。

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
