import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import torch
import math

@dataclass
class GPTConfig:
    block_size:int = 1024
    vocab_size:int = 50257
    n_layer:int = 12
    n_head:int = 12
    n_embed:int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)

        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self, x):
        
        B,T,C = x.size()

        qkv = self.c_attn(x)

        q,k,v = qkv.split(self.n_embed,dim=2)

        k = k.view(B,T,self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head, C // self.n_head).transpose(1,2)

        attn = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))

        attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0,float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v

        y = y.transpose(1,2).contiguous().view(B,T,C)

        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x



class Block(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size,config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))

        self.lm_head = nn.Linear(config.n_embed,config.vocab_size,bias=False)

        

    def forward(self, idx):
        B,T = idx.size()

        assert T <= self.config.block_size

        pos = torch.arange(0,T, dtype=torch.long, device=idx.device)

        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits



    @classmethod
    def from_pretrained(cls, model_type):

        assert model_type in {"gpt2","gpt2-medium","gpt2-large","gpt2-xl"}

        from transformers import GPT2LMHeadModel


        config_args = {
            'gpt2':        dict(n_layer=12,n_head=12,n_embed=768),
            'gpt2-medium': dict(n_layer=24,n_head=16,n_embed=1024),
            'gpt2-large':  dict(n_layer=36,n_head=20,n_embed=1280),
            'gpt2-xl':     dict(n_layer=48,n_head=25,n_embed=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        
        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()

        sd_keys = sd.keys()

        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]


        model_hf = GPT2LMHeadModel.from_pretrained(model_type)

        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        transposed = ["attn.c_attn.weight","attn.c_proj.weight","mlp.c_fc.weight","mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_hf)

        for k in sd_keys_hf:
            # print(k)
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1]==sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape==sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

if __name__=="__main__":
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    max_length = 30

    model = GPT.from_pretrained("gpt2")
    print("loaded")
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # model_state = model.state_dict()
    # print(*model_state.keys(),sep="\n")

    # print(model_state["transformer.h.11.attn.c_attn.weight"].shape)

    prompt = "Hello I am Catalina,"

    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens,dtype=torch.long)
    x = tokens.unsqueeze(0)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)

            logits = logits[:,-1,:]

            probs = F.softmax(logits,dim=-1)

            topk_probs, topk_indices = torch.topk(probs,50,dim=-1)

            ix = torch.multinomial(topk_probs,1)
            xcol = torch.gather(topk_indices,-1,ix)

            x = torch.cat((x,xcol),dim=1)

    tokens = x[0,:max_length].tolist()

    decoded = tokenizer.decode(tokens)
    print(decoded)