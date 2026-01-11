import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class FeedForwardNet(nn.Module):
    def __init__(self, d_model, dff, dropout, activation="gelu", bias=False):
        super().__init__()
        
        if activation == "swiglu":
            # For parameter parity with standard FFN of hidden dim 4*d_model:
            # SwiGLU uses (8/3)*d_model ≈ 2.67*d_model
            self.w1 = nn.Linear(d_model, dff, bias=bias)  # Gate
            self.w3 = nn.Linear(d_model, dff, bias=bias)  # Value
            self.w2 = nn.Linear(dff, d_model, bias=bias)  # Down
            self.activation_fn = F.silu
            self.use_swiglu = True
        else:
            self.linear1 = nn.Linear(d_model, dff, bias=bias)
            self.linear2 = nn.Linear(dff, d_model, bias=bias)
            self.activation_fn = nn.GELU(approximate="tanh") if activation == "gelu" else F.relu
            self.use_swiglu = False
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if self.use_swiglu:
            # SwiGLU: silu(W1 @ x) ⊙ (W3 @ x)
            return self.w2(self.dropout(self.activation_fn(self.w1(x)) * self.w3(x)))
        else:
            return self.linear2(self.dropout(self.activation_fn(self.linear1(x))))





class GPT2Attention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()


        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        assert d_model % n_heads == 0, "d_model mod n_heads not equal to zero"

        self.d_k = d_model // n_heads

        # self.rope = RotaryEmbedding(self.d_k)


        self.c_attn = nn.Linear(d_model, 3 * d_model)

        self.w_o = nn.Linear(d_model, d_model)

        # self.use_flash_attn = use_flash_attn and hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.register_buffer("bias",torch.tril(torch.ones(block_size,block_size)).view(1,1,block_size, block_size))

    def forward(self, x):
        
        
        B,T,C = x.size()

        qkv = self.c_attn(x)

        q,k,v = qkv.split(self.d_model,dim=2)

        k = k.view(B,T,self.n_heads, C // self.n_heads).transpose(1,2)
        q = q.view(B,T,self.n_heads, C // self.n_heads).transpose(1,2)
        v = v.view(B,T,self.n_heads, C // self.n_heads).transpose(1,2)

        attn = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))

        attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0,float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v

        y = y.transpose(1,2).contiguous().view(B,T,C)

        y = self.w_o(y)

        return y



class GPTDecoderBlock(nn.Module):
    def __init__(self, self_attention, feed_forward_block: FeedForwardNet, dropout, d_model):
        super().__init__()

        self.self_attention = self_attention
        self.feed_forward = feed_forward_block

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # self.residual_conns = nn.ModuleList([GPTResidualConn(dropout,d_model) for _ in range(2)])

    def forward(self, x):
        # x = self.residual_conns[0](x, lambda x: self.self_attention(x,trgt_mask))
        # x = self.residual_conns[1](x, self.feed_forward)
        x = x + self.dropout(self.self_attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class GPTDecoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model):
        super().__init__()

        self.layers = layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class GPTTransformer(nn.Module):
    def __init__(self, decoder, embed, pos, projection):
        super().__init__()
        self.decoder = decoder
        self.embed = embed
        self.pos = pos
        self.proj = projection

    def forward(self, x):
        B,T = x.size()
        pos = torch.arange(0,T, dtype=torch.long, device=x.device)

        token_emb = self.embed(x)
        pos_emb = self.pos(pos)

        x = token_emb + pos_emb
    
        x = self.decoder(x)
        return self.proj(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size, bias=True):
        super().__init__()

        self.projection_layer = nn.Linear(d_model, vocab_size,bias=bias)

    def forward(self, x):
        return self.projection_layer(x)
    


def gpt2_like_model(
    vocab_size,
    block_size=1024,
    d_model=512,
    n_layers=6,
    n_heads=8,
    dropout=0.1,
    bias_projection=False,
    mlp_activation="gelu"
):
    embed = nn.Embedding(vocab_size,d_model)
    pos = nn.Embedding(block_size,d_model)

    decoder_blocks = []
    for _ in range(n_layers):
        self_attn = GPT2Attention(d_model, n_heads, block_size, dropout)
        ff = FeedForwardNet(d_model, 4 * d_model, dropout,activation=mlp_activation,bias=True)
        decoder_blocks.append(GPTDecoderBlock(self_attn, ff, dropout, d_model))

    decoder = GPTDecoder(nn.ModuleList(decoder_blocks),d_model)
    projection = ProjectionLayer(d_model, vocab_size,bias=bias_projection)

    model = GPTTransformer(decoder, embed,pos, projection)

    # for p in model.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform_(p)

    return model
