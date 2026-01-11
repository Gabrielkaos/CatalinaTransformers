import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # creates a 2d list for the vocab size and the d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # multiply by sqrt of dmodel


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires even dimension"

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute for max_seq_len (optional optimization)
        self._seq_len_cached = max_seq_len
        positions = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)  # (seq_len, dim//2)
        
        # Duplicate freqs for complex-valued rotation
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])  # (1, 1, seq, dim)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, q, k, seq_len):
        # If seq_len exceeds cache, recompute
        if seq_len > self._seq_len_cached:
            positions = torch.arange(seq_len, device=q.device).float()
            freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()[None, None, :, :]
            sin = emb.sin()[None, None, :, :]
        else:
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
        
        # Apply rotary embeddings
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout, device=None):
        super().__init__()

        if device is None:
            device = torch.device("cpu")

        self.device = device
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create 2d list of size (seq_len,d_model)

        self.pos_enc = torch.zeros(seq_len, d_model).to(device)

        # create list of size seq_len
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(device)  # (seq_len,1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)

        # apply the formula
        # apply sin to the even of pos

        self.pos_enc[:, 0::2] = torch.sin(position * denominator)
        self.pos_enc[:, 1::2] = torch.cos(position * denominator)

        # batching so reshape
        self.pos_enc = self.pos_enc.unsqueeze(0)  # (1,seq_len,d_model)

        self.register_buffer('pos_enc_buffer', self.pos_enc)

    def forward(self, x):
        if x.shape[1] < self.seq_len:
            sliced_pos_enc = self.pos_enc[:, :x.shape[0], :]
        else:
            sliced_pos_enc = self.pos_enc[:, :x.shape[1], :]

        # print(x.shape)
        # print(self.pos_enc.shape)
        # print(sliced_pos_enc.shape)

        x = x + sliced_pos_enc.requires_grad_(False)

        return self.dropout(x)
"""
LayerNorm

    Normalizes using mean + variance
    Removes both scale and bias
    Slightly more expensive
    Very stable across many tasks
    Use if Small 

RMSNorm

    Normalizes using root mean square only
    Keeps the mean (no centering)
    Cheaper (fewer ops)
    Preserves signal magnitude better
"""
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x / rms

class LayerNormalization(nn.Module):
    def __init__(self, d_model, epsilon=10 ** -6, bias=True):
        super().__init__()

        self.use_bias=bias

        self.eps = epsilon

        self.alpha = nn.Parameter(torch.ones(d_model))  # multiplied
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        if self.use_bias:
            return self.alpha * normalized  + self.bias
        return self.alpha * normalized

#better than GELU (idk)
# class SwiGLU(nn.Module):
#     def forward(self, x):
#         x, gate = x.chunk(2, dim=-1)
#         return F.silu(gate) * x

# class FeedForwardNet(nn.Module):
#     def __init__(self, d_model, dff, dropout, activation="relu"):
#         super().__init__()

#         # self.linear1 = nn.Linear(d_model, dff)

        

#         # self.linear2 = nn.Linear(dff, d_model)

#         # if activation=="gelu":
#         #     self.activation = nn.GELU()
#         # elif activation=="swiglu":
#         #     self.activation = SwiGLU()
#         # else:
#         #     self.activation = nn.ReLU()

#         if activation == "swiglu":
#             self.linear1 = nn.Linear(d_model, 2 * dff)
#             self.activation = SwiGLU()
#             self.linear2 = nn.Linear(dff, d_model)
#         else:
#             self.linear1 = nn.Linear(d_model, dff)
#             self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
#             self.linear2 = nn.Linear(dff, d_model)

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         return self.linear2(self.dropout(self.activation(self.linear1(x))))

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


class MultiHeadBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, use_flash_attn=False, is_causal=True):
        super().__init__()

        self.is_causal=is_causal

        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        assert d_model % n_heads == 0, "d_model mod n_heads not equal to zero"

        self.d_k = d_model // n_heads

        self.rope = RotaryEmbedding(self.d_k)


        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

        self.use_flash_attn = use_flash_attn and hasattr(torch.nn.functional, 'scaled_dot_product_attention')


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value

    # def forward(self, q, k, v, mask):
    #     query = self.w_q(q)
    #     key = self.w_k(k)
    #     value = self.w_v(v)

    #     query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
    #     key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
    #     value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)

    #     x = MultiHeadBlock.attention(query, key, value, mask, self.dropout)

    #     x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

    #     return self.w_o(x)

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)

        seq_len = query.shape[-2]
        query, key = self.rope(query, key, seq_len)
        
        # Use Flash Attention if available (PyTorch 2.0+)
        if self.use_flash_attn:
            # Convert mask format for scaled_dot_product_attention
            attn_mask = None
            if not self.is_causal and mask is not None:
                attn_mask = mask.bool() if mask.dtype != torch.bool else mask
            
            x = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=self.is_causal  # We provide our own mask
            )
        else:
            # Fallback to manual attention
            x = MultiHeadBlock.attention(query, key, value, mask, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)
        return self.w_o(x)


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

    def forward(self, x, mask):
        
        
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


    # @staticmethod
    # def attention(query, key, value, mask, dropout: nn.Dropout):
    #     d_k = query.shape[-1]

    #     attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    #     if mask is not None:
    #         attention_scores.masked_fill_(mask == 0, float("-inf"))

    #     attention_scores = attention_scores.softmax(dim=-1)

    #     if dropout is not None:
    #         attention_scores = dropout(attention_scores)

    #     return attention_scores @ value

    # def forward(self, x, mask):

    #     qkv = self.c_attn(x)

    #     query,key,value = qkv.split(self.d_model,dim=2)

        
    #     query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
    #     key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
    #     value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        
    #     x = GPT2Attention.attention(query, key, value, mask, self.dropout)
        
    #     x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)
    #     return self.w_o(x)



class ResidualConn(nn.Module):
    def __init__(self, dropout, d_model, bias=True,norm="layernorm"):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        if norm=="layernorm":
            self.norm = LayerNormalization(d_model,bias=bias)
        else:
            self.norm = RMSNorm(d_model)

    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.norm(x)))
    

class GPTResidualConn(nn.Module):
    def __init__(self, dropout, d_model):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadBlock, feed_forward_block: FeedForwardNet, dropout, d_model,bias=True,norm="layernorm"):
        super().__init__()

        self.attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_conn = nn.ModuleList([ResidualConn(dropout, d_model,bias=bias,norm=norm) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_conn[0](x, lambda x: self.attention_block(x, x, x, src_mask))
        x = self.residual_conn[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model,bias=True,norm="layernorm"):
        super().__init__()

        self.layers = layers
        if norm=="layernorm":
            self.norm = LayerNormalization(d_model,bias=bias)
        else:
            self.norm = RMSNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadBlock, cross_attention: MultiHeadBlock, feed_forward_block, dropout, d_model,
                 bias=True, norm="layernorm"):
        super().__init__()

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward_block

        self.residual_conns = nn.ModuleList([ResidualConn(dropout,d_model,bias=bias,norm=norm) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, trgt_mask):
        x = self.residual_conns[0](x, lambda x: self.self_attention(x, x, x, trgt_mask))
        x = self.residual_conns[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_conns[2](x, self.feed_forward)
        return x

#modified decoder block(removed cross attention) used for decoder only transformer
class DecoderOnlyBlock(nn.Module):
    def __init__(self, self_attention, feed_forward_block: FeedForwardNet, dropout, d_model,bias=True,norm="layernorm"):
        super().__init__()

        self.self_attention = self_attention
        self.feed_forward = feed_forward_block

        self.residual_conns = nn.ModuleList([ResidualConn(dropout,d_model,bias=bias,norm=norm) for _ in range(2)])

    def forward(self, x, trgt_mask):
        x = self.residual_conns[0](x, lambda x: self.self_attention(x,x,x, trgt_mask))
        x = self.residual_conns[1](x, self.feed_forward)
        return x
    
class GPTDecoderBlock(nn.Module):
    def __init__(self, self_attention, feed_forward_block: FeedForwardNet, dropout, d_model):
        super().__init__()

        self.self_attention = self_attention
        self.feed_forward = feed_forward_block

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # self.residual_conns = nn.ModuleList([GPTResidualConn(dropout,d_model) for _ in range(2)])

    def forward(self, x, trgt_mask):
        # x = self.residual_conns[0](x, lambda x: self.self_attention(x,trgt_mask))
        # x = self.residual_conns[1](x, self.feed_forward)
        x = x + self.dropout(self.self_attention(self.norm1(x),trgt_mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model, bias=True,norm="layernorm"):
        super().__init__()

        self.layers = layers
        if norm=="layernorm":
            self.norm = LayerNormalization(d_model,bias=bias)
        else:
            self.norm = RMSNorm(d_model)

    def forward(self, x, encoder_out, src_mask, trgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, trgt_mask)
        return self.norm(x)

#used mainly for the decoder only transformer
class DecoderOnly(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model,bias=True,norm="layernorm"):
        super().__init__()

        self.layers = layers
        if norm=="layernorm":
            self.norm = LayerNormalization(d_model,bias=bias)
        else:
            self.norm = RMSNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class GPTDecoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model):
        super().__init__()

        self.layers = layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size, bias=True):
        super().__init__()

        self.projection_layer = nn.Linear(d_model, vocab_size,bias=bias)

    def forward(self, x):
        return self.projection_layer(x)


#full encoder + decoder
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbedding, trgt_embed: InputEmbedding,
                #  src_pos: PositionalEncoding, trgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trgt_embed = trgt_embed
        # self.src_pos = src_pos
        # self.trgt_pos = trgt_pos
        self.proj = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        # src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_out, src_mask, trgt, trgt_mask):
        trgt = self.trgt_embed(trgt)
        # trgt = self.trgt_pos(trgt)
        return self.decoder(trgt, encoder_out, src_mask, trgt_mask)

    def project(self, x):
        return self.proj(x)


class TransformerEncoderOnly(nn.Module):
    def __init__(self, encoder: Encoder,
                 src_embed: InputEmbedding,
                #  src_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()

        self.encoder = encoder
        self.src_embed = src_embed
        # self.src_pos = src_pos
        self.proj = projection_layer

    def forward(self, x, mask):
        x = self.src_embed(x)
        # x = self.src_pos(x)
        x = self.encoder(x, mask)[:,0,:]
        return self.proj(x)



    # def encode(self, src, src_mask):
    #     src = self.src_embed(src)
    #     # src = self.src_pos(src)
    #     return self.encoder(src, src_mask)

    # def project(self, x):
    #     return self.proj(x)
    


class TransformerDecoderOnly(nn.Module):
    def __init__(self, decoder, embed, projection):
        super().__init__()
        self.decoder = decoder
        self.embed = embed
        # self.pos = pos
        self.proj = projection

        #weight tying
        self.proj.projection_layer.weight=self.embed.embedding.weight

    def forward(self, x, mask):
        x = self.embed(x)
        # x = self.pos(x)
        x = self.decoder(x, mask)
        return self.proj(x)
    
class GPTTransformer(nn.Module):
    def __init__(self, decoder, embed, pos, projection):
        super().__init__()
        self.decoder = decoder
        self.embed = embed
        self.pos = pos
        self.proj = projection

        #weight tying
        # self.proj.weight=self.embed.weight

    def forward(self, x, mask):
        B,T = x.size()
        pos = torch.arange(0,T, dtype=torch.long, device=x.device)

        token_emb = self.embed(x)
        pos_emb = self.pos(pos)

        x = token_emb + pos_emb
    
        x = self.decoder(x, mask)
        return self.proj(x)

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



#decoder only transformer (used for next token prediction)
def build_transformer_next_token(
    vocab_size,
    d_model=512,
    n_layers=6,
    n_heads=8,
    dropout=0.1,
    bias_projection=False,
    norm="rms",
    mlp_activation="swiglu",
    use_flash_attn=True
):
    embed = InputEmbedding(d_model, vocab_size)
    # pos = PositionalEncoding(d_model, seq_len, dropout, device=device)

    decoder_blocks = []
    for _ in range(n_layers):
        self_attn = MultiHeadBlock(d_model, n_heads, dropout,use_flash_attn=use_flash_attn)
        ff = FeedForwardNet(d_model, 4 * d_model, dropout,activation=mlp_activation)
        decoder_blocks.append(DecoderOnlyBlock(self_attn, ff, dropout, d_model, bias=False,norm=norm))

    decoder = DecoderOnly(nn.ModuleList(decoder_blocks), d_model,bias=False,norm=norm)
    projection = ProjectionLayer(d_model, vocab_size,bias=bias_projection)

    model = TransformerDecoderOnly(decoder, embed, projection)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


#encoder only
def build_transformer_encoder(vocab_size, num_classes, d_model=512, n_layers=6,
                          n_heads=8, dropout=0.1, dff=2048, use_flash_attn=True):
    # create embed layers
    src_embed = InputEmbedding(d_model, vocab_size)

    # position encoders
    # src_pos = PositionalEncoding(d_model, src_seq_length, dropout, device=device)

    # encoder_blocks
    encoder_blocks = []
    for _ in range(n_layers):
        encoder_self_attention_block = MultiHeadBlock(d_model, n_heads, dropout, use_flash_attn=use_flash_attn, is_causal=False)
        feed_f_block = FeedForwardNet(d_model, dff, dropout, activation="gelu",bias=False)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_f_block, dropout, d_model,bias=False)
        encoder_blocks.append(encoder_block)

    # encoder decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks),d_model, bias=False)

    # project layer
    projection_layer = ProjectionLayer(d_model, num_classes)

    transformer = TransformerEncoderOnly(encoder, src_embed, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


# encoder + decoder architecture
def build_transformer(src_vocab_size, trgt_vocab_size, src_seq_length, trgt_seq_length, d_model=512, n_layers=6,
                      n_heads=8, dropout=0.1, dff=2048, device=None):
    # create embed layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    trgt_embed = InputEmbedding(d_model, trgt_vocab_size)

    # position encoders
    # src_pos = PositionalEncoding(d_model, src_seq_length, dropout, device=device)

    # trgt_pos = PositionalEncoding(d_model, trgt_seq_length, dropout, device=device)

    # encoder_blocks
    encoder_blocks = []
    for _ in range(n_layers):
        encoder_self_attention_block = MultiHeadBlock(d_model, n_heads, dropout)
        feed_f_block = FeedForwardNet(d_model, dff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_f_block, dropout, d_model)
        encoder_blocks.append(encoder_block)

    # decoder_blocks
    decoder_blocks = []
    for _ in range(n_layers):
        decoder_self_attention_block = MultiHeadBlock(d_model, n_heads, dropout)
        cross_self_attention_block = MultiHeadBlock(d_model, n_heads, dropout)
        feed_f_block = FeedForwardNet(d_model, dff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, cross_self_attention_block, feed_f_block, dropout, d_model)
        decoder_blocks.append(decoder_block)

    # encoder decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks),d_model)
    decoder = Decoder(nn.ModuleList(decoder_blocks), d_model)

    # project layer
    projection_layer = ProjectionLayer(d_model, trgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, trgt_embed, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer





if __name__ == "__main__":
    # Test
    dim = 64
    rope = RotaryEmbedding(dim)

    q = torch.randn(2, 8, 100, dim)  # (batch, heads, seq, d_k)
    k = torch.randn(2, 8, 100, dim)

    q_rot, k_rot = rope(q, k, 100)

    print(q_rot.shape)  # Should be (2, 8, 100, 64)
    print(torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5))  # Should be True (rotation preserves norm)