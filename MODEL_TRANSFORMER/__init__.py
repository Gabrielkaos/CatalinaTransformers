import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # creates a 2d list for the vocab size and the d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # multiply by sqrt of dmodel


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


class FeedForwardNet(nn.Module):
    def __init__(self, d_model, dff, dropout, activation="relu"):
        super().__init__()

        self.linear1 = nn.Linear(d_model, dff)

        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(dff, d_model)

        if activation=="gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class MultiHeadBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, use_flash_attn=False):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        assert d_model % n_heads == 0, "d_model mod n_heads not equal to zero"

        self.d_k = d_model // n_heads

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
        
        # Use Flash Attention if available (PyTorch 2.0+)
        if self.use_flash_attn:
            # Convert mask format for scaled_dot_product_attention
            attn_mask = None
            if mask is not None:
                attn_mask = mask.bool() if mask.dtype != torch.bool else mask
            
            x = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False  # We provide our own mask
            )
        else:
            # Fallback to manual attention
            x = MultiHeadBlock.attention(query, key, value, mask, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)
        return self.w_o(x)


class ResidualConn(nn.Module):
    def __init__(self, dropout, d_model, bias=True):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model,bias=bias)

    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadBlock, feed_forward_block: FeedForwardNet, dropout, d_model,bias=True):
        super().__init__()

        self.attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_conn = nn.ModuleList([ResidualConn(dropout, d_model,bias=bias) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_conn[0](x, lambda x: self.attention_block(x, x, x, src_mask))
        x = self.residual_conn[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model,bias=True):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(d_model, bias=bias)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadBlock, cross_attention: MultiHeadBlock, feed_forward_block, dropout, d_model,bias=True):
        super().__init__()

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward_block

        self.residual_conns = nn.ModuleList([ResidualConn(dropout,d_model,bias=bias) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, trgt_mask):
        x = self.residual_conns[0](x, lambda x: self.self_attention(x, x, x, trgt_mask))
        x = self.residual_conns[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_conns[2](x, self.feed_forward)
        return x
    
class DecoderOnlyBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadBlock, feed_forward_block, dropout, d_model,bias=True):
        super().__init__()

        self.self_attention = self_attention
        self.feed_forward = feed_forward_block

        self.residual_conns = nn.ModuleList([ResidualConn(dropout,d_model,bias=bias) for _ in range(2)])

    def forward(self, x, trgt_mask):
        x = self.residual_conns[0](x, lambda x: self.self_attention(x, x, x, trgt_mask))
        x = self.residual_conns[1](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model, bias=True):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(d_model,bias=bias)

    def forward(self, x, encoder_out, src_mask, trgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, trgt_mask)
        return self.norm(x)
    
class DecoderOnly(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model,bias=True):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(d_model,bias=bias)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.projection_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.projection_layer(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbedding, trgt_embed: InputEmbedding,
                 src_pos: PositionalEncoding, trgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trgt_embed = trgt_embed
        self.src_pos = src_pos
        self.trgt_pos = trgt_pos
        self.proj = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_out, src_mask, trgt, trgt_mask):
        trgt = self.trgt_embed(trgt)
        trgt = self.trgt_pos(trgt)
        return self.decoder(trgt, encoder_out, src_mask, trgt_mask)

    def project(self, x):
        return self.proj(x)


class TransformerEncoderOnly(nn.Module):
    def __init__(self, encoder: Encoder,
                 src_embed: InputEmbedding,
                 src_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()

        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.proj = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def project(self, x):
        return self.proj(x)
    
class TransformerDecoderOnly(nn.Module):
    def __init__(self, decoder, embed, pos, projection):
        super().__init__()
        self.decoder = decoder
        self.embed = embed
        self.pos = pos
        self.proj = projection

        self.proj.projection_layer.weight=self.embed.embedding.weight

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pos(x)
        x = self.decoder(x, mask)
        return self.proj(x)

def build_transformer_next_token(
    vocab_size,
    seq_len,
    d_model=512,
    n_layers=6,
    n_heads=8,
    dropout=0.1,
    dff=2048,
    device=None
):
    embed = InputEmbedding(d_model, vocab_size)
    pos = PositionalEncoding(d_model, seq_len, dropout, device=device)

    decoder_blocks = []
    for _ in range(n_layers):
        self_attn = MultiHeadBlock(d_model, n_heads, dropout)
        ff = FeedForwardNet(d_model, dff, dropout,activation="gelu")
        decoder_blocks.append(
            DecoderOnlyBlock(self_attn, ff, dropout, d_model, bias=False)
        )

    # decoder_blocks = []
    # for _ in range(n_layers):
    #     decoder_self_attention_block = MultiHeadBlock(d_model, n_heads, dropout)
    #     cross_self_attention_block = MultiHeadBlock(d_model, n_heads, dropout)
    #     feed_f_block = FeedForwardNet(d_model, dff, dropout)
    #     decoder_block = DecoderBlock(decoder_self_attention_block, cross_self_attention_block, feed_f_block, dropout)
    #     decoder_blocks.append(decoder_block)

    decoder = DecoderOnly(nn.ModuleList(decoder_blocks), d_model,bias=False)
    projection = ProjectionLayer(d_model, vocab_size)

    model = TransformerDecoderOnly(decoder, embed, pos, projection)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model



def build_transformer_encoder(src_vocab_size, trgt_vocab_size, src_seq_length, d_model=512, n_layers=6,
                          n_heads=8, dropout=0.1, dff=2048, device=None):
    # create embed layers
    src_embed = InputEmbedding(d_model, src_vocab_size)

    # position encoders
    src_pos = PositionalEncoding(d_model, src_seq_length, dropout, device=device)

    # encoder_blocks
    encoder_blocks = []
    for _ in range(n_layers):
        encoder_self_attention_block = MultiHeadBlock(d_model, n_heads, dropout)
        feed_f_block = FeedForwardNet(d_model, dff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_f_block, dropout, d_model)
        encoder_blocks.append(encoder_block)

    # encoder decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks),d_model)

    # project layer
    projection_layer = ProjectionLayer(d_model, trgt_vocab_size)

    transformer = TransformerEncoderOnly(encoder, src_embed, src_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer


def build_transformer(src_vocab_size, trgt_vocab_size, src_seq_length, trgt_seq_length, d_model=512, n_layers=6,
                      n_heads=8, dropout=0.1, dff=2048, device=None):
    # create embed layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    trgt_embed = InputEmbedding(d_model, trgt_vocab_size)

    # position encoders
    src_pos = PositionalEncoding(d_model, src_seq_length, dropout, device=device)

    trgt_pos = PositionalEncoding(d_model, trgt_seq_length, dropout, device=device)

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

    transformer = Transformer(encoder, decoder, src_embed, trgt_embed, src_pos, trgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer





class RNN(nn.Module):
    def __init__(self, n_input, n_hiddens, num_layers, n_output, device=None):
        super(RNN, self).__init__()

        if device is None:
            device = torch.device("cpu")

        self.device = device
        self.n_hidden = n_hiddens
        self.n_layers = num_layers
        self.embed = nn.Embedding(n_input, n_hiddens).to(self.device)
        self.lstm = nn.LSTM(n_hiddens, n_hiddens, num_layers, batch_first=True, bidirectional=True).to(self.device)

        self.fc = nn.Linear(n_hiddens * 2, n_output).to(self.device)

    def forward(self, x, hiddens, cells):
        output_n = self.embed(x)

        output_n, (hiddens, cells) = self.lstm(output_n.unsqueeze(1), (hiddens, cells))

        output_n = self.fc(output_n.reshape(output_n.shape[0], -1))
        return output_n, (hiddens, cells)

    def init_hidden(self, batch_num):
        hiddens = torch.zeros(self.n_layers * 2, batch_num, self.n_hidden).to(self.device)
        cells = torch.zeros(self.n_layers * 2, batch_num, self.n_hidden).to(self.device)
        return hiddens, cells
