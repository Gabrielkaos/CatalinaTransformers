import torch
import torch.nn as nn
from torch.nn import functional as F
import math


with open("sheakespear.txt","r") as f:
    text = f.read()



chars = sorted(list(set(text)))
vocab_size = len(chars)


stoi = {char:i for i,char in enumerate(chars)}
iost = {i:char for char, i in stoi.items()}

encoder = lambda s:[stoi[i] for i in s]
decoder = lambda i:"".join([iost[s] for s in i])


data = torch.tensor(encoder(text),dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# seq_len
block_size = 16
batch_size = 32
dropout = 0.2

def get_batch(split="train"):
    data = train_data if split=="train" else val_data

    offset = torch.randint(len(data)-block_size,(batch_size,))
    inputs = torch.stack([data[i:i+block_size] for i in offset])
    targets = torch.stack([data[i+1:i+block_size+1] for i in offset])
    return inputs,targets

inputs, targets = get_batch()


class Head(nn.Module):
    def __init__(self, head_size, d_model):
        super().__init__()

        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1)
        wei/=math.sqrt(C)

        wei = wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))

        wei = torch.softmax(wei,dim=-1)
        wei = self.dropout(wei)

        out = wei @ v

        # softmax(QK^T/sqrt(head_size)) * v

        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, d_model, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, d_model)  for _ in range(num_heads)])
        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.projection(out))
    

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model,d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()

        head_size = d_model // n_head

        self.heads = MultiHeadAttention(head_size, d_model, n_head)
        self.feed_forward = FeedForward(d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.heads(self.ln1(x))
        return x + self.feed_forward(self.ln2(x))

class Bigram(nn.Module):
    def __init__(self, d_model, n_layer, n_head):
        super().__init__()

        self.embed = nn.Embedding(vocab_size,d_model) #embedding layer
        self.position_embed = nn.Embedding(block_size,d_model) #positional encoding
        
        self.decoder_blocks = nn.Sequential(
            *[DecoderBlock(d_model, n_head) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model,vocab_size) #projection layer

    def forward(self, idx):

        B,T = idx.shape

        tokens_embed = self.embed(idx) #b,t,d_model
        pos_embed = self.position_embed(torch.arange(T)) #t,d_model

        x = tokens_embed + pos_embed
        x = self.ln(self.decoder_blocks(x))
        logits = self.lm_head(x) #b,t,vocab_size

        return logits
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            idx_cond = idx[:,-block_size:]
            logits = self(idx_cond)
            logits = logits[:,-1,:]

            probs = F.softmax(logits,dim=-1)

            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)

        return idx
d_model = 32
n_layer = 6
n_head = 4

model = Bigram(d_model, n_layer, n_head)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)

#train
print("Training...")
model.train()
n_epoch = 5000
for i in range(n_epoch):
    x,y=get_batch("train")

    out = model(x)
    B,T,C = out.shape
    loss = criterion(out.view(B*T,C),y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(i+1) % 100==0:print(f"{(loss.item()):.3f}-{i+1}/{n_epoch}")

print("Testing...\n")
model.eval()
print(decoder(model.generate(torch.zeros(1,1,dtype=torch.long),100)[0].tolist()))


# B,T,C = 4,8,32

# x = torch.randn(B,T,C)

# xbow = torch.zeros((B,T,C))


# for b in range(B):
#     for t in range(T):
#         xprev = x[b,:t+1]
#         xbow[b,t] = torch.mean(xprev,0)

# wei = torch.tril(torch.ones(T,T))
# wei = wei / torch.sum(wei,1, keepdim=True)

# # xbow2 = wei @ x


#single head
# n_head = 16
# key = nn.Linear(C,n_head,bias=False)
# query = nn.Linear(C,n_head,bias=False)
# value = nn.Linear(C,n_head,bias=False)
# k = key(x) #, B,T, n_head
# q = query(x) #B,T,n_head

# wei = q @ k.transpose(-2,-1)
# wei /= math.sqrt(n_head)


# tril = torch.tril(torch.ones(T,T))
# wei = wei.masked_fill(tril==0,float('-inf'))

# wei = F.softmax(wei, dim=-1)


# v = value(x)

# out = wei @ v

# B,T, T x B, C, T
# print(out[0])
# print(wei.shape)
# print(x.shape)


