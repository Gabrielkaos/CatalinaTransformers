import torch


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
block_size = 8
batch_size = 4

def get_batch(split="train"):
    data = train_data if split=="train" else val_data

    offset = torch.randint(len(data)-block_size,(batch_size,))
    inputs = torch.stack([data[i:i+block_size] for i in offset])
    targets = torch.stack([data[i+1:i+block_size+1] for i in offset])
    return inputs,targets

inputs, targets = get_batch()

print(inputs.shape)
print(targets.shape)