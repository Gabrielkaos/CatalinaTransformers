import torch
from torch.utils.data import DataLoader, Dataset
from MODEL_TRANSFORMER import build_transformer_next_token
import torch.nn as nn

class LanguageModelDataset(Dataset):
    def __init__(self, sequences, pad_idx):
        self.sequences = sequences
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            "input": seq[:-1],
            "label": seq[1:]
        }

def causal_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).int()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")

    seq_len = 72
    batch_size = 64
    lr = 3e-4
    epochs = 50

    data = torch.load("data.pth")
    sequences = data["x"]  # use same tokens
    vocab = data["vocab"]
    tokenizer = data["tokenizer"]

    pad_idx = tokenizer["<PAD>"]

    dataset = LanguageModelDataset(sequences, pad_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = build_transformer_next_token(
        vocab_size=len(vocab),
        seq_len=seq_len - 1,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["label"].to(device)

            print(x.shape,y.shape)
            print(x,y)

            mask = causal_mask(x.size(1)).to(device)

            logits = model(x, mask)

            loss = criterion(
                logits.view(-1, len(vocab)),
                y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
        torch.save({"model": model.state_dict()}, "lm.pth")


if __name__ == '__main__':
    train()