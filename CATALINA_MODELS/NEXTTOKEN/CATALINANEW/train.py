import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.amp import autocast, GradScaler
from MODEL_TRANSFORMER import build_transformer_next_token

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
    batch_size = 130
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
        vocab_size=len(vocab)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    scaler = GradScaler()


    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_i,batch in enumerate(loader):
            x = batch["input"].to(device)
            y = batch["label"].to(device)


            mask = causal_mask(x.size(1)).to(device)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                logits = model(x, mask)
                loss = criterion(
                    logits.view(-1, len(vocab)),
                    y.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if (batch_i + 1) % 5 == 0:
              print(f"{batch_i+1}/{len(loader)}")

        print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
        torch.save({"model_state": model.state_dict()}, "lm.pth")


if __name__ == '__main__':
    train()