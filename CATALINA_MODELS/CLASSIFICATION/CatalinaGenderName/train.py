import torch
# from data_cleaning import get_dialogue_data_for_transformer
from torch.utils.data import DataLoader, Dataset
from MODEL_TRANSFORMER import build_transformer_encoder
import torch.nn as nn
import warnings


class TransformerData(Dataset):
    def __init__(self, x, y, label, src_tokenizer):
        super().__init__()

        self.x = x
        self.y = y
        self.label = label

        self.pad_token = torch.tensor([src_tokenizer.get("<PAD>")], dtype=torch.int64)
        self.eos_token = torch.tensor([src_tokenizer.get("<EOS>")], dtype=torch.int64)
        self.sos_token = torch.tensor([src_tokenizer.get("<SOS>")], dtype=torch.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]
        label = self.label[item]

        return {
            "encoder_input": x,
            "decoder_input": y,
            "encoder_mask": (x != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (y != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(y.size(0)),
            "label": label
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


def save_file(name, model_to_be_saved):
    data1 = {
        "model_state": model_to_be_saved.state_dict()
    }

    torch.save(data1, name)

    print("MODEL SAVED")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Device={device}")
    if str(device) == "cuda": print(f"Device memory={(torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3):.3f} GB")
    # hyper params
    batch_size = 400
    max_seq_src = 27
    lr = 5e-1
    num_epoch = 10_000

    data = torch.load("data.pth")
    x = data["x"]
    y = data["y"]
    label = data["label"]
    src_vocab = data["src_vocab"]
    trgt_vocab = data["trgt_vocab"]
    tokenizer_src = data["tokenizer_src"]

    print(f"x={len(x)}")
    print(f"y={len(y)}")
    print(f"src_vocab={len(src_vocab)}")
    print(f"trgt_vocab={len(trgt_vocab)}")
    print("Data processed")

    # data
    train_loader = DataLoader(TransformerData(x, y, label, tokenizer_src), batch_size=batch_size, shuffle=True)

    model = build_transformer_encoder(len(src_vocab), len(trgt_vocab), max_seq_src, device=device).to(device)
    model.load_state_dict(torch.load("brain.pth")["model_state"])
    model.train()

    # optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-9)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    print()
    for epoch in range(100):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):

            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            label = batch["label"].to(device)
            label = label.squeeze(dim=1)


            encoder_output = model.encode(encoder_input, encoder_mask)
            # encoder_output_collapsed = torch.mean(encoder_output,dim=1)
            proj_output = model.project(encoder_output)
            proj_output_collapsed = torch.mean(proj_output,dim=1)

            loss = criterion(proj_output_collapsed,label)


            if (batch_idx + 1) % 5 == 0:
                print(
                    f"[TRAIN] Epoch:{epoch + 1}/{num_epoch}, Batch:{batch_idx + 1}/{len(train_loader)}, Loss:{loss.item():.4f}")

            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(
            f"[EPOCH] Epoch:{epoch + 1}/{num_epoch}, Accumulated Loss:{(running_loss/len(train_loader)):.4f}")
        save_file(f"saved_from_loss/{epoch+1}-{(running_loss/len(train_loader)):.4f}.pth", model)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
