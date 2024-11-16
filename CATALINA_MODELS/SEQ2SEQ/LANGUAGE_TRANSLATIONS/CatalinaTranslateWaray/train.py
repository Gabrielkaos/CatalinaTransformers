import torch
from data_cleaning import get_dialogue_data_for_transformer
from torch.utils.data import DataLoader, Dataset
from MODEL_TRANSFORMER import build_transformer
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
    print(f"Device={device}")
    print(f"Device memory={(torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3):.3f} GB")
    # hyper params
    batch_size = 50
    max_seq_length = 40
    lr = 3e-4
    num_epoch = 10_000

    # data
    x, y, label, src_vocab, trgt_vocab, tokenizer_src, tokenizer_trgt = get_dialogue_data_for_transformer("script.txt",
                                                                                                          max_seq_length)
    train_loader = DataLoader(TransformerData(x, y, label, tokenizer_src), batch_size=batch_size, shuffle=True)

    # model
    model = build_transformer(len(src_vocab), len(trgt_vocab), max_seq_length, max_seq_length,device=device).to(device)
    # model.load_state_dict(torch.load("brain.pth")["model_state"])
    model.train()

    # optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.index("<PAD>"), label_smoothing=0.1).to(device)

    print()
    lowest = 9999
    for epoch in range(num_epoch):
        for batch_idx, batch in enumerate(train_loader):

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = criterion(proj_output.view(-1, len(tokenizer_trgt)), label.view(-1))

            if loss.item() < lowest:
                lowest = loss.item()
                print(
                    f"[SAVED] Epoch:{epoch + 1}/{num_epoch}, Batch:{batch_idx + 1}/{len(train_loader)}, Loss:{loss.item():.5f}")
                save_file(f"saved_from_loss/{loss.item():.4f}-{epoch+1}-{batch_idx+1}.pth", model)

            if batch_idx % 4 == 0:
                print(
                    f"[TRAIN] Epoch:{epoch + 1}/{num_epoch}, Batch:{batch_idx + 1}/{len(train_loader)}, Loss:{loss.item():.5f}")

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
