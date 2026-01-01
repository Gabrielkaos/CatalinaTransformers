import torch
from torch.utils.data import DataLoader, Dataset
from MODEL_TRANSFORMER.OLD import build_transformer
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


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")
    print(f"Device memory={(torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3):.3f} GB")
    # hyper params
    batch_size = 50
    max_seq_length = 30

    # data
    real_data = torch.load("30-30_100k_data.pth")
    src_vocab, trgt_vocab, tokenizer_src, tokenizer_trgt = real_data["src_vocab"], real_data["trgt_vocab"], real_data["tokenizer_src"], real_data["tokenizer_trgt"]
    test_data = torch.load("test_30-30_100k_data.pth")
    x_test, y_test, label_test = test_data["x"], test_data["y"], test_data["label"]
    test_loader = DataLoader(TransformerData(x_test, y_test, label_test, tokenizer_src), batch_size=batch_size, shuffle=True)

    # model
    model = build_transformer(len(src_vocab), len(trgt_vocab), max_seq_length, max_seq_length,device=device).to(device)
    model.load_state_dict(torch.load("epoch/1.4614-30-149.pth")["model_state"])
    model.eval()

    # optimizer and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.index("<PAD>"), label_smoothing=0.1).to(device)

    print()
    with torch.no_grad():
        running_loss = 0.0
        for batch_idx, batch in enumerate(test_loader):

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = criterion(proj_output.view(-1, len(tokenizer_trgt)), label.view(-1))

            running_loss+=loss.item()

        print(f"Running Loss:{(running_loss/len(test_loader)):.4f}")

        # old/10.8830

        # 11.7158


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    test()
