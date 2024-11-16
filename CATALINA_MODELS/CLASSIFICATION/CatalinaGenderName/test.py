import torch
from MODEL_TRANSFORMER import build_transformer_encoder
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import warnings


class TransformerTestData(Dataset):
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
            "decoder_mask": (y != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "label": label
        }


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")
    print(f"Device memory={(torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3):.3f} GB")

    batch_size = 400
    max_seq_src = 27

    data = torch.load("data.pth")
    src_vocab = data["src_vocab"]
    trgt_vocab = data["trgt_vocab"]
    print("Data processed")

    model = build_transformer_encoder(len(src_vocab), len(trgt_vocab), max_seq_src, device=device).to(device)
    model.load_state_dict(torch.load("brain.pth")["model_state"])
    model.eval()

    test_data = torch.load("test_data.pth")
    x_test = test_data["x"]
    y_test = test_data["y"]
    label_test = test_data["label"]
    tokenizer_src_test = test_data["tokenizer_src"]
    test_loader = DataLoader(TransformerTestData(x_test, y_test, label_test, tokenizer_src_test), batch_size=batch_size,
                             shuffle=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    print()
    with torch.no_grad():
        running_loss = 0.0
        n_correct = 0
        n_total = 0
        for batch_idx, batch in enumerate(test_loader):

            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            label = batch["label"].to(device)
            label = label.squeeze(dim=1)

            encoder_output = model.encode(encoder_input, encoder_mask)
            # collapsed_encoder_output = torch.mean(encoder_output, dim=1)  # Collapse the sequence dimension

            proj_output = model.project(encoder_output).mean(dim=1)

            loss = criterion(proj_output, label)

            _, preds = torch.max(proj_output, dim=1)
            # print(preds.shape)
            print(preds.shape,label.shape,batch_idx)
            n_correct+= sum(preds==label).item()

            n_total+=len(label)

            running_loss += loss.item()

        print(f"Accuracy:{((n_correct/n_total)*100):.2f}")
        print(f"Accumulated Loss:{(running_loss / len(test_loader)):.4f}")



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    test()
