import torch
# from data_cleaning import get_dialogue_data_for_transformer
from torch.utils.data import DataLoader, Dataset
from MODEL_TRANSFORMER import build_transformer_encoder
import torch.nn as nn
import warnings
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler


class TransformerData(Dataset):
    def __init__(self, x, label, src_tokenizer):
        super().__init__()

        self.x = x
        self.label = label
        self.pad_token = torch.tensor([src_tokenizer.get("<PAD>")], dtype=torch.int32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        label = self.label[item]

        return {
            "encoder_input": x,
            "encoder_mask": (x != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "label": label
        }


def count_parameters(model1):
    return sum(p.numel() for p in model1.parameters() if p.requires_grad)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()
    print(f"Device={device}")
    if str(device) == "cuda":
        print(f"Device memory={(torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3):.3f} GB")

    # Hyperparameters
    batch_size = 100
    max_seq_src = 100

    data = torch.load("data_simplified.pth")
    test_data = torch.load("test_data.pth")
    x = test_data["x"]
    label = test_data["label"]
    src_vocab = data["src_vocab"]
    tokenizer_src = data["tokenizer_src"]
    num_labels = data["num_labels"]

    print(f"x={len(x)}")
    print(f"src_vocab={len(src_vocab)}")
    print("Data processed")

    # Data loader
    test_loader = DataLoader(TransformerData(x, label, tokenizer_src), batch_size=batch_size, shuffle=False)

    # Model
    model = build_transformer_encoder(len(src_vocab), num_labels, max_seq_src, device=device, dropout=0.05, 
                                      n_layers=15, 
                                      n_heads=15, 
                                      d_model=900,
                                      dff=2600).to(device)
    print(count_parameters(model))
    model.load_state_dict(torch.load("D://Catalina/Emotion/checkpoint/CHECKPOINT-4-800.pth")["model_state"])
    model.eval()

    # Optimizer and loss function
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Training loop
    print("\nStarting testing...")
    with torch.no_grad():
        running_loss = 0.0
        for i, batch in enumerate(test_loader):
            
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            label = batch["label"].to(device)

            with autocast(device_type='cuda'):
                encoder_output = model.encode(encoder_input, encoder_mask)
                proj_output = model.project(encoder_output)
                proj_output_collapsed = proj_output[:, 0, :]
                loss = criterion(proj_output_collapsed, label)

            running_loss += loss.item()


            if (i + 1) % 2 == 0:
                print(f"Batch:{i + 1}/{len(test_loader)}")
            

        avg_loss = running_loss / len(test_loader)
        print(f"Avg Loss:{avg_loss:.4f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
