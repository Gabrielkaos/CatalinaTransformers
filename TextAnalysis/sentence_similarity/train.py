import torch
# from data_cleaning import get_dialogue_data_for_transformer
from torch.utils.data import DataLoader, Dataset
from MODEL_TRANSFORMER import build_transformer_encoder
import torch.nn as nn
import warnings
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler


class TransformerData(Dataset):
    def __init__(self, x, label, segment, src_tokenizer):
        super().__init__()

        self.x = x
        self.segment = segment
        self.label = label
        self.pad_token = torch.tensor([src_tokenizer.get("<PAD>")], dtype=torch.int32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        segment = self.segment[item]
        label = self.label[item]

        return {
            "encoder_input": x,
            "encoder_mask": (x != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "label": label,
            "segment": segment
        }


def save_file(name, model:nn.Module, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.LinearLR):
    data = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(data, name)
    print("MODEL SAVED")


def count_parameters(model1):
    return sum(p.numel() for p in model1.parameters() if p.requires_grad)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()
    print(f"Device={device}")
    if str(device) == "cuda":
        print(f"Device memory={(torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3):.3f} GB")

    # Hyperparameters
    batch_size = 200
    max_seq_src = 100
    lr = 3e-4
    num_epochs = 10

    data = torch.load("data.pth")
    x = data["x"]
    label = data["label"]
    src_vocab = data["src_vocab"]
    tokenizer_src = data["tokenizer_src"]
    num_labels = data["num_labels"]
    segment_ids = data["segment_ids"]
    print(len(segment_ids))

    print(f"x={len(x)}")
    print(f"src_vocab={len(src_vocab)}")
    

    # Data loader
    train_loader = DataLoader(TransformerData(x, label, segment_ids, tokenizer_src), batch_size=batch_size, shuffle=True)
    print("Data processed")

    # Model
    model = build_transformer_encoder(len(src_vocab), num_labels, max_seq_src, 
                                      device=device, is_segmented=True).to(device)
    print(count_parameters(model))
    model.train()

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-9, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,end_factor=0.01, total_iters=num_epochs * len(train_loader))
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Training loop
    scaler = GradScaler(device=device)
    print("\nStarting training...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            
            optimizer.zero_grad()
            
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            label = batch["label"].to(device)
            segment_id = batch["segment"].to(device)

            # print(encoder_input)
            print(segment_id.shape)

            with autocast(device_type='cuda'):
                encoder_output = model.encode(encoder_input, encoder_mask, segments=segment_id)
                proj_output = model.project(encoder_output)
                proj_output_collapsed = proj_output[:,0,:]
                loss = criterion(proj_output_collapsed, label)

            running_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if (batch_idx + 1) % 2 == 0:
                print(
                        f"[TRAIN] Epoch:{epoch + 1}/{num_epochs}, Batch:{batch_idx + 1}/{len(train_loader)}, Loss:{loss.item():.4f}"
                    )

        avg_loss = running_loss / len(train_loader)
        print(f"[EPOCH] Epoch:{epoch + 1}/{num_epochs}, Avg Loss:{avg_loss:.4f}")

        # Save model at the end of each epoch
        save_file(f"epoch/{epoch + 1}-{avg_loss:.4f}.pth", model, optimizer, scheduler)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
