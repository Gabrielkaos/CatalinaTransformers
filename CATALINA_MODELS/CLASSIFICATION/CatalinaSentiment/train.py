import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.amp import autocast, GradScaler
import time
from pathlib import Path
from tqdm import tqdm
# from MODEL_TRANSFORMER.gpt_architecture import gpt_classifier
import math
from transformers import GPT2LMHeadModel


class CustomDataset(Dataset):
    def __init__(self, sequences, labels, pad_idx):
        self.sequences = sequences
        self.labels = labels
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        mask = (seq != self.pad_idx)

        return {
            "input": seq,
            "mask": mask,
            "label": label
        }



def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, filepath):

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict(),
        "metrics": metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    if scaler and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_multilabel=False):

    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        x = batch["input"].to(device)
        mask = batch["mask"].to(device)
        y = batch["label"].to(device)


        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            hidden = model(x,mask=mask,return_hidden=True)   # [B, T, D]
            mask_f = mask.unsqueeze(-1).float()               # [B, T, 1]
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1)  # [B, D]
            logits = model.last_projection(pooled)
            loss = criterion(logits, y)


        if is_multilabel:
            predictions = (torch.sigmoid(logits) > 0.2).float()
            accuracy = (predictions == y).float().mean()
        else:
            if y.dim() > 1:
                y = y.squeeze(-1)
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == y).float().mean()

        total_loss += loss.item()
        total_accuracy += accuracy.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_accuracy / num_batches
    }


def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, device,
                gradient_accumulation_steps=1, max_grad_norm=1.0, is_multilabel=False):

    model.train()
    total_loss = 0
    total_accuracy = 0
    start_time = time.time()

    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        x = batch["input"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            hidden = model(x,mask=mask,return_hidden=True)   # [B, T, D]
            mask_f = mask.unsqueeze(-1).float()               # [B, T, 1]
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1)  # [B, D]
            logits = model.last_projection(pooled)
            # print(logits.shape)
            loss = criterion(logits, y)
            loss = loss / gradient_accumulation_steps


        scaler.scale(loss).backward()


        if (batch_idx + 1) % gradient_accumulation_steps == 0:

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()


        with torch.no_grad():
            if is_multilabel:
                predictions = (torch.sigmoid(logits) > 0.2).float()
                accuracy = (predictions == y).float().mean()
            else:
                if y.dim() > 1:
                    y_flat = y.squeeze(-1)
                else:
                    y_flat = y
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == y_flat).float().mean()


        total_loss += loss.item() * gradient_accumulation_steps
        total_accuracy += accuracy.item()


        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'acc': f'{accuracy.item():.4f}',
            'lr': f'{current_lr:.2e}'
        })

    elapsed_time = time.time() - start_time

    return {
        "loss": total_loss / len(loader),
        "accuracy": total_accuracy / len(loader),
        "time": elapsed_time
    }


def param_groups(model, base_lr, n_layers=12, layer_decay=0.8):
    groups = []
    no_decay = ["bias", "norm"]

    def trainable(params):
        return [p for p in params if p.requires_grad]

    # === embeddings (slowest) ===
    emb_lr = base_lr * (layer_decay ** n_layers)

    groups.append({
        "params": trainable(
            p for n, p in model.embed.named_parameters()
            if not any(nd in n for nd in no_decay)
        ),
        "lr": emb_lr,
        "weight_decay": 0.01
    })

    groups.append({
        "params": trainable(
            p for n, p in model.embed.named_parameters()
            if any(nd in n for nd in no_decay)
        ),
        "lr": emb_lr,
        "weight_decay": 0.0
    })

    # === transformer layers ===
    for i, layer in enumerate(model.decoder.layers):
        depth = n_layers - i - 1
        lr = base_lr * (layer_decay ** depth)

        groups.append({
            "params": trainable(
                p for n, p in layer.named_parameters()
                if not any(nd in n for nd in no_decay)
            ),
            "lr": lr,
            "weight_decay": 0.01
        })

        groups.append({
            "params": trainable(
                p for n, p in layer.named_parameters()
                if any(nd in n for nd in no_decay)
            ),
            "lr": lr,
            "weight_decay": 0.0
        })

    # === classifier head (fastest) ===
    groups.append({
        "params": trainable(model.last_projection.parameters()),
        "lr": base_lr,
        "weight_decay": 0.01
    })

    return groups


def train():
    # ========== Configuration ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


    config = {
        "vocab_size": 50257,
        "num_class": None,
        "d_model" : 768,
        "n_layers" : 12,
        "n_heads" : 12,
        "is_causal" : False,
        "block_size" : 1024,
        "dropout" : 0.2,
        "mlp_activation" : "gelu"
    }


    batch_size = 24
    gradient_accumulation_steps = 10
    lr = 3e-5
    weight_decay = 0.001
    epochs = 4
    max_grad_norm = 1.0


    val_split = 0.1
    val_every = 1


    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    save_every = 5
    resume_from = None

    # ========== Load Data ==========
    print("\nLoading data...")
    data = torch.load("data.pth")
    sequences = data["x"]
    labels = data["label"]

    vocab = 50257
    num_classes = data["num_classes"]
    pad_idx = 50256

    config["num_class"] = num_classes

    print(f"Vocab size: {vocab}")
    print(f"Num classes: {num_classes}")
    print(f"Total sequences: {len(sequences)}")


    is_multilabel = labels[0].dim() > 0 and labels[0].size(-1) > 1
    print(f"Task type: {'Multi-label' if is_multilabel else 'Multi-class'} classification")


    dataset = CustomDataset(sequences, labels, pad_idx)


    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train size: {train_size}, Val size: {val_size}")


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True
    )

    # ========== Build Model ==========
    print("\nBuilding model...")
    model = gpt_classifier(
        **config
    )

    #load gpt2
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = model_hf.state_dict()

    print("Copying gpt2's weights")
    print("Copying embedding...")
    #copy gpt2's embedding
    model.embed.weight.data.copy_(sd_hf["transformer.wte.weight"])
    model.pos.weight.data.copy_(sd_hf["transformer.wpe.weight"])

    #copy gpt2 attention projection
    print("Copying attention...")
    for i in range(config["n_layers"]):
        layer = model.decoder.layers[i]
        #proj
        layer.self_attention.w_o.weight.data.copy_(sd_hf[f"transformer.h.{i}.attn.c_proj.weight"].t())
        layer.self_attention.w_o.bias.data.copy_(sd_hf[f"transformer.h.{i}.attn.c_proj.bias"])

        #attn
        layer.self_attention.c_attn.weight.data.copy_(sd_hf[f"transformer.h.{i}.attn.c_attn.weight"].t())
        layer.self_attention.c_attn.bias.data.copy_(sd_hf[f"transformer.h.{i}.attn.c_attn.bias"])

        #mlp
        layer.feed_forward.linear1.weight.data.copy_(sd_hf[f"transformer.h.{i}.mlp.c_fc.weight"].t())
        layer.feed_forward.linear2.weight.data.copy_(sd_hf[f"transformer.h.{i}.mlp.c_proj.weight"].t())
        layer.feed_forward.linear1.bias.data.copy_(sd_hf[f"transformer.h.{i}.mlp.c_fc.bias"])
        layer.feed_forward.linear2.bias.data.copy_(sd_hf[f"transformer.h.{i}.mlp.c_proj.bias"])


        #copy layer norms
        layer.norm1.weight.data.copy_(sd_hf[f"transformer.h.{i}.ln_1.weight"])
        layer.norm1.bias.data.copy_(sd_hf[f"transformer.h.{i}.ln_1.bias"])
        layer.norm2.weight.data.copy_(sd_hf[f"transformer.h.{i}.ln_2.weight"])
        layer.norm2.bias.data.copy_(sd_hf[f"transformer.h.{i}.ln_2.bias"])

    #last norm copy
    model.decoder.norm.weight.data.copy_(sd_hf["transformer.ln_f.weight"])
    model.decoder.norm.bias.data.copy_(sd_hf["transformer.ln_f.bias"])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:_}/{total:_} ({100*trainable/total:.2f}%)")

    model = model.to(device)

    #freeze embed, unfreeze after one epoch
    for p in model.embed.parameters():
        p.requires_grad = False


    if hasattr(torch, 'compile') and resume_from is None:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ========== Setup Training ==========
    optimizer = torch.optim.AdamW(
        param_groups(model,lr,n_layers=config["n_layers"]),
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        eps=1e-8
    )


    if is_multilabel:
        counts = torch.tensor([
            4129, 2328, 1567, 2470, 2939, 1087, 1368, 2191,
            641, 1269, 2022, 793, 303, 853, 596, 2662,
            77, 1452, 2086, 164, 1581, 111, 1110, 153,
            545, 1326, 1060, 14216
        ], dtype=torch.float)

        N = len(sequences)
        pos_weight = (N - counts) / counts

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        print("Using BCEWithLogitsLoss for multi-label classification")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss for multi-class classification")

    scaler = GradScaler(enabled=(device.type == "cuda"))


    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )


    start_epoch = 0
    best_val_acc = 0.0

    if resume_from and Path(resume_from).exists():
        print(f"\nResuming from {resume_from}")
        start_epoch, metrics = load_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, device
        )
        best_val_acc = metrics.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    # ========== Training Loop ==========
    print("\n" + "="*60)
    print("Starting training...")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print("="*60 + "\n")

    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")

        if epoch==1:
            for p in model.embed.parameters():
                p.requires_grad = True
            
            emb_params = sum(p.numel() for p in model.embed.parameters() if p.requires_grad)
            print(f"Unfroze embeddings: {emb_params:,} params")
        else:
            pass



        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            scaler, device, gradient_accumulation_steps, max_grad_norm,
            is_multilabel
        )

        print(f"\nTrain - Loss: {train_metrics['loss']:.4f} | "
              f"Accuracy: {train_metrics['accuracy']:.4f} | "
              f"Time: {train_metrics['time']:.1f}s")


        if (epoch + 1) % val_every == 0:
            print("\nValidating...")
            val_metrics = evaluate(model, val_loader, criterion, device, is_multilabel)

            print(f"Val - Loss: {val_metrics['loss']:.4f} | "
                  f"Accuracy: {val_metrics['accuracy']:.4f}")


            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_path = "best_model.pth"
                print("Saving best model")
                # save_checkpoint(
                #     model, optimizer, scheduler, scaler, epoch + 1,
                #     {"train": train_metrics, "val": val_metrics, "best_val_acc": best_val_acc},
                #     best_path
                # )
                torch.save({"model_state":model.state_dict()},best_path)
                print(f"✓ Saved best model (val_acc: {best_val_acc:.4f})")


        if (epoch + 1) % save_every == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1,
                {"train": train_metrics, "best_val_acc": best_val_acc},
                checkpoint_path
            )
            print(f"✓ Saved checkpoint: {checkpoint_path}")

        print("SAving latest")
        latest_path = "latest.pth"
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch + 1,
            {"train": train_metrics, "best_val_acc": best_val_acc},
            latest_path
        )
        print("Saved latest")

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("="*60)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    train()