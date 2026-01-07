import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.amp import autocast, GradScaler
import time
from pathlib import Path
from tqdm import tqdm
from MODEL_TRANSFORMER import build_transformer_encoder
import math


class EmotionDataset(Dataset):
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


def compute_metrics(logits, labels, num_classes):
    
    
    
    if labels.dim() > 1 and labels.size(-1) > 1:
        
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        
        
        predictions = (torch.sigmoid(logits) > 0.5).float()
        accuracy = (predictions == labels).float().mean()
        
    else:
        
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        
        loss = nn.functional.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == labels).float().mean()
    
    return {
        "loss": loss.item(),
        "accuracy": accuracy.item()
    }


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
            logits = model(x, mask)
            loss = criterion(logits, y)
        
        
        if is_multilabel:
            predictions = (torch.sigmoid(logits) > 0.5).float()
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
            logits = model(x, mask)
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
                predictions = (torch.sigmoid(logits) > 0.5).float()
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


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0):
   
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
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


def train():
    # ========== Configuration ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    
    config = {
        "vocab_size": None,
        "num_classes": None,
    }
    
    
    batch_size = 32
    gradient_accumulation_steps = 4  
    lr = 3e-4
    weight_decay = 0.01
    epochs = 10
    warmup_steps = 500
    max_grad_norm = 1.0
    
    
    val_split = 0.1
    val_every = 1
    
    
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    save_every = 2
    resume_from = None  
    
    # ========== Load Data ==========
    print("\nLoading data...")
    data = torch.load("data_simplified.pth")
    sequences = data["x"]
    labels = data["label"]
    vocab = data["vocab"]
    tokenizer = data["tokenizer"]
    num_classes = data["num_classes"]
    pad_idx = tokenizer.get("<PAD>")
    
    config["vocab_size"] = len(vocab)
    config["num_classes"] = num_classes
    
    print(f"Vocab size: {len(vocab)}")
    print(f"Num classes: {num_classes}")
    print(f"Total sequences: {len(sequences)}")
    
    
    is_multilabel = labels[0].dim() > 0 and labels[0].size(-1) > 1
    print(f"Task type: {'Multi-label' if is_multilabel else 'Multi-class'} classification")
    
   
    dataset = EmotionDataset(sequences, labels, pad_idx)
    
    
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
    model = build_transformer_encoder_classifier(
        vocab_size=config["vocab_size"],
        num_classes=config["num_classes"],
        seq_length=config["seq_length"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        dff=config["dff"],
        dropout=config["dropout"],
        use_flash_attn=config["use_flash_attn"],
        device=device
    ).to(device)
    
   
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    
    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # ========== Setup Training ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        eps=1e-8
    )
    
    
    if is_multilabel:
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCEWithLogitsLoss for multi-label classification")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss for multi-class classification")
    
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=lr * 0.1
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
                best_path = save_dir / "best_model.pth"
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch + 1,
                    {"train": train_metrics, "val": val_metrics, "best_val_acc": best_val_acc},
                    best_path
                )
                print(f"✓ Saved best model (val_acc: {best_val_acc:.4f})")
        
        
        if (epoch + 1) % save_every == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1,
                {"train": train_metrics, "best_val_acc": best_val_acc},
                checkpoint_path
            )
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        
        latest_path = save_dir / "latest.pth"
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch + 1,
            {"train": train_metrics, "best_val_acc": best_val_acc},
            latest_path
        )
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("="*60)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    train()