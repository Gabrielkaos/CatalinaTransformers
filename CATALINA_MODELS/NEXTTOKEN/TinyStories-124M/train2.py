import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.amp import autocast, GradScaler
import math
import time
from pathlib import Path
from tqdm import tqdm
# from MODEL_TRANSFORMER import gpt2_like_model

def freeze_for_dialogue(model, freeze_until_layer=8):
    """
    freeze_until_layer:
      GPT-2 small (12 layers): 8
      GPT-2 medium (24 layers): 18
    """

    for i, layer in enumerate(model.decoder.layers):
        if i < freeze_until_layer:
            for name, param in layer.named_parameters():
                # keep LayerNorms trainable
                if "norm" in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # top layers fully trainable
            for param in layer.parameters():
                param.requires_grad = True

    # Freeze embeddings
    for param in model.embed.parameters():
        param.requires_grad = False
    for param in model.pos.parameters():
        param.requires_grad = False

    # Final LayerNorm stays trainable
    for param in model.decoder.norm.parameters():
        param.requires_grad = True

    # Print stats
    # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total = sum(p.numel() for p in model.parameters())
    # print(f"Trainable params: {trainable:_}/{total:_} ({100*trainable/total:.2f}%)")


class LanguageModelDataset(Dataset):
    def __init__(self, inputs,labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]
        return {
            "input": x,
            "label": y
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


def compute_metrics(logits, labels, pad_idx):
    
    
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    
    
    loss = nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=pad_idx)
    
    
    perplexity = torch.exp(loss)
    
    
    predictions = logits_flat.argmax(dim=-1)
    mask = labels_flat != pad_idx
    correct = (predictions == labels_flat) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return {
        "loss": loss.item(),
        "perplexity": perplexity.item(),
        "accuracy": accuracy.item()
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    
    model.eval()
    total_loss = 0
    total_perplexity = 0
    total_accuracy = 0
    num_batches = 0
    
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["label"].to(device)
       
        with autocast(device_type=device.type, enabled=(device.type == "cuda"),dtype=torch.bfloat16):
            logits = model(x)
            criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        
        metrics = compute_metrics(logits, y, criterion.ignore_index)
        total_loss += metrics["loss"]
        total_perplexity += metrics["perplexity"]
        total_accuracy += metrics["accuracy"]
        num_batches += 1
    
    return {
        "loss": total_loss / num_batches,
        "perplexity": total_perplexity / num_batches,
        "accuracy": total_accuracy / num_batches
    }


def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, 
                gradient_accumulation_steps=1, max_grad_norm=1.0):
   
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        x = batch["input"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
       
        with autocast(device_type=device.type, enabled=(device.type == "cuda"),dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
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
        
        
        total_loss += loss.item() * gradient_accumulation_steps
        total_tokens += (y != criterion.ignore_index).sum().item()
        
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'lr': f'{current_lr:.2e}'
        })
    
    
    elapsed_time = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed_time
    
    return {
        "loss": total_loss / len(loader),
        "tokens_per_sec": tokens_per_sec,
        "time": elapsed_time
    }


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
    torch.set_float32_matmul_precision('high')
    
   
    config = {
        "vocab_size": None,
        "d_model":768,
        "n_layers":12,
        "n_heads":12,
        "dropout":0.1,
        "mlp_activation":"gelu"
    }
    
   
    batch_size = 12
    gradient_accumulation_steps = 10 
    lr = 5e-5
    weight_decay = 0.001
    epochs = 2
    warmup_steps = 1000
    max_grad_norm = 1.0
    
    
    val_split = 0.1  
    val_every = 1  
    
   
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    save_every = 5
    resume_from = None  
    
    # ========== Load Data ==========
    print("Loading data...")
    data = torch.load("data.pth")
    inputs = data["x"]
    labels = data["y"]
    vocab = 50257
    # tokenizer = tiktoken.get_encoding("gpt2")
    
    config["vocab_size"] = vocab
    print(f"Vocab size: {vocab}")
    print(f"Total sequences: {len(inputs)}")
    
   
    dataset = LanguageModelDataset(inputs,labels)
    
    
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True
    )
    
    # ========== Build Model ==========
    print("Building model...")
    model = gpt2_like_model(**config).to(device)
    model.load_state_dict(torch.load("gpt2-124M.pth",map_location=device)["model_state"])
    print("Model loaded successfully!")
    
    #freeze some layers
    freeze_for_dialogue(model,freeze_until_layer=4)

    # Wrap with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total parameters: {total_params:,}")
    # print(f"Trainable parameters: {trainable_params:,}")
    
    print("Compiling...")
    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # ========== Setup Training ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95), 
        weight_decay=weight_decay
    )
    
    #lets ignore the pad idx or eos_token
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
   
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1*warmup_steps),
        num_training_steps=total_steps
    )
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from {resume_from}")
        start_epoch, metrics = load_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, device
        )
        best_val_loss = metrics.get("best_val_loss", float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # ========== Training Loop ==========
    print("\nStarting training...")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}\n")
    
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*50}")
        
      
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            scaler, device, gradient_accumulation_steps, max_grad_norm
        )
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f} | "
              f"Throughput: {train_metrics['tokens_per_sec']:.0f} tokens/s | "
              f"Time: {train_metrics['time']:.1f}s")
        
       
        if (epoch + 1) % val_every == 0:
            print("\nValidating...")
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            print(f"Val - Loss: {val_metrics['loss']:.4f} | "
                  f"Perplexity: {val_metrics['perplexity']:.2f} | "
                  f"Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_path = "best_model.pth"
                print("Saving best model...")
                torch.save({"model_state":model.state_dict()}, best_path)
                print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            print("Saving checkpoint...")
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1,
                {"train": train_metrics, "best_val_loss": best_val_loss},
                checkpoint_path
            )
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Always save latest
        latest_path = "latest.pth"
        print("Saving latest...")
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch + 1,
            {"train": train_metrics, "best_val_loss": best_val_loss},
            latest_path
        )
        print("Saved latest")

    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*50)


if __name__ == '__main__':
    train()