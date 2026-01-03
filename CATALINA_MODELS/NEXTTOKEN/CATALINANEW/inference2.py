import torch
import torch.nn.functional as F
from data_cleaning import split_string_with_special_characters
from MODEL_TRANSFORMER import build_transformer_next_token


def causal_mask(size, device):
    """Create causal mask once and reuse."""
    return torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))


def sample_token(logits, temperature=1.0, top_k=None, top_p=None):
    """
    Sample next token with various strategies.
    
    Args:
        logits: Raw logits from model (batch_size, vocab_size)
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k tokens
        top_p: If set, nucleus sampling with cumulative probability p
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Top-k filtering
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    # Top-p (nucleus) filtering
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    # Sample from distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token


@torch.no_grad()
def generate(
    model, 
    tokenizer, 
    prompt, 
    max_len=50, 
    device="cpu",
    temperature=1.0,
    top_k=None,
    top_p=0.9,
    repetition_penalty=1.0
):
    """
    Generate text from prompt with optimized inference.
    
    Args:
        model: Trained transformer model
        tokenizer: Token to ID mapping dict
        prompt: Input text string
        max_len: Maximum tokens to generate
        device: Device to run on
        temperature: Sampling temperature (0.1-2.0)
        top_k: Top-k sampling (e.g., 50)
        top_p: Nucleus sampling threshold (e.g., 0.9)
        repetition_penalty: Penalty for repeating tokens (>1.0 reduces repetition)
    """
    model.eval()
    
    # Build reverse tokenizer once
    id_to_token = {v: k for k, v in tokenizer.items()}
    
    # Tokenize prompt
    tokens = split_string_with_special_characters(prompt.lower())
    tokens = ["<SOS>"] + tokens
    ids = [tokenizer.get(t, tokenizer.get("<UNK>", tokenizer["<PAD>"])) for t in tokens]
    
    # Initialize sequence
    x = torch.tensor([ids], dtype=torch.long, device=device)
    seq_len = model.pos.seq_len if hasattr(model, 'pos') else 512
    
    # Generate tokens
    for step in range(max_len):
        current_len = x.size(1)
        
        # Truncate if exceeds max sequence length
        if current_len > seq_len:
            x = x[:, -seq_len:]
            current_len = seq_len
        
        # Create mask (reuse if same size)
        if step == 0 or current_len != prev_len:
            mask = causal_mask(current_len, device).unsqueeze(0)
            prev_len = current_len
        
        # Forward pass
        logits = model(x, mask)
        next_token_logits = logits[:, -1, :]
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(x[0].tolist()):
                next_token_logits[:, token_id] /= repetition_penalty
        
        # Sample next token
        if temperature == 0.0:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # Stochastic sampling
            next_token = sample_token(
                next_token_logits, 
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Check for end of sequence
        if next_token.item() == tokenizer.get("<EOS>"):
            break
        
        # Append token
        x = torch.cat([x, next_token], dim=1)
    
    # Decode tokens efficiently
    decoded_tokens = [id_to_token.get(i, "<UNK>") for i in x[0].tolist()]
    
    # Remove special tokens for cleaner output
    decoded_tokens = [t for t in decoded_tokens if t not in ["<SOS>", "<PAD>"]]
    
    return " ".join(decoded_tokens)


def generate_batch(
    model,
    tokenizer,
    prompts,
    max_len=50,
    device="cpu",
    temperature=1.0,
    top_k=None,
    top_p=0.9
):
    """Generate text for multiple prompts in parallel."""
    model.eval()
    
    id_to_token = {v: k for k, v in tokenizer.items()}
    batch_size = len(prompts)
    
    # Tokenize all prompts
    all_ids = []
    for prompt in prompts:
        tokens = split_string_with_special_characters(prompt.lower())
        tokens = ["<SOS>"] + tokens
        ids = [tokenizer.get(t, tokenizer.get("<UNK>", tokenizer["<PAD>"])) for t in tokens]
        all_ids.append(ids)
    
    # Pad to same length
    max_prompt_len = max(len(ids) for ids in all_ids)
    padded = []
    for ids in all_ids:
        padded.append(ids + [tokenizer["<PAD>"]] * (max_prompt_len - len(ids)))
    
    x = torch.tensor(padded, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_len):
            mask = causal_mask(x.size(1), device).unsqueeze(0).expand(batch_size, -1, -1)
            logits = model(x, mask)
            next_token_logits = logits[:, -1, :]
            
            if temperature == 0.0:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_tokens = sample_token(
                    next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
            
            # Mark finished sequences
            eos_id = tokenizer.get("<EOS>")
            if eos_id is not None:
                finished |= (next_tokens.squeeze(1) == eos_id)
            
            x = torch.cat([x, next_tokens], dim=1)
            
            if finished.all():
                break
    
    # Decode all sequences
    results = []
    for seq in x:
        decoded = [id_to_token.get(i.item(), "<UNK>") for i in seq]
        decoded = [t for t in decoded if t not in ["<SOS>", "<PAD>"]]
        results.append(" ".join(decoded))
    
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    data = torch.load("data.pth", map_location=device)
    vocab = data["vocab"]
    tokenizer = data["tokenizer"]
    
    # Model configuration (MUST match training config!)
    config = {
        "vocab_size": len(vocab)
    }
    
    # Build and load model
    model = build_transformer_next_token(**config).to(device)
    
    try:
        checkpoint = torch.load("checkpoints/checkpoint_epoch_29.pth", map_location=device)
        state_dict = checkpoint["model_state"]

        # Handle compiled checkpoints: remove the `_orig_mod.` prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_k = k[len("_orig_mod."):]
            else:
                new_k = k
            new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict)  # optionally strict=False if needed
        print("Model loaded successfully!")

    except FileNotFoundError:
        print("Warning: Model checkpoint not found. Using random weights.")
    except KeyError:
        print("Warning: Checkpoint format incorrect. Expected 'model_state' key.")
    
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    prompt = "Who are you?"
    # Example: Greedy decoding (deterministic)
    print("\n=== Greedy Decoding ===")
    print("Prompt:",prompt,"\n")
    output = generate(
        model, tokenizer, prompt, 
        max_len=100, 
        device=device,
        temperature=0.0  # Greedy
    )
    print(output)
    
    # Example: Creative sampling
    print("\n=== Sampling (temp=0.8, top_p=0.9) ===")
    print("Prompt:",prompt,"\n")
    output = generate(
        model, tokenizer, prompt,
        max_len=100,
        device=device,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2
    )
    print(output)
    
    # # Example: Batch generation
    # print("\n=== Batch Generation ===")
    # prompts = ["We", "The", "In"]
    # outputs = generate_batch(
    #     model, tokenizer, prompts,
    #     max_len=50,
    #     device=device,
    #     temperature=0.8
    # )
    # for prompt, output in zip(prompts, outputs):
    #     print(f"{prompt}: {output}")