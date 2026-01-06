import torch
import torch.nn.functional as F
from MODEL_TRANSFORMER import build_transformer_next_token
from unidecode import unidecode
from tqdm import tqdm

def causal_mask(size, device):
    
    return torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))


def sample_token(logits, temperature=1.0, top_k=None, top_p=None):
    
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
    
    
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token


@torch.no_grad()
def generate(
    model, 
    tokenizer, 
    prompt, 
    max_len=50,
    seq_len=256, 
    device="cpu",
    temperature=1.0,
    top_k=None,
    top_p=0.9,
    repetition_penalty=1.0
):
    
    model.eval()
    
    bow = list(unidecode(prompt))
    token_ids = [tokenizer.get(i,tokenizer.get("<PAD>")) for i in bow]
    
   
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    predicted = []
    
    for step in tqdm(range(max_len)):
        current_len = x.size(1)
        
        
        if current_len > seq_len:
            x = x[:, -seq_len:]
            current_len = seq_len
        
        
        if step == 0 or current_len != prev_len:
            mask = causal_mask(current_len, device).unsqueeze(0)
            prev_len = current_len
        
        logits = model(x, mask)
        next_token_logits = logits[:, -1, :]
        
        
        if repetition_penalty != 1.0:
            for token_id in set(x[0].tolist()):
                next_token_logits[:, token_id] /= repetition_penalty
        
        
        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
           
            next_token = sample_token(
                next_token_logits, 
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        
        if next_token.item() == tokenizer.get("<EOS>"):
            break
        
        
        x = torch.cat([x, next_token], dim=1)
    
    
    decoder = {idx:char for char,idx in tokenizer.items()}
    decoded_tokens = "".join(decoder[i] for i in x[0].tolist())


    return decoded_tokens

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    
    data = torch.load("data.pth", map_location=device)
    vocab = data["vocab"]
    tokenizer = data["tokenizer"]
    
    
    config = {
        "vocab_size": None,
        "d_model":640,
        "n_layers":10,
        "n_heads":640//64,
        "dff":640*4,
        "dropout":0.2
    }
    
    config["vocab_size"] = len(vocab)
    model = build_transformer_next_token(**config).to(device)
    print(f"Model vocab:{model.embed.vocab_size}")
    print(f"Data vocab:{len(vocab)}")
    try:
        checkpoint = torch.load("best_model.pth", map_location=device)
        state_dict = checkpoint["model_state"]

        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_k = k[len("_orig_mod."):]
            else:
                new_k = k
            new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict) 
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
    
    
    # print("\n=== Greedy Decoding ===")
    # print("Prompt:",prompt,"\n")
    # output = generate(
    #     model, tokenizer, prompt, 
    #     max_len=100,seq_len=256, 
    #     device=device,
    #     temperature=0.1,
    #     repetition_penalty=1.2 
    # )
    # print(output)
    while True:
        print("\n\n")
        prompt = input("prompt:")
        
        
        
        # print("\n=== Sampling (temp=0.8, top_p=0.9) ===")
        # print("Prompt:",prompt,"\n")
        print("response:\n")
        output = generate(
            model, tokenizer, prompt,
            max_len=300,seq_len=512,
            device=device,
            temperature=0.0,
            top_p=0.9,
            repetition_penalty=1.2
        )
        print(output)