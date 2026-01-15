import torch
import torch.nn.functional as F
from MODEL_TRANSFORMER.gpt_architecture import gpt2_like_model
import tiktoken
from transformers import GPT2LMHeadModel


# def freeze_bottom_layers(model, n_layers_to_freeze=12):
#     if not hasattr(model, 'decoder') or not hasattr(model.decoder, 'layers'):
#         print("Warning: Model structure not recognized. Cannot freeze layers.")
#         return
    
#     total_layers = len(model.decoder.layers)
#     n_layers_to_freeze = min(n_layers_to_freeze, total_layers)
    
#     for i in range(n_layers_to_freeze):
#         layer = model.decoder.layers[i]
#         print(f"Freezing {i} layer")
#         for param in layer.parameters():
#             param.requires_grad = False
    
#     print(f"Froze bottom {n_layers_to_freeze}/{total_layers} layers")
    
#     # Show trainable parameter count after freezing
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Trainable parameters: {trainable_params:_}/{total_params:_} ({100*trainable_params/total_params:.1f}%)")


# def freeze_embeddings(model):
#     if hasattr(model, 'embed'):
#         print("Embeddings frozen")
#         for param in model.embed.parameters():
#             param.requires_grad = False
#     if hasattr(model, 'pos'):
#         print("Position Embeddings frozen")
#         for param in model.pos.parameters():
#             param.requires_grad = False


# def freeze_bottom_and_embeddings(model, n_layers_to_freeze=12):
#     freeze_embeddings(model)
#     freeze_bottom_layers(model, n_layers_to_freeze)
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:_}/{total:_} ({100*trainable/total:.2f}%)")


@torch.no_grad()
def generate_story(
    model, 
    tokenizer: tiktoken.Encoding, 
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    stop_sequences: list = None,
    device: str = "cpu",
    verbose: bool = False
):
    """
    Enhanced story generation with better sampling controls.
    
    Args:
        model: The trained transformer model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Starting prompt for the story
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Only sample from top k tokens
        top_p: Nucleus sampling (cumulative probability threshold)
        stop_sequences: List of strings that stop generation
        device: Device to run on
        verbose: Print generation steps
    
    Returns:
        Generated story text
    """
    model.eval()
    
    # Default stop sequences for stories
    if stop_sequences is None:
        stop_sequences = ["\n\n", "<|endoftext|>", "The End", "THE END"]
    
    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    generated_tokens = []
    stop_reached = False
    
    # Track recent tokens for repetition penalty
    recent_tokens = []
    max_recent = 20  # Look back window for repetition penalty
    
    if verbose:
        print(f"Starting generation with prompt: '{prompt}'")
        print(f"Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}")
    
    for step in range(max_tokens):
        # Get model predictions
        logits = model(input_tensor)
        logits = logits[:, -1, :]  # Last token's logits
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
        
        # Apply top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('Inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Decode and check for stop sequences
        next_token_id = next_token.item()
        generated_tokens.append(next_token_id)
        
        # Update recent tokens for repetition penalty
        recent_tokens.append(next_token_id)
        if len(recent_tokens) > max_recent:
            recent_tokens.pop(0)
        
        # Append token to input for next iteration
        input_tensor = torch.cat([input_tensor, next_token], dim=1)
        
        # Decode full sequence so far to check for stop sequences
        current_text = tokenizer.decode(token_ids + generated_tokens)
        
        # Check for stop sequences
        for stop_seq in stop_sequences:
            if stop_seq in current_text[len(prompt):]:
                if verbose:
                    print(f"Stopping generation: found '{stop_seq}'")
                stop_reached = True
                break
        
        if stop_reached:
            break
        
        # Print progress if verbose
        if verbose and step % 10 == 0:
            new_text = tokenizer.decode([next_token_id])
            print(f"Step {step}: '{new_text}'", end="", flush=True)
    
    # Combine prompt and generated tokens
    full_tokens = token_ids + generated_tokens
    full_text = tokenizer.decode(full_tokens)
    
    # Clean up the output (optional)
    # full_text = clean_story_text(full_text, prompt)
    
    return full_text


def clean_story_text(text: str, original_prompt: str) -> str:
    """
    Clean up generated story text.
    """
    # Remove any trailing incomplete sentences
    sentences = text.split('. ')
    if len(sentences) > 1 and not sentences[-1].endswith('.'):
        text = '. '.join(sentences[:-1]) + '.'
    
    # Ensure the prompt is included exactly as given
    if text.startswith(original_prompt):
        return text
    
    # Sometimes tokenizer adds spaces - handle this
    prompt_clean = original_prompt.strip()
    text_clean = text.strip()
    
    if text_clean.startswith(prompt_clean):
        return text
    
    # If not, prepend the prompt
    return original_prompt + text




# Example usage function
def interactive_story_generation(model, tokenizer, device="cpu"):
    """
    Interactive mode for story generation.
    """
    print("\nStory Generator")
    print("Type 'quit' to exit")
    print("Type 'settings' to change generation parameters")
    
    # Default parameters
    params = {
        'max_tokens': 20,
        'temperature': 0.7,
        'top_k': 40,
        'top_p': 0.85
    }
    
    while True:
        print("\n" + "="*60)
        prompt = input("\nEnter a story prompt: ").strip()
        
        if prompt.lower() == 'quit':
            break
        elif prompt.lower() == 'settings':
            print("\nCurrent settings:")
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            change = input("\nChange a setting? (e.g., 'temperature 0.8' or 'no'): ")
            if change.lower() != 'no':
                try:
                    key, value = change.split()
                    if key in params:
                        params[key] = float(value) if '.' in value else int(value)
                        print(f"Updated {key} to {params[key]}")
                except:
                    print("Invalid format. Use: 'parameter value'")
            continue
        
        print("\nGenerating story...")
        
        story = generate_story(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            **params
        )
        
        print("\n" + "="*60)
        print("📖 Generated Story:")
        print("="*60)
        print(story)
        print("="*60)
        
        
        while True:
            continue_story = input("\nContinue this story? (yes/no): ").lower()
            if continue_story != "yes":break

            print("\nContinuing the story...")
            continuation = generate_story(
                model=model,
                tokenizer=tokenizer,
                prompt=story,
                device=device,
                **params
            )
            print("\n" + continuation)
            story = continuation
            
            



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    
    vocab = 50257
    tokenizer = tiktoken.get_encoding("gpt2")
    
    
    config = {
        "vocab_size": None,
        "d_model":768,
        "n_layers":12,
        "n_heads":12,
        "dropout":0.1,
        "mlp_activation":"gelu"
    }
        
    config["vocab_size"] = vocab
    model = gpt2_like_model(**config).to(device)
    
    print(f"Data vocab:{vocab}")
    try:
        # data_model=torch.load("best_model.pth",map_location="cpu")
        # model.load_state_dict(data_model["model_state"])
        
        checkpoint = torch.load("best_model.pth", map_location=device)
        state_dict = checkpoint["model_state"]

        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                k = k[len("_orig_mod."):]
            if k.startswith("module."):
                k = k[len("module."):]
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict) 

        #load gpt2
        # model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        # sd_hf = model_hf.state_dict()
        
        # print("Copying gpt2's weights")
        # print("Copying embedding...")
        # #copy gpt2's embedding
        # model.embed.weight.data.copy_(sd_hf["transformer.wte.weight"])
        # model.pos.weight.data.copy_(sd_hf["transformer.wpe.weight"])
        
        # #copy gpt2 attention projection
        # print("Copying attention...")
        # for i in range(config["n_layers"]):
        #     layer = model.decoder.layers[i]
        #     #proj
        #     layer.self_attention.w_o.weight.data.copy_(sd_hf[f"transformer.h.{i}.attn.c_proj.weight"].t())
        #     layer.self_attention.w_o.bias.data.copy_(sd_hf[f"transformer.h.{i}.attn.c_proj.bias"])
            
        #     #attn
        #     layer.self_attention.c_attn.weight.data.copy_(sd_hf[f"transformer.h.{i}.attn.c_attn.weight"].t())
        #     layer.self_attention.c_attn.bias.data.copy_(sd_hf[f"transformer.h.{i}.attn.c_attn.bias"])
            
        #     #mlp
        #     if config["mlp_activation"]=="gelu":
        #         layer.feed_forward.linear1.weight.data.copy_(sd_hf[f"transformer.h.{i}.mlp.c_fc.weight"].t())
        #         layer.feed_forward.linear2.weight.data.copy_(sd_hf[f"transformer.h.{i}.mlp.c_proj.weight"].t())
        #         layer.feed_forward.linear1.bias.data.copy_(sd_hf[f"transformer.h.{i}.mlp.c_fc.bias"])
        #         layer.feed_forward.linear2.bias.data.copy_(sd_hf[f"transformer.h.{i}.mlp.c_proj.bias"])
        #     # transformer.h.9.mlp.c_fc.weight

        #     #copy layer norms
        #     layer.norm1.weight.data.copy_(sd_hf[f"transformer.h.{i}.ln_1.weight"])
        #     layer.norm1.bias.data.copy_(sd_hf[f"transformer.h.{i}.ln_1.bias"])
        #     layer.norm2.weight.data.copy_(sd_hf[f"transformer.h.{i}.ln_2.weight"])
        #     layer.norm2.bias.data.copy_(sd_hf[f"transformer.h.{i}.ln_2.bias"])

        # #last norm copy
        # model.decoder.norm.weight.data.copy_(sd_hf["transformer.ln_f.weight"])
        # model.decoder.norm.bias.data.copy_(sd_hf["transformer.ln_f.bias"])

        
        print("Model loaded successfully!")

    except FileNotFoundError:
        print("Warning: Model checkpoint not found. Using random weights.")
    except KeyError:
        print("Warning: Checkpoint format incorrect. Expected 'model_state' key.")


    model.eval()
    # freeze_for_dialogue(model,freeze_until_layer=8)

    interactive_story_generation(model,tokenizer)