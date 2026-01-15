import torch
import torch.nn.functional as F
from MODEL_TRANSFORMER.gpt_architecture import gpt2_like_model
import tiktoken
from transformers import GPT2LMHeadModel



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
    
    model.eval()
    
    if stop_sequences is None:
        stop_sequences = ["\n\n", "<|endoftext|>", "The End", "THE END"]
    
    token_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    generated_tokens = []
    stop_reached = False
    
    if verbose:
        print(f"Starting generation with prompt: '{prompt}'")
        print(f"Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}")
    
    for step in range(max_tokens):
        logits = model(input_tensor)
        logits = logits[:, -1, :]  
        
        
        if temperature != 1.0:
            logits = logits / temperature
        
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        next_token_id = next_token.item()
        generated_tokens.append(next_token_id)
        
        input_tensor = torch.cat([input_tensor, next_token], dim=1)
        
        current_text = tokenizer.decode(token_ids + generated_tokens)
        
        for stop_seq in stop_sequences:
            if stop_seq in current_text[len(prompt):]:
                if verbose:
                    print(f"Stopping generation: found '{stop_seq}'")
                stop_reached = True
                break
        
        if stop_reached:
            break
        
        if verbose and step % 10 == 0:
            new_text = tokenizer.decode([next_token_id])
            print(f"Step {step}: '{new_text}'", end="", flush=True)
    
    full_tokens = token_ids + generated_tokens
    full_text = tokenizer.decode(full_tokens)
    if verbose:
        print("Initial text before cleaning")
        print(full_text)
    
    full_text = clean_story_text(full_text, prompt)
    
    return full_text


def clean_story_text(text: str, original_prompt: str) -> str:
    
    sentences = text.split('. ')
    if len(sentences) > 1 and not sentences[-1].endswith('.'):
        text = '. '.join(sentences[:-1]) + '.'
    
    
    if text.startswith(original_prompt):
        return text
    
    
    prompt_clean = original_prompt.strip()
    text_clean = text.strip()
    
    if text_clean.startswith(prompt_clean):
        return text
    
    return original_prompt + text


# Example usage function
def interactive_story_generation(model, tokenizer, device="cpu"):
    
    print("\nStory Generator")
    print("Type 'quit' to exit")
    print("Type 'settings' to change generation parameters")
    
    params = {
        'max_tokens': 256,
        'temperature': 1.0,
        'top_k': 50,
        'top_p': 1.0
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
                    if key in params.keys():
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
        print("Generated Story:")
        print("="*60)
        print(story)
        print("="*60)
        
        continue_story = input("\nContinue this story? (yes/no): ").lower()
        if continue_story in ['yes', 'y']:
            print("\nContinuing the story...")
            continuation = generate_story(
                model=model,
                tokenizer=tokenizer,
                prompt=story,
                device=device,
                **params
            )
            print("\n" + continuation[len(story):])

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
        
        checkpoint = torch.load("best_model1.pth", map_location=device)
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