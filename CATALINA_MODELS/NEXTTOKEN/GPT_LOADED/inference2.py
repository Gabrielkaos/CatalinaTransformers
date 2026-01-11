import torch
import torch.nn.functional as F
# from data_cleaning2 import tokenize_with_tiktoken
from MODEL_TRANSFORMER import gpt2_like_model
# from unidecode import unidecode
import tiktoken
from transformers import GPT2LMHeadModel

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
    seq_len=1024, 
    device="cpu",
    temperature=1.0,
    top_k=None,
    top_p=0.9,
    repetition_penalty=1.0
):
    
    model.eval()
    
    # _,bow = tokenize_with_tiktoken(unidecode(prompt.strip()))
    token_ids = tokenizer.encode(prompt)

    # print("token ids",token_ids)
    
   
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    predicted = []
    
    for step in range(max_len):
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
        
        
        # if next_token.item() == tokenizer.eos_token:
        #     break
        
        predicted.append(next_token.item())
        
        x = torch.cat([x, next_token], dim=1)
    
    x = x[0][len(token_ids):].tolist()
    # decoder = {idx:char for char,idx in tokenizer.items()}
    # decoded_tokens = "".join(decoder[i] for i in predicted)



    return tokenizer.decode(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    
    # data = torch.load("data_triple.pth", map_location=device)
    vocab = 50257
    tokenizer = tiktoken.get_encoding("gpt2")
    
    
    config = {
        "vocab_size": None,
        "d_model":768,
        "n_layers":12,
        "n_heads":12,
        "dropout":0.2,  
        "bias_projection":False,
        "norm":"rms",
        "mlp_activation":"gelu",
        "use_flash_attn":True
    }
    
    config["vocab_size"] = vocab
    model = gpt2_like_model(**config).to(device)
    print(f"Model vocab:{model.embed.vocab_size}")
    print(f"Data vocab:{vocab}")
    try:
        # checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
        # state_dict = checkpoint["model_state"]

        
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith("_orig_mod."):
        #         new_k = k[len("_orig_mod."):]
        #     else:
        #         new_k = k
        #     new_state_dict[new_k] = v

        # model.load_state_dict(new_state_dict) 

        #load gpt2
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        sd_hf = model_hf.state_dict()
        
        print("Copying gpt2's weights")
        print("Copying embedding...")
        #copy gpt2's embedding
        model.state_dict()["embed.embedding.weight"].copy_(sd_hf["transformer.wte.weight"])
        
        #copy gpt2 attention projection
        print("Copying attention...")
        for i in range(config["n_layers"]):
            #proj
            model.state_dict()[f"decoder.layers.{i}.self_attention.w_o.weight"].copy_(sd_hf[f"transformer.h.{i}.attn.c_proj.weight"].t())
            model.state_dict()[f"decoder.layers.{i}.self_attention.w_o.bias"].copy_(sd_hf[f"transformer.h.{i}.attn.c_proj.bias"])
            
            #attn
            model.state_dict()[f"decoder.layers.{i}.self_attention.c_attn.weight"].copy_(sd_hf[f"transformer.h.{i}.attn.c_attn.weight"].t())
            model.state_dict()[f"decoder.layers.{i}.self_attention.c_attn.bias"].copy_(sd_hf[f"transformer.h.{i}.attn.c_attn.bias"])
            
            #mlp
            if config["mlp_activation"]=="gelu":
                model.state_dict()[f"decoder.layers.{i}.feed_forward.linear1.weight"].copy_(sd_hf[f"transformer.h.{i}.mlp.c_fc.weight"].t())
                model.state_dict()[f"decoder.layers.{i}.feed_forward.linear2.weight"].copy_(sd_hf[f"transformer.h.{i}.mlp.c_proj.weight"].t())
            # transformer.h.9.mlp.c_fc.weight

        #copy gpt2's lm_head
        print("Copying lm head...")
        model.state_dict()["proj.projection_layer.weight"].copy_(sd_hf["lm_head.weight"])

        # print(*model.state_dict().keys(),sep="\n")
        # print()
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

    # print(*model.state_dict().keys(), sep="\n")
    # exit()
    
    
    print("\n=== Greedy Decoding ===")
    output = generate(
        model, tokenizer, "Hello I am, ", 
        max_len=100,seq_len=256, 
        device=device,
        temperature=0.0,
    )
    print(output)
    # print(output)
#     while True:
#         print("\n\n")
#         instruction = input("instruction:")
#         if instruction=="quit":break
#         input1 = input("input:")

#         if input1 is not None:
#             prompt = f"""
# Instruction: {instruction}
# Input: {input1}
# Response:
#                 """
#         else:
#             prompt = f"""
# Instruction: {instruction}
# Response:
#                 """
        
        
#         # print("\n=== Sampling (temp=0.8, top_p=0.9) ===")
#         # print("Prompt:",prompt,"\n")
#         print("response:\n")
#         output = generate(
#             model, tokenizer, prompt,
#             max_len=500,seq_len=256,
#             device=device,
#             temperature=0.8,
#             top_p=0.9,
#             repetition_penalty=1.2
#         )
#         print(output)