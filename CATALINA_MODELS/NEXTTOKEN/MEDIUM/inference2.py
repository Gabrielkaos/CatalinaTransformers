import torch
import torch.nn.functional as F
from MODEL_TRANSFORMER.gpt_architecture import gpt2_like_model
import tiktoken
from transformers import GPT2LMHeadModel


def freeze_bottom_layers(model, n_layers_to_freeze=12):
    if not hasattr(model, 'decoder') or not hasattr(model.decoder, 'layers'):
        print("Warning: Model structure not recognized. Cannot freeze layers.")
        return
    
    total_layers = len(model.decoder.layers)
    n_layers_to_freeze = min(n_layers_to_freeze, total_layers)
    
    for i in range(n_layers_to_freeze):
        layer = model.decoder.layers[i]
        print(f"Freezing {i} layer")
        for param in layer.parameters():
            param.requires_grad = False
    
    print(f"Froze bottom {n_layers_to_freeze}/{total_layers} layers")
    
    # Show trainable parameter count after freezing
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:_}/{total_params:_} ({100*trainable_params/total_params:.1f}%)")


def freeze_embeddings(model):
    if hasattr(model, 'embed'):
        print("Embeddings frozen")
        for param in model.embed.parameters():
            param.requires_grad = False
    if hasattr(model, 'pos'):
        print("Position Embeddings frozen")
        for param in model.pos.parameters():
            param.requires_grad = False


def freeze_bottom_and_embeddings(model, n_layers_to_freeze=12):
    freeze_embeddings(model)
    freeze_bottom_layers(model, n_layers_to_freeze)


@torch.no_grad()
def generate(
    model, 
    tokenizer:tiktoken.Encoding, 
    prompt, 
    max_len=50,
    greedy=False,
    seq_len=1024, 
    device="cpu",
    top_k=0.2
):
    
    model.eval()
    
    token_ids = tokenizer.encode(prompt)
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    
    for _ in range(max_len):
        current_len = x.size(1)
        
        
        if current_len > seq_len:
            x = x[:, -seq_len:]
            current_len = seq_len
        
        logits = model(x)
        logits = logits[:, -1, :]

        if not greedy:
            probs = F.softmax(logits,dim=-1)
            topk_probs, topk_indices = torch.topk(probs,int(top_k * 100),dim=-1)
            ix = torch.multinomial(topk_probs,1)
            xcol = torch.gather(topk_indices,-1,ix)
            # print(xcol.shape)
        else:
            xcol = torch.argmax(logits,dim=-1,keepdim=True)
            # print(xcol.shape)

        out =  tokenizer.decode([xcol[0].item()])
        if out == "<|endoftext|>":break

        x = torch.cat((x,xcol),dim=1)

        
    
    x = x[0][len(token_ids):].tolist()

    return tokenizer.decode(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    
    vocab = 50257
    tokenizer = tiktoken.get_encoding("gpt2")
    
    
    config = {
        "vocab_size": None,
        "d_model":1280,
        "n_layers":36,
        "n_heads":20,
        "dropout":0.1,
    }
        
    config["vocab_size"] = vocab
    model = gpt2_like_model(**config).to(device)
    
    print(f"Data vocab:{vocab}")
    try:
        data_model=torch.load("gpt-large.pth",map_location=device)
        model.load_state_dict(data_model["model_state"])
        
        # checkpoint = torch.load("brain.pth", map_location=device)
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
        # model_hf = GPT2LMHeadModel.from_pretrained("gpt2-large")
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
        #     layer.feed_forward.linear1.weight.data.copy_(sd_hf[f"transformer.h.{i}.mlp.c_fc.weight"].t())
        #     layer.feed_forward.linear2.weight.data.copy_(sd_hf[f"transformer.h.{i}.mlp.c_proj.weight"].t())
        #     layer.feed_forward.linear1.bias.data.copy_(sd_hf[f"transformer.h.{i}.mlp.c_fc.bias"])
        #     layer.feed_forward.linear2.bias.data.copy_(sd_hf[f"transformer.h.{i}.mlp.c_proj.bias"])
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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # freeze_bottom_and_embeddings(model)

    # torch.save({"model_state":model.state_dict()},"gpt-large.pth")
    

    print("\n=== Generating ===")
    output = generate(
        model, tokenizer, "1 + 1 = ", 
        max_len=10,
        device=device,
        top_k=0.5
    )
    print(output)