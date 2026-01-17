import torch
import torch.nn.functional as F
import tiktoken
from MODEL_TRANSFORMER.gpt_architecture import gpt_classifier
from pathlib import Path

# -------------------------
# Configuration
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 1024
PAD_IDX = 50256

# -------------------------
# Load metadata
# -------------------------
data = torch.load("label_map.pth", map_location=DEVICE)
label_map = data["label_map"]  # {idx: label}
num_classes = len(label_map)
# -------------------------
# Tokenizer
# -------------------------
tokenizer = tiktoken.get_encoding("gpt2")

# -------------------------
# Build model (same as training)
# -------------------------
config = {
    "vocab_size": 50257,
    "num_class": num_classes,
    "d_model" : 768,
    "n_layers" : 12,
    "n_heads" : 12,
    "is_causal" : False,
    "block_size" : 1024,
    "dropout" : 0.1,
    "mlp_activation" : "gelu"
}
model = gpt_classifier(
    **config
)

# Load weights
ckpt = torch.load("best_model.pth", map_location=DEVICE)
state_dict = ckpt["model_state"]
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        new_k = k[len("_orig_mod."):]
    else:
        new_k = k
    new_state_dict[new_k] = v

model.load_state_dict(new_state_dict) 
model.to(DEVICE)
model.eval()

# -------------------------
# Inference loop
# -------------------------
@torch.no_grad()
def predict(text: str):
    # tokenize
    tokens = tokenizer.encode(text)
    tokens = tokens[:MAX_LEN]

    input_ids = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    mask = (input_ids != PAD_IDX)
    # forward
    # logits = model(input_ids,mask)
    # mask = mask.unsqueeze(-1)        # [B, T, 1]
    # pooled = (logits * mask).sum(dim=1) / mask.sum(dim=1)

    hidden = model(input_ids,mask=mask,return_hidden=True)   # [B, T, D]
    # print(hidden.shape)
    mask_f = mask.unsqueeze(-1).float()               # [B, T, 1]
    # print(mask_f.sum(dim=1))
    pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1)  # [B, D]
    # print(pooled.shape)
    logits = model.last_projection(pooled)
    # print(logits.shape)
    probs = torch.softmax(logits,dim=-1)[0]


    # mask = probs >= 0.5

    # # selected_probs = probs[mask]
    # selected_indices = mask.nonzero(as_tuple=True)[0]

    return {
        label_map[i]: probs[i].item()
        for i in range(num_classes)
    }



# -------------------------
# Interactive CLI
# -------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    while True:
        text = input("\n> ").strip()
        if not text:
            continue

        probs = predict(text)
        for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"{k}: {v:6.2f}%")
