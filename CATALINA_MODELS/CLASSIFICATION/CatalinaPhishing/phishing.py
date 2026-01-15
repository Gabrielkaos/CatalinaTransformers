import torch
import torch.nn.functional as F
from MODEL_TRANSFORMER import build_transformer_encoder
import string


# -------------------------
# Tokenizer
# -------------------------
VOCAB = list(string.ascii_lowercase + string.digits + "/.-_?=&%:")
PAD = "<PAD>"
UNK = "<UNK>"

itos = [PAD, UNK] + VOCAB
stoi = {c: i for i, c in enumerate(itos)}

pad_idx = stoi[PAD]
unk_idx = stoi[UNK]



def normalize_url(url):
    url = url.lower()
    url = url.replace("http://", "")
    url = url.replace("https://", "")
    return url


def encode_url(url, max_len=256):
    url = normalize_url(url)
    ids = [stoi.get(c, unk_idx) for c in url[:max_len]]
    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))
    return ids

label_map = {0:"legitimate",1:"phishing"}

# -------------------------
# Configuration
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Build model (same as training)
# -------------------------
config = {
    "vocab_size": len(itos),
    "num_classes": 2,
    "d_model" : 256,
    "n_layers" : 4,
    "n_heads" : 4
}
model = build_transformer_encoder(
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
    tokens = encode_url(text)

    input_ids = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    mask = (input_ids != pad_idx)
    # forward
    logits = model(input_ids,mask=mask)[:,0,:]
    probs = torch.softmax(logits,dim=-1)[0]
    # mask = probs >= 0.5

    # # selected_probs = probs[mask]
    # selected_indices = mask.nonzero(as_tuple=True)[0]

    return {
        label_map[i]: probs[i].item()
        for i in range(2)
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
