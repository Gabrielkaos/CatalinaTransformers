import torch
from data_cleaning import split_string_with_special_characters
from MODEL_TRANSFORMER import build_transformer_next_token

def causal_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).int()


def generate(model, tokenizer, prompt, max_len=50, device="cpu"):
    model.eval()

    tokens = split_string_with_special_characters(prompt.lower())
    tokens = ["<SOS>"] + tokens
    ids = [tokenizer.get(t, tokenizer["<PAD>"]) for t in tokens]

    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_len):
        mask = causal_mask(x.size(1)).to(device)
        with torch.no_grad():
            logits = model(x, mask)
            next_token = torch.argmax(logits[:, -1], dim=-1)

        x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == tokenizer["<EOS>"]:
            break

    decoded = [k for i in x[0].tolist() for k, v in tokenizer.items() if v == i]
    return " ".join(decoded)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")

    seq_len = 72

    data = torch.load("data.pth")
    vocab = data["vocab"]
    tokenizer = data["tokenizer"]

    model = build_transformer_next_token(
        vocab_size=len(vocab),
        seq_len=seq_len - 1,
        device=device
    ).to(device)
    model.load_state_dict(torch.load("lm-10.pth",map_location=torch.device(device))["model_state"])
    model.eval()

    print(generate(model,tokenizer,"Hello I am about to"))