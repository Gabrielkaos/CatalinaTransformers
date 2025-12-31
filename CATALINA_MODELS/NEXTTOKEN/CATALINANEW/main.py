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
