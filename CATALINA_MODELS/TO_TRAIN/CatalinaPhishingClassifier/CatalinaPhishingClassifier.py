import warnings
import torch
from data_cleaning import tokens_to_tensor
from MODEL_TRANSFORMER import build_transformer_encoder
from unidecode import unidecode


def decode_category(label, categories):
    return categories[label.item()]


def count_parameters(model1):
    return sum(p.numel() for p in model1.parameters() if p.requires_grad)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")

    max_seq_src = 1700

    # data
    data = torch.load("data.pth")
    src_vocab = data["src_vocab"]
    trgt_vocab = data["trgt_vocab"]
    tokenizer_src = data["tokenizer_src"]
    categories = data["categories"]
    print("Data processed")

    model = build_transformer_encoder(len(src_vocab), len(trgt_vocab), max_seq_src, device=device).to(device)
    model.load_state_dict(torch.load("brain.pth")["model_state"])
    model.eval()
    print(count_parameters(model))

    with torch.no_grad():
        while True:
            input_text = input("\nGab:")
            line = list(unidecode(input_text))
            line = line[:max_seq_src - 2]
            line.insert(0, "<SOS>")
            line.append("<EOS>")
            line += ["<PAD>"] * (max_seq_src - len(line))

            x = tokens_to_tensor([line], tokenizer_src, max_seq_src)[0].to(device)

            pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int64).to(device)

            source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

            encoder_output = model.encode(x, source_mask).to(device)

            proj_output = model.project(encoder_output).mean(dim=1)

            _, predicted_label = torch.max(proj_output, dim=1)

            predicted_category = decode_category(predicted_label, categories)

            print(f"Predicted: {predicted_category}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
