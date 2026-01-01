import warnings
import torch
from data_cleaning import tokens_to_tensor, split_string_with_special_characters
from MODEL_TRANSFORMER.OLD import build_transformer_encoder
from unidecode import unidecode


def decode_category(label, categories):
    return categories[label.item()]


# def add(prompt, mode):
#     with open("/CatalinaCentralCommand/mode_selection_data.txt", "a") as f:
#         f.writelines(f"\n{prompt}--->{mode}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")

    max_seq_src = 80

    # data
    data = torch.load("data.pth")
    src_vocab = data["src_vocab"]
    trgt_vocab = data["trgt_vocab"]
    tokenizer_src = data["tokenizer_src"]
    categories = data["categories"]
    print("Data processed")
    print(len(src_vocab))
    # x, y, label, src_vocab, trgt_vocab, tokenizer_src, tokenizer_trgt = get_dialogue_data_for_transformer(max_seq_length)
    model = build_transformer_encoder(len(src_vocab), len(trgt_vocab), max_seq_src, device=device).to(device)
    model.load_state_dict(torch.load("saved_from_loss/2-0.4830.pth",map_location=torch.device(device))["model_state"])
    model.eval()

    with torch.no_grad():
        while True:
            input_text = input("\nGab:")
            # add(input_text,"emotion")
            line = split_string_with_special_characters(unidecode(input_text).lower().strip())
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
