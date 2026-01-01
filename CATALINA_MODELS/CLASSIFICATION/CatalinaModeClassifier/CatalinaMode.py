import warnings
import torch
from data_cleaning import tokens_to_tensor
from MODEL_TRANSFORMER.OLD import build_transformer_encoder
from unidecode import unidecode


def decode_category(label, categories):
    return categories[label.item()]



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")

    max_seq_trgt = 2
    max_seq_src = 20

    # data
    data = torch.load("data.pth")
    src_vocab = data["src_vocab"]
    trgt_vocab = data["trgt_vocab"]
    tokenizer_src = data["tokenizer_src"]
    tokenizer_trgt = data["tokenizer_trgt"]
    categories = data["categories"]
    print(categories)
    print("Data processed")
    print(len(data["x"]))
    # x, y, label, src_vocab, trgt_vocab, tokenizer_src, tokenizer_trgt = get_dialogue_data_for_transformer(max_seq_length)
    model = build_transformer_encoder(len(src_vocab), len(trgt_vocab), max_seq_src, device=device).to(device)
    model.load_state_dict(torch.load("saved_from_loss/9-0.5888.pth",map_location=torch.device(device))["model_state"])
    model.eval()

    with torch.no_grad():
        while True:
            input_text = input("\nGab:")
            # add(input_text,"emotion")
            line = unidecode(input_text.strip()).split()
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
            print(predicted_category)



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
