import warnings
import torch
from data_cleaning import tokens_to_tensor, remove_special
from MODEL_TRANSFORMER import build_transformer_encoder
from unidecode import unidecode


def decode_category(label, categories):
    return categories[label.item()]


# def add(prompt, mode):
#     with open("C:/Users/Gabriel Montes/PycharmProjects/TransformersNN/CatalinaCentralCommand/mode_selection_data.txt", "a") as f:
#         f.writelines(f"\n{prompt}--->{mode}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")

    max_seq_src = 150

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

    with torch.no_grad():
        while True:
            input_text = input("\nGab:")
            what_type = input("Type[c/j]:")
            if what_type.lower() != "j":type_input = "commonsense"
            else:type_input = "justice"
            # add(input_text,"ethics")
            line = unidecode(remove_special(input_text.lower().strip())).split()
            line.append("<t>")
            line.append(type_input)
            line.append("</t>")
            line = line[:max_seq_src - 2]
            line.insert(0, "<SOS>")
            line.append("<EOS>")
            print(line)
            line += ["<PAD>"] * (max_seq_src - len(line))


            x = tokens_to_tensor([line], tokenizer_src, max_seq_src)[0].to(device)
            # print(x)
            pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int64).to(device)

            source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)
            encoder_output = model.encode(x, source_mask).to(device)

            proj_output = model.project(encoder_output).mean(dim=1)

            _, predicted_label = torch.max(proj_output, dim=1)

            print(predicted_label)

            predicted_category = decode_category(predicted_label, categories)

            print(f"Predicted: {predicted_category}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
