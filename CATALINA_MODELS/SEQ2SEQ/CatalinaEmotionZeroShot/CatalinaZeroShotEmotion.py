import warnings
import torch
from data_cleaning import tokens_to_tensor, split_string_with_special_characters
from MODEL_TRANSFORMER.OLD import build_transformer
from unidecode import unidecode


def decode_word(token_trgt, idx,skip_special=False):
    decoded = {v: k for k, v in token_trgt.items()}

    word = decoded[idx]

    if skip_special:
        if word == "<PAD>" or word == "<SOS>": return ""

    return word


# def add(prompt, mode):
#     with open("/CatalinaCentralCommand/mode_selection_data.txt", "a") as f:
#         f.writelines(f"\n{prompt}--->{mode}")


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")

    max_seq_src = 72
    max_seq_trgt = 14

    # data
    data = torch.load("data.pth")
    src_vocab = data["src_vocab"]
    trgt_vocab = data["trgt_vocab"]
    tokenizer_src = data["tokenizer_src"]
    tokenizer_trgt = data["tokenizer_trgt"]
    print("Data processed")
    print(len(src_vocab))
    # x, y, label, src_vocab, trgt_vocab, tokenizer_src, tokenizer_trgt = get_dialogue_data_for_transformer(max_seq_length)
    model = build_transformer(len(src_vocab), len(trgt_vocab), max_seq_src, max_seq_trgt,device=device).to(device)
    model.load_state_dict(torch.load("epoch/8-1.3558.pth",map_location=torch.device(device))["model_state"])
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
            # print(line)

            x = tokens_to_tensor([line], tokenizer_src, max_seq_src)[0].to(device)
            # print(x)

            pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int64).to(device)

            source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

            encoder_output = model.encode(x, source_mask).to(device)
            decoder_input = torch.empty(1, 1).fill_(tokenizer_trgt.get("<SOS>")).type_as(x).to(device)
            # print(decoder_input)

            print("\nCatalinaZeroShotEmotion:", end='')
            predicteds = []
            while decoder_input.size(1) < max_seq_trgt:
                decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
                out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask).to(device)

                prob = model.project(out[:, -1]).to(device)
                _, next_word = torch.max(prob, dim=1)
                decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(x).fill_(next_word.item())], dim=1)

                if decode_word(tokenizer_trgt, next_word.item()) == "<EOS>": break
                decoded = decode_word(tokenizer_trgt,next_word.item(),skip_special=True)
                if decoded not in predicteds:
                    predicteds.append(decoded)
                # print(decode_word(tokenizer_trgt, next_word.item()), end=" ")
            print(*predicteds)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
