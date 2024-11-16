import warnings
import torch
from data_cleaning import tokens_to_tensor, remove_special
from MODEL_TRANSFORMER import build_transformer
from unidecode import unidecode


def decode_word(token_trgt,idx):
    decoded = {v:k for k,v in token_trgt.items()}

    word = decoded[idx]

    if word=="<PAD>" or word=="<SOS>":return ""
    return word


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")

    max_seq_length = 70

    # data
    real_data = torch.load("70-70_374k_data.pth")
    src_vocab, trgt_vocab, tokenizer_src, tokenizer_trgt = real_data["src_vocab"], real_data["trgt_vocab"], real_data["tokenizer_src"], real_data["tokenizer_trgt"]
    model = build_transformer(len(src_vocab), len(trgt_vocab), max_seq_length, max_seq_length,device=device).to(device)
    model.load_state_dict(torch.load("brain.pth")["model_state"])
    model.eval()

    with torch.no_grad():
        while True:
            input_text = input("\nGab:")
            # add(input_text,"paraphrase")
            line = unidecode(remove_special(input_text.lower().strip())).split()
            line = line[:max_seq_length - 2]
            line.insert(0, "<SOS>")
            line.append("<EOS>")
            line += ["<PAD>"] * (max_seq_length - len(line))

            x = tokens_to_tensor([line], tokenizer_src, max_seq_length)[0].to(device)
            # print(x)

            pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int64).to(device)

            source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

            encoder_output = model.encode(x, source_mask).to(device)
            decoder_input = torch.empty(1, 1).fill_(tokenizer_trgt.get("<SOS>")).type_as(x).to(device)

            print("\nCatalinaGeneral:", end='')

            while decoder_input.size(1) < max_seq_length:
                decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
                out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask).to(device)

                prob = model.project(out[:, -1]).to(device)
                _, next_word = torch.max(prob, dim=1)
                decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(x).fill_(next_word.item())], dim=1)

                if decode_word(tokenizer_trgt,next_word.item()) == "<EOS>":break
                print(f"{decode_word(tokenizer_trgt,next_word.item())}", end=' ')
            print()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
