import warnings
import torch
from data_cleaning import tokens_to_tensor, tokenize_with_tiktoken, get_segment_ids
from MODEL_TRANSFORMER import build_transformer_encoder
from unidecode import unidecode


def count_parameters(model1):
    return sum(p.numel() for p in model1.parameters() if p.requires_grad)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")

    max_seq_src = 100

    # data
    data = torch.load("data.pth")
    src_vocab = data["src_vocab"]
    tokenizer_src = data["tokenizer_src"]
    num_labels = data["num_labels"]
    label_map = data["label_map"]

    # Model
    model = build_transformer_encoder(len(src_vocab), num_labels, max_seq_src, 
                                      device=device, is_segmented=True).to(device)
    model.load_state_dict(torch.load("brain.pth")["model_state"])
    model.eval()
    print(count_parameters(model))

    with torch.no_grad():
        while True:
            sentence1 = input("\nsetence 1:"); sentence2 = input("\nsetence 2:")
            
            _,line1 = tokenize_with_tiktoken(unidecode(sentence1.strip().lower()))
            _,line2 = tokenize_with_tiktoken(unidecode(sentence2.strip().lower()))
            line = ["<CLS>"] + line1 + ["<SEP>"] + line2 + ["<SEP>"]
            segments = get_segment_ids(line)
            line += ["<PAD>"] * (max_seq_src - len(line))
            segments += [0] * (max_seq_src - len(segments))
            print(line)
            print(segments)

            x = tokens_to_tensor([line], tokenizer_src, max_seq_src)[0].to(device)
            # print(x)

            pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int32).to(device)

            source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

            encoder_output = model.encode(x, source_mask, segments=torch.tensor([segments]).to(device)).to(device)

            proj_output = model.project(encoder_output)[:,0,:]

            probs = torch.softmax(proj_output.to('cpu'), dim=1)[0] * 100
            
            probs = {label_map[idx]: value.item() for idx, value in enumerate(probs)}
            probs_sorted = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            print()
            for label in probs_sorted:
                print(label[0], ":", f"{label[1]:.2f}%")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
