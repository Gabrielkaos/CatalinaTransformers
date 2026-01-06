import warnings
import torch
from data_cleaning import tokens_to_tensor
from MODEL_TRANSFORMER import build_transformer_encoder
from unidecode import unidecode


def count_parameters(model1):
    return sum(p.numel() for p in model1.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")

    max_seq_src = 1000

    # data
    data = torch.load("data.pth")
    src_vocab = data["src_vocab"]
    tokenizer_src = data["tokenizer_src"]
    num_labels = data["num_labels"]
    label_map = data["label_map"]

    # Model
    model = build_transformer_encoder(len(src_vocab), num_labels, max_seq_src, 
                                      device=device).to(device)
    model.load_state_dict(torch.load("brain.pth")["model_state"])
    model.eval()
    print(count_parameters(model))

    with torch.no_grad():
        while True:
            input_text = input("\n:")
            line = list(unidecode(input_text.strip()))
            line = line[:max_seq_src]
            line += ["<PAD>"] * (max_seq_src - len(line))
            # print(line)

            x = tokens_to_tensor([line], tokenizer_src, max_seq_src)[0].to(device)
            # print(x)

            pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int32).to(device)

            source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

            encoder_output = model.encode(x, source_mask).to(device)

            proj_output = model.project(encoder_output)[:, 0, :]

            probs = torch.softmax(proj_output.to('cpu'), dim=1)[0] * 100
            
            emotion_probs = {label_map[idx]: value.item() for idx, value in enumerate(probs)}
            emotion_probs_sorted = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
            # print(*emotion_probs_sorted, sep="\n")
            
            print()
            for emotion in emotion_probs_sorted:
                print(emotion[0], ":", f"{emotion[1]:.2f}%")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
