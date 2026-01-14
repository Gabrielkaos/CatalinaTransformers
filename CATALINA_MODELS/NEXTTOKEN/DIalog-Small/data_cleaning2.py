from datasets import load_dataset
import torch
import tiktoken
from tqdm import tqdm

def build_sliding_dialogue(dialog):
    samples = []

    for end in range(2, len(dialog) + 1):
        convo = []
        for i in range(end):
            speaker = "Person1" if i % 2 == 0 else "Person2"
            convo.append(f"{speaker}: {dialog[i]}")
        samples.append("\n".join(convo))

    return samples




def process_dailydialog(
    max_seq_len,
    split
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("roskoN/dailydialog", split=split)

    input_seqs = []
    label_seqs = []
    skipped = 0

    for item in tqdm(dataset):
        dialog = item["utterances"]

        samples = build_sliding_dialogue(dialog)

        for text in samples:
            
            tokens = tokenizer.encode(text)

            if len(tokens) >= max_seq_len:
                skipped += 1
                continue
            
            pad_len = (max_seq_len - len(tokens))
            inputs = tokens[:-1] + [50256]
            inputs += [50256] * pad_len

            labels = tokens[1:] + [50256]
            labels += [-100] * pad_len

            assert len(inputs)==len(labels),"Length differ"


            input_seqs.append(inputs)
            label_seqs.append(labels)

    print(f"Total samples: {len(input_seqs)} | Skipped: {skipped}")
    return torch.tensor(input_seqs, dtype=torch.int64),torch.tensor(label_seqs, dtype=torch.int64)


if __name__ == "__main__":
    
    x,y = process_dailydialog(
        max_seq_len=256, 
        split="train"
    )
    
    file_name = "data.pth"
    torch.save({
        "x": x,
        "y":y
    }, file_name)

    print(x.shape)
    print(y.shape)

    print(x[0])
    print(y[0])
    
    print("Done")