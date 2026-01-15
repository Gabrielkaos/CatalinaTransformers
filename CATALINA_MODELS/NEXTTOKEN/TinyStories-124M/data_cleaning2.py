from datasets import load_dataset
import torch
import tiktoken
from tqdm import tqdm

def process_data(
    max_seq_len,
    split
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("roneneldan/TinyStories", split=split,streaming=True)
    dataset = dataset.shuffle(seed=12123)

    input_seqs = []
    label_seqs = []
    skipped = 0

    for item in tqdm(dataset):
        tiny_story = item["text"]

        tokens = tokenizer.encode(tiny_story)

        if len(tokens) >= max_seq_len or len(tokens) < 10:
            skipped += 1
            continue
        
        pad_len = (max_seq_len - len(tokens))
        inputs = tokens[:-1]
        inputs += [50256] * pad_len

        labels = tokens[1:]
        labels += [-100] * pad_len

        assert len(inputs)==len(labels),"Length differ"


        input_seqs.append(inputs)
        label_seqs.append(labels)

        if len(input_seqs)>=20_000:break


    print(f"Total samples: {len(input_seqs)} | Skipped: {skipped}")
    return torch.tensor(input_seqs, dtype=torch.int64),torch.tensor(label_seqs, dtype=torch.int64)


if __name__ == "__main__":
    
    x,y = process_data(
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