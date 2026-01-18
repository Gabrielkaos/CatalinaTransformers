from datasets import load_dataset
import torch
import tiktoken
from tqdm import tqdm

def process_data(
    max_seq_len,
    split
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("abisee/cnn_dailymail","3.0.0", split=split,streaming=True)
    dataset = dataset.shuffle(seed=43)

    input_seqs = []
    label_seqs = []
    skipped = 0

    for item in tqdm(dataset):
        text = item["article"]

        tokens = tokenizer.encode(text)[:max_seq_len]

        if len(tokens) < 100:
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

        if len(input_seqs)>=50_000:break


    print(f"Total samples: {len(input_seqs)} | Skipped: {skipped}")
    return torch.tensor(input_seqs, dtype=torch.int64),torch.tensor(label_seqs, dtype=torch.int64)


if __name__ == "__main__":
    
    x,y = process_data(
        max_seq_len=257, 
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