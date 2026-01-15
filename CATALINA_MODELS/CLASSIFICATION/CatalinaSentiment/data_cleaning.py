import torch
from datasets import load_dataset
import tiktoken

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def get_data_simplified(max_seq_src,split="train"):
    print("\nFetching Dataset...")
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)

    tokenizer = tiktoken.get_encoding("gpt2")

    # Holders for input and labels
    inputs = []
    labels = []

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    num_labels = len(label_map)

    print("Processing dataset...")
    for data in dataset:
        token_ids = tokenizer.encode(preprocess(data["text"].strip()))

        if len(token_ids) > max_seq_src:
            continue
        
        pad_length = max_seq_src - len(token_ids)
        token_ids = token_ids + [50256] * pad_length

        inputs.append(token_ids)
        labels.append(int(data["label"]))

    

    print("Number of inputs = ", len(inputs))
    print(f"Number of labels = {num_labels}")


    print("Processing tensors...")
    x = torch.tensor(inputs,dtype=torch.int64)
    label_tensor = torch.tensor(labels, dtype=torch.int64)

    return x, label_tensor, label_map, num_labels



if __name__ == "__main__":
    max_seq_src = 128

    x, label, label_map, num_labels = get_data_simplified(max_seq_src,split="test")

    inputs_dict = {
        "x": x,
        "label": label,
        "label_map":label_map, 
        "num_classes":num_labels
    }

    torch.save(inputs_dict, "data_test.pth")

    # data = torch.load("data.pth",map_location=torch.device("cpu"))
    # tokenizer = tiktoken.get_encoding("gpt2")

    # x = data["x"]
    # y = data["label"]
    # label_map = data["label_map"]

    # print(x.shape)
    # print(y.shape)