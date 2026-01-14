import torch
from datasets import load_dataset
import tiktoken


def get_data_simplified(max_seq_src):
    print("\nFetching Dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified", split='train')

    tokenizer = tiktoken.get_encoding("gpt2")

    # Holders for input and labels
    inputs = []
    labels = []

    label_map = {0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval", 5: "caring", 6: "confusion",
                 7: "curiosity", 8: "desire", 9: "disappointment", 10: "disapproval", 11: "disgust",
                 12: "embarrassment", 13: "excitement", 14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
                 19: "nervousness", 20: "optimism", 21: "pride", 22: "realization", 23: "relief", 24: "remorse",
                 25: "sadness", 26: "surprise", 27: "neutral"}
    
    num_labels = len(label_map)

    print("Processing dataset...")
    for data in dataset:
        token_ids = tokenizer.encode(data["text"].strip())

        if len(token_ids) > max_seq_src:
            continue

        
        label_vector = [0] * num_labels
        labels1 = data["labels"]
        for i in labels1:
            label_vector[i] = 1

        if sum(label_vector)==0:
            continue
        
        pad_length = max_seq_src - len(token_ids)
        token_ids = token_ids + [50256] * pad_length

        inputs.append(token_ids)
        labels.append(label_vector)

    

    print("Number of inputs = ", len(inputs))
    print(f"Number of labels = {num_labels}")


    print("Processing tensors...")
    x = torch.tensor(inputs,dtype=torch.int64)
    label_tensor = torch.tensor(labels, dtype=torch.float32)

    return x, label_tensor, label_map, num_labels



if __name__ == "__main__":
    max_seq_src = 128

    # x, label, label_map, num_labels = get_data_simplified(max_seq_src)

    # inputs_dict = {
    #     "x": x,
    #     "label": label,
    #     "label_map":label_map, 
    #     "num_classes":num_labels
    # }

    # torch.save(inputs_dict, "data.pth")

    data = torch.load("data.pth",map_location=torch.device("cpu"))
    tokenizer = tiktoken.get_encoding("gpt2")

    x = data["x"]
    y = data["label"]
    label_map = data["label_map"]


    only_neutral = [0] * 28

    for i in y:
        for j in range(28):
            only_neutral[j] += i[j].item()==1


    print(only_neutral)
    print(len(y))