import torch
from datasets import load_dataset
import string


VOCAB = list(string.ascii_lowercase + string.digits + "/.-_?=&%:")
PAD = "<PAD>"
UNK = "<UNK>"

itos = [PAD, UNK] + VOCAB
stoi = {c: i for i, c in enumerate(itos)}

pad_idx = stoi[PAD]
unk_idx = stoi[UNK]



def normalize_url(url):
    url = url.lower()
    url = url.replace("http://", "")
    url = url.replace("https://", "")
    return url


def encode_url(url, max_len=256):
    url = normalize_url(url)
    ids = [stoi.get(c, unk_idx) for c in url[:max_len]]
    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))
    return ids





def get_data_simplified():
    print("\nFetching Dataset...")
    dataset = load_dataset("stanpony/phishing_urls", split='train',streaming=True)
    dataset = dataset.shuffle(seed=213)

    # Holders for input and labels
    inputs = []
    labels = []

    label_map = {0: "legitimate", 1: "phishing"}
    
    num_labels = len(label_map)

    positive_count = 0
    negative_count = 0

    dont_accept_negative = False
    dont_accept_positive = False

    print("Processing dataset...")
    for data in dataset:
        token_ids = encode_url(data["text"])        
        label = int(data["label"])

        #0 for legitimate and 1 for phishing
        if dont_accept_positive and dont_accept_negative:break
        if (dont_accept_negative and label==1) or (dont_accept_positive and label==0):
            continue
        
        inputs.append(token_ids)
        labels.append(label)

        positive_count += label==0
        negative_count += label==1

        if positive_count>=50_000:
            dont_accept_positive=True
        if negative_count>=50_000:
            dont_accept_negative=True

    print("Number of inputs = ", len(inputs))
    print(f"Number of labels = {num_labels}")


    print("Processing tensors...")
    x = torch.tensor(inputs,dtype=torch.int64)
    label_tensor = torch.tensor(labels, dtype=torch.float32)

    return x, label_tensor, label_map, num_labels



if __name__ == "__main__":

    x, label, label_map, num_labels = get_data_simplified()

    inputs_dict = {
        "x": x,
        "label": label,
        "label_map":label_map, 
        "num_classes":num_labels
    }

    torch.save(inputs_dict, "data.pth")

    # data = torch.load("data.pth",map_location=torch.device("cpu"))

    # x = data["x"]
    # y = data["label"]
    # label_map = data["label_map"]
