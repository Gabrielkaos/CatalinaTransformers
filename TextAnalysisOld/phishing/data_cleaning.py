import torch
from unidecode import unidecode
from datasets import load_dataset
from collections import OrderedDict

def get_unique(list1):
    unique = OrderedDict()

    for line in list1:
        for word in line:
            if word not in {"<PAD>"}:
                unique[word] = None

    return list(unique.keys())


def get_data(max_seq_src):
    print("\nFetching Dataset...")
    dataset = load_dataset("pirocheto/phishing-url",split='train')

    # Holders for input and labels
    inputs = []
    labels = []

    label_map = {0:"legitimate",1:"phishing"}
    map_labels = {v: k for k, v in label_map.items()}

    num_labels = len(label_map)

    # longest = 0

    print("Processing dataset...")
    for data in dataset:
        tokenized_text = list(unidecode(data["url"]).strip())

        # longest = max(longest, len(tokenized_text))

        if len(tokenized_text) > max_seq_src:
            continue
        
        label_vector = [0] * num_labels
        label_vector[map_labels[data['status']]] = 1

        inputs.append(tokenized_text)
        labels.append(label_vector)


    # print(longest)
    # exit()
    extra = [['https://x4q866fp-8000.asse.devtunnels.ms/','legitimate']]
    for extra_url in extra:
        tokenized_text = list(unidecode(extra_url[0]).strip())

        if len(tokenized_text) > max_seq_src:
            continue
        
        label_vector = [0] * num_labels
        label_vector[map_labels[extra_url[1]]] = 1

        inputs.append(tokenized_text)
        labels.append(label_vector)
    
    print("Processing tokens...")
    src_vocab = ["<PAD>"]
    src_vocab.extend(get_unique(inputs))

    print("Number of inputs = ", len(inputs))
    print(f"Number of input words = {len(src_vocab)}")
    print(f"Number of labels = {num_labels}")

    tokenizer_src = {token: idx for idx, token in enumerate(src_vocab)}

    print("Processing tensors...")
    x = tokens_to_tensor(inputs, tokenizer_src, max_seq_src)
    label_tensor = torch.tensor(labels, dtype=torch.float32)

    return x, label_tensor, src_vocab, tokenizer_src, label_map, num_labels


def tokens_to_tensor(tokens_list, word_to_index, max_sequence_length, dtype=torch.int32):
    indexed_sequences = [
        [word_to_index.get(token, word_to_index["<PAD>"]) for token in sequence] for sequence in tokens_list
    ]

    padded_sequences = [
        sequence + [word_to_index["<PAD>"]] * (max_sequence_length - len(sequence)) for sequence in indexed_sequences
    ]
    return torch.tensor(padded_sequences, dtype=dtype)


if __name__ == "__main__":
    max_seq_src = 1000

    x, label, src_vocab, tokenizer_src, label_map, num_labels = get_data(max_seq_src)

    inputs_dict = {
        "x": x,
        "label": label,
        "src_vocab": src_vocab,
        "tokenizer_src": tokenizer_src,
        "label_map":label_map, 
        "num_labels":num_labels
    }

    torch.save(inputs_dict, "data.pth")
