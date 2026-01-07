import torch
from unidecode import unidecode
from datasets import load_dataset
from collections import OrderedDict
import tiktoken


def tokenize_with_tiktoken(text, max_length=None):
    """
    Tokenizes the input text using a tokenizer similar to ChatGPT's tokenizer (tiktoken).

    Args:
        text (str): The input text to be tokenized.
        max_length (int, optional): Maximum number of tokens to return. If None, return all tokens.

    Returns:
        list[int]: List of token IDs.
        list[str]: List of corresponding token strings.
    """
    # Load the tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")  # This encoding is similar to GPT-3.5/4.

    # Tokenize the text
    token_ids = tokenizer.encode(text)
    token_strings = [tokenizer.decode([token_id]) for token_id in token_ids]

    # Optionally truncate to max_length
    if max_length:
        token_ids = token_ids[:max_length]
        token_strings = token_strings[:max_length]

    return token_ids, token_strings

def get_unique(list1):
    unique = OrderedDict()

    for line in list1:
        for word in line:
            if word not in {"<PAD>"}:
                unique[word] = None

    return list(unique.keys())

def get_data(max_seq_src):
    print("\nFetching Dataset...")
    dataset = load_dataset("go_emotions", "raw", split='train')

    # Holders for input and labels
    inputs = []
    labels = []

    label_map = {0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval", 5: "caring", 6: "confusion",
                 7: "curiosity", 8: "desire", 9: "disappointment", 10: "disapproval", 11: "disgust",
                 12: "embarrassment", 13: "excitement", 14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
                 19: "nervousness", 20: "optimism", 21: "pride", 22: "realization", 23: "relief", 24: "remorse",
                 25: "sadness", 26: "surprise", 27: "neutral"}
    
    map_label = {value: key for key, value in label_map.items()}

    num_labels = len(label_map)

    print("Processing dataset...")
    for data in dataset:
        _, tokenized_text = tokenize_with_tiktoken(unidecode(data["text"]).strip())

        if len(tokenized_text) > max_seq_src:
            continue

        
        label_vector = [0] * num_labels

        has_ones = False
        for value in label_map.values():
            labeled = data[value]
            if labeled==1:
                has_ones = True
                label_vector[map_label[value]] = 1

        if not has_ones:
            label_vector[map_label["neutral"]] = 1

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


def get_data_simplified(max_seq_src):
    print("\nFetching Dataset...")
    dataset = load_dataset("go_emotions", "simplified", split='train')

    # Holders for input and labels
    inputs = []
    labels = []

    label_map = {0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval", 5: "caring", 6: "confusion",
                 7: "curiosity", 8: "desire", 9: "disappointment", 10: "disapproval", 11: "disgust",
                 12: "embarrassment", 13: "excitement", 14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
                 19: "nervousness", 20: "optimism", 21: "pride", 22: "realization", 23: "relief", 24: "remorse",
                 25: "sadness", 26: "surprise", 27: "neutral"}
    
    map_label = {value: key for key, value in label_map.items()}

    num_labels = len(label_map)

    print("Processing dataset...")
    for data in dataset:
        _, tokenized_text = tokenize_with_tiktoken(unidecode(data["text"]).strip())

        if len(tokenized_text) > max_seq_src:
            continue

        
        label_vector = [0] * num_labels
        labels1 = data["labels"]
        for i in labels1:
            label_vector[i] = 1
        

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
    max_seq_src = 128

    x, label, src_vocab, tokenizer_src, label_map, num_labels = get_data_simplified(max_seq_src)

    inputs_dict = {
        "x": x,
        "label": label,
        "vocab": src_vocab,
        "tokenizer": tokenizer_src,
        "label_map":label_map, 
        "num_classes":num_labels
    }

    torch.save(inputs_dict, "data.pth")

    # data = torch.load("data.pth")

    # x = data["x"]
    # label = data["label"]
    # de_tokenizer = {idx:char for char, idx in data["tokenizer"].items()}

    # print([de_tokenizer.get(i) for i in x[0].tolist()])

    
