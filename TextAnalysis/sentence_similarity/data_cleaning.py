import torch
from unidecode import unidecode
from datasets import load_dataset
from collections import OrderedDict
import tiktoken


def get_segment_ids(tokens):
    """
    Extracts segment IDs for the given input string formatted as:
    "<CLS> " + first sentence + " <SEP> " + second sentence + " <SEP>"
    
    Args:
        input_string (str): The input string.
    
    Returns:
        list: A list of segment IDs corresponding to each token.
    """
    
    segment_ids = []
    segment = 0
    
    for token in tokens:
        segment_ids.append(segment)
        # Switch to segment 1 after the first "<SEP>"
        if token == "<SEP>":
            segment = 1
    
    return segment_ids


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
    dataset = load_dataset("jatinmehra/MIT-PLAGAIRISM-DETECTION-DATASET",split='train').shuffle(12)

    # Holders for input and labels
    inputs = []
    labels = []

    label_map = {0:"nonplagiarized",1:"plagiarized"}

    num_labels = len(label_map)
    segment_ids = []

    # longests = [0,0]

    print("Processing dataset...")
    for data in dataset:
        text = unidecode(data["text"]).strip().lower()

        try:
            first = text.split(".")[0]
            second = text.split(".")[1]
        except IndexError:
            first = text.split("\t")[0]
            second = text.split("\t")[1]
        

        label = text[-1]

        _, tokenized_text1 = tokenize_with_tiktoken(first)
        _, tokenized_text2 = tokenize_with_tiktoken(second)
        tokenized_text = ["<CLS>"] + tokenized_text1 + ["<SEP>"] + tokenized_text2 + ["<SEP>"]
        segment = get_segment_ids(tokenized_text)
        segment += [0] * (max_seq_src - len(segment))

        # assert len(tokenized_text) == len(segment)
        
        if len(tokenized_text) > max_seq_src:
            continue

        label_vector = [0] * num_labels
        label_vector[int(label)] = 1
        inputs.append(tokenized_text)
        labels.append(label_vector)
        segment_ids.append(torch.tensor(segment))

        if len(inputs) >= 150000:
            break

    # print(longests)
    # exit()
    
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

    return x, label_tensor, src_vocab, tokenizer_src, label_map, num_labels, segment_ids


def tokens_to_tensor(tokens_list, word_to_index, max_sequence_length, dtype=torch.int32):
    indexed_sequences = [
        [word_to_index.get(token, word_to_index["<PAD>"]) for token in sequence] for sequence in tokens_list
    ]

    padded_sequences = [
        sequence + [word_to_index["<PAD>"]] * (max_sequence_length - len(sequence)) for sequence in indexed_sequences
    ]
    return torch.tensor(padded_sequences, dtype=dtype)


if __name__ == "__main__":
    max_seq_src = 100

    x, label, src_vocab, tokenizer_src, label_map, num_labels, segment_ids = get_data(max_seq_src)

    inputs_dict = {
        "x": x,
        "label": label,
        "src_vocab": src_vocab,
        "tokenizer_src": tokenizer_src,
        "label_map":label_map, 
        "num_labels":num_labels,
        "segment_ids":segment_ids
    }

    torch.save(inputs_dict, "data.pth")
