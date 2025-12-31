from unidecode import unidecode
from datasets import load_dataset
import torch
from collections import OrderedDict

def split_string_with_special_characters(input_str):
    char_not = ["`", "~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=", ",", "<", ">", ".", "/",
                "?", "'", ";", '"', ":", "[", "]", "{", "}", "|", "\"", "\\"]
    words = []
    current_word = ''

    for char in input_str:
        if char == ' ':
            if current_word:
                words.append(current_word)
                current_word = ''
        elif char in char_not:
            if current_word:
                words.append(current_word)
                current_word = ''
            words.append(char)
        else:
            current_word += char

    if current_word:
        words.append(current_word)

    return words


def get_unique(list1):
    unique = OrderedDict()

    for line in list1:
        for word in line:
            if word not in {"<PAD>", "<EOS>", "<SOS>"}:
                unique[word] = None

    return list(unique.keys())


def tokens_to_tensor(tokens_list, word_to_index, max_sequence_length):
    indexed_sequences = [
        [word_to_index.get(token, word_to_index["<PAD>"]) for token in sequence] for sequence in tokens_list
    ]

    padded_sequences = [
        sequence + [word_to_index["<PAD>"]] * (max_sequence_length - len(sequence)) for sequence in indexed_sequences
    ]
    return torch.tensor(padded_sequences, dtype=torch.int64)




def get_language_model_data(max_seq_len):
    print("\nFetching Dataset (Language Modeling)...")
    dataset = load_dataset("go_emotions", "raw", split="train")

    sequences = []

    print("Processing dataset...")
    for data in dataset:
        tokens = split_string_with_special_characters(
            unidecode(data["text"]).lower().strip()
        )

        if len(tokens) > max_seq_len - 2:
            continue

        tokens.insert(0, "<SOS>")
        tokens.append("<EOS>")
        tokens += ["<PAD>"] * (max_seq_len - len(tokens))

        # print(tokens)
        sequences.append(tokens)

    print("Building vocabulary...")
    vocab = ["<PAD>", "<EOS>", "<SOS>"]
    vocab.extend(get_unique(sequences))
    print(vocab)

    tokenizer = {token: idx for idx, token in enumerate(vocab)}

    print("Converting to tensors...")
    x = tokens_to_tensor(sequences, tokenizer, max_seq_len)

    print(f"Total sequences: {len(x)}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Max sequence length: {max_seq_len}")

    return x, vocab, tokenizer


if __name__ == "__main__":
    max_seq_len = 72

    x, vocab, tokenizer = get_language_model_data(max_seq_len)

    torch.save(
        {
            "x": x,
            "vocab": vocab,
            "tokenizer": tokenizer
        },
        "data.pth"
    )
