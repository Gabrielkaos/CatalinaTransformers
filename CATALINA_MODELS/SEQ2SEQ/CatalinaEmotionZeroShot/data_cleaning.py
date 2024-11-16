import torch
from unidecode import unidecode
from datasets import load_dataset
from collections import OrderedDict


# def add(prompt, mode):
#     with open("/CatalinaCentralCommand/mode_selection_data.txt", "a") as f:
#         f.writelines(f"\n{prompt}--->{mode}")


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




def get_dialogue_data_for_transformer(max_seq_src, max_seq_trgt):
    print("\nFetching Dataset Conversational...")
    dataset = load_dataset("go_emotions", "raw", split='train')

    # holder for convo
    gabs = []
    cats = []

    label_map = {0:"admiration",1:"amusement",2:"anger",3:"annoyance",4:"approval",5:"caring",6:"confusion",
                 7:"curiosity",8:"desire",9:"disappointment",10:"disapproval",11:"disgust",12:"embarrassment",13:"excitement",
                 14:"fear",15:"gratitude",16:"grief",17:"joy",18:"love",19:"nervousness",20:"optimism",
                 21:"pride",22:"realization",23:"relief",24:"remorse",25:"sadness",26:"surprise",27:"neutral"}


    print("Processing dataset...")
    # longest_output = 0
    # count = 0
    for index,data in enumerate(dataset):
        inputs = split_string_with_special_characters(unidecode(data["text"]).lower().strip())
        if len(inputs) > (max_seq_src - 2):
            continue

        ones = []
        for value in label_map.values():
            labeled = data[value]
            if labeled==1:ones.append(value)

        # longest_output = max(len(inputs),longest_output)
        if len(ones)>1:
            print(inputs)
            print(ones)


        if len(ones)==0:
            ones.append("example_not_clear")

        gabs.append(inputs)
        cats.append(ones)
    exit()

    # print(count)


    # input output
    input_sequences = []
    target_sequences = []
    label_sequences = []

    print("Processing tokens...")

    for gab_tokens, cat_tokens in zip(gabs, cats):
        gab_tokens = gab_tokens[:max_seq_src - 2]
        cat_tokens = cat_tokens[:max_seq_trgt - 1]
        label_tokens = cat_tokens[:max_seq_trgt - 1]

        gab_tokens.insert(0, "<SOS>")
        gab_tokens.append("<EOS>")


        cat_tokens.insert(0, "<SOS>")
        label_tokens.append("<EOS>")

        gab_tokens += ["<PAD>"] * (max_seq_src - len(gab_tokens))
        cat_tokens += ["<PAD>"] * (max_seq_trgt - len(cat_tokens))
        label_tokens += ["<PAD>"] * (max_seq_trgt - len(label_tokens))

        input_sequences.append(gab_tokens)
        target_sequences.append(cat_tokens)
        label_sequences.append(label_tokens)

    print("Processing Vocabulary...")
    src_vocab = ["<PAD>", "<EOS>", "<SOS>"]
    src_vocab.extend(get_unique(input_sequences))
    trgt_vocab = ["<PAD>", "<EOS>", "<SOS>"]
    trgt_vocab.extend(get_unique(target_sequences))


    print(f"X length = {len(input_sequences)}")
    print(f"y length = {len(target_sequences)}")
    print(f"Number of input words = {len(src_vocab)}")
    print(f"Number of target words = {len(trgt_vocab)}")
    print(f"max sequence length[src, target] = {max_seq_src},{max_seq_trgt}")

    print("Processing tokenizers...")
    tokenizer_src1 = {token: idx for idx, token in enumerate(src_vocab)}
    tokenizer_trgt1 = {token: idx for idx, token in enumerate(trgt_vocab)}

    print("Processing tensors...")
    x = tokens_to_tensor(input_sequences, tokenizer_src1, max_seq_src)
    y = tokens_to_tensor(target_sequences, tokenizer_trgt1, max_seq_trgt)
    label = tokens_to_tensor(label_sequences, tokenizer_trgt1, max_seq_trgt)

    return x, y, label, src_vocab, trgt_vocab, tokenizer_src1, tokenizer_trgt1


def tokens_to_tensor(tokens_list, word_to_index, max_sequence_length):
    indexed_sequences = [
        [word_to_index.get(token, word_to_index["<PAD>"]) for token in sequence] for sequence in tokens_list
    ]

    padded_sequences = [
        sequence + [word_to_index["<PAD>"]] * (max_sequence_length - len(sequence)) for sequence in indexed_sequences
    ]
    return torch.tensor(padded_sequences, dtype=torch.int64)


if __name__ == "__main__":
    x, y, label, src_vocab, trgt_vocab, tokenizer_src, tokenizer_trgt = get_dialogue_data_for_transformer(72, 14)

    inputs_dict = {
        "x": x,
        "y": y,
        "label": label,
        "src_vocab": src_vocab,
        "trgt_vocab": trgt_vocab,
        "tokenizer_src": tokenizer_src,
        "tokenizer_trgt": tokenizer_trgt
    }

    torch.save(inputs_dict, "data.pth")
