import torch
from unidecode import unidecode
from datasets import load_dataset
from collections import OrderedDict


# def add(prompt, mode):
#     with open("/CatalinaCentralCommand/mode_selection_data.txt", "a") as f:
#         f.writelines(f"\n{prompt}--->{mode}")
#


def get_unique(list1):
    unique = OrderedDict()

    for line in list1:
        for word in line:
            if word not in {"<PAD>", "<EOS>", "<SOS>"}:
                unique[word] = None

    return list(unique.keys())


def get_unique_old(list1):
    unique = []

    for lines in list1:
        for word in lines:
            if word in ["<PAD>", "<EOS>", "<SOS>"]: continue
            if word not in unique:
                unique.append(word)

    return unique


def remove_special(line: str):
    char_not = ["`", "~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=", ",", "<", ">", ".", "/",
                "?", "'", ";", '"', ":", "[", "]", "{", "}", "|", "\"", "\\"]
    new = []
    for i in line:
        if i in char_not:
            continue
        new.append(i)
    return "".join(new)


def get_dialogue_data_for_transformer(max_seq_src, max_seq_trgt):
    print("\nFetching Dataset Conversational...")
    dataset_justice = load_dataset("metaeval/ethics","justice",split='train')
    dataset_justice1 = load_dataset("metaeval/ethics","justice",split='test')
    dataset_justice2 = load_dataset("metaeval/ethics","justice",split='validation')
    dataset_commonsense = load_dataset("metaeval/ethics","commonsense",split='train')
    dataset_commonsense1 = load_dataset("metaeval/ethics","commonsense",split='test')
    dataset_commonsense2 = load_dataset("metaeval/ethics","commonsense",split='validation')

    # holder for convo
    gabs = []
    cats = []

    # longest = 0

    label_map = {0:"acceptable",1:"unacceptable"}

    print("Processing dataset...")
    for data in dataset_justice:
        inputs = unidecode(remove_special(data["text"].lower())).split()
        outputs = label_map[data["label"]].split()

        inputs.append("<t>")
        inputs.append("justice")
        inputs.append("</t>")


        # longest = max(len(inputs),longest)

        if len(inputs) > max_seq_src - 2:
            continue


        gabs.append(inputs)
        cats.append(outputs)

    for data in dataset_justice1:
        inputs = unidecode(remove_special(data["text"].lower())).split()
        outputs = label_map[data["label"]].split()

        inputs.append("<t>")
        inputs.append("justice")
        inputs.append("</t>")

        # longest = max(len(inputs),longest)

        if len(inputs) > max_seq_src - 2:
            continue



        gabs.append(inputs)
        cats.append(outputs)

    for data in dataset_justice2:
        inputs = unidecode(remove_special(data["text"].lower())).split()
        outputs = label_map[data["label"]].split()

        inputs.append("<t>")
        inputs.append("justice")
        inputs.append("</t>")

        # longest = max(len(inputs),longest)

        if len(inputs) > max_seq_src - 2:
            continue


        gabs.append(inputs)
        cats.append(outputs)


    for data in dataset_commonsense:
        inputs = unidecode(remove_special(data["text"].lower())).split()
        outputs = label_map[data["label"]].split()

        inputs.append("<t>")
        inputs.append("commonsense")
        inputs.append("</t>")

        # longest = max(len(inputs),longest)

        if len(inputs) > max_seq_src - 2:
            continue


        gabs.append(inputs)
        cats.append(outputs)

    for data in dataset_commonsense1:
        inputs = unidecode(remove_special(data["text"].lower())).split()
        outputs = label_map[data["label"]].split()

        inputs.append("<t>")
        inputs.append("commonsense")
        inputs.append("</t>")

        # longest = max(len(inputs),longest)

        if len(inputs) > max_seq_src - 2:
            continue


        gabs.append(inputs)
        cats.append(outputs)

    for data in dataset_commonsense2:
        inputs = unidecode(remove_special(data["text"].lower())).split()
        outputs = label_map[data["label"]].split()

        inputs.append("<t>")
        inputs.append("commonsense")
        inputs.append("</t>")

        # longest = max(len(inputs),longest)

        if len(inputs) > max_seq_src - 2:
            continue


        gabs.append(inputs)
        cats.append(outputs)

    # print(longest)
    # exit()

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

        gab_tokens += ["<PAD>"] * (max_seq_src - len(gab_tokens))

        input_sequences.append(gab_tokens)
        target_sequences.append(cat_tokens)
        label_sequences.append(label_tokens)

    print("Processing Vocabulary...")
    src_vocab = ["<PAD>", "<EOS>", "<SOS>"]
    src_vocab.extend(get_unique(input_sequences))
    trgt_vocab = []
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
    y = tokens_to_tensor_y(target_sequences, tokenizer_trgt1, max_seq_trgt)
    label = tokens_to_tensor_y(label_sequences, tokenizer_trgt1, max_seq_trgt)

    return x, y, label, src_vocab, trgt_vocab, tokenizer_src1, tokenizer_trgt1, trgt_vocab



def tokens_to_tensor(tokens_list, word_to_index, max_sequence_length):
    indexed_sequences = [
        [word_to_index.get(token, word_to_index["<PAD>"]) for token in sequence] for sequence in tokens_list
    ]

    padded_sequences = [
        sequence + [word_to_index["<PAD>"]] * (max_sequence_length - len(sequence)) for sequence in indexed_sequences
    ]
    return torch.tensor(padded_sequences, dtype=torch.int64)


def tokens_to_tensor_y(tokens_list, word_to_index, max_sequence_length):
    indexed_sequences = [
        [word_to_index.get(token) for token in sequence] for sequence in tokens_list
    ]

    return torch.tensor(indexed_sequences, dtype=torch.int64)



if __name__ == "__main__":
    x, y, label, src_vocab, trgt_vocab, tokenizer_src, tokenizer_trgt, categories = get_dialogue_data_for_transformer(150, 2)

    inputs_dict = {
        "x": x,
        "y": y,
        "label": label,
        "src_vocab": src_vocab,
        "trgt_vocab": trgt_vocab,
        "tokenizer_src": tokenizer_src,
        "tokenizer_trgt": tokenizer_trgt,
        "categories": categories
    }

    torch.save(inputs_dict, "data.pth")
