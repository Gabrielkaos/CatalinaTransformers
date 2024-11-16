import torch
from unidecode import unidecode


# def add(prompt, mode):
#     with open("/CatalinaCentralCommand/mode_selection_data.txt", "a") as f:
#         f.writelines(f"\n{prompt}--->{mode}")



def remove_blank_space(data_lines):
    new_data = []
    for i in data_lines:
        if i == "\n":
            continue
        new_data.append(i)

    return new_data


def get_unique(list1):
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


# removes second :
def remove_second_splitter(line):
    new_line = []
    count = 0
    for char in line:
        count += char == ":"

        if count > 1 and char == ":":
            continue
        new_line.append(char)

    return "".join(new_line)


def get_dialogue_data_for_transformer(path, max_sequence_length):
    with open(path, "r") as f:
        data_lines = f.readlines()

    # gabs is the input and cats are the outputs
    data_lines = remove_blank_space(data_lines)

    # holder for convo
    gabs = []
    cats = []

    for i, line in enumerate(data_lines):
        line = remove_second_splitter(line)
        assert len(line.split(":")) == 2, f"What the fuck multiple : detected in line {line}"
        removed = line.split(":")[0].lower()

        line = unidecode(remove_special(line.split(":")[1].lower().strip()))

        assert removed == "gab" or removed == "catalina", "Who the fuck is this?"

        if removed == "gab":
            gabs.append(line.split())
        else:
            cats.append(line.split())

    # input output
    input_sequences = []
    target_sequences = []
    label_sequences = []

    for gab_tokens, cat_tokens in zip(gabs, cats):
        gab_tokens = gab_tokens[:max_sequence_length - 2]
        cat_tokens = cat_tokens[:max_sequence_length - 1]
        label_tokens = cat_tokens[:max_sequence_length - 1]

        gab_tokens.insert(0, "<SOS>")
        cat_tokens.insert(0, "<SOS>")

        gab_tokens.append("<EOS>")
        label_tokens.append("<EOS>")

        gab_tokens += ["<PAD>"] * (max_sequence_length - len(gab_tokens))
        cat_tokens += ["<PAD>"] * (max_sequence_length - len(cat_tokens))
        label_tokens += ["<PAD>"] * (max_sequence_length - len(label_tokens))

        input_sequences.append(gab_tokens)
        target_sequences.append(cat_tokens)
        label_sequences.append(label_tokens)

    src_vocab = ["<PAD>", "<EOS>", "<SOS>"]
    src_vocab.extend(get_unique(input_sequences))
    trgt_vocab = ["<PAD>", "<EOS>", "<SOS>"]
    trgt_vocab.extend(get_unique(target_sequences))

    print(f"X length = {len(input_sequences)}")
    print(f"y length = {len(target_sequences)}")
    print(f"Number of input words = {len(src_vocab)}")
    print(f"Number of target words = {len(trgt_vocab)}")
    print(f"max sequence length = {max_sequence_length}")

    tokenizer_src1 = {token: idx for idx, token in enumerate(src_vocab)}
    tokenizer_trgt1 = {token: idx for idx, token in enumerate(trgt_vocab)}

    x = tokens_to_tensor(input_sequences, tokenizer_src1, max_sequence_length)
    y = tokens_to_tensor(target_sequences, tokenizer_trgt1, max_sequence_length)
    label = tokens_to_tensor(label_sequences, tokenizer_trgt1, max_sequence_length)

    return x, y, label, src_vocab, trgt_vocab, tokenizer_src1, tokenizer_trgt1


def tokens_to_tensor(tokens_list, word_to_index, max_sequence_length):
    indexed_sequences = [[word_to_index.get(token, word_to_index["<PAD>"]) for token in sequence] for sequence in
                         tokens_list]
    padded_sequences = [sequence + [word_to_index["<PAD>"]] * (max_sequence_length - len(sequence)) for sequence in
                        indexed_sequences]
    return torch.tensor(padded_sequences, dtype=torch.int64)


if __name__ == "__main__":
    # pass
    x, y, label, src_vocab, trgt_vocab, tokenizer_src, tokenizer_trgt = get_dialogue_data_for_transformer("script.txt", 40)
