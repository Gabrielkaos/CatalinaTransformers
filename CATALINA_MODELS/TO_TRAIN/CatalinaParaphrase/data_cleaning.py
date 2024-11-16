import torch
from unidecode import unidecode
import csv
from collections import OrderedDict


def get_unique(list1):
    unique = OrderedDict()

    for line in list1:
        for word in line:
            if word not in {"<PAD>", "<EOS>", "<SOS>"}:
                unique[word] = None

    return list(unique.keys())


def remove_special(line: str):
    char_not = ["`", "~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=", ",", "<", ">", ".", "/",
                "?", "'", ";", '"', ":", "[", "]", "{", "}", "|", "\"", "\\"]
    new = []
    for i in line:
        if i in char_not:
            continue
        new.append(i)
    return "".join(new)


def get_dialogue_data_csv_for_transformer(path, max_sequence_length, length):


    # holder for convo
    inputs = []
    outputs = []
    with open(path,"r",encoding="utf-8") as f:
        csv_reader = csv.reader(f)

        print("\nFetching data...")
        for row in csv_reader:
            if row[0] != "text":
                input_line = remove_special(unidecode(row[0].lower())).split()
                output_line = remove_special(unidecode(row[1].split("', '")[0].lower())).split()

                if len(input_line)>(max_sequence_length-2) or len(output_line)>(max_sequence_length-1):
                    continue

                inputs.append(input_line)
                outputs.append(output_line)

                if len(inputs)>=length:
                    break

    # input output
    input_sequences = []
    target_sequences = []
    label_sequences = []


    print("Processing tokens...")
    for gab_tokens, cat_tokens in zip(inputs, outputs):
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

    print("Building Vocabularies...")
    src_vocab = ["<PAD>", "<EOS>", "<SOS>"]
    src_vocab.extend(get_unique(input_sequences))
    trgt_vocab = ["<PAD>", "<EOS>", "<SOS>"]
    trgt_vocab.extend(get_unique(target_sequences))

    print("Making tokenizers...")
    tokenizer_src1 = {token: idx for idx, token in enumerate(src_vocab)}
    tokenizer_trgt1 = {token: idx for idx, token in enumerate(trgt_vocab)}


    print("Building tokenizers...")
    x = tokens_to_tensor(input_sequences, tokenizer_src1, max_sequence_length)
    y = tokens_to_tensor(target_sequences, tokenizer_trgt1, max_sequence_length)
    label1 = tokens_to_tensor(label_sequences, tokenizer_trgt1, max_sequence_length)

    print(f"\nX length = {len(input_sequences)}")
    print(f"y length = {len(target_sequences)}")
    print(f"Number of input words = {len(src_vocab)}")
    print(f"Number of target words = {len(trgt_vocab)}")
    print(f"max sequence length = {max_sequence_length}")

    return x, y, label1, src_vocab, trgt_vocab, tokenizer_src1, tokenizer_trgt1


def get_test(path, max_sequence_length):


    # holder for convo
    inputs = []
    outputs = []
    with open(path,"r",encoding="utf-8") as f:
        csv_reader = csv.reader(f)

        print("\nFetching data...")
        found = 0
        for row in csv_reader:
            if row[0] != "text":
                input_line = remove_special(unidecode(row[0].lower())).split()
                output_line = remove_special(unidecode(row[1].split("', '")[0].lower())).split()

                if len(input_line)>(max_sequence_length-2) or len(output_line)>(max_sequence_length-2):
                    continue

                found+=1
                if (found + 1) % 10000 == 0:print("skipped 10000")
                if found>100_000 and len(inputs)<1000:
                    inputs.append(input_line)
                    outputs.append(output_line)
                # inputs.append(input_line)
                # outputs.append(output_line)

    # input output
    input_sequences = []
    target_sequences = []
    label_sequences = []

    print("Processing tokens...")
    for gab_tokens, cat_tokens in zip(inputs, outputs):
        gab_tokens = gab_tokens[:max_sequence_length - 2]
        cat_tokens = cat_tokens[:max_sequence_length - 2]
        label_tokens = cat_tokens[:max_sequence_length - 2]

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


    print("Building Vocabularies...")
    src_vocab = ["<PAD>", "<EOS>", "<SOS>"]
    src_vocab.extend(get_unique(input_sequences))
    trgt_vocab = ["<PAD>", "<EOS>", "<SOS>"]
    trgt_vocab.extend(get_unique(target_sequences))

    print("Making tokenizers...")
    tokenizer_src1 = {token: idx for idx, token in enumerate(src_vocab)}
    tokenizer_trgt1 = {token: idx for idx, token in enumerate(trgt_vocab)}


    print("Building tokenizers...")
    x = tokens_to_tensor(input_sequences, tokenizer_src1, max_sequence_length)
    y = tokens_to_tensor(target_sequences, tokenizer_trgt1, max_sequence_length)
    label1 = tokens_to_tensor(label_sequences, tokenizer_trgt1, max_sequence_length)

    print(f"\nX length = {len(input_sequences)}")
    print(f"y length = {len(target_sequences)}")
    print(f"Number of input words = {len(src_vocab)}")
    print(f"Number of target words = {len(trgt_vocab)}")
    print(f"max sequence length = {max_sequence_length}")

    return x, y, label1, src_vocab, trgt_vocab, tokenizer_src1, tokenizer_trgt1


def tokens_to_tensor(tokens_list, word_to_index, max_sequence_length):
    indexed_sequences = [[word_to_index.get(token, word_to_index["<PAD>"]) for token in sequence] for sequence in
                         tokens_list]
    padded_sequences = [sequence + [word_to_index["<PAD>"]] * (max_sequence_length - len(sequence)) for sequence in
                        indexed_sequences]
    return torch.tensor(padded_sequences, dtype=torch.int64)


if __name__ == "__main__":
    length = 1_000_000
    x, y, label, src_vocab, trgt_vocab, tokenizer_src, tokenizer_trgt = get_dialogue_data_csv_for_transformer(
        "data/data_paraphrases.csv", 70, length)

    # inputs_dict = {
    #     "x": x,
    #     "y": y,
    #     "label": label,
    #     "src_vocab": src_vocab,
    #     "trgt_vocab": trgt_vocab,
    #     "tokenizer_src": tokenizer_src,
    #     "tokenizer_trgt": tokenizer_trgt
    # }
    #
    # torch.save(inputs_dict, f"test_30-30_{int(length/1000)}k(1)_data.pth")
