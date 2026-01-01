import torch
from MODEL_TRANSFORMER.OLD import build_transformer, RNN, build_transformer_encoder
from unidecode import unidecode
import warnings
import string
import os


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


def remove_special(line: str):
    char_not = ["`", "~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=", ",", "<", ">", ".", "/",
                "?", "'", ";", '"', ":", "[", "]", "{", "}", "|", "\"", "\\"]
    new = []
    for i in line:
        if i in char_not:
            continue
        new.append(i)
    return "".join(new)


def count_parameters(model1):
    return sum(p.numel() for p in model1.parameters() if p.requires_grad)


def clean_input_text_final(input_text: str, tokenizer_src, max_seq, device, type_input=None):
    if type_input is not None:
        if type_input == "emotion":
            line = split_string_with_special_characters(unidecode(input_text).lower().strip())
            line = line[:max_seq - 2]
            line.insert(0, "<SOS>")
            line.append("<EOS>")
            line += ["<PAD>"] * (max_seq - len(line))

            return tokens_to_tensor([line], tokenizer_src, max_seq)[0].to(device)

    line = unidecode(remove_special(input_text.lower().strip())).split()
    line = line[:max_seq - 2]
    line.insert(0, "<SOS>")
    line.append("<EOS>")
    line += ["<PAD>"] * (max_seq - len(line))

    return tokens_to_tensor([line], tokenizer_src, max_seq)[0].to(device)


def clean_input_name_gender(input_text: str, tokenizer_src, max_seq, device):
    line = list(unidecode(input_text.lower()))
    line = line[:max_seq - 2]
    line.insert(0, "<SOS>")
    line.append("<EOS>")
    line += ["<PAD>"] * (max_seq - len(line))

    return tokens_to_tensor([line], tokenizer_src, max_seq)[0].to(device)


def clean_input_link(input_text: str, tokenizer_src, max_seq, device):
    line = list(unidecode(input_text))
    line = line[:max_seq - 2]
    line.insert(0, "<SOS>")
    line.append("<EOS>")
    line += ["<PAD>"] * (max_seq - len(line))

    return tokens_to_tensor([line], tokenizer_src, max_seq)[0].to(device)


def tokens_to_tensor(tokens_list, word_to_index, max_sequence_length):
    indexed_sequences = [[word_to_index.get(token, word_to_index["<PAD>"]) for token in sequence] for sequence in
                         tokens_list]
    padded_sequences = [sequence + [word_to_index["<PAD>"]] * (max_sequence_length - len(sequence)) for sequence in
                        indexed_sequences]
    return torch.tensor(padded_sequences, dtype=torch.int64)


def decode_word(token_trgt, idx):
    decoded = {v: k for k, v in token_trgt.items()}

    word = decoded[idx]

    if word == "<PAD>" or word == "<SOS>": return ""
    return word


def get_index(all_words1, char):
    try:
        return all_words1.index(char)
    except (IndexError, ValueError):
        return -1


def char_tensor(sentence, all_characters1):
    tensor = torch.zeros(len(sentence)).long()

    for i in range(len(sentence)):
        tensor[i] = get_index(all_characters1, sentence[i])

    return tensor


def generate_text(model1, all_characters1, init_str="A", pred_len=100, temp=0.50, device=None):
    if device is None:
        device = torch.device("cpu")
    hiddens, cells = model1.init_hidden(1)
    init_input = char_tensor(init_str, all_characters1).to(device)

    predicted = init_str

    for p in range(len(init_str) - 1):
        _, (hiddens, cells) = model1(init_input[p].view(1), hiddens.to(device), cells.to(device))

    last_char = init_input[-1].to(device)

    for p in range(pred_len):
        outs, (hiddens, cells) = model1(last_char.view(1).to(device), hiddens.to(device), cells.to(device))
        out_distance = outs.data.view(-1).div(temp).exp()
        top_char = torch.multinomial(out_distance, 1)[0]
        pred_char = all_characters1[top_char]
        predicted += pred_char
        last_char = char_tensor(pred_char, all_characters1).to(device)

    return predicted


def decode_category(label, categories):
    return categories[label.item()]


class Catalina:
    def __init__(self, device=None, broken=None):
        print("Initializing Catalina...")
        self.device=device
        if device is None:
            device = torch.device("cuda")

        if str(device) == "cuda":
            print(f"Using={device}")
            print(f"Device memory={(torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3):.3f} GB")

        if broken is None:
            self.broken = []
        else:
            self.broken = broken

        self.language_translatable = {
            "english": ["tagalog"],
            "tagalog": ["english", "cebuano"],
            "cebuano": ["tagalog"]
        }

        self.models_dict = {
            
            "paraphrase": {},
            "conversational": {},
            "translate": {
                "cebuano_tagalog": {},
                "tagalog_cebuano": {},
                "tagalog_english": {},
                "english_tagalog": {},
            },
            "poem": {},
            "emotion": {},
        }

        self.models_dict_classifiers = {
            "ethics": {},
            "sentiment": {},
            "mode": {},
            "documentlevel": {},
            "names": {},
            "language_detector": {},
            "linkclassification": {},
            "aihumantext": {}
        }

        self.lengths = {
            "emotion": (72, 14),
            "ethics": (150, 2),
            "names": (27, 2),
            "paraphrase": (70, 70),
            "sentiment": (80, 2),
            "conversational": (70, 70),
            "translate": (40, 40),
            "mode": (20, 2),
            "linkclassification": (1700, 2),
            "documentlevel": (1400, 2),
            "aihumantext": (1400, 2),
            "language_detector": (70, 2)
        }
        self.total_params = 0
        self.load_models_classifier()
        self.load_models()

        print(f"Catalina Initialized Total Parameters: {(self.total_params / 1_000_000_000):.2f} Billion(s)\n\n")

        # mode
        self.mode = None

    def load_models_classifier(self):
        device = self.device

        for key in self.models_dict_classifiers.keys():
            if key in self.broken:continue
            data = torch.load(f"DIRS/{key}/data.pth")

            src_vocab = data["src_vocab"]
            trgt_vocab = data["trgt_vocab"]
            tokenizer_src = data["tokenizer_src"]
            tokenizer_trgt = data["tokenizer_trgt"]
            categories = data["categories"]
            src_len, trgt_len = self.lengths[key]

            model = build_transformer_encoder(len(src_vocab), len(trgt_vocab), src_len, device=device).to(device)
            model.load_state_dict(torch.load(f"DIRS/{key}/brain.pth", map_location=device)["model_state"])
            model.eval()
            self.total_params += count_parameters(model)
            self.models_dict_classifiers[key] = {
                "src_vocab": src_vocab,
                "trgt_vocab": trgt_vocab,
                "tokenizer_src": tokenizer_src,
                "tokenizer_trgt": tokenizer_trgt,
                "model": model,
                "src_len": src_len,
                "trgt_len": trgt_len,
                "categories": categories
            }

        # print(f"Catalina Initialized Total Parameters: {(self.total_params/1_000_000_000):.2f} Billion(s)\n\n")

    def load_models(self):
        device = self.device

        for key in self.models_dict.keys():
            if key in self.broken:continue
            if key == "poem":
                # DATA
                all_characters = string.printable
                n_characters = len(all_characters)

                # MODEL
                model = RNN(n_characters, 200, 8, n_characters).to(device)
                # model.load_state_dict(torch.load("DIRS/poem/brain.pth",map_location=device)["model_state"])
                # model.eval()

                self.total_params += count_parameters(model)

                self.models_dict[key] = {
                    "model": model,
                    "all_characters": all_characters
                }
            else:
                if key == "translate":
                    path = "DIRS/translate"
                    for dir_files in os.listdir(path):
                        f_name = os.path.join(path, dir_files)

                        data = torch.load(f"{f_name}/data.pth")

                        src_vocab = data["src_vocab"]
                        trgt_vocab = data["trgt_vocab"]
                        tokenizer_src = data["tokenizer_src"]
                        tokenizer_trgt = data["tokenizer_trgt"]
                        src_len, trgt_len = self.lengths[key]

                        model = build_transformer(len(src_vocab), len(trgt_vocab), src_len, trgt_len,
                                                  device=device).to(device)
                        model.load_state_dict(torch.load(f"{f_name}/brain.pth", map_location=device)["model_state"])
                        model.eval()
                        self.total_params += count_parameters(model)
                        self.models_dict[key][dir_files] = {
                            "src_vocab": src_vocab,
                            "trgt_vocab": trgt_vocab,
                            "tokenizer_src": tokenizer_src,
                            "tokenizer_trgt": tokenizer_trgt,
                            "model": model,
                            "src_len": src_len,
                            "trgt_len": trgt_len
                        }
                else:
                    data = torch.load(f"DIRS/{key}/data.pth")

                    src_vocab = data["src_vocab"]
                    trgt_vocab = data["trgt_vocab"]
                    tokenizer_src = data["tokenizer_src"]
                    tokenizer_trgt = data["tokenizer_trgt"]
                    src_len, trgt_len = self.lengths[key]

                    model = build_transformer(len(src_vocab), len(trgt_vocab), src_len, trgt_len, device=device).to(
                        device)
                    model.load_state_dict(torch.load(f"DIRS/{key}/brain.pth", map_location=device)["model_state"])
                    model.eval()
                    self.total_params += count_parameters(model)
                    self.models_dict[key] = {
                        "src_vocab": src_vocab,
                        "trgt_vocab": trgt_vocab,
                        "tokenizer_src": tokenizer_src,
                        "tokenizer_trgt": tokenizer_trgt,
                        "model": model,
                        "src_len": src_len,
                        "trgt_len": trgt_len,
                    }

        # print(f"Catalina Initialized Total Parameters: {(self.total_params / 1_000_000_000):.2f} Billion(s)\n\n")

    def get_system_mode(self, system_mode):

        device=self.device

        mode_of_model = self.models_dict_classifiers["mode"]

        max_seq_src = mode_of_model["src_len"]
        model = mode_of_model["model"]
        tokenizer_src = mode_of_model["tokenizer_src"]
        categories = mode_of_model["categories"]
        with torch.no_grad():
            x = clean_input_text_final(system_mode, tokenizer_src, max_seq_src, device)

            pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int64).to(device)
            source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)
            encoder_output = model.encode(x, source_mask).to(device)

            proj_output = model.project(encoder_output).mean(dim=1)

            _, predicted_label = torch.max(proj_output, dim=1)

            self.mode = decode_category(predicted_label, categories).lower()

    def get_response(self, input_text):
        device = self.device

        if self.mode in self.broken:return f"Sorry '{self.mode}' model is currently unusable."

        if self.mode not in self.models_dict_classifiers.keys():
            if self.mode == "poem":
                with torch.no_grad():
                    model = self.models_dict["poem"]["model"]
                    if "love" in input_text:
                        model.load_state_dict(torch.load("DIRS/poem/lovebrain.pth", map_location=device)["model_state"])
                    elif "life" in input_text:
                        model.load_state_dict(torch.load("DIRS/poem/lifebrain.pth", map_location=device)["model_state"])
                    else:
                        model.load_state_dict(torch.load("DIRS/poem/brain.pth", map_location=device)["model_state"])
                    model.eval()
                    all_characters = self.models_dict["poem"]["all_characters"]
                    return generate_text(model, all_characters, init_str=input_text, pred_len=500, temp=0.5,
                                         device=device)
            if self.mode == "translate":
                language_of_input = self._get_language(input_text, device=device)

                print(f"detected language {language_of_input}")
                translate_to = input(f"translate to {self.language_translatable[language_of_input]}:")
                if translate_to not in self.language_translatable[language_of_input]:
                    translate_to = self.language_translatable[language_of_input][0]
                    print(f"wrong input default translate to {translate_to}")

                mode_of_model = self.models_dict[self.mode][f"{language_of_input}_{translate_to}"]

                max_seq_src = mode_of_model["src_len"]
                max_seq_trgt = mode_of_model["trgt_len"]
                model = mode_of_model["model"]
                tokenizer_src = mode_of_model["tokenizer_src"]
                tokenizer_trgt = mode_of_model["tokenizer_trgt"]
            else:

                mode_of_model = self.models_dict[self.mode]

                max_seq_src = mode_of_model["src_len"]
                max_seq_trgt = mode_of_model["trgt_len"]
                model = mode_of_model["model"]
                tokenizer_src = mode_of_model["tokenizer_src"]
                tokenizer_trgt = mode_of_model["tokenizer_trgt"]

            if len(input_text.split()) > max_seq_src - 2:
                raise Exception(
                    f"This model only accepts input of length {max_seq_src - 2}. [{len(input_text.split())}>{max_seq_src - 2}]")

            with torch.no_grad():
                if self.mode == "emotion":
                    x = clean_input_text_final(input_text, tokenizer_src, max_seq_src, device, type_input=self.mode)
                else:
                    x = clean_input_text_final(input_text, tokenizer_src, max_seq_src, device)

                pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int64).to(device)
                source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

                encoder_output = model.encode(x, source_mask).to(device)
                decoder_input = torch.empty(1, 1).fill_(tokenizer_trgt.get("<SOS>")).type_as(x).to(device)

                output_text = ""
                predicteds = []
                while decoder_input.size(1) < max_seq_trgt:
                    decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))),
                                              diagonal=1).type(torch.int).type_as(source_mask).to(device)
                    out1 = model.decode(encoder_output, source_mask, decoder_input, decoder_mask).to(device)

                    prob = model.project(out1[:, -1]).to(device)
                    _, next_word = torch.max(prob, dim=1)
                    decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(x).fill_(next_word.item())],
                                              dim=1)

                    if decode_word(tokenizer_trgt, next_word.item()) == "<EOS>": break
                    word_text = decode_word(tokenizer_trgt, next_word.item()) + " "
                    if word_text not in predicteds:
                        predicteds.append(word_text)
                    output_text += word_text
                if self.mode=="emotion":return predicteds
                return output_text
        else:
            mode_of_model = self.models_dict_classifiers[self.mode]

            max_seq_src = mode_of_model["src_len"]
            model = mode_of_model["model"]
            tokenizer_src = mode_of_model["tokenizer_src"]
            categories = mode_of_model["categories"]

            if len(input_text.split()) > max_seq_src - 2:
                raise Exception(
                    f"This model only accepts input of length {max_seq_src - 2}. [{len(input_text.split())}>{max_seq_src - 2}]")

            with torch.no_grad():
                if self.mode == "names":
                    x = clean_input_name_gender(input_text, tokenizer_src, max_seq_src, device)
                elif self.mode == "ethics":
                    what_type = input("Type[c/j]:")
                    if what_type.lower() != "j":
                        type_input = "commonsense"
                    else:
                        type_input = "justice"
                    line = unidecode(remove_special(input_text.lower().strip())).split()
                    line.append("<t>")
                    line.append(type_input)
                    line.append("</t>")
                    line = line[:max_seq_src - 2]
                    line.insert(0, "<SOS>")
                    line.append("<EOS>")
                    line += ["<PAD>"] * (max_seq_src - len(line))

                    x = tokens_to_tensor([line], tokenizer_src, max_seq_src)[0].to(device)
                elif self.mode == "linkclassification":
                    x = clean_input_link(input_text, tokenizer_src, max_seq_src, device)
                else:
                    x = clean_input_text_final(input_text, tokenizer_src, max_seq_src, device)

                pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int64).to(device)
                source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)
                encoder_output = model.encode(x, source_mask).to(device)

                proj_output = model.project(encoder_output).mean(dim=1)

                _, predicted_label = torch.max(proj_output, dim=1)

                return decode_category(predicted_label, categories)

    def _get_language(self, input_text):
        device = self.device

        mode_of_model = self.models_dict_classifiers["language_detector"]

        max_seq_src = mode_of_model["src_len"]
        model = mode_of_model["model"]
        tokenizer_src = mode_of_model["tokenizer_src"]
        categories = mode_of_model["categories"]

        if len(input_text.split()) > max_seq_src - 2:
            raise Exception(
                f"This model only accepts input of length {max_seq_src - 2}. [{len(input_text.split())}>{max_seq_src - 2}]")

        with torch.no_grad():
            x = clean_input_text_final(input_text, tokenizer_src, max_seq_src, device)

            pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int64).to(device)
            source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)
            encoder_output = model.encode(x, source_mask).to(device)

            proj_output = model.project(encoder_output).mean(dim=1)

            _, predicted_label = torch.max(proj_output, dim=1)

            return decode_category(predicted_label, categories)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    catalina = Catalina(broken=["conversational","paraphrase","translate","language_detector","aihumantext"],device="cpu")

    while True:
        system_input = input("System input:")
        if len(system_input)>1:catalina.get_system_mode(system_input)
        out = catalina.get_response(input("Prompt:"))
        print(out)
