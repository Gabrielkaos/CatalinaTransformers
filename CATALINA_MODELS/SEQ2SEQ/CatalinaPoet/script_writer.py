import torch
import unidecode
import string
from MODEL_TRANSFORMER.OLD import RNN
from data_utils import char_tensor



def generate_text(model1, all_characters1, init_str="A", pred_len=100, temp=0.50):
    hiddens, cells = model1.init_hidden(1)
    init_input = char_tensor(init_str, all_characters1)

    predicted = init_str

    for p in range(len(init_str) - 1):
        _, (hiddens, cells) = model1(init_input[p].view(1), hiddens, cells)

    last_char = init_input[-1]

    for p in range(pred_len):
        outs, (hiddens, cells) = model1(last_char.view(1), hiddens, cells)
        out_distance = outs.data.view(-1).div(temp).exp()
        top_char = torch.multinomial(out_distance, 1)[0]
        pred_char = all_characters1[top_char]
        predicted += pred_char
        last_char = char_tensor(pred_char, all_characters1)

    return predicted


if __name__ == "__main__":

    # hyper parameters
    n_layers = 8
    n_hidden = 200

    # DATA
    all_characters = string.printable
    n_characters = len(all_characters)

    # MODEL
    model = RNN(n_characters, n_hidden, n_layers, n_characters)
    model.load_state_dict(torch.load("brain.pth")["model_state"])
    model.eval()
    with torch.no_grad():

        while True:
            input_str = unidecode.unidecode(input("\nGab:"))
            #
            # add(input_str,"poet")

            print("CatalinaPoet:\n")
            print(generate_text(model, all_characters, init_str=input_str, pred_len=500, temp=0.5))
