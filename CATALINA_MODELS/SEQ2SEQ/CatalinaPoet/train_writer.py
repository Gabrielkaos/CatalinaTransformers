import torch
import torch.nn as nn
import unidecode
from MODEL_TRANSFORMER import RNN
import string
from data_utils import char_tensor


def get_random_batch(batch_num, all_characters1, file1, chunk_len1, counter_idx1, len_file1):
    text_input = torch.zeros(batch_num, chunk_len1)
    text_target = torch.zeros(batch_num, chunk_len1)

    for i in range(batch_num):
        end_idx = counter_idx1 + chunk_len1 + 1
        str_input = file1[counter_idx1:end_idx]

        text_input[i, :] = char_tensor(str_input[:-1], all_characters1)
        text_target[i, :] = char_tensor(str_input[1:], all_characters1)

        counter_idx1 += 1
        if counter_idx1 >= (len_file1 - chunk_len1): counter_idx1 = 0

    return text_input.long(), text_target.long(), counter_idx1


def save(save_file, model1):
    data_model = {
        "model_state": model1.state_dict()
    }

    torch.save(data_model, save_file)

    print("MODEL SAVED\n")


if __name__ == "__main__":
    # hyper parameters
    chunk_len = 200
    batch_size = 250
    lr = 3e-4
    n_epoch = 1_000_000
    n_layers = 8
    n_hidden = 200

    # DATA
    file = unidecode.unidecode(open("data/poem/poem.txt", encoding='utf-8').read())
    counter_idx = 0
    len_file = len(file)
    all_characters = string.printable
    n_characters = len(all_characters)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")
    print(f"Device memory={(torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3):.3f} GB")

    # MODEL
    model = RNN(n_characters, n_hidden, n_layers, n_characters,device=device).to(device)
    model.load_state_dict(torch.load("data/poem/brain.pth")["model_state"])

    # LOSS and OPTIMIZER
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TRAINING
    model.train()
    print("\nTraining Started")
    print(f"Chunk len={len_file-chunk_len}")
    last_counter = counter_idx
    epoch = 0
    running_loss = 0.0
    length_loss = 0
    while epoch < n_epoch:
        inputs, target, counter_idx = get_random_batch(batch_size, all_characters, file, chunk_len, counter_idx, len_file)


        inputs = inputs.to(device)
        target = target.to(device)

        hidden, cell = model.init_hidden(batch_size)

        model.zero_grad()
        loss = 0

        for c in range(chunk_len):

            if c == 0:
                out, (hidden, cell) = model(inputs[:, c], hidden, cell)
            else:
                out, (hidden, cell) = model(target[:, c - 1], hidden, cell)

            loss += criterion(out, target[:, c])

        loss.backward()
        optimizer.step()
        loss = loss.item() / chunk_len

        if ((length_loss + 1) % 5) == 0:
            print(f"-->Epoch: {epoch + 1}, counter: {counter_idx}, Loss: {loss:.4f}")

        running_loss+=loss
        length_loss+=1

        if last_counter > counter_idx:
            print(f"<--Epoch: {epoch + 1}, Running Loss: {(running_loss/length_loss):.4f}")
            save(f"loss_update/script{epoch+1}-{(running_loss/length_loss):.4f}.pth",model)
            running_loss=0.0
            length_loss=0
            epoch+=1
        last_counter = counter_idx
