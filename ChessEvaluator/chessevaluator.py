import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
import warnings


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc01 = nn.Linear(832, 8000)
        self.bn01 = nn.BatchNorm1d(8000)
        self.dropout01 = nn.Dropout(0.3)

        self.fc0 = nn.Linear(8000, 5000)
        self.bn0 = nn.BatchNorm1d(5000)
        self.dropout0 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(5000, 3000)
        self.bn1 = nn.BatchNorm1d(3000)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(3000, 1500)
        self.bn2 = nn.BatchNorm1d(1500)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(1500, 800)
        self.bn3 = nn.BatchNorm1d(800)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(800, 400)
        self.bn4 = nn.BatchNorm1d(400)
        self.dropout4 = nn.Dropout(0.3)

        self.fc5 = nn.Linear(400, 200)
        self.bn5 = nn.BatchNorm1d(200)
        self.dropout5 = nn.Dropout(0.3)

        self.fc6 = nn.Linear(200, 100)
        self.bn6 = nn.BatchNorm1d(100)
        self.dropout6 = nn.Dropout(0.3)

        self.fc7 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = self.dropout01(F.relu(self.bn01(self.fc01(x))))
        x = self.dropout0(F.relu(self.bn0(self.fc0(x))))
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
        x = self.dropout5(F.relu(self.bn5(self.fc5(x))))
        x = self.dropout6(F.relu(self.bn6(self.fc6(x))))
        x = self.fc7(x)

        return x



class ChessDataset(Dataset):
    def __init__(self, data_frame):
        self.fens = torch.from_numpy(np.array([*map(fen_to_bit_vector, data_frame["FEN"])], dtype=np.float32))
        self.evals = torch.Tensor([[x] for x in data_frame["Evaluation"]])
        self._len = len(self.evals)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.fens[index], self.evals[index]


def eval_to_int(evaluation):
    try:
        res = int(evaluation)
    except ValueError:
        res = 10000 if evaluation[1] == '+' else -10000
    return res / 100


def fen_to_bit_vector(fen):
    parts = re.split(" ", fen)
    piece_placement = re.split("/", parts[0])
    active_color = parts[1]
    castling_rights = parts[2]
    en_passant = parts[3]
    halfmove_clock = int(parts[4])
    fullmove_clock = int(parts[5])

    bit_vector = np.zeros((13, 8, 8), dtype=np.uint8)

    piece_to_layer = {
        'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5, 'P': 6,
        'p': 7, 'k': 8, 'q': 9, 'b': 10, 'n': 11, 'r': 12
    }

    castling = {
        'K': (7, 7), 'Q': (7, 0), 'k': (0, 7), 'q': (0, 0),
    }

    for r, row in enumerate(piece_placement):
        c = 0
        for piece in row:
            if piece in piece_to_layer:
                bit_vector[piece_to_layer[piece], r, c] = 1
                c += 1
            else:
                c += int(piece)

    if en_passant != '-':
        bit_vector[0, ord(en_passant[0]) - ord('a'), int(en_passant[1]) - 1] = 1

    if castling_rights != '-':
        for char in castling_rights:
            bit_vector[0, castling[char][0], castling[char][1]] = 1

    bit_vector[0, 7, 4] = 1 if active_color == 'w' else 0

    for value, start_row in zip([halfmove_clock, fullmove_clock], [3, 4]):
        c = 7
        while value > 0:
            bit_vector[0, start_row, c] = value % 2
            value //= 2
            c -= 1
            if c < 0:
                break

    return bit_vector


def train_model(net, trainloader, device, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'Epoch [{epoch+1}], Batch [{i+1}], Loss: {running_loss / (2000 * len(labels)):.3f}')
                running_loss = 0.0
        print("Saving the model...")
        torch.save(net.state_dict(), f'epoch/chess{epoch+1}.pth')


def evaluate_model(net, testloader, device, criterion):
    total_loss, count = 0, 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            count += len(labels)
    print(f'Average error of the model on {count} positions: {total_loss / count:.3f}')


def predict_fen(net, fen, device):
    net.eval()
    with torch.no_grad():
        bit_vector = torch.from_numpy(np.array([fen_to_bit_vector(fen)], dtype=np.float32)).to(device)
        evaluation = net(bit_vector)
    return evaluation.item() * 100


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # MAX_DATA = 1000000
    # print("Preparing Training Data...")
    # train_data = pd.read_csv("chessData.csv").sample(frac=1, random_state=42).head(MAX_DATA)
    # train_data["Evaluation"] = train_data["Evaluation"].map(eval_to_int)
    # trainset = ChessDataset(train_data)

    # print("Preparing Test Data...")
    # test_data = pd.read_csv("tactic_evals.csv").sample(frac=1, random_state=42).head(200000)
    # test_data["Evaluation"] = test_data["Evaluation"].map(eval_to_int)
    # testset = ChessDataset(test_data)

    # torch.save(trainset, "trainset.pt")
    # torch.save(testset, "testset.pt")

    # batch_size = 100
    # trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)
    # testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net().to(device)
    net.load_state_dict(torch.load('chess10.pth', map_location=device))
    print(f"Number of trainable parameters: {count_parameters(net)}")
    # exit()
    # criterion = nn.MSELoss()
    # optimizer = optim.AdamW(net.parameters())

    # print("Training the model...")
    # train_model(net, trainloader, device, optimizer, criterion)

    # print("Evaluating the model...")
    # evaluate_model(net, testloader, device, criterion)

    test_fen = "r3k2r/ppp2p1p/n7/N1bP3p/5pbq/6n1/PPP3PP/R1B2R1K w kq - 4 14"
    print(f"Prediction for FEN: {test_fen} is {predict_fen(net, test_fen, device):.3f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
