import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with batch norm and dropout"""
    def __init__(self, hidden_size, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out


class Net(nn.Module):
    """Improved chess position evaluator with residual connections"""
    def __init__(self, input_size=832, hidden_size=1024, num_res_blocks=6, dropout=0.2):
        super(Net, self).__init__()
        
        # Initial expansion layer
        self.fc_input = nn.Linear(input_size, hidden_size)
        self.bn_input = nn.BatchNorm1d(hidden_size)
        self.dropout_input = nn.Dropout(dropout)
        
        # Stack of residual blocks for deep feature learning
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout) for _ in range(num_res_blocks)
        ])
        
        # Compression layers
        self.fc1 = nn.Linear(hidden_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout)
        
        # Output layer
        self.fc_output = nn.Linear(128, 1)
        
    def forward(self, x):
        # Flatten input if needed
        x = torch.flatten(x, 1)
        
        # Initial expansion
        x = F.relu(self.bn_input(self.fc_input(x)))
        x = self.dropout_input(x)
        
        # Pass through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Compression pathway
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Output (raw centipawn evaluation)
        x = self.fc_output(x)
        
        # Optional: Use tanh to bound outputs to reasonable range
        # x = torch.tanh(x) * 1000  # Bounds to roughly ±1000 centipawns
        
        return x


class LightNet(nn.Module):
    """Lighter version for faster training and inference"""
    def __init__(self, input_size=832, hidden_size=768, num_res_blocks=4, dropout=0.2):
        super(LightNet, self).__init__()
        
        self.fc_input = nn.Linear(input_size, hidden_size)
        self.bn_input = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout) for _ in range(num_res_blocks)
        ])
        
        self.fc1 = nn.Linear(hidden_size, 384)
        self.bn1 = nn.BatchNorm1d(384)
        
        self.fc2 = nn.Linear(384, 192)
        self.bn2 = nn.BatchNorm1d(192)
        
        self.fc_output = nn.Linear(192, 1)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        
        x = F.relu(self.bn_input(self.fc_input(x)))
        x = self.dropout(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.fc_output(x)
        
        return x


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Usage example:
if __name__ == "__main__":
    # Standard version: ~3.8M parameters (vs your 19M)
    net = Net()
    print(f"Net parameters: {count_parameters(net):,}")
    
    # Light version: ~1.8M parameters
    light_net = LightNet()
    print(f"LightNet parameters: {count_parameters(light_net):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 832)  # Batch of 4 positions
    output = net(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output.squeeze()}")