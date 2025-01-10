import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations=6, n_actions=4, n_layers=5, hidden_size=256):
        super(DQN, self).__init__()
        
        self.input_layer = nn.Linear(n_observations, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x