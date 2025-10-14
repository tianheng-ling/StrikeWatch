import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)

    def forward(self, inputs: torch.FloatTensor):
        outputs = self.fc1(inputs)
        outputs = F.relu(outputs)
        outputs = self.fc2(outputs)
        return outputs
