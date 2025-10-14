import torch
import torch.nn as nn

from .RNNLayer import RNNLayer
from config import DEVICE


class StackedRNN(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.inputs_size = kwargs.get("inputs_size")
        self.hidden_size = kwargs.get("hidden_size")
        self.window_size = kwargs.get("window_size")
        self.batch_size = kwargs.get("batch_size")
        self.num_rnn_layers = kwargs.get("num_rnn_layers")
        self.cell_type = kwargs.get("cell_type")
        self.rnn_layers = nn.ModuleList()

        for i in range(self.num_rnn_layers):
            self.rnn_layers.append(
                RNNLayer(
                    inputs_size=self.inputs_size if i == 0 else self.hidden_size,
                    hidden_size=self.hidden_size,
                    window_size=self.window_size,
                    cell_type=self.cell_type,
                    batch_size=self.batch_size,
                )
            )

    def forward(
        self,
        inputs: torch.FloatTensor,
    ) -> torch.FloatTensor:

        h_prev = torch.zeros(self.batch_size, self.hidden_size).to(DEVICE)
        c_prev = torch.zeros(self.batch_size, self.hidden_size).to(DEVICE)

        for layer in self.rnn_layers:
            outputs, h_next, c_next = layer(inputs, h_prev, c_prev)
            inputs, h_prev, c_prev = outputs, h_next, c_next

        return outputs[:, -1, :]
