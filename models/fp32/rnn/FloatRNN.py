import torch
import torch.nn as nn

from config import DEVICE
from .StackedRNN import StackedRNN


class FloatRNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.num_in_features = kwargs.get("num_in_features")
        self.hidden_size = kwargs.get("hidden_size")
        self.window_size = kwargs.get("seq_len")
        self.batch_size = kwargs.get("batch_size")
        self.num_rnn_layers = kwargs.get("num_rnn_layers")
        self.cell_type = kwargs.get("cell_type")
        self.num_out_features = kwargs.get("num_out_features")

        self.stacked_rnn = StackedRNN(
            inputs_size=self.num_in_features,
            hidden_size=self.hidden_size,
            window_size=self.window_size,
            cell_type=self.cell_type,
            batch_size=self.batch_size,
            num_rnn_layers=self.num_rnn_layers,
        ).to(DEVICE)

        self.output_layer = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_out_features,
            bias=True,
        ).to(DEVICE)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:

        outputs = self.stacked_rnn(inputs)
        outputs = self.output_layer(outputs)
        return outputs
