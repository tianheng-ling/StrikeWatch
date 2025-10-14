import torch
import torch.nn as nn

from .LSTMCell import LSTMCell


class RNNLayer(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.inputs_size = kwargs.get("inputs_size")
        self.hidden_size = kwargs.get("hidden_size")
        self.window_size = kwargs.get("window_size")
        self.batch_size = kwargs.get("batch_size")
        self.cell_type = kwargs.get("cell_type")

        self.rnn_cell = LSTMCell(
            inputs_size=self.inputs_size, hidden_size=self.hidden_size
        )

    def forward(
        self,
        inputs: torch.FloatTensor,
        h_prev: torch.FloatTensor,
        c_prev: torch.FloatTensor,
    ) -> torch.FloatTensor:

        outputs = torch.zeros(self.batch_size, self.window_size, self.hidden_size).to(
            inputs.device
        )
        for t in range(self.window_size):
            h_next, c_next = self.rnn_cell(inputs[:, t, :], h_prev, c_prev)
            outputs[:, t, :] = h_next
            h_prev, c_prev = h_next, c_next

        return outputs, h_next, c_next
