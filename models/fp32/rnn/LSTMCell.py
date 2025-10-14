import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, inputs_size: int, hidden_size: int) -> None:
        super().__init__()

        self.i_gate_linear = nn.Linear(inputs_size + hidden_size, hidden_size)
        self.f_gate_linear = nn.Linear(inputs_size + hidden_size, hidden_size)
        self.c_gate_linear = nn.Linear(inputs_size + hidden_size, hidden_size)
        self.o_gate_linear = nn.Linear(inputs_size + hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(
        self,
        inputs: torch.FloatTensor,
        h_prev: torch.FloatTensor,
        c_prev: torch.FloatTensor,
    ) -> torch.FloatTensor:

        combined = torch.cat((inputs, h_prev), dim=1)

        i_gate = self.sigmoid(self.i_gate_linear(combined))
        f_gate = self.sigmoid(self.f_gate_linear(combined))
        c_gate = self.tanh(self.c_gate_linear(combined))
        o_gate = self.sigmoid(self.o_gate_linear(combined))

        c_next = f_gate * c_prev + i_gate * c_gate
        h_next = o_gate * self.tanh(c_next)

        return h_next, c_next
