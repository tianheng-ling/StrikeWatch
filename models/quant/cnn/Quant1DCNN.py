import torch
import torch.nn as nn

from elasticai.creator.nn.integer.conv1dbn import Conv1dBN
from elasticai.creator.nn.integer.maxpooling import MaxPooling1d
from elasticai.creator.nn.integer.avgpooling1dflatten import AVGPooling1dFlatten
from elasticai.creator.nn.integer.linear import Linear
from elasticai.creator.nn.integer.relu import ReLU
from elasticai.creator.nn.integer.linearrelu import LinearReLU
from elasticai.creator.nn.integer.sequential import Sequential
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class Quant1DCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs.get("num_in_features")
        seq_len = kwargs.get("seq_len")
        num_blocks = kwargs.get("num_blocks")
        assert num_blocks >= 1, "num_blocks should be bigger or equal to 1"
        le_classes = kwargs.get("num_out_features")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.enable_int_forward = kwargs.get("enable_int_forward")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        self.layers = nn.ModuleList()
        tmp_seq_len = seq_len
        tmp_channels = in_channels

        for i in range(num_blocks):
            out_channels = tmp_channels if i < 2 else tmp_channels * 2

            self.layers.append(
                Conv1dBN(
                    in_channels=tmp_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding="same",
                    seq_len=tmp_seq_len,
                    name=f"conv1dbn_{i}",
                    quant_bits=self.quant_bits,
                    quant_data_dir=self.quant_data_dir,
                    device=device,
                )
            )
            self.layers.append(
                ReLU(
                    name=f"relu_{i}",
                    quant_bits=self.quant_bits,
                    quant_data_dir=self.quant_data_dir,
                    device=device,
                )
            )
            if i != num_blocks - 1:
                self.layers.append(
                    MaxPooling1d(
                        in_features=tmp_seq_len,
                        out_features=tmp_seq_len // 2,
                        in_num_dimensions=out_channels,
                        out_num_dimensions=out_channels,
                        kernel_size=2,
                        name=f"maxpooling1d_{i}",
                        quant_bits=self.quant_bits,
                        quant_data_dir=self.quant_data_dir,
                        device=device,
                    )
                )
                tmp_seq_len = tmp_seq_len // 2

            tmp_channels = out_channels

        # Global Average Pooling with Flattening
        self.layers.append(
            AVGPooling1dFlatten(
                in_features=tmp_seq_len,
                out_features=1,
                in_num_dimensions=tmp_channels,
                out_num_dimensions=tmp_channels,
                name="avgpooling1dflatten_0",
                quant_bits=self.quant_bits,
                quant_data_dir=self.quant_data_dir,
                device=device,
            )
        )

        # Fully Connected Layers
        self.layers.append(
            LinearReLU(
                name="linearrelu_0",
                in_features=tmp_channels,
                out_features=tmp_channels // 2,
                bias=True,
                quant_bits=self.quant_bits,
                quant_data_dir=self.quant_data_dir,
                device=device,
            )
        )
        self.layers.append(
            Linear(
                name="linear_0",
                in_features=tmp_channels // 2,
                out_features=le_classes,
                bias=True,
                quant_bits=self.quant_bits,
                quant_data_dir=self.quant_data_dir,
                device=device,
            )
        )

        self.sequential = Sequential(
            *self.layers,
            name=self.name,
            quant_data_dir=self.quant_data_dir,
        )

    def forward(
        self,
        inputs: torch.FloatTensor,
        enable_simquant: bool = True,
    ) -> torch.FloatTensor:

        # If the input shape is (batch_size, seq_len, in_channels),
        # convert it to (batch_size, in_channels, seq_len) for the Conv1d layer
        if inputs.shape[1] != self.layers[0].in_channels:
            inputs = inputs.permute(0, 2, 1)

        if self.enable_int_forward:
            self.sequential.precompute()
            inputs = inputs.to("cpu")
            q_inputs = self.sequential.quantize_inputs(inputs)
            save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

            q_outputs = self.sequential.int_forward(q_inputs)

            save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
            dq_outputs = self.sequential.dequantize_outputs(q_outputs)
            return dq_outputs
        else:
            outputs = self.sequential.forward(inputs, enable_simquant=enable_simquant)
            return outputs
