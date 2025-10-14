import torch.nn as nn

from .FlattenedAdaptiveAvgPool1d import FlattenedAdaptiveAvgPool1d


class Float1DCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs.get("num_in_features")
        num_blocks = kwargs.get("num_blocks")
        assert num_blocks >= 1, "num_blocks should be bigger or equal to 1"
        le_classes = kwargs.get("num_out_features")

        self.layers = nn.ModuleList()
        tmp_channels = in_channels

        for i in range(num_blocks):
            out_channels = tmp_channels if i < 2 else tmp_channels * 2
            self.layers.append(
                nn.Conv1d(
                    in_channels=tmp_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding="same",
                )
            )
            self.layers.append(nn.BatchNorm1d(num_features=out_channels))
            self.layers.append(nn.ReLU())

            if i != num_blocks - 1:
                self.layers.append(nn.MaxPool1d(kernel_size=2))

            tmp_channels = out_channels

        # Global Average Pooling with Flattening
        self.layers.append(FlattenedAdaptiveAvgPool1d(output_size=1))

        # Fully Connected Layers (FC + ReLU + Dropout+ FC)
        self.layers.append(
            nn.Linear(in_features=tmp_channels, out_features=tmp_channels // 2)
        )
        self.layers.append(nn.ReLU())
        self.layers.append(
            nn.Linear(in_features=tmp_channels // 2, out_features=le_classes)
        )

    def forward(self, x):
        # If the input shape is (batch_size, seq_len, in_channels),
        # convert it to (batch_size, in_channels, seq_len) for the Conv1d layer
        if x.shape[1] != self.layers[0].in_channels:
            x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        return x
