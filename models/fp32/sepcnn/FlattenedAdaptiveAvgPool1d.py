import torch.nn as nn


class FlattenedAdaptiveAvgPool1d(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x):
        x = self.avg_pool(x)
        return x.view(x.size(0), -1)
