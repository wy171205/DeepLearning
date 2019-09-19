import torch.nn as nn


# [b,512,1,1,] --> [b,512]
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shape = x.size(1)
        x = x.reshape(-1, shape)
        return x
