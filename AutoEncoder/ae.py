import torch
from torch import nn


class AE(nn.Module):

    def __init__(self):
        super().__init__()

        # [b,784]  -->  [b,20]
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
        # [b , 20]  -->  [b,784]
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

        # x : [b,1,28,28]
    def forward(self, x):
        # x : [b,1,28,28]  -->  [b,784]

        batchsize = x.size(0)

        x = x.reshape(batchsize,-1)

        h=self.encoder(x)
        r=self.decoder(h)

        r=r.reshape(batchsize,1,28,28)

        return r