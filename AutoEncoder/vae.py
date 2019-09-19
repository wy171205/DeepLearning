import torch
from torch import nn


class VAE(nn.Module):

    def __init__(self):
        super().__init__()

        # [b,784]  -->  [b,20]
        # u [b,10]
        # sigma [b,10]

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
            nn.Linear(10, 64),
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

        h=self.encoder(x) # h [b,20]

        # [b,20]  -->  [b,10]  and  [b,10]
        u , sigma = h.chunk(2,dim=1)

        h=u+sigma*torch.randn_like(sigma)

        r=self.decoder(h)

        r_=r.reshape(batchsize,1,28,28)

        # kld = torch.zeros(1,1).cuda()
        kld = 0.5 * torch.sum(
            torch.pow(u, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (32 * 28 * 28)

        # for i in range(x.size(0)):
        #     px=(x[i]/torch.sum(x[i])).unsqueeze(0)
        #     pr=(r[i]/torch.sum(r[i])).unsqueeze(0)
        #     kld+=px@torch.log(1e-8+px/pr).t()

            # print(px.shape , pr.shape,kld)

        return r_,kld