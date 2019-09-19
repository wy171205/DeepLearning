import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim
from ae import AE
import visdom
from vae import VAE
from torchvision import models


def main():
    vis = visdom.Visdom()

    tf = transforms.Compose([
        transforms.ToTensor(),  # 转换成tensor 格式，并且压缩到[0 ,1]

    ])
    # 构建训练集  测试集
    mnist_train = datasets.MNIST('mnist', train=True, transform=tf, download=True)

    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)

    mnist_test = datasets.MNIST('mnist', train=False, transform=tf, download=True)

    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)

    x, _ = next(iter(mnist_train))

    print("x:", x.shape)

    # **************************  train  **************************
    device = torch.device('cuda:0')

    model_vae = VAE().to(device)
    criteon = nn.MSELoss().to(device)
    optimizer = optim.Adam(model_vae.parameters(), lr=1e-3)
    print(model_vae)
    loss = None
    kld=None
    for epoch in range(20 ):
        for idx, (x, _) in enumerate(mnist_train):
            x = x.to(device)
            x_hat,kld = model_vae(x)

            # print(x.shape , x_hat.shape)

            loss = criteon(x, x_hat)+1.0*kld



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch {} loss: {} include kld loss: {}'.format(epoch, loss.item(),kld.item()))


        if epoch % 1 == 0:  # 每1次epoch做一次可视化
            vis.line([loss.item()], [epoch], win='train loss', update='append', opts=dict(
                title='train loss', xlabel='epoch', ylabel='loss'
            ))

    #******************** test ************************
    x,_=next(iter(mnist_test))
    x=x.to(device)
    with torch.no_grad():
        x_hat,_ = model_vae(x)  # x : [32,1,28,28]  32 张图片
    vis.images(x,nrow=8,win='x source',opts=dict(
        title = 'x source'
    ))
    vis.images(x_hat,win='x hat',nrow=8,opts=dict(title = 'x hat'))




if __name__ == '__main__':
    main()


