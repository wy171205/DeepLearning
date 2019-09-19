from PIL import Image
from torchvision import transforms, datasets
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# tf = transforms.Compose([
#         transforms.ToTensor(),  # 转换成tensor 格式，并且压缩到[0 ,1]
#
#     ])
#     # 构建训练集  测试集
# mnist_train = datasets.MNIST('mnist', train=True, transform=tf, download=True)
#
# mnist_train_dataloader = DataLoader(mnist_train,batch_size=32)
#
# x,_ =next(iter(mnist_train_dataloader))
#
#
#
# print(x.shape,x[1].shape)
#
# tf2 = transforms.ToPILImage()
#
#
# print('x[1]')
#
# plt.imshow(np.array(x[1]).reshape(28,28)*255)
#
#
# fig = plt.figure()
#
# for i in range(x.size(0)):  # x: [0 ~ 31]
#     plt.subplot(4,8,i+1)
#     img = tf2(x[i]).convert('L')  # to PIL img
#     img=np.array(img)
#     print(img.shape,type(img),type(x[i]))
#     # img=img.reshape(28,28,)
#     # # print(img.shape, type(img))
#     plt.imshow(img)
#
# plt.show()

# a = torch.randn(3,4)
# print(a)
# print(a.argmax(dim= 1))
#
# print(a.topk(2,dim=1)[1])

dir = os.listdir(os.path.join('mnist', 'MNIST'))

print(dir)
