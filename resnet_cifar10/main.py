import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
from PIL import Image
from utils import Flatten

cifar_train = datasets.CIFAR10(download=True, root='cifar10', train=True, transform=transforms.Compose([
    # lambda x: Image.open(x).convert('RGB'),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet 数据中的均值和方差

]))
cifar_test = datasets.CIFAR10(download=True, root='cifar10', train=False, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet 数据中的均值和方差

]))

print(len(cifar_train), len(cifar_test))

train_Dataloader = DataLoader(cifar_train, batch_size=100, num_workers=4, shuffle=True)
test_Dataloader = DataLoader(cifar_test, batch_size=100, num_workers=4, shuffle=True)

device = torch.device('cuda:0')
epochs = 100


def main():
    base_model = models.resnet18(pretrained=True)
    model = nn.Sequential(
        *(list(base_model.children())[:-1]),  # 输出 torch.Size([b, 512, 1, 1])
        Flatten(),  # [b,512]
        nn.Linear(512, 10)  # --> [b,10]
    ).to(device)
    # x=torch.randn(100,3,224,224)
    # y=model(x)
    # print(y.shape)

    # ****************************** train ******************************
    model.train()
    model.load_state_dict(torch.load(r'./best_model'))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criteon = nn.CrossEntropyLoss().to(device)

    epoch_acc, best_acc, best_epoch = 0, 0, 0

    corrects = []
    for epoch in range(epochs):
        print('epoch : ', epoch)
        for idx, (x, y) in enumerate(train_Dataloader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criteon(logits, y)

            pred = logits.argmax(dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct = torch.eq(pred, y).sum().float().item() / 100  # 每个batch的正确率,每个batch 100张

            corrects.append(correct)

            if (idx + 1) % 100 == 0:
                print('100 batch acc : ', np.mean(corrects))

        epoch_acc = np.mean(corrects)  # 一个epoch上的正确率
        print('epoch {} acc is {}'.format(epoch, epoch_acc))
        if epoch_acc > best_acc:
            best_epoch = epoch
            best_acc = epoch_acc
            # save model state_dict
            torch.save(model.state_dict(), r'./checkpoint/best_model')
    print('\n\n\n')
    print('*****************************')
    print('best epoch {} , best acc {}'.format(best_epoch, best_acc))

    # ************************** test **************************

    model.load_state_dict(torch.load(r'./checkpoint/best_model'))
    print('load done!')

    model.eval()

    acc = 0

    for x, y in test_Dataloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        acc += torch.eq(pred, y).sum().float().item()
    acc = acc / len(test_Dataloader.dataset)

    print('test acc is {}'.format(acc))


if __name__ == '__main__':
    main()
