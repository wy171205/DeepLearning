import torch
import os, csv
import random, glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class Pokeman(Dataset):
    def __init__(self, root, resize, mode):
        """
        数据集初始化
        :param root: path
        :param resize: 重新设定图片的大小
        :param mode:  train 或 test
        """
        super().__init__()

        self.root = root
        self.resize = resize
        self.mode = mode

        self.name2label = {}  # 目录名到数字标签的映射
        for name in sorted(os.listdir(os.path.join(root))):
            # 把不是目录的过滤掉
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())

        self.images, self.labels = self.load_csv('images_path_and_label.csv')

        if mode == 'train':  # 0.6 作为train_data

            self.images = self.images[:int(.6 * len(self.images))]
            self.labels = self.labels[:int(.6 * len(self.labels))]

        elif mode == 'val':  # .6 ~ .8 --> valdation
            self.images = self.images[int(.6 * len(self.images)):int(.8 * len(self.images))]
            self.labels = self.labels[int(.6 * len(self.labels)):int(.8 * len(self.labels))]
        else:  # .8 ~ --> test
            self.images = self.images[int(.8 * len(self.images)):]
            self.labels = self.labels[int(.8 * len(self.labels)):]

        # print(self.name2label)
        # {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}

    # 用csv文件存储 image_path ， label
    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):

            image_path = []
            for name in self.name2label.keys():
                image_path.extend(glob.glob(os.path.join(self.root, name, '*')))

            # print('length:',len(image_path),image_path[1])
            # length: 1168 .\data\pokeman\bulbasaur\00000001.png

            random.shuffle(image_path)  # 打乱顺序
            # 写入CSV文件

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in image_path:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # [.\data\pokeman\bulbasaur\00000001.png , 0]
                    writer.writerow([img, label])
            print('write done!')

        # 读取scv文件
        image_paths, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img_path, label = row
                label = int(label)

                image_paths.append(img_path)
                labels.append(label)

        assert len(image_paths) == len(labels)
        assert isinstance(labels[1], int)

        return image_paths, labels

    def __len__(self):  # 不能有str
        # TODO 返回数据集的长度
        return len(self.images)

    def __getitem__(self, idx):
        # TODO 根据索引获取一个数据
        # idx~[0~len(self.images)]
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),  # 数据被归一到【0,1】 ,
            # 【-1 ，1】 ， 想要显示图片要进行反归一化
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet 数据中的均值和方差

        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label

    def denormalize(self, x_hat):
        """
        反normalize
        :return:
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean) / std
        mean = torch.tensor(mean).reshape(3, 1, 1)
        std = torch.tensor(std).reshape(3, 1, 1)

        x = x_hat * std + mean

        return x


def main():
    import visdom
    import time
    import torchvision

    # tf = transforms.Compose([
    #     lambda x: Image.open(x).convert('RGB'),
    #     transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
    #     transforms.RandomRotation(15),
    #     transforms.CenterCrop(self.resize),
    #     transforms.ToTensor(),  # 数据被归一到【0,1】 ,
    #     # 【-1 ，1】 ， 想要显示图片要进行反归一化
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet 数据中的均值和方差
    #
    # ])
    #
    # db = torchvision.datasets.ImageFolder(r'.\data\pokeman',tf)
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    #
    # for x, y in loader:  # 每次加载一个batch
    #     vis.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
    #     vis.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    #
    #     time.sleep(10)



    vis = visdom.Visdom()

    db = Pokeman(r'.\data\pokeman', resize=70, mode='train', )

    img, label = next(iter(db))

    print('sample:', img.shape, label.shape)
    # vis.image(img,win='sample1',opts=dict(title = 'sample label : {}'.format(label),))

    vis.image(db.denormalize(img), win='sample1', opts=dict(title='sample label : {}'.format(label), ))

    loader = DataLoader(db, batch_size=32, shuffle=True)

    for x, y in loader:  # 每次加载一个batch
        vis.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        vis.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(10)


if __name__ == '__main__':
    main()


