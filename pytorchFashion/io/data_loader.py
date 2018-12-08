#encoding:utf-8
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from ..preprocessing import image


class ImageDataIter(object):
    '''
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    '''
    def __init__(self, data_dir,
                 image_size,
                 batch_size,
                 shuffle,
                 num_workers,
                 mode,
                 resize = None,
                 rotate = None,
                 random_crop=None,
                 normailizer = None,
                 horizontallyFlip = None,
                 verticalFlip = None,
                 pin_memory = True,
                 brightness=0,
                 contrast=0,
                 saturation=0,
                 hue=0):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.mode = mode
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.trfs = []

        # 对PIL image做resize操作的
        # 这里的输入可以是int，此时表示将输入图像的
        # 短边resize到这个int数，长边则根据数据进行调整
        # 保证图像的长宽比不变
        if resize:
            self.trfs.append(transforms.Resize(resize))

        # 随机旋转图像
        if rotate:
            self.trfs.append(transforms.RandomRotation(rotate))

        # random size的crop
        if random_crop:
            self.trfs.append(transforms.RandomSizedCrop(image_size))

        # 图像像素的调整
        if brightness or contrast or saturation or hue:
            self.trfs.append(transforms.ColorJitter(brightness = brightness, #亮度
                                                    contrast = contrast,     # 对比度
                                                    saturation = saturation, # 饱和度
                                                    hue = hue))              # 色度
        # 随机的图像水平翻转，即图像的左右对调
        if horizontallyFlip:
            self.trfs.append(transforms.RandomHorizontalFlip())

        # 随机的将图像竖直翻转，即图像的上下对调
        if verticalFlip:
            self.trfs.append(transforms.RandomVerticalFlip())
        # 转化为Tensor
        # 注意：ToTensor()操作中，已经默认对图像像素 / 255.0操作了
        self.trfs.append(transforms.ToTensor())

        # 标准化
        if normailizer:
            if isinstance(normailizer,dict):
                self.trfs.append(transforms.Normalize(mean=normailizer['mean'],std = normailizer['std']))
            else:
                self.trfs.append(image.normalize())

    def _dataLoad(self):
        dataset = datasets.ImageFolder(
                root = self.data_dir,
                transform=transforms.Compose(self.trfs)
            )
        data_loader = DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory
        )
        return data_loader

    def data_loader(self):
        return self._dataLoad()

