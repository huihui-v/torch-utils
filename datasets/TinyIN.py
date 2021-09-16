import os

import glob
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


class TinyIN(Dataset):
    def __init__(self, dataroot, transform=T.Compose([]), train=True, max_n_per_class=10000):
        self.dataroot_param = dataroot
        self.dataroot = os.path.join(dataroot, 'tiny-imagenet-200')
        self.train = train
        # self.split = split

        self.max_n_per_class = max_n_per_class

        self.transform = transform

        with open(os.path.join(self.dataroot, "wnids.txt"), "r") as f:
            self.classes = f.readlines()
            self.classes = [c.strip() for c in self.classes]
        self.class_num = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}

        if train:
            self.img_paths = glob.glob(os.path.join(self.dataroot, "train", "*", "images", "*.JPEG"))
            self.targets = [p.split("/")[-3] for p in self.img_paths]
            self.targets = [self.class_to_idx[t] for t in self.targets]
        else:
            with open(os.path.join(self.dataroot, "val", "val_annotations.txt"), "r") as f:
                datapairs = f.readlines()
            self.img_paths = [c.split('\t')[0] for c in datapairs]
            self.targets = [c.split('\t')[1] for c in datapairs]
            self.targets = [self.class_to_idx[t] for t in self.targets]

        # self.classes = self.data.classes
        # self.class_num = len(self.data.classes)
        # self.class_to_idx = self.data.class_to_idx
        # self.idx_to_class = {self.class_to_idx[cls]:cls for cls in self.class_to_idx}
    
    def __getitem__(self, idx):
        # idx_in_ori_data = self.subset_indices[idx]
        # return self.data.__getitem__(idx_in_ori_data)
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        target = self.targets[idx]

        return (img_tensor, target)

    def __len__(self):
        return len(self.img_paths)

    def __repr__(self):
        repr = """Tiny ImageNet Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, "Train" if self.train else "Val", self.class_num, self.__len__())
        return repr

if __name__ == '__main__':
    data = TinyIN(os.environ["DATAROOT"])
    print(data)
    traindata = TinyIN(os.environ["DATAROOT"], train=True)
    print(traindata)
    testdata = TinyIN(os.environ["DATAROOT"], train=False)
    print(testdata)
