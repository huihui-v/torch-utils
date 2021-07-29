import os

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

class Cifar10(Dataset):
    def __init__(self, dataroot, transform=T.Compose([]), train=True, max_n_per_class=10000):
        self.dataroot_param = dataroot
        self.dataroot = dataroot
        self.train = train

        self.max_n_per_class = max_n_per_class

        self.transform = transform

        self.data = datasets.CIFAR10(root=self.dataroot, train=train, transform=self.transform, download=True)

        self.classes = self.data.classes
        self.class_num = len(self.data.classes)
        self.class_to_idx = self.data.class_to_idx
        self.idx_to_class = {self.class_to_idx[cls]:cls for cls in self.class_to_idx}

        self.subset_mask = np.array(self.data.targets)
        for i in range(10):
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
    
    def __getitem__(self, idx):
        idx_in_ori_data = self.subset_indices[idx]
        return self.data.__getitem__(idx_in_ori_data)

    def __len__(self):
        return len(self.subset_indices)

    def __repr__(self):
        repr = """Cifar10 Dataset(subset):
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr


if __name__ == '__main__':
    # Use cases.
    data = Cifar10(os.environ["DATAROOT"])
    print(data)
    testdata = Cifar10(os.environ["DATAROOT"], train=False)
    print(testdata)
    # When using subset of ImageNet dataset, you should set `subset` and `max_n_per_class`.
    # Get first N classes
    subsetdata_topclass = Cifar10(os.environ["DATAROOT"], train=True, max_n_per_class=100)
    print(subsetdata_topclass)
