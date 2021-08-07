import os

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


class STL10(Dataset):
    def __init__(self, dataroot, transform=T.Compose([]), split='train+unlabeled', max_n_per_class=10000):
        self.dataroot_param = dataroot
        self.dataroot = dataroot
        # self.train = train
        self.split = split

        self.max_n_per_class = max_n_per_class

        self.transform = transform

        self.data = datasets.STL10(root=self.dataroot, split=split, transform=self.transform, download=True)

        self.classes = self.data.classes
        self.class_num = len(self.data.classes)
        # self.class_to_idx = self.data.class_to_idx
        # self.idx_to_class = {self.class_to_idx[cls]:cls for cls in self.class_to_idx}

        self.subset_mask = np.array(self.data.labels)
        for i in range(10):
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -2
        self.subset_indices = np.where(self.subset_mask != -2)[0]
    
    def __getitem__(self, idx):
        idx_in_ori_data = self.subset_indices[idx]
        return self.data.__getitem__(idx_in_ori_data)

    def __len__(self):
        return len(self.subset_indices)

    def __repr__(self):
        repr = """STL10 Dataset(subset):
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, self.split, self.class_num, self.__len__())
        return repr

if __name__ == '__main__':
    data = STL10(os.environ["DATAROOT"])
    print(data)
    traindata = STL10(os.environ["DATAROOT"], split='train')
    print(traindata)
    testdata = STL10(os.environ["DATAROOT"], split='test')
    print(testdata)
