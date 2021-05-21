import glob
import os

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class CelebA(Dataset):
    """Dataset CelebA
    Class number: 1
    Train data number: 202599
    Test data number: 0

    """
    def __init__(self, dataroot, n_data=None, transform=None, train=True):
        # Initial parameters
        self.dataroot = os.path.join(dataroot, "CelebA")
        self.train = train
        if n_data:
            self.n_data = n_data
        if transform: # Set default transforms if no transformation provided.
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize((64, 64)),
                T.ToTensor(),
                T.Normalize((.5, .5, .5), (.5, .5, .5))
            ])
        
        # Metadata of dataset
        classes = ["face"]
        self.class_num = len(classes)
        self.classes = classes
        self.class_to_idx = {"face": 0}
        self.idx_to_class = {0: "face"}
        
        # Split file and image path list.
        if self.train:
            if n_data:
                self.img_paths = glob.glob(os.path.join(self.dataroot, 'data', '*.jpg'))[:n_data]
            else:
                self.img_paths = glob.glob(os.path.join(self.dataroot, 'data', '*.jpg'))
            self.targets = [0] * len(self.img_paths)
        else:
            self.img_paths = []
            self.targets = []

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        target = self.targets[idx]

        return (img_tensor, target)

    def __repr__(self):
        repr = """CelebA Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr


if __name__ == '__main__':
    data = CelebA(os.environ["DATAROOT"])
    print(data)
    sample = CelebA(os.environ["DATAROOT"], n_data=100)
    print(sample)
    testdata = CelebA(os.environ["DATAROOT"], train=False)
    print(testdata)
