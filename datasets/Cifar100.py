import glob
import os

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class Cifar100(Dataset):
    """Dataset Cifar100
    Class number: 100
    Train data number: 50000
    Test data number: 10000

    """
    def __init__(self, dataroot, transform=None, train=True):
        # Initial parameters
        self.dataroot_param = dataroot
        self.dataroot = os.path.join(dataroot, "Cifar100")
        self.train = train
        if transform: # Set default transforms if no transformation provided.
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation((0, 30)),
                T.Resize((32, 32)),
                T.ToTensor()
            ])
        
        # Metadata of dataset
        classes = [i.split('/')[-1] for i in glob.glob(os.path.join(self.dataroot, 'train', '*'))]
        self.class_num = len(classes)
        self.classes = [i.split('.')[1] for i in classes]
        self.class_to_idx = {i.split('.')[1]: int(i.split('.')[0]) for i in classes}
        self.idx_to_class = {int(i.split('.')[0]): i.split('.')[1] for i in classes}
        
        self.img_paths = glob.glob(os.path.join(self.dataroot, 'train' if self.train else 'test', '*', '*.jpg'))
        self.targets = [int(i.split('/')[-2].split('.')[0]) for i in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        target = self.targets[idx]

        return (img_tensor, target)

    def __repr__(self):
        repr = """Cifar100 Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

if __name__ == '__main__':
    data = Cifar100(os.environ["DATAROOT"])
    print(data)
    testdata = Cifar100(os.environ["DATAROOT"], train=False)
    print(testdata)
