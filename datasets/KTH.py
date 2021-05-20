import glob
import os

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class KTH(Dataset):
    def __init__(self, dataroot, transform=None, train=True):
        self.dataroot = os.path.join(dataroot, "KTH")
        self.train = train

        if transform:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation((0, 30)),
                T.Resize((256, 256)),
                T.RandomResizedCrop((224, 224)),
                T.ToTensor(),
                T.Normalize((.485, .456, .406), (.229, .224, .225))
            ])

        # classes = 
        classes = [i.split('/')[-1] for i in glob.glob(os.path.join(self.dataroot, 'data', 'train', '*'))]
        self.class_num = len(classes)
        self.classes = classes
        self.class_to_idx = {item: idx for (idx, item) in enumerate(classes)}
        self.idx_to_class = {idx: item for (idx, item) in enumerate(classes)}

        if train:
            self.img_paths = glob.glob(os.path.join(self.dataroot, 'data', 'train', '*', '*'))
            self.targets = [self.class_to_idx[i.split('/')[-2]] for i in self.img_paths]
        else:
            self.img_paths = glob.glob(os.path.join(self.dataroot, 'data', 'test', '*', '*'))
            self.targets = [self.class_to_idx[i.split('/')[-2]] for i in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        target = self.targets[idx]

        return (img_tensor, target)

    def __repr__(self):
        repr = """KTH Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

if __name__ == '__main__':
    data = KTH(os.environ["DATAROOT"])
    print(data)
    testdata = KTH(os.environ["DATAROOT"], train=False)
    print(testdata)
