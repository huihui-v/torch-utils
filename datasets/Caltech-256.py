import glob
import os

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class Caltech256(Dataset):
    """Dataset Caltech 256
    Class number: 257
    Train data number: 24582
    Test data number: 6027

    """
    def __init__(self, dataroot, transform=None, train=True):
        # Initial parameters
        self.dataroot = os.path.join(dataroot, "Caltech-256")
        self.train = train
        if transform: # Set default transforms if no transformation provided.
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
        
        # Metadata of dataset
        classes = [i.split('/')[-1] for i in glob.glob(os.path.join(self.dataroot, 'train' if train else 'test', '*'))]
        self.class_num = len(classes)
        self.classes = [i.split('.')[1] for i in classes]
        self.class_to_idx = {i.split('.')[1]: int(i.split('.')[0])-1 for i in classes}
        self.idx_to_class = {int(i.split('.')[0])-1: i.split('.')[1] for i in classes}

        self.img_paths = glob.glob(os.path.join(self.dataroot, 'train' if train else 'test', '*', '*'))
        self.targets = [self.class_to_idx[p.split('/')[-2].split('.')[1]] for p in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        target = self.targets[idx]

        return (img_tensor, target)

    def __repr__(self):
        repr = """Caltech-256 Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr


if __name__ == '__main__':
    data = Caltech256(os.environ["DATAROOT"])
    print(data)
    testdata = Caltech256(os.environ["DATAROOT"], train=False)
    print(testdata)
