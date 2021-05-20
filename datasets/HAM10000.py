import os
from collections import Counter

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class HAM10000(Dataset):
    """Dataset HAM10000
    Class number: 7
    Train data number: 9017
    Test data number: 998
    """
    def __init__(self, dataroot, transform=None, train=True, toy=False):
        # Initial parameters
        self.dataroot = os.path.join(dataroot, "HAM10000")
        self.train = train
        self.toy = toy
        if transform: # Set default transform if no transformation provided.
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                # T.RandomRotation((0, 30)),
                T.Resize((256, 256)),
                T.RandomResizedCrop((224, 224)),
                T.ToTensor(),
                T.Normalize((.485, .456, .406), (.229, .224, .225))
            ])
        
        self.split_file = os.path.join(self.dataroot, 'trainset.csv') if train else os.path.join(self.dataroot, 'testset.csv')
        if toy:
            self.split_file = os.path.join(self.dataroot, 'toy.csv')
        
        with open(self.split_file, 'r') as f:
            lines = f.readlines()
            # lines = [line.strip() for line in lines]
            img_ids = [i.split(',')[0].strip() for i in lines]
            self.img_paths = [os.path.join(self.dataroot, 'data', i+'.jpg') for i in img_ids]
            self.targets = [i.split(',')[1].strip() for i in lines]
        
        classes = list(set(self.targets))
        self.class_num = len(classes)
        self.classes = classes
        self.class_to_idx = {v: k for k, v in enumerate(classes)}
        self.idx_to_class = {k: v for k, v in enumerate(classes)}

        self.targets = [self.class_to_idx[i] for i in self.targets]

        self.class_count = Counter(self.targets)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('YCbCr')
        img_tensor = self.transform(img)
        target = self.targets[idx]

        return (img_tensor, target)

    def __repr__(self):
        split = 'Train'
        if not self.train:
            split = 'Test'
        if self.toy:
            split = 'Toy'
        repr = """HAM10000 Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, split, self.class_num, self.__len__())
        return repr

if __name__ == '__main__':
    data = HAM10000(os.environ["DATAROOT"])
    print(data)
    testdata = HAM10000(os.environ["DATAROOT"], train=False)
    print(testdata)
    toydata = HAM10000(os.environ["DATAROOT"], toy=True)
    print(toydata)