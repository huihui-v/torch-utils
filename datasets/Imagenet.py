import os

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


class ImageNet(Dataset):
    """ ImageNet dataset with subset and MAX #data-per-class settings.
    If use default parameters, it will just return a dataset with all ImageNet data.
    Otherwise, it will return a subset of ImageNet dataset.
    """
    def __init__(self, dataroot, transform=None, train=True, subset=1000, max_n_per_class=10000):
        # Initial parameters
        self.dataroot_param = dataroot
        self.dataroot = os.path.join(dataroot, "ilsvrc2012")
        self.train = train

        # Class number of subset. Take top-N classes as a subset(Torchvision official implementation sorting).
        self.subset = subset
        # Max number of data per class. If it was set more than the total number of that class, all the data will be taken.
        # Otherwise, it will take top-N data of that class(Torchvision official implementation sorting).
        self.max_n_per_class = max_n_per_class

        if transform: # Set default transforms if no transformation provided.
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation((0, 30)),
                T.Resize((256, 256)),
                T.RandomCrop((224, 224)),
                T.ToTensor(),
                # T.Normalize((.485, .456, .406), (.229, .224, .225))
            ])
        
        self.data = datasets.ImageNet(root=self.dataroot, split='train' if train else 'val', transform=self.transform)
        
        # Metadata of dataset
        self.classes = self.data.classes
        self.class_num = len(self.data.classes)
        self.class_to_idx = self.data.class_to_idx
        self.idx_to_class = {self.class_to_idx[c]:c for c in self.class_to_idx}
        
        # Subset process.
        self.class_subset = list(range(subset))
        self.class_count = {i: 0 for i in self.class_subset}
        self.subset_indices = []
        self.targets = []
        self.img_paths = []

        for i, target in enumerate(self.data.targets):
            if target not in self.class_subset:
                break
            if self.class_count[target] >= self.max_n_per_class:
                continue
            else:
                self.class_count[target] += 1
                self.subset_indices.append(i)
                self.targets.append(target)
                self.img_paths.append(self.data.imgs[i][0])

        self.classes = [self.classes[i] for i in self.class_subset]
        self.class_num = len(self.class_subset)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        target = self.targets[idx]

        return (img_tensor, target)

    def __repr__(self):
        repr = """ImageNet Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr


if __name__ == '__main__':
    # Use cases.
    data = ImageNet(os.environ["DATAROOT"])
    print(data)
    testdata = ImageNet(os.environ["DATAROOT"], train=False)
    print(testdata)
    # When using subset of ImageNet dataset, you should set `subset` and `max_n_per_class`.
    subsetdata = ImageNet(os.environ["DATAROOT"], train=True, subset=50, max_n_per_class=100)
    print(subsetdata)
