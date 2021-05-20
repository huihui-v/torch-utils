import glob
import os

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class CatvsDog(Dataset):
    def __init__(self, dataroot="data", train=True, transform=None):
        self.dataroot = os.path.join(dataroot, "cat-vs-dog")
        self.train = train
        
        self.dataroot = os.path.join(self.dataroot, 'training_set' if train else 'test_set')
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize((224, 224), interpolation=Image.NEAREST),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        
        self.classes = ['cats', 'dogs']
        self.class_to_idx = {'cats': 0, 'dogs': 1}
        self.class_num = len(self.classes)
        self.idx_to_class = {0: "cats", 1: "dogs"}

        self.img_paths = glob.glob(os.path.join(self.dataroot, '*', '*.jpg')) + glob.glob(os.path.join(self.dataroot, '*', '*.png'))
        self.targets = [self.class_to_idx[p.split('/')[-2]] for p in self.img_paths]

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        # idx = self.class_to_idx[data_path.split('/')[-2]]
        target = self.targets[index]


        return (img_tensor, target)

    def __len__(self):
        return len(self.img_paths)

    def __repr__(self):
        repr = """Dataset Cat-vs-Dog
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

if __name__ == '__main__':
    data = CatvsDog(os.environ["DATAROOT"])
    print(data)
    testdata = CatvsDog(os.environ["DATAROOT"], train=False)
    print(testdata)
