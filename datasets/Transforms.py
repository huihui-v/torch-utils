import torchvision.transforms as T

def common_train(image_size=(224, 224)):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResizedCrop((224, 224)),
        T.RandomApply([
            T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([
            T.GaussianBlur((3, 3))
        ], p=1.0),
        T.ToTensor()
    ])

def common_test(image_size=(224, 224)):
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor()
    ])
