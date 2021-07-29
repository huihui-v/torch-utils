import torch
import torch.nn as nn
from torchvision import models
from .utils import _init_weight
from .utils import Normalization, Classification

class resnet18(Classification):
    def __init__(self, mean, std, class_num, n_channels=3):
        super(resnet18, self).__init__(mean, std, n_channels=n_channels)

        self.feature_extractor = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-1]+[nn.Flatten()])
        self.classifier = nn.Linear(in_features=512, out_features=class_num, bias=True)

        self._init()