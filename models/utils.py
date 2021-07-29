import torch
from torch.functional import norm
import torch.nn as nn
from torchvision import models

def _init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1 and len(m.weight.shape) > 1:
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.weight.bias)

class Normalization(nn.Module):
    def __init__(self, mean, std, n_channels=3):
        super(Normalization, self).__init__()
        self.n_channels=n_channels
        if mean is None:
            mean = [.5] * n_channels
        if std is None:
            std = [.5] * n_channels
        self.mean = torch.tensor(list(mean))
        self.std = torch.tensor(list(std))
        self.mean = self.mean.reshape((1, self.n_channels, 1, 1))
        self.std = self.std.reshape((1, self.n_channels, 1, 1))
        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
    
    def forward(self, x):
        # y = torch.empty_like(x).to(x.device)
        # for i in range(self.n_channels):
        #     y[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        y = (x - self.mean / self.std)

        return y

class Classification(nn.Module):
    def __init__(self, mean, std, n_channels=3):
        super(Classification, self).__init__()
        self.norm = Normalization(mean, std, n_channels=n_channels)
        self.feature_extractor = nn.Sequential()
        self.classifier = nn.Sequential()

    def forward(self, x):
        x_norm = self.norm(x)
        feature = self.feature_extractor(x_norm)
        y = self.classifier(feature)

        return y

    def _init(self):
        self.feature_extractor.apply(_init_weight)
        self.classifier.apply(_init_weight)