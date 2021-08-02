
import os

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch

from datasets.Imagenet import ImageNet
from datasets.Transforms import common_test, common_train
from models.resnet import resnet18
from runners.classification import DistRunner

def get_device_id():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args.local_rank

def main():
    # Distributed setting.
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    device_id = get_device_id()
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'

    # Automatic mixed precision.
    scaler = torch.cuda.amp.GradScaler()

    batch_size = 128
    normalize = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    lr = 1e-1
    epochs = 200

    trainset = ImageNet(os.environ['DATAROOT'], transform=common_train((224, 224)), train=True, subset=50)
    testset = ImageNet(os.environ['DATAROOT'], transform=common_test((224, 224)), train=False, subset=50)
    # Use distributed sampler to map data parts to different CUDA devices.
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    testsampler = torch.utils.data.distributed.DistributedSampler(testset)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=trainsampler, pin_memory=False, num_workers=12)
    test_loader = DataLoader(testset, batch_size=batch_size, sampler=testsampler, pin_memory=False, num_workers=12)


    model = resnet18(**normalize, class_num=50).to(device)
    # Distributed: convert the BN layers of the model into sync-BN layers.
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Distributed: create DDP model.
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-4, -1)
    
    # Use DistRunner to run DDP train and evaluation.
    runner = DistRunner(model, train_loader, test_loader, criterion, optimizer, scheduler, scaler, epochs, 10)
    
    tqdm.write("Start training with Resnet18.")
    runner.train()

if __name__ == '__main__':
    # python -m torch.distributed.launch --nproc_per_node=2 run-classification-distributed.py
    main()
