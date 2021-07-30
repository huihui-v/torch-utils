# torch-utils
Tools in pytorch.


## Benchmarks

|   Model  |          Dataset         |  LR  |                      Optimizer                      |            Scheduler            | Batch Size | Epochs |  Acc.  |
|:--------:|:------------------------:|:----:|:---------------------------------------------------:|:-------------------------------:|:----------:|:------:|:------:|
| Resnet18 | ImageNet(Top 50 Classes) | 1e-1 | SGD(momentum=0.9, weight_decay=1e-4, nesterov=True) | CosineAnnealingLR(eta_min=1e-4) | 256        | 200    | 0.7404 |
|          |                          |      |                                                     |                                 |            |        |        |
|          |                          |      |                                                     |                                 |            |        |        |