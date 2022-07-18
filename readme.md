# ST-ResNet

Pytorch version of ST-ResNet

The code implementation refers to [[ST-ResNet-Pytorch](https://github.com/BruceBinBoxing/ST-ResNet-Pytorch)](https://github.com/BruceBinBoxing/ST-ResNet-Pytorch)

## Dataset

Four types of input data:

- **C**loseness: (bs, 2 * l_c, h, w)
- **P**eriod: (bs, 2 * l_p, h, w)
- **T**rend: (bs, 2 * l_t, h, w)
- **E**xternal-Feature: (bs, 28)

TaxiBJ:

```python
l_c, l_p, l_t = 3, 1, 1
h, w = 32, 32
bs = 500
```

## Network

Four Way Network

C-Way, P-Way, T-Way are stack by ResUnit

E-Way is stack by nn.linear() and nn.ReLU()

## Experiment


| bn | TaxiBJ(rmse) | BikeNYC(rmse) |
| ---- | -------------- | --------------- |
| v  | -            | -             |
| x  | -            | -             |
