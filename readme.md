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


| ID        | resume     | structure | training mode | TaxiBJ(rmse) | BikeNYC(rmse) |
|-----------|------------|-----------|---------------| -------------- | --------------- |
| 20220725A | random(42) | L4-woBN   | training      | 33.1854      | -             |
| 20220725B | 20220725A  | L4-woBN   | finetuning    | -            | -             |

Note:
- training: train set, 500epoch;
- finetuning: train+val set, 100epoch;
- woBN: without BN layers;
