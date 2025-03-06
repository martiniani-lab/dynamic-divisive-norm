# Unconditional stability of a recurrent neural circuit implementing divisive normalization

This repository contains code associated with our 2024 NeurIPS paper.

<!-- ![](./figures/readme.svg){width="200px"} -->
<div style="text-align: center;">
<img src="./figures/github_image.svg" alt="Description" width="800px">
</div>

## Project Website

For more details, visit the project website:  
[https://shivangrawat.github.io/dynamic-divisive-norm/](https://shivangrawat.github.io/dynamic-divisive-norm/)


## Installation

Follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/martiniani-lab/dynamic-divisive-norm.git
    cd dynamic-divisive-norm
    ```
2. Add the current directory to your Python path:
    ```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    ```

## Example
A minimal example of PyTorch code to implement ORGaNICs is shown below,
```python
import torch
from torch import nn
import torch.nn.functional as F


class rnnCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        Wr_identity=False,
    ):
        super(rnnCell, self).__init__()

        # Define the parameters for the weight matrices
        self.Wzx = nn.Parameter(torch.randn((hidden_size, input_size)))
        self.Wr = nn.Parameter(torch.eye(hidden_size), requires_grad=not Wr_identity)
        self.log_Way = nn.Parameter(torch.zeros((hidden_size, hidden_size)))

        # Define independent parameters for gain
        self.b0 = nn.Parameter(torch.rand(hidden_size))
        self.b1 = nn.Parameter(torch.rand(hidden_size))

        # Define the other parameters
        self.dt_tau = 0.001 * nn.Parameter(torch.ones((hidden_size)), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones((hidden_size)), requires_grad=False)

    def Way(self):
        return self.log_Way.exp()

    def forward(self, x, y, a):
        z = F.relu(F.linear(x, self.Wzx, bias=None))
        y_hat = F.relu(F.linear(y, self.Wr, bias=None))

        y_new = y + self.dt_tau * (- y + self.b1 * z + (1 - torch.sqrt(F.relu(a))) * y_hat)
        a_new = a + self.dt_tau * (- a + self.sigma**2 * self.b0**2 + F.linear(F.relu(y) ** 2 * F.relu(a), self.Way(), bias=None))

        return y_new, a_new
```

## ORGaNICs implementation
The classes for feedforward and convolutional implementation of ORGaNICs can be found at,
```bash
    models/fixed_point/
```
The classes for ORGaNICs implemented as a recurrent neural network (RNN), defined by the explicit Euler discretization of the underlying system of nonlinear differential equations, can be found at,
```bash
    models/dynamical/
```
Implementation of the time-continuous dynamical system of ORGaNICs can be found at,
```bash
    models/ORGaNICs_model/
```

## Experiment code
PyTorch Lightning code for fitting ORGaNICs on static MNIST dataset can be found at,
```bash
    training_scripts/MNIST/
```
PyTorch Lightning code for fitting ORGaNICs on sequential MNIST dataset can be found at,
```bash
    training_scripts/sMNIST/
```

## Training/inference code
Code to generate all the figures in the paper can be found at,
```bash
    examples/
```

## Reference and Citation

> *Unconditional stability of a recurrent neural circuit implementing divisive normalization*
> 
> Shivang Rawat, David J. Heeger and Stefano Martiniani
>
> https://proceedings.neurips.cc/paper_files/paper/2024/file/1abed6ee581b9ceb4e2ddf37822c7fcb-Paper-Conference.pdf

```bibtex
@article{rawat2025unconditional,
  title={Unconditional stability of a recurrent neural circuit implementing divisive normalization},
  author={Rawat, Shivang and Heeger, David and Martiniani, Stefano},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={14712--14750},
  year={2025}
}
```
