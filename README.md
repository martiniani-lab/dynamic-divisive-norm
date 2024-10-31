# Unconditional stability of a recurrent neural circuit implementing divisive normalization

This repository contains code associated with our 2024 NeurIPS paper.

<!-- ![](./figures/readme.svg){width="200px"} -->
<div style="text-align: center;">
<img src="./figures/github_image.svg" alt="Description" width="800px">
</div>

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

## ORGaNICs implementation
The classes for feedforward and convolutional implementation of ORGaNICs can be found at,
```bash
    models/fixed_point/
```
The classes for ORGaNICs implemented as a recurrent neural network (RNN), defined by the explicit Euler discretization of the underlying system of nonlinear differential equations, can be found at,
```bash
    models/dynamical/
```
Implementation of the dynamical system of ORGaNICs can be found at,
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
> https://arxiv.org/abs/2409.18946

```bibtex
@article{rawat2024unconditional,
  title={Unconditional stability of a recurrent neural circuit implementing divisive normalization},
  author={Rawat, Shivang and Heeger, David J and Martiniani, Stefano},
  journal={arXiv preprint arXiv:2409.18946},
  year={2024}
}
```
