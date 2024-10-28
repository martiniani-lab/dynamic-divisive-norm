import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.func import vmap


class conv(nn.Module):
    def __init__(
        self,
        H_input: int,
        W_input: int,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        attention_kernel_size: tuple,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        device=None,
        dtype=None,
        sigma=1.0,
        n=2.0,
        attention=True,
        set_norm_diag_one=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}

        super(conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.attention_kernel_size = attention_kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.attention = attention
        self.set_norm_diag_one = set_norm_diag_one

        if padding != "same" and kernel_size != attention_kernel_size:
            raise NotImplementedError(
                "Mismatch in kernel size and attention kernel size. Please use same kernel size for both."
            )

        # Calculate the shape of the output
        if padding == "same":
            self.H_out = H_input
            self.W_out = W_input
        else:
            self.H_out = math.floor(
                (H_input + 2 * padding - dilation * (kernel_size[0] - 1) - 1) // stride
                + 1
            )
            self.W_out = math.floor(
                (H_input + 2 * padding - dilation * (kernel_size[1] - 1) - 1) // stride
                + 1
            )

        # Define the input kernel weights
        self.input_kernel = nn.Parameter(
            torch.randn(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs
            )
        )

        # define the attention kernel weights and the base attention weights
        if self.attention:
            self.attention_kernel = nn.Parameter(
                torch.randn(
                    (out_channels, in_channels // groups, *attention_kernel_size),
                    **factory_kwargs
                ), requires_grad=True
            )
        self.initilize_kernels()
        self.b0 = nn.Parameter(
            torch.zeros((out_channels, self.H_out, self.W_out), **factory_kwargs),
            requires_grad=True,
        )

        # Define the normalization weight parameters
        self.log_Way = nn.Parameter(
            0.0 * torch.ones(
                (out_channels, self.H_out * self.W_out, self.H_out * self.W_out),
                **factory_kwargs
            ), requires_grad=False
        )
        # self.initilize_norm_matrix()

        # Define the semi-saturation constant and exponent of activation
        self.sigma = nn.Parameter(
            torch.tensor(sigma, **factory_kwargs), requires_grad=False
        )
        self.n = n

        # define a small value to prevent division by zero in norm_eqn and sqrt backprop
        self.eps = 1e-8

        # Define a variable for MSE loss between input gain and attention parameter.
        self.input_gain_mse_loss = 0.0

    def initilize_kernels(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.input_kernel.data.uniform_(-stdv, stdv)
        if self.attention:
            n = self.in_channels
            for k in self.attention_kernel_size:
                n *= k
            stdv = 1.0 / math.sqrt(n)
            self.attention_kernel.data.uniform_(-stdv, stdv)
        return

    def initilize_norm_matrix(self):
        self.log_Way.data.uniform_(-10.0, 0.0)
        self.fill_diagonal(self.log_Way.data, 0.0)
        return

    @staticmethod
    def mat_mul(x, y):
        return x @ y.t()

    @staticmethod
    def fill_diagonal(t, value):
        set_diag_one = lambda t: t.fill_diagonal_(value)
        _ = torch.func.vmap(set_diag_one)(t)
        return

    def Way(self):
        return self.log_Way.exp()

    def B0(self):
        return torch.sigmoid(self.b0)

    def B1(self, x):
        if self.attention:
            return torch.sigmoid(
                F.conv2d(
                    x,
                    self.attention_kernel,
                    bias=self.bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )
        else:
            return self.B0()

    def forward(self, x):
        z = F.conv2d(
            x,
            self.input_kernel,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.set_norm_diag_one:
            with torch.no_grad():
                self.fill_diagonal(self.log_Way.data, 0.0)

        B0 = self.B0()
        B1 = self.B1(x)

        # Update the input gain loss
        self.input_gain_mse_loss = self.input_gain_mse(B0, B1)

        gated_z = B1**self.n * F.relu(z) ** self.n
        pooled_activation = vmap(self.mat_mul, (1, 0), out_dims=1)(
            gated_z.view(-1, self.out_channels, self.H_out * self.W_out), self.Way()
        )
        pooled_response = (
            pooled_activation.reshape(-1, self.out_channels, self.H_out, self.W_out)
            + self.eps
        )
        norm_response = gated_z / ((self.sigma * B0) ** self.n + pooled_response)

        # Note the activation (firing rates) are squared
        return norm_response

    def input_gain_mse(self, B0, B1):
        """
        This function computes the input gain MSE loss. Find the norm of the difference in B1 and B0.
        """
        num_elements = np.prod(B1.shape)
        return torch.norm(B1 - B0) / num_elements
