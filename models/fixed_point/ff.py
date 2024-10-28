import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from .. import utils


class ff(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        sigma=1.0,
        n=2.0,
        set_norm_diag_one=False,
        Wr_identity=True,
        attention=True,
        num_iterations=10,
    ):
        super(ff, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_iterations = num_iterations
        self.set_norm_diag_one = set_norm_diag_one
        self.Wr_identity = Wr_identity

        # Define the parameters for the weight matrices
        self.attention = attention
        self.Wzx = nn.Parameter(
            torch.randn((output_size, input_size))
        )
        if self.attention:
            self.Wbx = nn.Parameter(
                torch.randn((output_size, input_size))
            )
        self.initilize_weights()

        self.Wr = utils.Identity(
            nn.Parameter(torch.eye(output_size), requires_grad=not Wr_identity)
        )
        if not Wr_identity:
            torch.nn.utils.parametrizations.spectral_norm(self.Wr, name="weight")

        # create an identity matrix
        self.identity_matrix = nn.Parameter(torch.eye(output_size), requires_grad=False)

        self.log_Way = nn.Parameter(
            -1.0 * torch.ones((output_size, output_size))
        )
        self.initilize_norm_matrix()
        if self.attention:
            self.Wbx = nn.Parameter(
                torch.randn((output_size, input_size))
            )

        # Define the base b0 vector and semi-saturation constant
        self.b0 = nn.Parameter(
            torch.zeros((output_size)), requires_grad=True
        )
        self.sigma = nn.Parameter(
            torch.tensor(sigma), requires_grad=False
        )
        self.n = n

        # define a small value to prevent division by zero in norm_eqn and sqrt backprop
        self.eps = 1e-10

        # Define a variable for MSE loss between input gain and attention parameter.
        self.input_gain_mse_loss = 0.0

    def initilize_weights(self):
        nn.init.kaiming_uniform_(self.Wzx, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wbx, a=math.sqrt(5))
        return

    def initilize_norm_matrix(self):
        self.log_Way.data.uniform_(-1.0, 0.0)
        self.log_Way.data.fill_diagonal_(0.0)
        return

    def Way(self):
        return self.log_Way.exp()

    def B0(self):
        return torch.sigmoid(self.b0)

    def B1(self, x):
        if self.attention:
            return torch.sigmoid(F.linear(x, self.Wbx, bias=None))
        else:
            return self.B0()

    def forward(self, x):
        z = F.linear(x, self.Wzx, bias=None)
        # make z to be norm 1
        z = z / torch.norm(z, dim=1, keepdim=True)

        if self.set_norm_diag_one:
            with torch.no_grad():
                self.log_Way.data.fill_diagonal_(0.0)

        B0 = self.B0()
        B1 = self.B1(x)

        # Calculate the fixed point
        y_s, _ = self.steady_state(x, B1, B0, z)
        norm_response = torch.relu(y_s) ** self.n

        return norm_response
    
    def steady_state(self, x, B1, B0, z):
        """In this function, we calculate the steady-states of the network."""
        if self.Wr_identity:
            a_s = (self.sigma * B0) ** self.n + F.linear((B1 * z) ** self.n, self.Way(), bias=None) + self.eps
            y_s = (B1 * z) / torch.sqrt(a_s)
        else:
            a_s = (self.sigma * B0) ** self.n + F.linear((F.linear((B1 * z), self.Wr(), bias=None)) ** self.n, self.Way(), bias=None) + self.eps
            y_s = F.linear((B1 * z), self.Wr(), bias=None) / torch.sqrt(a_s)
            for i in range(self.num_iterations):
                y_s = torch.linalg.solve((self.identity_matrix  - self.Wr() + torch.diag_embed(a_s) @ self.Wr()), (B1 * z))
                a_s = (self.sigma * B0) ** self.n + F.linear(y_s ** self.n * a_s, self.Way(), bias=None) + self.eps
        return y_s, a_s

    def input_gain_mse(self, B0, B1):
        """
        This function computes the input gain MSE loss. Find the norm of the difference in B1 and B0.
        """
        num_elements = np.prod(B1.shape)
        return torch.norm(B1 - B0) / num_elements
