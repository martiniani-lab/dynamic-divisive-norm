import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import models.utils as utils


class rnnCell_unrectified(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        Wr_identity=False,
        learn_tau=True,
        dt_tau_max_y=0.01,
        dt_tau_max_a=0.01,
        dt_tau_max_b=0.1,
        sigma=1.0,
        n=2.0,
        set_norm_diag_one=False,
    ):
        """This code is implemented with batch_first=True"""
        super(rnnCell_unrectified, self).__init__()
        self.learn_tau = learn_tau
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wr_identity = Wr_identity
        self.set_norm_diag_one = set_norm_diag_one

        # Define the parameters for the weight matrices
        self.Wzx = utils.Identity(
            nn.Parameter(torch.randn((hidden_size, input_size)), requires_grad=True)
        )
        # parametrize the spectral norm of the input drive to be maximum of 1
        # torch.nn.utils.parametrizations.spectral_norm(self.Wzx, name="weight")

        self.Wr = utils.Identity(
            nn.Parameter(torch.eye(hidden_size), requires_grad=not Wr_identity)
        )
        # parameterize Wr to have a max singular value of 1
        # torch.nn.utils.parametrizations.spectral_norm(self.Wr, name="weight")
        
        # Define the weight matrices for the gains
        self.Wbx0 = nn.Parameter(torch.randn((hidden_size, input_size)), requires_grad=True)
        self.Wby0 = nn.Parameter(torch.randn((hidden_size, hidden_size)), requires_grad=True)
        self.Wba0 = nn.Parameter(torch.randn((hidden_size, hidden_size)), requires_grad=True)

        self.Wbx1 = nn.Parameter(torch.randn((hidden_size, input_size)))
        self.Wby1 = nn.Parameter(torch.randn((hidden_size, hidden_size)))
        self.Wba1 = nn.Parameter(torch.randn((hidden_size, hidden_size)))

        # Define the normalization matrix
        self.log_Way = nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.sparsity = 0.0

        self.initilize_weights()

        # Define the semi-saturation constant
        # self.sigma = nn.Parameter(torch.randn((hidden_size)), requires_grad=True)
        self.sigma = nn.Parameter(sigma * torch.ones((hidden_size)), requires_grad=False)
        # self.B0_const = nn.Parameter(torch.ones((hidden_size)), requires_grad=False)
        self.n = n

        # Define the dimensionless time_step parameters
        self.dt_tau_max_y = dt_tau_max_y
        self.dt_tau_max_a = dt_tau_max_a
        self.dt_tau_max_b = dt_tau_max_b
        self.beta = 5.0

        # Define the time constant parameters
        if self.learn_tau:
            self.param_dt_tauy = nn.Parameter(
                torch.randn((hidden_size)), requires_grad=learn_tau
            )
            self.param_dt_taua = nn.Parameter(
                torch.randn((hidden_size)), requires_grad=learn_tau
            )
            self.param_dt_taub0 = nn.Parameter(
                torch.randn((hidden_size)), requires_grad=learn_tau
            )
            self.param_dt_taub1 = nn.Parameter(
                torch.randn((hidden_size)), requires_grad=learn_tau
            )
        else:
            self.param_dt_tauy = nn.Parameter(
                torch.ones((hidden_size)), requires_grad=learn_tau
            )
            self.param_dt_taua = nn.Parameter(
                torch.ones((hidden_size)), requires_grad=learn_tau
            )
            self.param_dt_taub0 = nn.Parameter(
                torch.ones((hidden_size)), requires_grad=learn_tau
            )
            self.param_dt_taub1 = nn.Parameter(
                torch.ones((hidden_size)), requires_grad=learn_tau
            )

        # define a small value to prevent division by zero in norm_eqn and sqrt backprop
        self.eps = 1e-8

        # Define a variable for MSE loss between input gain and attention parameter.
        self.input_gain_mse_loss = 0.0

    @staticmethod
    def get_activation_y(y):
        return y ** 2

    @staticmethod
    def get_activation_a(a):
        return torch.sqrt(torch.relu(a))

    def initilize_weights(self):
        nn.init.kaiming_uniform_(self.Wzx(), a=math.sqrt(5))
        # make the spectral norm of Wzx() to be 1
        spectral_norm = torch.svd(self.Wzx()).S[0].item()
        # # print(spectral_norm)
        self.Wzx.weight.data = self.Wzx.weight.data / spectral_norm
        # spectral_norm = torch.svd(self.Wzx()).S[0].item()
        # print(spectral_norm)

        # make orthogonal initialization for Wr
        if self.Wr_identity:
            self.Wr.weight.data = torch.eye(self.hidden_size)
        else:
            nn.init.orthogonal_(self.Wr(), gain=1.0)

        # nn.init.kaiming_uniform_(self.Wr.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wbx0, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wby0, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wba0, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.Wbx1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wby1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wba1, a=math.sqrt(5))

        # initialize the Way matrix
        mask = torch.rand((self.hidden_size, self.hidden_size)) > self.sparsity
        uniform_values = torch.empty((self.hidden_size, self.hidden_size)).uniform_(0.0, 1.0)
        sparse_values = uniform_values * mask.float()
        log_sparse_values = torch.log(torch.clamp(sparse_values, min=1e-8))
        self.log_Way.data.copy_(log_sparse_values)
        self.log_Way.data.fill_diagonal_(0.0)
        return

    def Way(self):
        return torch.clamp(self.log_Way.exp(), min=0.0, max=1.0)
        # return self.log_Way.exp()

    def B0(self, x, y, a):
        return torch.sigmoid(F.linear(x, self.Wbx0, bias=None) + F.linear(y, self.Wby0, bias=None) + F.linear(a, self.Wba0, bias=None))

    # def B0(self, z):
    #     return torch.sigmoid(1 / (torch.norm(z, dim=-1, keepdim=True) + self.eps))
    
    def B1(self, x, y, a):
        return torch.sigmoid(F.linear(x, self.Wbx1, bias=None) + F.linear(y, self.Wby1, bias=None) + F.linear(a, self.Wba1, bias=None))
        # return torch.sigmoid(F.linear(x, self.Wbx1, bias=None))

    @staticmethod
    def tau_func(param, tau_max, beta):
        # return tau_max / (1.0 + F.softplus(param, beta=beta))
        return tau_max * torch.sigmoid(param)

    def dt_tauy(self):
        return rnnCell_unrectified.tau_func(self.param_dt_tauy, self.dt_tau_max_y, self.beta)

    def dt_taua(self):
        return rnnCell_unrectified.tau_func(self.param_dt_taua, self.dt_tau_max_a, self.beta)
    
    def dt_taub0(self):
        return rnnCell_unrectified.tau_func(self.param_dt_taub0, self.dt_tau_max_b, self.beta)
    
    def dt_taub1(self):
        return rnnCell_unrectified.tau_func(self.param_dt_taub1, self.dt_tau_max_b, self.beta)

    def set_norm_diag_one(self):
        if self.set_norm_diag_one:
            with torch.no_grad():
                self.log_Way.data.fill_diagonal_(0.0)

    def forward(self, x, y, a, b0, b1):
        """
        x: (batch_size, input_size)
        hidden: (batch_size, hidden_size)
        """
        # z = F.relu(F.linear(x, self.Wzx(), bias=None))
        z = F.linear(x, self.Wzx(), bias=None)

        # Scale the norm of z
        # norm_z = torch.norm(z, dim=1, keepdim=True) + 1e-5
        # z = (z / norm_z) * (torch.norm(x, dim=1, keepdim=True) / math.sqrt(self.input_size))

        # z = torch.where(norm_z > 0.0, z / norm_z, z) * x 
        # print(torch.mean(torch.norm(z, dim=1)))

        # Wr = torch.eye(self.hidden_size, device=self.Wr.weight.device) + self.Wr()
        Wr = self.Wr()
        y_hat = F.linear(y, Wr, bias=None)

        B0 = self.B0(x, y, a)
        # B0 = self.B0(z)
        # B0 = self.B0_const
        B1 = self.B1(x, y, a)

        # Update the input gain loss
        # self.input_gain_mse_loss = self.input_gain_mse(B0, B1)

        # Integrate the diff. equation by one step and find the activations
        b_new0 = b0 + self.dt_taub0() * (-b0 + B0)
        b_new1 = b1 + self.dt_taub1() * (-b1 + B1)
        y_new = y + self.dt_tauy() * (-y + b1 * z + (1 - self.get_activation_a(a)) * y_hat)
        a_new = a + self.dt_taua() * (
            - a
            + self.sigma ** 2 * b0 ** 2
            + F.linear(self.get_activation_y(y) * torch.relu(a), self.Way(), bias=None)
        )
        return y_new, a_new, b_new0, b_new1

    def input_gain_mse(self, B0, B1):
        """
        This function computes the input gain MSE loss. Find the norm of the difference in B1 and B0.
        """
        num_elements = np.prod(B1.shape)
        return torch.norm(B1 - B0) / num_elements

