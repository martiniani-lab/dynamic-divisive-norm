import numpy as np
import torch
import sympy as sp
import math
from ._dyn_models import _dyn_models
import os


class ORGaNICs(_dyn_models):
    def __init__(self, params):
        super().__init__(device=params.get("device"))
        """
        This function initializes the general parameters of the ORGaNICs.
        """

        """More model parameters"""
        self._sigma = params.get("sigma")
        self._b0 = params.get("b0")
        self._norm_band = params.get("norm_band")

        """Parameters of the input"""
        self._input_dim = params.get("input_dim")
        self._c = params.get("contrast")
        self._grating_angle = params.get("grating_angle")

        """Make the input stimulus"""
        self.input = None
        self.z = None

        """Type of noise"""
        self._noise_type = params.get("noise_type")

        """Make the weight matrices"""
        self.Wzx = None
        self.Wyy = None

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        if 0 <= sigma <= 1:
            self._sigma = sigma
            _ = self.jacobian_autograd()
        else:
            raise ValueError("Contrast gain must be between 0 and 1")

    @property
    def b0(self):
        return self._b0

    @b0.setter
    def b0(self, b0):
        if b0 >= 0:
            self._b0 = b0
            _ = self.jacobian_autograd()
        else:
            raise ValueError("Input gain parameter must be > 0")

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, input_dim):
        if input_dim > 0:
            self._input_dim = input_dim
            self.Wzx, self.Wyy = ORGaNICs.make_weight_matrices(self.input_dim, self.Ny)
            _ = self.jacobian_autograd()
        else:
            raise ValueError("Input dimension must be a positive integer")

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c):
        if 0 <= c <= 1:
            self._c = c
            self.input = self.make_input()
            _ = self.jacobian_autograd()
        else:
            raise ValueError("Input grating contrast must be between 0 and 1")

    @property
    def grating_angle(self):
        return self._grating_angle

    @grating_angle.setter
    def grating_angle(self, grating_angle):
        if 0 <= grating_angle <= 180:
            self._grating_angle = grating_angle
            self.input = self.make_input()
            _ = self.jacobian_autograd()
        else:
            raise ValueError("Input grating angle must be between 0 and 180 degrees")

    @property
    def noise_type(self):
        return self._noise_type

    @noise_type.setter
    def noise_type(self, noise_type):
        if noise_type in ["additive", "multiplicative"]:
            self._noise_type = noise_type
            self.make_noise_mats()
        else:
            raise ValueError("Noise type must be either 'additive' or 'multiplicative'")

    @staticmethod
    def make_weight_matrices(n_input, n_output):
        kernel = torch.tensor(
            [
                0.02807382,
                -0.060944743,
                -0.073386624,
                0.41472545,
                0.7973934,
                0.41472545,
                -0.073386624,
                -0.060944743,
                0.02807382,
            ]
        )

        # Make the recurrent weight matrix
        k = kernel.shape[0]
        conv_mat = torch.zeros(n_output, n_output)
        for row in range(n_output):
            kc = 0
            for col in range(
                row - int(np.floor(k / 2)) + 1, row + int(np.floor(k / 2)) + 1, 1
            ):
                conv_mat[row, int(np.remainder(col - 1 + n_output, n_output))] = kernel[
                    kc
                ]
                kc = kc + 1
        u, s, vh = torch.linalg.svd(conv_mat)
        s1 = torch.ones(s.size())
        s = torch.where(s > 1, s1, s)
        wyy = u @ torch.diag_embed(s) @ vh
        wyy = 0.5 * (wyy + wyy.T)  # Make symmetric

        # Make the feedforward weight matrix
        theta_step = 2 * np.pi / n_input
        theta = torch.linspace(0, 2 * np.pi - theta_step, steps=n_input)

        wd = torch.linalg.eigvals(wyy)

        d = wd[torch.abs(torch.real(wd) - 1) < 1e-3].shape[0]
        p = int(d - 1)

        # Find the constant
        p1 = sp.Rational(p, 1)
        constant = float(
            sp.sqrt(d / n_output)
            * sp.sqrt(
                (2 ** (2 * p1) * (sp.factorial(p1)) ** 2)
                / (sp.factorial(2 * p1) * (p1 + 1))
            )
        )

        wzx = torch.zeros(n_output, n_input)
        for angleIndex in range(n_output):
            theta_offset = angleIndex * 2 * np.pi / n_output
            theta_diff = (theta - theta_offset) / 2
            rf = constant * torch.cos(theta_diff) ** p
            wzx[angleIndex, :] = abs(rf)

        # sum_theta_RFs = torch.sqrt(torch.mean(torch.sum(wzx ** 2, 1)))
        # wzx = wzx / sum_theta_RFs

        return wzx, wyy

    @staticmethod
    def make_Wuy(n_input, n_output, norm_band):
        """
        This function makes the weight matrix from the auditory input to the
        output neurons.
        :param n_input: Number of input neurons
        :param n_output: Number of output neurons
        :param norm_band: Spatial extent of the normalization weights
        :return: Wuy
        """
        # Wuy = torch.ones(n_output, n_input)
        Wuy = torch.zeros((n_output, n_input))
        bandwidth = norm_band * n_input

        for row in range(n_output):
            x_prime = (n_input / n_output) * (row - 0.5)  # center of the gaussian
            for col in range(n_input):
                if (col - 0.5 - x_prime) >= 0:
                    if (col - 0.5 - x_prime) <= n_input / 2:
                        Wuy[row, col] = ORGaNICs.gaussian_no_scale(
                            abs(col - 0.5 - x_prime), bandwidth
                        )
                    else:
                        Wuy[row, col] = ORGaNICs.gaussian_no_scale(
                            abs(n_input + x_prime - (col - 0.5)), bandwidth
                        )
                else:
                    if (col - 0.5 - x_prime) >= -n_input / 2:
                        Wuy[row, col] = ORGaNICs.gaussian_no_scale(
                            abs(col - 0.5 - x_prime), bandwidth
                        )
                    else:
                        Wuy[row, col] = ORGaNICs.gaussian_no_scale(
                            abs(n_input + (col - 0.5) - x_prime), bandwidth
                        )

        return Wuy

    @staticmethod
    def gaussian_no_scale(d, sigma):
        """
        Returns the value of the Gaussian function given mean and variance parameters
        without the constant scaling.
        :param d: Distance from the mean
        :param sigma: Width of the Gaussian function
        :return: The value of the Gaussian function
        """
        G = math.exp(-(d**2) / (2 * sigma**2))
        return G

    @staticmethod
    def gaussian(x, y, sigma):
        """
        This function creates the gaussian function.
        """
        d = torch.abs(x - y)
        return torch.exp(-torch.pow(d / sigma, 2) / 2)

    @staticmethod
    def check_eig(mat):
        """Checks if all the eigenvalues of a matrix have real part<0, ei, they describe
        a stable dynamical system."""
        return torch.all(torch.real(torch.linalg.eigvals(mat)) < 0)

    def make_input(self):
        """
        This function generates the one-hot stimulus.
        :param N: Dimensionality of the stimulus.
        :param c: Contrast of the grating stimulus
        :param grating_angle: Grating angle
        :return: The one-hot input stimulus.
        """
        input = torch.zeros(self.input_dim)
        idx = int(self.input_dim * self.grating_angle / 180)
        input[idx] = self.c
        return input


if __name__ == "__main__":
    pass
