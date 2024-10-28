import torch
from .ORGaNICs import ORGaNICs
from models.utils.utils import dynm_fun
import os


class ORGaNICs2D(ORGaNICs):
    def __init__(self, params):
        super().__init__(params)
        """
        This function initializes the various parameters of the 2D model.
        """
        self._Ny = params.get("N_y")
        self._Na = params.get("N_a")

        """Noise strength"""
        self._eta = params.get("eta")

        """Time constants"""
        self._tauY = params.get("tauY")
        self._tauA = params.get("tauA")

        """Make the weight matrices"""
        self.Way = None

    @property
    def Ny(self):
        return self._Ny

    @Ny.setter
    def Ny(self, Ny):
        if Ny > 0:
            self._Ny = Ny
            self.dim = self.calculate_dim()
            self.initialize_circuit()
            self.make_noise_mats()
        else:
            raise ValueError("Number of neurons must be a positive integer")

    @property
    def Na(self):
        return self._Na

    @Na.setter
    def Na(self, Na):
        if Na > 0:
            self._Na = Na
            self.dim = self.calculate_dim()
            self.initialize_circuit()
            self.make_noise_mats()
        else:
            raise ValueError("Number of neurons must be a positive integer")

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, eta):
        if eta >= 0:
            self._eta = eta
            self.make_noise_mats()
        else:
            raise ValueError("Noise S.D. must be a positive float")

    @property
    def tauY(self):
        return self._tauY

    @tauY.setter
    def tauY(self, tauY):
        if tauY > 0:
            self._tauY = tauY
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def tauA(self):
        return self._tauA

    @tauA.setter
    def tauA(self, tauA):
        if tauA > 0:
            self._tauA = tauA
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def norm_band(self):
        return self._norm_band

    @norm_band.setter
    def norm_band(self, norm_band):
        if norm_band >= 0:
            self._norm_band = norm_band
            self.Way = ORGaNICs.make_Wuy(self.Ny, self.Na, self.norm_band)
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Normalization bandwidth must be larger than 0")

    def get_instance_variables(self):
        return (
            self._Ny,
            self._Na,
            self._eta,
            self._tauY,
            self._tauA,
            self._sigma,
            self._b0,
            self._norm_band,
            self._input_dim,
            self._c,
            self._grating_angle,
        )

    def calculate_dim(self):
        return self._Ny + self._Na

    def initialize_circuit(self):
        """
        This function makes the input stimulus, the weight matrices and the jacobian
        corresponding to the system.
        :return: None
        """

        """Make the input stimulus"""
        self.input = self.make_input()

        """Make the feedforward and recurrent weight matrices"""
        self.Wzx, self.Wyy = ORGaNICs.make_weight_matrices(self.input_dim, self.Ny)

        """Make the normalization matrix"""
        self.Way = ORGaNICs.make_Wuy(self.Ny, self.Na, self.norm_band)

        """Make the jacobian"""
        _ = self.jacobian_autograd()
        return None

    def make_Ly(self, t, x):
        """
        This function defines the most general form of the noise matrix.
        :param N: The number of units in the network.
        :param eta: The strength (coefficient) of the noise.
        :return: The N x N noise matrix.
        """
        return self.eta * torch.eye(self.dim)

    def make_D(self):
        """
        This function creates the D matrix for the noise.
        :return: The D matrix
        """
        D = torch.zeros(self.dim)
        D[0 : self.Ny] = 1 / self.tauY
        D[self.Ny :] = 1 / self.tauA
        return torch.diag(D)

    @dynm_fun
    def _dynamical_fun(self, t, x):
        """
        This function defines the dynamics of the ring ORGaNICs model.
        :param x: The state of the network.
        :return: The derivative of the network at the current time-step.
        """
        x = x.squeeze(0)  # Remove the extra dimension
        y = x[0 : self.Ny]
        a = x[self.Ny :]
        cc = ((self.sigma * self.b0) / (1 + self.b0)) ** 2
        z = torch.relu((self.Wzx @ self.input))
        dydt = (1 / self.tauY) * (
            -y
            + (self.b0 / (1 + self.b0)) * z
            + (1 - torch.sqrt(torch.relu(a))) * (self.Wyy @ y)
        )
        dadt = (1 / self.tauA) * (-a + self.Way @ (torch.relu(a) * torch.relu(y) ** 2) + cc)
        return torch.cat((dydt, dadt))

    def ss_norm(self):
        """
        This function calculates the norm of the steady state of the firing rates of the
        principal neurons using the analytical solution.
        :return: The steady state of the network.
        """
        z = torch.relu(self.Wzx @ self.input)
        ss = z**2 / (self.sigma**2 + self.Way @ z**2)
        return ss


