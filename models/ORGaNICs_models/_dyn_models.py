from abc import ABC, abstractmethod
import torch
from scipy import linalg
from torch.func import jacrev
from models.utils.sim_spectrum import sim_solution


class _dyn_models(ABC):
    def __init__(self, device=None):
        """Dimensionality of the system"""
        self.dim = None

        """Define the Jacobian matrix"""
        self.J = None

        """Make the noise matrix"""
        self.L = None
        self.D = None
        self.S = None

        """Type of noise"""
        self._noise_type = None

        """Dictionary to store the simulation"""
        self.simulation = {}

        """Device to run the simulation on"""
        self.device = device if device is not None else torch.device("cpu")

        """Steady-state solution"""
        self.ss = None

    @abstractmethod
    def get_instance_variables(self):
        pass

    @abstractmethod
    def initialize_circuit(self):
        pass

    def make_noise_mats(self):
        """
        This function creates the noise matrices for the SDE simulation.
        :return: None
        """
        self.L = self.make_L()
        self.D = self.make_D()
        self.S = torch.sqrt(self.D)
        return

    @abstractmethod
    def make_Ly(self, t, x):
        pass

    def make_L(self, time=1, points=10000, method="euler"):
        """
        Calculates the steady-state noise matrix.
        :param time: Time to simulate the circuit.
        :param points: How many points to discretize the simulation.
        :return: The steady-state noise matrix.
        """
        ss = self.ss
        if self.ss is None:
            if not self._noise_type == "additive":
                sim_obj = sim_solution(self)
                ss = sim_obj.steady_state(time=time, points=points, method=method)
        return self.make_Ly(0, ss)

    @abstractmethod
    def make_D(self):
        pass

    @abstractmethod
    def _dynamical_fun(self, t, x):
        pass

    def jacobian_autograd(self, time=1, points=10000, method="euler", ss=None):
        """
        Calculates the Jacobian of the dynamical system using torch autograd.
        """
        if ss is None:
            sim_obj = sim_solution(self)
            ss = sim_obj.steady_state(time=time, points=points)
        J = jacrev(self._dynamical_fun, argnums=1)(torch.tensor([0]), ss)
        self.J = J
        self.ss = ss
        return J, ss
