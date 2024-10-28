import torch
from .ORGaNICs2D import ORGaNICs2D
from models.utils.utils import dynm_fun


class ORGaNICs2Dgeneral(ORGaNICs2D):
    def __init__(self, 
                 params,
                 Way=None, 
                 Wyy=None,
                 b0=None,
                 b1=None,
                 sigma=None,
                 tauA=None,
                 tauY=None,
                 z=None,
                 method="euler",
                 run_jacobian=True):
        super().__init__(params)

        self._b0 = b0
        self._b1 = b1

        self._sigma = sigma

        self._tauA = tauA
        self._tauY = tauY

        self.Wyy = Wyy
        self.Way = Way

        self.z = z

        self.method = method
        self.run_jacobian = run_jacobian

        """Initialize the circuit"""
        self.dim = self.calculate_dim()
        self.initialize_circuit(method=method, run_jacobian=run_jacobian)
    
    @property
    def b1(self):
        return self._b1

    def initialize_circuit(self, method="euler", run_jacobian=True):
        """
        This function makes the input stimulus, the weight matrices and the jacobian
        corresponding to the system.
        :return: None
        """

        """Make the feedforward and recurrent weight matrices"""
        if self.Wyy is None:
            self.Wyy = torch.eye(self.Ny)

        """Make the input stimulus"""
        if self.z is None:
            self.z = torch.zeros(self.Ny)
            self.z[self.Ny // 2] = 1.0

        """Make the normalization matrix"""
        if self.Way is None:
            self.Way = torch.ones(self.Na, self.Ny)

        """Make the attention gains"""
        if self.b0 is None:
            self._b0 = 0.5 * torch.ones(self.Ny)

        if self.b1 is None:
            self._b1 = self.b0

        """Make the contrast gain"""
        if self.sigma is None:
            self._sigma = 0.1

        """Make the time constants"""
        if self.tauA is None:
            self._tauA = torch.tensor([0.001])
        
        if self.tauY is None:
            self._tauY = torch.tensor([0.002])

        """Make the jacobian"""
        if run_jacobian:
            tau_min = min(torch.min(self.tauA), torch.min(self.tauY))
            tau_max= max(torch.max(self.tauA), torch.max(self.tauY))
            time = tau_max * 200
            dt = 0.05 * tau_min
            points = int(time / dt)
            _ = self.jacobian_autograd(time=time, points=points, method=method)
        return None
    
    @dynm_fun
    def _dynamical_fun(self, t, x):
        """
        This function defines the dynamics of the ring ORGaNICs model.
        :param x: The state of the network.
        :return: The derivative of the network at the current time-step.
        """
        x = x.squeeze(0)  # Remove the extra dimension
        y = x[0:self.Ny]
        a = x[self.Ny:]
        dydt = (1 / self.tauY) * (-y + self.b1 * self.z
                + (1 - torch.sqrt(torch.relu(a))) * (self.Wyy @ y))
        dadt = (1 / self.tauA) * (-a + (self.sigma * self.b0) ** 2 + self.Way @ (torch.relu(a) * y ** 2))
        return torch.cat((dydt, dadt))
    
    def analytical_ss(self):
        """
        This function calculates the steady-state of the network analytically.
        :return: The steady-state of the network.
        """
        a_s = self.sigma ** 2 * self.b0 ** 2 + self.Way @ (self.b1 * self.z) ** 2
        y_s = (self.b1 * self.z) / torch.sqrt(a_s)
        return y_s, a_s
    
class ORGaNICs2DgeneralRectified(ORGaNICs2Dgeneral):
    def __init__(self, 
                 params,
                 Way=None, 
                 Wyy=None,
                 b0=None,
                 b1=None,
                 sigma=None,
                 tauA=None,
                 tauY=None,
                 z=None,
                 method="euler",
                 run_jacobian=True):
        super().__init__(params,
                         Way=Way,
                         Wyy=Wyy,
                         b0=b0,
                         b1=b1,
                         sigma=sigma,
                         tauA=tauA,
                         tauY=tauY,
                         z=z,
                         method=method,
                         run_jacobian=run_jacobian)
    
    @dynm_fun
    def _dynamical_fun(self, t, x):
        """
        This function defines the dynamics of the ring ORGaNICs model.
        :param x: The state of the network.
        :return: The derivative of the network at the current time-step.
        """
        x = x.squeeze(0)  # Remove the extra dimension
        y = x[0:self.Ny]
        a = x[self.Ny:]
        dydt = (1 / self.tauY) * (-y + self.b1 * self.z
                + (1 - torch.sqrt(torch.relu(a))) * (self.Wyy @ torch.relu(y)))
        dadt = (1 / self.tauA) * (-a + (self.sigma * self.b0) ** 2 + self.Way @ (torch.relu(a) * torch.relu(y) ** 2))
        return torch.cat((dydt, dadt))


class ORGaNICs2D0(ORGaNICs2D):
    def __init__(self, params):
        super().__init__(params)

        """Initialize the circuit"""
        self.dim = self.calculate_dim()
        self.initialize_circuit()
        self.make_noise_mats()


class ORGaNICs2D1(ORGaNICs2D):
    def __init__(self, params):
        super().__init__(params)
        """
        This model implements the 2D ORGaNICs model with noise in input gain and a 
        dynamical equation for the firing variables.
        """
        self._Nb = params["N_b"]
        self._tauB = params["tauB"]

        """Type of noise"""
        self._noise_type = "multiplicative"

        """Initialize the circuit"""
        self.dim = self.calculate_dim()
        self.initialize_circuit()
        self.make_noise_mats()

    @property
    def Nb(self):
        return self._Nb

    @Nb.setter
    def Nb(self, Nb):
        if Nb > 0:
            self._Nb = Nb
            self.dim = self.calculate_dim()
            self.initialize_circuit()
            self.make_noise_mats()
        else:
            raise ValueError("Number of neurons must be a positive integer")

    @property
    def tauB(self):
        return self._tauB

    @tauB.setter
    def tauB(self, tauB):
        if tauB > 0:
            self._tauB = tauB
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def noise_type(self):
        return self._noise_type

    @noise_type.setter
    def noise_type(self, noise_type):
        if not noise_type == "multiplicative":
            raise ValueError("Noise type for the model can only be multiplicative.")

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c):
        if 0 <= c <= 1:
            self._c = c
            self.input = self.make_input()
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Input grating contrast must be between 0 and 1")

    def get_instance_variables(self):
        return (
            self._Ny,
            self._Na,
            self._Nb,
            self._eta,
            self._tauY,
            self._tauA,
            self._tauB,
            self._sigma,
            self._b0,
            self._norm_band,
            self._input_dim,
            self._c,
            self._grating_angle,
        )

    def calculate_dim(self):
        return self._Ny + self._Na + self._Nb + self._Ny

    def make_Ly(self, t, x):
        """
        This function defines the most general form of the noise matrix.
        :return: The N x N noise matrix.
        """
        y = x[0 : self.Ny]
        L = self.eta * torch.eye(self.dim)
        L[(self.Ny + self.Na + self.Nb) :, (self.Ny + self.Na + self.Nb) :] = (
            torch.sqrt(y**2) * torch.eye(self.Ny)
        )
        return L

    def make_D(self):
        """
        This function creates the D matrix for the noise.
        :return: The D matrix
        """
        D = torch.zeros(self.dim)
        D[0 : self.Ny] = 1 / self.tauY
        D[self.Ny : (self.Ny + self.Na)] = 1 / self.tauA
        D[(self.Ny + self.Na) : (self.Ny + self.Na + self.Nb)] = 1 / self.tauB
        D[(self.Ny + self.Na + self.Nb) :] = 1 / self.tauY
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
        a = x[self.Ny : (self.Ny + self.Na)]
        b = x[(self.Ny + self.Na) : (self.Ny + self.Na + self.Nb)]
        yf = x[(self.Ny + self.Na + self.Nb) :]
        cc = ((self.sigma * self.b0) / (1 + self.b0)) ** 2
        dydt = (1 / self.tauY) * (
            -y
            + (b / (1 + b)) * (self.Wzx @ self.input)
            + (1 - torch.sqrt(torch.relu(a))) * (self.Wyy @ y)
        )
        dadt = (1 / self.tauA) * (-a + self.Way @ (a * y**2) + cc)
        dbdt = (1 / self.tauB) * (-b + self.b0)
        dyfdt = (1 / self.tauY) * (-yf + y**2)
        return torch.cat((dydt, dadt, dbdt, dyfdt))


class ORGaNICs2D2(ORGaNICs2D):
    def __init__(self, params):
        super().__init__(params)
        """
        This model implements the 2D ORGaNICs model with noise in input gain and a 
        dynamical equation for the firing variables.
        """
        self._Nb = params["N_b"]
        self._tauB = params["tauB"]

        """Type of noise"""
        self._noise_type = "additive"

        """Initialize the circuit"""
        self.dim = self.calculate_dim()
        self.initialize_circuit()
        self.make_noise_mats()

    @property
    def Nb(self):
        return self._Nb

    @Nb.setter
    def Nb(self, Nb):
        if Nb > 0:
            self._Nb = Nb
            self.dim = self.calculate_dim()
            self.initialize_circuit()
            self.make_noise_mats()
        else:
            raise ValueError("Number of neurons must be a positive integer")

    @property
    def tauB(self):
        return self._tauB

    @tauB.setter
    def tauB(self, tauB):
        if tauB > 0:
            self._tauB = tauB
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Time constant must be a positive float")

    @property
    def noise_type(self):
        return self._noise_type

    @noise_type.setter
    def noise_type(self, noise_type):
        if not noise_type == "multiplicative":
            raise ValueError("Noise type for the model can only be multiplicative.")

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c):
        if 0 <= c <= 1:
            self._c = c
            self.input = self.make_input()
            _ = self.jacobian_autograd()
            self.make_noise_mats()
        else:
            raise ValueError("Input grating contrast must be between 0 and 1")

    def get_instance_variables(self):
        return (
            self._Ny,
            self._Na,
            self._Nb,
            self._eta,
            self._tauY,
            self._tauA,
            self._tauB,
            self._sigma,
            self._b0,
            self._norm_band,
            self._input_dim,
            self._c,
            self._grating_angle,
        )

    def calculate_dim(self):
        return self._Ny + self._Na + self._Nb

    def make_Ly(self, t, x):
        """
        This function defines the most general form of the noise matrix.
        :return: The N x N noise matrix.
        """
        y = x[0 : self.Ny]
        L = self.eta * torch.eye(self.dim)
        return L

    def make_D(self):
        """
        This function creates the D matrix for the noise.
        :return: The D matrix
        """
        D = torch.zeros(self.dim)
        D[0 : self.Ny] = 1 / self.tauY
        D[self.Ny : (self.Ny + self.Na)] = 1 / self.tauA
        D[(self.Ny + self.Na) : (self.Ny + self.Na + self.Nb)] = 1 / self.tauB
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
        a = x[self.Ny : (self.Ny + self.Na)]
        b = x[(self.Ny + self.Na) :]
        cc = ((self.sigma * self.b0) / (1 + self.b0)) ** 2
        dydt = (1 / self.tauY) * (
            -y
            + (b / (1 + b)) * (self.Wzx @ self.input)
            + (1 - torch.sqrt(torch.relu(a))) * (self.Wyy @ y)
        )
        dadt = (1 / self.tauA) * (-a + self.Way @ (a * y**2) + cc)
        dbdt = (1 / self.tauB) * (-b + self.b0)
        return torch.cat((dydt, dadt, dbdt))
