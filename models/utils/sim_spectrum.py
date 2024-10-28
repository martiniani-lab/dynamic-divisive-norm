import numpy as np
import torch
from torchdiffeq import odeint
from torchsde import sdeint
from .simulation_class import SDE, SDE_mul, SDE_cor_mul
import scipy.signal


class sim_solution:
    def __init__(self, obj=None):
        """
        This function initializes the nonlinear dynamical function and noise matrices
        required for SDE simulation.
        :param obj: Dynamical function object
        """
        """Object containing dynamical function to be used for simulation"""
        self.obj = obj
        self.noise_type = obj.noise_type

    def steady_state(self, time=1, points=10000, tol=1e-5, atol=1e-8, rtol=1e-9, y0=None, method='euler'):
        """
        This function calculates the steady-state of the function by simulating the
        dynamical system over time.
        :return: The steady-state circuit. Raises an error if steady-state is not found.
        """
        t = self.time_points(time, points)
        sim = self.simulate(t, atol, rtol, y0=y0, method=method)
        # find if steady state is reached in a given tolerance else raise exception
        if torch.any(torch.isnan(sim[-1, :])):
            raise Exception("The simulation has NaNs")
        if torch.all(torch.abs(sim[-1, :] - sim[-2, :]) < tol):
            return sim[-1, :]
        else:
            raise Exception("The simulation did not converge.")

    def time_points(self, time, points):
        t = torch.linspace(0, time, points)
        return t

    def simulate(self, t, atol=1e-8, rtol=1e-9, y0=None, method='euler'):
        """
        This function simulates the circuit dynamics over time.
        :param t: The time tensor to simulate over.
        :return: The state of the circuit over time
        """
        if y0 is None:
            y0 = torch.zeros(self.obj.dim) + 1e-2
        return odeint(self.obj._dynamical_fun, y0, t, method=method, atol=atol, rtol=rtol)

    def simulate_sde(self, t, n_points=int(1e5), time=10, dt=1e-4, save=True):
        """
        This function simulates the dynamical function using the SDE solver.
        :param t: The time at which the simulation is to be done.
        :param save: This param is true when you want to look from the saved simulations.
        :return: The simulation of the circuit.
        """
        key = self.obj.get_instance_variables() + (n_points, time, dt, self.noise_type, save)
        if key not in self.obj.simulation:
            if hasattr(self.obj, "steady_state"):
                x0 = self.obj.steady_state().unsqueeze(0)
            else:
                x0 = self.steady_state(time=time/10, points=n_points//10).unsqueeze(0)
            if self.noise_type == "additive" or self.noise_type == "cor_add":
                sde = SDE(self.obj)
            elif self.noise_type == "multiplicative":
                sde = SDE_mul(self.obj)
            elif self.noise_type == "cor_mul":
                sde = SDE_cor_mul(self.obj)
            else:
                raise Exception("Noise type not recognized")
            with torch.no_grad():
                sol = sdeint(sde, x0, t, dt=dt, method='euler')
            solution = sol.squeeze(1)
            if save:
                self.obj.simulation[key] = solution
        else:
            solution = self.obj.simulation[key]
        return solution
    
    def ss_variance(self, n_points=int(1e3), time=1, dt=1e-4, n_trials=100):
        """
        This function calculates the steady-state variance of the neurons by simulating
        multiple trials. This code can be operated in two modes:
        1. When you sample points from the end of the simulation from multiple trials.
        2. When you sample all points from the simulation of a single trial.
        :return: The steady-state variance of the circuit.
        """
        t = torch.linspace(0, time, n_points)
        x0 = self.steady_state(time=time, points=n_points).unsqueeze(0)
        if self.noise_type == "additive":
            sde = SDE(self.obj)
        elif self.noise_type == "multiplicative":
            sde = SDE_mul(self.obj)
        else:
            raise Exception("Noise type not recognized")

        final_sol = torch.zeros(n_trials, self.obj.dim)
        for i in range(n_trials):
            with torch.no_grad():
                sol_sde = sdeint(sde, x0, t, dt=dt, method='euler')
            sol_sde = sol_sde - torch.mean(sol_sde, dim=0)
            sol_sde = sol_sde.squeeze(1)
            if n_trials == 1:
                final_sol = sol_sde
            else:
                final_sol[i, :] = sol_sde[-1, :]
        return torch.var(final_sol, dim=0)

    @staticmethod
    def spectrum(i=None, j=None, nperseg=1000, sampling_freq=torch.tensor([1e4]), sol_sde=None):
        sol_sde = sol_sde - torch.mean(sol_sde, dim=0)
        sol_i = sol_sde[:, i]

        if j is None:
            freq, S = scipy.signal.welch(sol_i.numpy(), sampling_freq.numpy(),
                                         nperseg=nperseg, scaling='density',
                                         return_onesided=False)
        else:
            sol_j = sol_sde[:, j]
            freq, S = scipy.signal.csd(sol_i.numpy(), sol_j.numpy(),
                                       sampling_freq.numpy(), nperseg=nperseg,
                                       scaling='density', return_onesided=False)

        f = freq[freq >= 0]
        S = S[freq >= 0]
        return torch.from_numpy(S), torch.from_numpy(f)

    @staticmethod
    def coh_spectrum(i=None, j=None, nperseg=1000, sampling_freq=1e4, sol_sde=None):
        sol_sde = sol_sde - torch.mean(sol_sde, dim=0)
        sol_i = sol_sde[:, i]
        sol_j = sol_sde[:, j]

        freq, S = scipy.signal.coherence(sol_i.numpy(), sol_j.numpy(),
                                         sampling_freq.numpy(), nperseg=nperseg,
                                         return_onesided=False)

        f = freq[freq >= 0]
        S = S[freq >= 0]
        return torch.from_numpy(S), torch.from_numpy(f)

    def simulation_spectrum(self, i=None, j=None, ndivs=15, n_points=int(1e5),
                            time=10, dt=1e-4):
        """
        This function calculates the power spectral density of the ith variable
        or cross-power spectral density between the i and j variables using simulation
        of the circuit with noise.
        :param i: First index of the variable.
        :param j: Second index of the variable.
        :return: The power spectral density or cross-power spectral density between the
        ith and jth variable.
        """
        i = 0 if i is None else i
        t = torch.linspace(0, time, n_points)

        sol_sde = self.simulate_sde(t, n_points, time, dt)

        nperseg = n_points // ndivs
        sampling_freq = 1 / (t[1] - t[0])
        S, f = self.spectrum(i=i, j=j, sol_sde=sol_sde,
                             nperseg=nperseg, sampling_freq=sampling_freq)
        return S, f

    def simulation_coherence(self, i=None, j=None, ndivs=15, n_points=int(1e5),
                             time=10, dt=1e-4):
        """
        This function calculates the coherence between the ith and jth variables,
        using simulation of the circuit with noise.
        :param i: The index of the first variable.
        :param j: The index of the second variable.
        :return: The coherence: |Sxy|^2 / (Sxx * Syy), between variables i and j.
        """
        i = 0 if i is None else i
        j = 1 if j is None else j

        Sxy, f = self.simulation_spectrum(i=i, j=j, ndivs=ndivs, n_points=n_points,
                                          time=time, dt=dt)
        Sxx, _ = self.simulation_spectrum(i=i, j=None, ndivs=ndivs, n_points=n_points,
                                          time=time, dt=dt)
        Syy, _ = self.simulation_spectrum(i=j, j=None, ndivs=ndivs, n_points=n_points,
                                          time=time, dt=dt)

        coherence = (torch.abs(Sxy) ** 2) / (Sxx * Syy)
        return coherence, f


if __name__ == '__main__':
    pass
