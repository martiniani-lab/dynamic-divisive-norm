import torch.nn as nn
import torch


class SDE(nn.Module):
    def __init__(self, outer_instance, noise_type="additive", sde_type="ito"):
        super().__init__()
        self.outer_instance = outer_instance
        self.noise_type = noise_type
        self.sde_type = sde_type

    def f(self, t, y):
        return self.outer_instance._dynamical_fun(t, y).unsqueeze(0)

    def g(self, t, y):
        return (self.outer_instance.L @ self.outer_instance.S).unsqueeze(0)


class SDE_mul(nn.Module):
    def __init__(self, outer_instance, noise_type="diagonal", sde_type="ito"):
        super().__init__()
        self.outer_instance = outer_instance
        self.noise_type = noise_type
        self.sde_type = sde_type

    def f(self, t, y):
        return self.outer_instance._dynamical_fun(t, y).unsqueeze(0)

    def g(self, t, y):
        """Note that the L taken here is the constant L, that is what comes out of the
        additive noise source"""
        Lt = self.outer_instance.make_Ly(t, y.squeeze(0))  # instantaneous L
        return (torch.diag(Lt @ self.outer_instance.S)).unsqueeze(0)


class SDE_cor_mul(nn.Module):
    def __init__(self, outer_instance, noise_type="general", sde_type="ito"):
        super().__init__()
        self.outer_instance = outer_instance
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.L = self.outer_instance.make_L("cor_add")

    def f(self, t, y):
        return self.outer_instance._dynamical_fun(t, y).unsqueeze(0)

    def g(self, t, y):
        """Note that the L taken here is the constant L, that is what comes out of the
        additive noise source"""
        return (((self.L @ self.outer_instance.S) @ y.T).squeeze(1)).unsqueeze(0)
