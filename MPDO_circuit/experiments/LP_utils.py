import numpy as np
import matplotlib.pyplot as plt 

Mc = lambda k: 1.6114 + 0.037 * k  # (photonic mode approximated as a straight line)
Exca = 1.6229  # (exciton energy)
OM = 3.4 * 1e-3 # (half the Rabi splitting)
hbar_eV_ps = 6.582 * 1e-4 # hbar, converting energy to frequency

def omega_LP(k: float):
    """
    Lower-polariton dispersion in eV and um!

    """

    LP_disp =  0.5 * (Mc(k) + Exca) - 0.5 * ((Mc(k) - Exca)**2 + 4. * OM ** 2) ** 0.5
    return LP_disp


def vg_LP(k: float, dk: float=1e-3) -> np.array:
    "LP group velocity in units um/ps"
    return (omega_LP(k + dk/2.) - omega_LP(k - dk/2.)) / dk / hbar_eV_ps

def curvature_LP(k: float, dk: float=1e-3) -> np.array:
    "branch curvature, in ps^-1, mass is inverse "
    curvature = (vg_LP(k + dk/2, dk) - vg_LP(k - dk/2, dk)) / dk # derivative of group velocity
    return curvature


def exciton_fraction(k):
    theta_k = np.arctan(OM / (omega_LP(k) - Mc(k)))
    return np.cos(theta_k) ** 2


