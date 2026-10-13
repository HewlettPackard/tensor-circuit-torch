import numpy as np
from typing import Iterable, Tuple
from scipy.integrate import solve_ivp

# function for ODE integraton
def solve_ODE_AS_mode(
    J: np.complex64, 
    U: float,
    n_photon: float, 
    phase: float, 
    tlist: Iterable[float], 
    Delta: float=0.,
    gamma: float=0.
    )-> Tuple[np.array]:
    
    # convert to antisymmetric field and symmetric density
    alpha_0 = 0.5 * np.sqrt(n_photon) * (np.exp(1j*phase/2) - np.exp(-1j*phase/2))

    def dF(t, v):

        # extract alpha, N, M from vector
        alpha, N, M = v[0], v[1], v[2]

        # intensities
        n_in = n_photon * np.exp( - gamma * t)  # total photon number
        na = np.abs(alpha) ** 2 + N             # antisymmetric contribution
        ns = n_in - na                          # symmetric contribution

        # compute differentials
        d_alpha = (-gamma/2. - 1j*Delta ) * alpha \
            -2j * J * alpha - 1j * 0.5 * U * (ns + na) * alpha - 1j * 0.5 * U * ns * alpha.conj() \
            #- 1j * U * N * alpha - 0.5 * 1j * U * M * np.sqrt(ns) # backactions
        d_N = -gamma * N - U * ns * M.imag
        d_M = (-gamma - 2j * Delta) * M \
            -4j * J * M - 1j * U * (ns + na) * M - 1j * U * ns * (N + 0.5)

        # return differentials in array format
        return np.array([d_alpha, d_N, d_M])

    # solve the system, with right initial state
    y0 = np.array([alpha_0, 0, 0])
    sol = solve_ivp(dF, [tlist[0], tlist[-1]], y0=y0, t_eval=tlist)

    # return alpha, N, M tuple
    return sol.y[0], sol.y[1], sol.y[2]


def get_g2(
        alpha: np.ndarray, N: np.ndarray, M: np.ndarray, include_fourth_order: bool=True
        ) -> np.ndarray:

    # MF density
    n_alpha = np.abs(alpha) ** 2

    # the g2
    numerator = (
        n_alpha ** 2
        + 4 * N * n_alpha
        + M * alpha.conj() ** 2
        + M.conj() * alpha ** 2
        + include_fourth_order * 2 * N ** 2
        + include_fourth_order * M * M.conj()
    )
    denominator = n_alpha ** 2 + 2 * N * n_alpha + include_fourth_order * N ** 2
    g2 = numerator / denominator

    return g2.real