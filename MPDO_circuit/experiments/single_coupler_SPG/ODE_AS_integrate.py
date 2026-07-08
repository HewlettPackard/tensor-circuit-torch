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
    alpha_0 = 0.5 * np.sqrt(n_photon) * (np.exp(1j*phase) - 1)
    ns = 0.25 * n_photon * np.abs(np.exp(1j*phase) + 1) ** 2

    def dF(t, v):

        # extract alpha, N, M from vector
        alpha, N, M = v[0], v[1], v[2]

        # MF energy
        U_ns = U * ns
        n_alpha = np.abs(alpha) ** 2

        # compute differentials
        d_alpha = (-gamma/2. - 1j*Delta - 1j * 0* U * n_alpha) * alpha \
            -2j * J * alpha - 1j * 0.5 * U_ns * alpha - 1j * 0.5 * U_ns * alpha.conj() \
            #- 1j * U * N * alpha - 0.5 * 1j * U * M * np.sqrt(ns) # backactions
        d_N = -gamma * N - U_ns * M.imag
        d_M = (-gamma - 2j * Delta) * M \
            -4j * J * M - 1j * U_ns * M - 1j * U_ns * (N + 0.5)

        # return differentials in array format
        return np.array([d_alpha, d_N, d_M])

    # solve the system, with right initial state
    y0 = np.array([alpha_0, 0, 0])
    sol = solve_ivp(dF, [tlist[0], tlist[-1]], y0=y0, t_eval=tlist)

    # return alpha, N, M tuple
    return sol.y[0], sol.y[1], sol.y[2]
