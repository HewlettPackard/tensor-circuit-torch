import numpy as np
from qutip import *
from typing import Iterable, Tuple
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


# function for ODE integraton
def solve_ODE_AS_mode(
    J: np.complex64, 
    U_ns: float, 
    alpha_0, 
    tlist: Iterable[float], 
    Delta: float=0.,
    gamma: float=0.
    )-> Tuple[np.array]:

  def dF(t, v):

    # extract alpha, N, M from vector
    alpha, N, M = v[0], v[1], v[2]

    # compute differentials
    d_alpha = (-gamma/2. - 1j*Delta) \
        -2j * J * alpha - 1j * 0.5 * U_ns * alpha - 1j * 0.5 * U_ns * alpha.conj()
    d_N = -gamma * N - U_ns * M.imag
    d_M = (-gamma - 2j*Delta) \
        -4j * J * M - 1j * U_ns * M - 1j * U_ns * (N + 0.5)
    # d_alpha = (-gamma/2. - 1j*Delta) \
    #     -2j * J * alpha - 1j * U_ns * alpha - 1j * 0.5 * U_ns * alpha.conj()
    # d_N = -gamma * N - U_ns * M.imag
    # d_M = (-gamma - 2j*Delta) \
    #     -4j * J * M - 2j * U_ns * M - 1j * U_ns * (N + 0.5)

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
    denominator = n_alpha ** 2 + include_fourth_order * 2 * N * n_alpha + N ** 2
    g2 = numerator / denominator

    return g2.real


def optimize_AS_SPG(J_init: float, phase_init: float, U: float, ns: float, target_n_out: float|None=None ):

    # ancillary function for running the ODEs and extract g2
    def AS_g2_MZI(x):

        # readout params
        J = x[0]
        phase = x[1]
        
        # input field
        alpha_0 = 1j * phase * np.sqrt(ns)

        # solve ODE
        alphas, Ns, Ms = solve_ODE_AS_mode(U_ns=U * ns, J=J, alpha_0=alpha_0, tlist=[0.,1.])

        # extract last time result
        alpha, N, M = alphas[-1], Ns[-1], Ms[-1]
        
        # compute g2                           
        n_alpha = np.abs(alpha) ** 2
        g2_ODE = (
            n_alpha ** 2 + 4 * N * n_alpha + M * alpha.conj() ** 2
                  + M.conj() * alpha ** 2 + 2 * N ** 2 + M * M.conj()
                  ) / (n_alpha ** 2 + 2 * N * n_alpha + N ** 2)
        
        # return value
        #print(f'{[alpha, N, M]}')
        return g2_ODE.real
    
    def callback(x):
        print(f"x={[f'{xi:.6f}' for xi in x]}, g2={AS_g2_MZI(x):.6f}")

    def constraint(x):
        # readout params
        J = x[0]
        phase = x[1]
        
        # input field
        alpha_0 = 1j * phase * np.sqrt(ns)

        # solve ODE
        alphas, Ns, _ = solve_ODE_AS_mode(U_ns=U * ns, J=J, alpha_0=alpha_0, tlist=[0.,1.])

        # extract last time result
        alpha, N = alphas[-1], Ns[-1]

        # return density and target density constraint
        return (np.abs(alpha) ** 2 + N - target_n_out) ** 2
    
    # construct initial conditions
    x0 = np.array([J_init, phase_init])

    # run Nelder-Mead optimizer (restrict to larger than 0 values)
    bounds = [(0, np.pi/2), (0, np.pi)]

    res = minimize(AS_g2_MZI, x0, method='SLSQP',
               options={'xatol': 1e-8, 'disp': True, 'maxiter': 50}, 
               bounds=bounds, 
               callback=callback,
               constraints={'type': 'eq', 'fun': constraint} if target_n_out is not None else None
               )
    
    return res


if __name__ == "__main__":
   
    vec_U = np.logspace(-3,-1, 10)
    vec_ns = np.logspace(-1, 2, 10)

    J_init = np.pi / 4.
    phase_init = 0.05 

    target_n_out = None

    res = optimize_AS_SPG(J_init, phase_init, U, ns, target_n_out)

    # evaluate

    J = res.x[0]
    phase = res.x[1]
    
    # input field
    alpha_0 = 1j * phase * np.sqrt(ns)

    # solve ODE
    alphas, Ns, Ms = solve_ODE_AS_mode(U_ns=U * ns, J=J, alpha_0=alpha_0, tlist=[0.,1.])

    # extract last time result
    alpha, N, M = alphas[-1], Ns[-1], Ms[-1]
    n_alpha = np.abs(alpha)**2

    print(f"\n\nIntensity: {n_alpha + N:.4f}")

    g2_opt = (
        n_alpha ** 2 + 4 * N * n_alpha + M * alpha.conj() ** 2
                + M.conj() * alpha ** 2 + 2 * N ** 2 + M * M.conj()
                ) / (n_alpha ** 2 + 2 * N * n_alpha + N ** 2)
    
    print(f"\ng2: {g2_opt:.4f}\n")
