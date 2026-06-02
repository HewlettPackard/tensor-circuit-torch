import numpy as np
from qutip import *
from typing import Iterable, Tuple
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product


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
    d_alpha = (-gamma/2. - 1j*Delta) * alpha \
        -2j * J * alpha - 1j * 0.5 * U_ns * alpha - 1j * 0.5 * U_ns * alpha.conj()
    d_N = -gamma * N - U_ns * M.imag
    d_M = (-gamma - 2j*Delta) * M \
        -4j * J * M - 1j * U_ns * M - 1j * U_ns * (N + 0.5)

    # return differentials in array format
    return np.array([d_alpha, d_N, d_M])

  # solve the system, with right initial state
  y0 = np.array([alpha_0, 0, 0])
  sol = solve_ivp(dF, [tlist[0], tlist[-1]], y0=y0, t_eval=tlist)

  # return alpha, N, M tuple
  return sol.y[0], sol.y[1], sol.y[2]


def optimize_AS_SPG(
        J_init: float, 
        phase_init: float, 
        U: float, 
        ns: float, 
        gamma: float=0.,
        target_n_out: float|None=None ):

    # ancillary function for running the ODEs and extract g2
    def AS_g2_MZI(x):

        # readout params
        J = x[0]
        phase = x[1]
        
        # input field
        alpha_0 = 1j * phase * np.sqrt(ns)

        # solve ODE
        alphas, Ns, Ms = solve_ODE_AS_mode(
            U_ns=U * ns, J=J, alpha_0=alpha_0, tlist=[0.,1.], gamma=gamma
            )

        # extract last time result
        alpha, N, M = alphas[-1], Ns[-1], Ms[-1]
        
        # compute g2                           
        n_alpha = np.abs(alpha) ** 2
        g2_ODE = (
            n_alpha ** 2 + 4 * N * n_alpha + M * alpha.conj() ** 2
                  + M.conj() * alpha ** 2 + 2 * N ** 2 + M * M.conj()
                  ) / (n_alpha ** 2 + 2 * N * n_alpha + N ** 2)
        
        # return value
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

        # total intensity
        n_total = np.abs(alpha) ** 2 + N

        # return density and target density constraint
        return (n_total - target_n_out) ** 2
    
    # construct initial conditions
    x0 = np.array([J_init, phase_init])

    # run Nelder-Mead optimizer (restrict to larger than 0 values)
    bounds = [(0, np.pi/2), (0, np.pi)]

    res = minimize(AS_g2_MZI, x0, method='SLSQP',
               options={'xatol': 1e-8, 'disp': False, 'maxiter': 50}, 
               bounds=bounds, 
               callback=callback,
               constraints={'type': 'eq', 'fun': constraint} if target_n_out is not None else None
               )
    
    return res


# ---------------------------------------------------------------------------
# Worker function (must be top-level for pickling with multiprocessing)
# ---------------------------------------------------------------------------

def run_single(args):
    """Run optimization + ODE for a single (U, ns) pair.

    Returns
    -------
    dict with keys: U, ns, g2, alpha, N, M
    """
    U, ns = args

    J_init     = np.pi / 4.0
    phase_init = 0.05
    gamma = 0.
    target_n_out = None

    try:
        res = optimize_AS_SPG(
            J_init=J_init, phase_init=phase_init, 
            U=U, ns=ns, target_n_out=target_n_out, gamma=gamma)

        J     = res.x[0]
        phase = res.x[1]

        alpha_0 = 1j * phase * np.sqrt(ns)

        alphas, Ns, Ms = solve_ODE_AS_mode(
            U_ns=U * ns, J=J, alpha_0=alpha_0, tlist=[0.0, 1.0]
        )

        alpha, N, M = alphas[-1], Ns[-1], Ms[-1]
        n_alpha = np.abs(alpha) ** 2

        numerator = (
            n_alpha ** 2
            + 4 * N * n_alpha
            + M * alpha.conj() ** 2
            + M.conj() * alpha ** 2
            + 2 * N ** 2
            + M * M.conj()
        )
        denominator = n_alpha ** 2 + 2 * N * n_alpha + N ** 2
        g2 = numerator / denominator

        return {
            "U":     U.real,
            "ns":    ns.real,
            "g2":    g2.real,
            "alpha": alpha,
            "N":     N.real,
            "M":     M,
            "error": None,
        }

    except Exception as exc:
        return {
            "U":     U.real,
            "ns":    ns.real,
            "g2":    None,
            "alpha": None,
            "N":     None,
            "M":     None,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Define the parameters for the scan
    vec_U  = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1] # nonlinearity
    vec_ns = [0.1, 0.5, 1. ,2. , 5., 10., 20., 40.] # symmetric occupation

    param_pairs = list(product(vec_U, vec_ns)) # product space of all parameter pairs

    # results dict keyed by (U, ns) tuple
    results: dict[tuple, dict] = {}

    with ProcessPoolExecutor() as executor:
        future_to_params = {
            executor.submit(run_single, pair): pair
            for pair in param_pairs
        }

        for future in as_completed(future_to_params):
            data = future.result()
            key  = (data["U"], data["ns"])
            results[key] = data

            if data["error"]:
                print(f"[WARN] U={data['U']:.3e}, ns={data['ns']:.3e} → {data['error']}")
            else:
                print(
                    f"U={data['U']:.3e}  ns={data['ns']:.3e}"
                    f"  g2={data['g2']:.4f}"
                    f"  |alpha|²={np.abs(data['alpha'])**2:.4f}"
                )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    # append parameter arrays to results
    results['Us'] = vec_U
    results['ns'] = vec_ns

    # store the results in given dir
    
    from pathlib import Path
    save_dir = Path(__file__).parent / "data_Gauss"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "results.npy"
    np.save(save_path, results)
    print(f"\nResults saved to {save_path}")