import numpy as np
from time import time
from typing import Iterable, Tuple
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

from ODE_AS_integrate import solve_ODE_AS_mode


def optimize_AS_SPG(
        J_init: float, 
        phase_init: float, 
        U: float, 
        n_photon: float, 
        gamma: float=0.,
        target_n_out: float|None=None ):

    # ancillary function for running the ODEs and extract g2
    def AS_g2_MZI(x):

        # readout params
        J = x[0]
        phase = x[1]
        

        # solve ODE
        alphas, Ns, Ms = solve_ODE_AS_mode(
            J, U, n_photon=n_photon, phase=phase, tlist=[0.,1.], gamma=gamma)

        # extract last time result
        alpha, N, M = alphas[-1], Ns[-1], Ms[-1]
        
        # compute g2                           
        n_alpha = np.abs(alpha) ** 2
        g2_ODE = (
            n_alpha ** 2 + 4 * N * n_alpha + M * alpha.conj() ** 2
                  + M.conj() * alpha ** 2 + 2 * N ** 2 + M * M.conj()
                  ) / (n_alpha ** 2 + 2 * N * n_alpha + N ** 2)
        
        return float(np.real(g2_ODE))
    
    def callback(x):
        if verbose:
            print(f"x={[f'{xi:.6f}' for xi in x]}, g2={AS_g2_MZI(x):.6f}")

    def constraint(x):
        J = x[0]
        phase = x[1]
        alphas, Ns, _ = solve_ODE_AS_mode(J, U, n_photon, phase, tlist=[0.,1.], gamma=gamma)
        alpha, N = alphas[-1], Ns[-1].real
        return (np.abs(alpha) ** 2 + N - target_n_out) ** 2
    
    x0 = np.array([J_init, phase_init])
    bounds = [(0, np.pi/2), (0, np.pi)]

    res = minimize(AS_g2_MZI, x0, method='SLSQP',
               options={'ftol': 1e-10, 'disp': False, 'maxiter': 500}, 
               bounds=bounds, 
               callback=callback,
               constraints={'type': 'eq', 'fun': constraint} if target_n_out is not None else None
               )
    
    return res


# ---------------------------------------------------------------------------
# Worker function (must be top-level for pickling with multiprocessing)
# ---------------------------------------------------------------------------

def run_single(args):

    """Run optimization + ODE for a single (U, ns, target_n_out, gamma) quadruple.

    Returns
    -------
    dict with keys: U, ns, target_n_out, g2, alpha, N, M, error
    """
    U, n_photon, target_n_out, gamma = args

    J_init     = np.pi / 4.0
    phase_init = 0.2

    try:
        # antisymmetric (field) and symmetric (density)
        alpha_0 = 0.5 * np.sqrt(n_photon) * (np.exp(1j*phase_init) - 1)
        ns = 0.25 * n_photon * np.abs(np.exp(1j*phase_init) + 1) ** 2

        res = optimize_AS_SPG(
            J_init=J_init, phase_init=phase_init,
            U=U, n_photon=n_photon, target_n_out=target_n_out, gamma=gamma)

        J_optim     = res.x[0]
        phase_optim = res.x[1]

        # solve AS ODE
        alphas, Ns, Ms = solve_ODE_AS_mode(
            J=J_optim, U=U, n_photon=n_photon, phase=phase_optim, tlist=[0.0, 1.0], gamma=gamma
        )

        # read out parameters
        alpha, N, M = alphas[-1], Ns[-1], Ms[-1]
        n_alpha = np.abs(alpha) ** 2

        # g2
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
            "U":            U,
            "n_photon":     n_photon,
            "ns":           ns,
            "na":           n_alpha + N,
            "target_n_out": target_n_out,
            "gamma":        gamma,     
            "g2":           g2.real,
            "alpha":        alpha,
            "N":            N.real,
            "M":            M,
            'coupling':     J_optim,
            'phase':        phase_optim,
            "error":        None,
        }

    except Exception as exc:
        return {
            "U":            U,
            "n_photon":     n_photon,
            "ns":           ns,
            "na":           n_alpha + N,
            "target_n_out": target_n_out,
            "gamma":        gamma,     
            "g2":           g2.real,
            "alpha":        alpha,
            "N":            N.real,
            "M":            M,
            'coupling':     J_optim,
            'phase':        phase_optim,
            "error":        str(exc),
        }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    t_start = time()

    verbose = False

    # vec_U           = [0.05]
    # vec_n_photon    = [40.]
    # vec_target_n_out = [0.01]   # <-- edit as needed
    # gammas          = [0.]

    vec_U           = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    vec_n_photon    = np.logspace(0, 2, 50)
    vec_target_n_out = [None, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1.]   # <-- edit as needed
    gammas          = [0., 0.1, 1.]

    param_quadruples = list(product(vec_U, vec_n_photon, vec_target_n_out, gammas))
    print(f"Total jobs: {len(param_quadruples)}")

    # results dict keyed by (U, ns, target_n_out)
    results: dict[tuple, dict] = {}

    with ProcessPoolExecutor() as executor:
        future_to_params = {
            executor.submit(run_single, quadruple): quadruple
            for quadruple in param_quadruples
        }

        for future in as_completed(future_to_params):
            data = future.result()
            key  = (data["U"], data["n_photon"], data["target_n_out"], data["gamma"])
            results[key] = data

            tgt_str = f"{data['target_n_out']:.3f}" if data["target_n_out"] is not None else "None"
            if data["error"]:
                print(f"[WARN] U={data['U']:.3e}  ns={data['ns']:.3e}  "
                      f"target={tgt_str} → {data['error']}")
            else:
                if verbose:
                    print(f"U={data['U']:.3e}  ns={data['ns']:.3e}  target={tgt_str}"
                        f"  g2={data['g2']:.4f}"
                        f"  |alpha|²={np.abs(data['alpha'])**2:.4f}")

    n_jobs = len(param_quadruples)
    t_run = time()-t_start
    with ProcessPoolExecutor() as executor:
        n_workers = executor._max_workers
    print(f"\n{n_jobs} jobs finished in {t_run:.2f}s, using {n_workers} workers")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    results['vec_U']            = vec_U
    results['vec_n_photon']    = vec_n_photon
    results['vec_target_n_out'] = vec_target_n_out
    results['gammas']           = gammas

    save_dir = Path(__file__).parent / "data_Gauss"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "results_optim.npy"
    np.save(save_path, results)

    print(f"\nResults saved to {save_path}")