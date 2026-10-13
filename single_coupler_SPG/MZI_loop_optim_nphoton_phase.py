import numpy as np
from time import time
from typing import Iterable, Tuple
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
import os

from ODE_AS_integrate import solve_ODE_AS_mode, get_g2


def optimize_AS_SPG(
        n_photon_init: float, 
        phase_init: float, 
        U: float, 
        J: float,
        gamma: float=0.,
        target_n_out: float|None=None ):

    # ancillary function for running the ODEs and extract g2
    def AS_g2_MZI(x):

        # readout params
        n_photon = x[0]
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
        if verbose_iter:
            print(f"x={[f'{xi:.6f}' for xi in x]}, g2={AS_g2_MZI(x):.6f}")

    def constraint(x):
        n_photon = x[0]
        phase = x[1]
        alphas, Ns, _ = solve_ODE_AS_mode(J, U, n_photon, phase, tlist=[0.,1.], gamma=gamma)
        alpha, N = alphas[-1], Ns[-1].real
        return (np.abs(alpha) ** 2 + N - target_n_out) ** 2
    
    x0 = np.array([n_photon_init, phase_init])
    bounds = [(.1, 100), (0., np.pi)]

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
    U, J, target_n_out, gamma = args

    n_photon_init     = 0.9
    phase_init        = np.sqrt(target_n_out / n_photon_init)

    try:

        res = optimize_AS_SPG(
            n_photon_init=n_photon_init, phase_init=phase_init,
            U=U, J=J, target_n_out=target_n_out, gamma=gamma)

        n_photon_optim      = res.x[0]
        phase_optim         = res.x[1]

        # solve AS ODE
        alphas, Ns, Ms = solve_ODE_AS_mode(
            J=J, U=U, n_photon=n_photon_optim, phase=phase_optim, tlist=[0.0, 1.0], gamma=gamma
        )

        # read out parameters
        alpha, N, M = alphas[-1], Ns[-1], Ms[-1]
        n_alpha = (np.abs(alpha) ** 2).real
        N = N.real

        # g2
        g2 = get_g2(alpha, N, M)

        return {
            "U":            U,
            "n_photon":     n_photon_optim,
            "na":           n_alpha + N,
            "target_n_out": target_n_out,
            "gamma":        gamma,     
            "g2":           g2.real,
            "alpha":        alpha,
            "N":            N,
            "M":            M,
            'J':            J,
            'phase':        phase_optim,
            "error":        None,
        }

    except Exception as exc:
        return {
            "U":            U,
            "n_photon":     n_photon_optim,
            "na":           n_alpha + N,
            "target_n_out": target_n_out,
            "gamma":        gamma,     
            "g2":           g2.real,
            "alpha":        alpha,
            "N":            N.real,
            "M":            M,
            'J':            J,
            'phase':        phase_optim,
            "error":        str(exc),
        }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    t_start = time()
    
    verbose_iter = False
    verbose = True

    # vec_U           = [0.01]
    # vec_J    = [np.pi/4]
    # vec_target_n_out = [0.01]   # <-- edit as needed
    # gammas          = [0.]

    vec_U           = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    vec_J           = np.linspace(np.pi/8, 3 * np.pi / 8, 50)
    vec_target_n_out = [0.01, 0.02, 0.05, 0.1, 0.2]  
    gammas          = [0., 0.1, 0.5]

    param_quadruples = list(product(vec_U, vec_J, vec_target_n_out, gammas))

    # number of jobs
    n_jobs = len(param_quadruples)
    max_workers = min(n_jobs, 350)

    print(f"Total jobs: {n_jobs}, running on {max_workers} workers")

    # results dict keyed by (U, ns, target_n_out)
    results: dict[tuple, dict] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(run_single, quadruple): quadruple
            for quadruple in param_quadruples
        }

        for future in as_completed(future_to_params):
            data = future.result()
            key  = (data["U"], data["J"], data["target_n_out"], data["gamma"])
            results[key] = data

            tgt_str = f"{data['target_n_out']:.3f}" if data["target_n_out"] is not None else "None"
            if data["error"]:
                print(f"[WARN] U={data['U']:.3e}  n_ph={data['n_photon']:.3e}  "
                      f"target={tgt_str} → {data['error']}")
            else:
                if verbose:
                    print(f"U={data['U']:.3e}   n_ph={data['n_photon']:.3e}   target={tgt_str}"
                        f"  g2={data['g2']:.4f}"
                        f"  |alpha|²={np.abs(data['alpha'])**2:.4f}")

    
    t_run = time() - t_start
    print(f"\n{n_jobs} jobs finished in {t_run:.2f}s, using {max_workers} workers")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    results['vec_U']            = vec_U
    results['vec_J']    = vec_J
    results['vec_target_n_out'] = vec_target_n_out
    results['gammas']           = gammas

    save_dir = Path(__file__).parent / "data_Gauss"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "results_optim_nphoton_phase.npy"
    np.save(save_path, results)

    print(f"\nResults saved to {save_path}")