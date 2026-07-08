import numpy as np
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

from ODE_AS_integrate import solve_ODE_AS_mode


# ---------------------------------------------------------------------------
# Worker function (must be top-level for pickling with multiprocessing)
# ---------------------------------------------------------------------------

def run_single(args):

    """Run optimization + ODE for a single (U, ns, target_n_out, gamma) quadruple.

    Returns
    -------
    dict with keys: U, ns, target_n_out, g2, alpha, N, M, error
    """

    # read out params
    U, n_photon, phase, gamma = args

    # coupler set equal, to 50:50
    J = np.pi / 4.0

    # solv e ODE
    alphas, Ns, Ms = solve_ODE_AS_mode(J, U, n_photon=n_photon, phase=phase, tlist=[0., 1.], gamma=gamma)

    # readout
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
        "U":            U,
        "n_photon":     n_photon,
        "phase":        phase,
        "na":           np.abs(alpha) ** 2 + N,
        "gamma":        gamma,     
        "g2":           g2.real,
        "alpha":        alpha,
        "N":            N.real,
        "M":            M,
        'coupling':     J,
        "error":        None,
    }




# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    t_start = time()

    verbose = False

    vec_U           = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    vec_n_photon    = np.logspace(0, 2, 50)
    vec_phase       = np.linspace(0.01, 0.2, 46) * np.pi  
    gammas          = [0., 0.1, 1.]

    # uncomment for single-run debugging
    # vec_U           = [0.05]
    # vec_n_photon    = [42.92]
    # vec_phase       = [0.628]  
    # gammas          = [0.]

    param_quadruples = list(product(vec_U, vec_n_photon, vec_phase, gammas))
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
            key  = (data["U"], data["n_photon"], data["phase"], data["gamma"])
            results[key] = data

        
            if verbose:
                print(f"U={data['U']:.3e}  ns={data['ns']:.3e}  phase={data['phase']}"
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
    results['vec_n_photon']     = vec_n_photon
    results['vec_phase']        = vec_phase
    results['gammas']           = gammas

    save_dir = Path(__file__).parent / "data_Gauss"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "results_run.npy"
    np.save(save_path, results)

    print(f"\nResults saved to {save_path}")