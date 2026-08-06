import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
import traceback
from typing import Dict, List, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for parallel worker processes
import matplotlib.pyplot as plt

from src.mpdo_circuit import MPDOCircuit
from src.mpdo_torch import StateCreator, MPDOtorch
from src.create_circuit import create_qubit_circuit
from src.tracker import Tracker
from src.mpdo_optimizer import epoch_optimize

from experiments.ising_solver.utils import (
    iter_func, read_ising_instance, get_zz_rz_params
)


# script path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# NOTE: do NOT compute TIMESTAMP / BASE_SAVE_DIR here at module (import) time.
# With ProcessPoolExecutor(mp_context=spawn), every worker starts as a fresh
# interpreter that re-imports this module, so a module-level `datetime.now()`
# gets re-evaluated independently in each worker -> one different timestamp
# folder per worker process (this is exactly the bug that produced multiple
# "..._loop" folders a few seconds apart). Instead, the timestamp is computed
# once in main() and passed down explicitly through `config`.


def run_iteration(iteration: int, config: Dict[str, Any]) -> str:
    """
    Run one fully independent optimization for a given sample/iteration index.

    Each iteration re-seeds numpy/torch with (base_seed + iteration) before the
    random initial-state sampling:

        theta = theta_start + eps_theta * (0.5 - np.random.rand())

    so every parallel run draws its own independent random initial state, and
    results are saved under base_save_dir/iter_{iteration}/.
    """
    base_seed = config['seed']
    iter_seed = base_seed + iteration
    np.random.seed(iter_seed)
    torch.manual_seed(iter_seed)

    device = config['device']
    filename = config['filename']
    num_layers = config['num_layers']
    theta_start = config['theta_start']
    eps_theta = config['eps_theta']
    eps_theta_gate = config['eps_theta_gate']
    options_MPDO = config['options_MPDO']
    options_ADAM = dict(config['options_ADAM'])  # copy: we pop keys out of it below
    base_save_dir = config['base_save_dir']

    save_dir = os.path.join(base_save_dir, f"iter_{iteration}")
    os.makedirs(save_dir, exist_ok=True)

    # initial circuit params (uniform)
    tz_init, theta_A1_init, theta_A2_init = (
        0.1,                                # tz's
        0. * np.pi / 2 / num_layers,         # A1's
        0. * np.pi / 2 / num_layers          # A2's
    )

    # initialize input state as vacuum
    state_creator = StateCreator(Nmax=1, num_batch=1)

    # load ising instance
    num_channels, num_edge, J_mat = read_ising_instance(filename)
    J_mat = torch.tensor(J_mat, device=device, dtype=torch.float64)
    target = 2. * (num_edge - 2 * 536)

    # initiate starting state (this is the sampled step: independent per iteration)
    coeffs = []
    for _ in range(num_channels):
        theta = theta_start + eps_theta * (0.5 - np.random.rand())
        coeffs += [[np.cos(theta / 2), np.sin(theta / 2)]]
    list_ten = state_creator.product_state(coeffs, device=device)
    rho = MPDOtorch(list_ten)

    # set gate params
    params = [
        [
            get_zz_rz_params(tz_init, theta_A1_init, theta_A2_init, device=device)
            if (d + l) % 2 == 0 else None
            for l in range(num_channels)
        ]
        for d in range(num_layers)
    ]

    # set up circuit
    circuit = create_qubit_circuit(
        num_layers,
        num_channels,
        params=params,
        device=device
    )

    # define the objective from iter_func
    tracker = Tracker()
    obj_params = circuit.get_params()
    objective = lambda it, rho, obj_params: iter_func(
        rho, circuit, J_mat,
        options_MPDO=options_MPDO,
        tracker=tracker, it=it,
        obj_params=obj_params,
        save_dir=save_dir,
        target=target
        )

    # set up optimizer (ADAM)
    max_epochs = options_ADAM.pop('max_epochs')
    param_lims = options_ADAM.pop('param_lims')
    lr_min = options_ADAM.pop('lr_min')
    optimizer = torch.optim.Adam(
        circuit.get_variables(), **options_ADAM
    )
    if lr_min is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr_min)
    else:
        scheduler = None

    # see if individual bounds must be given to Ry gates
    if eps_theta_gate > 1e-4:
        param_lims_list = []
        for i, _ in enumerate(optimizer.param_groups[0]['params']):
            if i % 3 == 2:
                param_lims_list.append(param_lims)
            else:
                param_lims_list.append(
                    (np.pi / 2 - eps_theta_gate, np.pi / 2 + eps_theta_gate)
                )
        used_param_lims = param_lims_list
    else:
        used_param_lims = param_lims

    # optimize (using iterative, epoch solver)
    epoch_optimize(
        objective,
        optimizer=optimizer,
        rho=rho,
        max_epochs=max_epochs,
        obj_params=obj_params,
        param_lims=used_param_lims,
        scheduler=scheduler,
        epoch_optim_clear=None,
        epoch_print=None
    )

    return save_dir


def _run_iteration_worker(args):
    """Top-level wrapper so ProcessPoolExecutor (spawn context) can pickle it."""
    iteration, config = args
    try:
        result_dir = run_iteration(iteration, config)
        print(f"[iter_{iteration}] DONE -> {result_dir}")
        return iteration, True, result_dir
    except Exception as e:
        err = traceback.format_exc()
        print(f"[iter_{iteration}] FAILED: {e}\n{err}")
        return iteration, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Massively parallel sampling runs of the MPDO Ising solver."
    )
    parser.add_argument("--num_samples", type=int, required=True,
                         help="Number of independent iterations (samples) to run.")
    parser.add_argument("--num_cpus", type=int, required=True,
                         help="Number of CPU worker processes to use in parallel.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Base random seed; iteration i uses seed + i (default: 42).")
    args = parser.parse_args()

    # Compute the batch timestamp ONCE, here in the parent process, and pass
    # it explicitly to every worker via `config`. Do not recompute this inside
    # run_iteration or at module import time (see NOTE above).
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') + '_loop'
    base_save_dir = os.path.join(FILE_DIR, "data", timestamp)
    os.makedirs(base_save_dir, exist_ok=True)

    print(f"Batch timestamp: {timestamp}")
    print(f"Saving all iterations under: {base_save_dir}")
    print(f"num_samples={args.num_samples}, num_cpus={args.num_cpus}, base_seed={args.seed}")

    # fixed run configuration, identical for every iteration except its seed/theta sampling
    config = {
        'device': "cpu",
        'filename': 'experiments/ising_solver/ising_instances/g05_60.0',
        'num_layers': 50,
        'theta_start': np.pi / 2,
        'eps_theta': 0.0 * np.pi / 2,
        'eps_theta_gate': np.pi / 2,  # maximal rotation angle theta = pi/2 +- eps
        'seed': args.seed,
        'base_save_dir': base_save_dir,
        'options_MPDO': {
            'max_BD': 2, 'max_PD': 100, 'cutoff_BD': 1e-12, 'cutoff_PD': 1e-12
        },
        'options_ADAM': {
            'lr': 1e-2, 'lr_min': None, 'max_epochs': 200,
            'param_lims': (0., np.pi)
        },
    }

    tasks = [(i, config) for i in range(args.num_samples)]
    num_workers = max(1, min(args.num_cpus, args.num_samples))

    results = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(_run_iteration_worker, t): t[0] for t in tasks}
        for f in as_completed(futures):
            iteration, success, info = f.result()
            results.append((iteration, success, info))

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    print(f"All iterations finished: {n_ok} succeeded, {n_fail} failed.")
    if n_fail:
        for iteration, ok, info in sorted(results):
            if not ok:
                print(f"  iter_{iteration} failed: {info}")


if __name__ == "__main__":
    main()