import numpy as np
from datetime import datetime
from time import time
import os
import argparse
import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Callable, Tuple

from src.mpdo_circuit import MPDOCircuit
from src.mpdo_torch import StateCreator, MPDOtorch
from src.utils import BosonOperatorsTorch, eye_like
from src.create_circuit import create_qubit_circuit
from src.tracker import Tracker
from experiments.MIP.utils import (
    epoch_optimize
)
from experiments.MIP.SAT.SAT_utils import evaluate_clauses, clauses_satisfied, load_cnf



from src.svd_trunc import svd_trunc
from src.utils import irescale, iregroup, sqrtm


# script path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# timestamp for data storage
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
#TIMESTAMP = 'test'


def iter_func(
    rho: MPDOtorch,
    clauses: List[Tuple[int,...]],
    hx: torch.Tensor,
    it: int,
    tracker: Tracker,
    obj_params: List[Any],
    save_dir: str,
    target: float | None = None,
    entropy_weight: float = 0.,
    verbose: bool = False,
    schedule_coeff_h: Callable | None = None,
    K_samples: int = 10
    ):

    ##################################
    # Compute objective
    ##################################

    # evaluate clauses
    E_clauses = evaluate_clauses(rho, clauses)
    E_err = E_clauses.sum()

    # the Pauli X operator
    X_op = torch.tensor([[0., 1.], [1., 0.]], device=rho.device, dtype=rho.dtype)

    # compute transverse magnetic field
    X_expect = rho.local_expectations(X_op).real

    # energy contribution
    E_h = (hx * X_expect).sum() # transverse field

    # total energy for iteration
    if schedule_coeff_h is not None:
        E = (schedule_coeff_h(it) * E_h + (1. - schedule_coeff_h(it)) * E_err)
    else:
        E = E_err

    # compute local_expect Z
    P1 = torch.tensor([[1., 0.], [0., 0.]], device=rho.device, dtype=rho.dtype)
    P1_expect = rho.local_expectations(P1).real

    # discretized bits
    bits_disc = P1_expect >= 0.5
    sat_disc = clauses_satisfied(bits_disc, clauses)

    # sample bit strings 
    #   -> sample max. prob. number outcome
    #   -> invert and make bool
    #   -> evaluate satisfied clauses   
    bits_top = (1 - rho.sample_outcome(max=True)).bool() 
    sat_top = clauses_satisfied(bits_top, clauses)

    err_sample = len(clauses)
    bits_sample = torch.zeros_like(bits_top)
    for _ in range(K_samples):
        bs = (1 - rho.sample_outcome(max=False)).bool()
        sat_case = clauses_satisfied(bs, clauses)
        err_case = (~sat_case).sum()
        if err_case < err_sample:
            sat_sample = sat_case
            bits_sample = bits_sample
            err_sample = err_case

    err_disc, err_top, err_sample = (~sat_disc).sum(), (~sat_top).sum(), (~sat_sample).sum()


    # compute entanglement penalty (mean square entropy profile)
    entropy_profile = rho.entropy_profile()
    entropy_penalty = entropy_profile.pow(2).sum()
    FOM = (
        E
        + entropy_weight * entropy_penalty 
    ) / (1. + entropy_weight)


    # store information of epochs
    tracker.add('FOM', FOM)
    tracker.add('E', E)
    tracker.add('E_clauses', E_clauses)
    tracker.add('E_err', E_err)
    tracker.add('E_h', E_h)
    tracker.add('err_disc', err_disc)
    tracker.add('err_sample', err_sample)
    tracker.add('err_top', err_top)
    tracker.add('sat_disc', sat_disc)
    tracker.add('sat_sample', sat_sample)
    tracker.add('sat_top', sat_top)
    tracker.add('bits_disc', bits_disc)
    tracker.add('bits_sample', bits_sample)
    tracker.add('bits_top', bits_top)
    tracker.add('entropy_penalty', entropy_penalty)
    tracker.add('P1_expect', P1_expect)
    tracker.add('X_expect', X_expect)
    tracker.add('entropy_profile', entropy_profile)

    # visuals
    tracker.visualize(
        ['FOM'],
        filename=os.path.join(save_dir, 'FOM.png'))

    tracker.visualize(
        ['E_err', 'err_disc', 'err_top', 'err_sample'],
        filename=os.path.join(save_dir, 'E.png'),
        title=f"E_err: {E_err:.2f}, err_disc: {err_disc:.0f}, err_top: {err_top:.0f}, err_sample: {err_sample:.0f}",
        ylim=[None, None])

    tracker.visualize(
        ['E', 'E_err', 'E_h'],
        filename=os.path.join(save_dir, 'E_contr.png'),
        title=f"E: {E:.2f}, E_err: {E_err:.2f}, , E_h: {E_h:.2f}",
        ylim=[None, None])

    tracker.visualize(
        ['entropy_penalty'],
        filename=os.path.join(save_dir, 'entropy.png'))

    plt.figure(figsize=(5, 3))
    plt.plot(tracker['entropy_profile'][-1])
    plt.xlabel("bond")
    plt.ylabel(r"$S_\text{vN}$")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'entropy_profile.png'))
    plt.close()

    plt.figure(figsize=(5, 3))
    plt.bar(range(rho.num_channels), tracker['P1_expect'][-1])
    plt.xlabel("site")
    plt.ylabel(r"$\langle P_1 \rangle$")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'expect_P1.png'))
    plt.close()

    plt.figure(figsize=(5, 3))
    dat_p1 = np.array(tracker['P1_expect'])
    plt.plot(range(it + 1), dat_p1)
    plt.xlabel("iter")
    plt.ylabel(r"$\langle P_1 \rangle$")
    plt.axhline(y=0.5, linestyle="--", color="k")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'expect_P1_iter.png'))
    plt.close()


    plt.figure(figsize=(5, 3))
    dat_cl = np.array(tracker['E_clauses'])
    plt.plot(range(it + 1), dat_cl)
    plt.xlabel("iter")
    plt.ylabel(r"$E_{clause}$")
    # plt.axhline(y=0.5, linestyle="--", color="k")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'SAT_iter.png'))
    plt.close()
    

    # store data until last run
    tracker.save(os.path.join(save_dir, 'data.pkl'))

    # print info
    if verbose:
        print(f"[{os.path.basename(save_dir)}] FOM: {FOM:.2f}, E_err: {E_err:.2f}, err_disc: {err_disc:.2f}, E: {E:.2f}")
        print(f"[{os.path.basename(save_dir)}] max. BD: {rho.get_BDs()}")
        print(f"[{os.path.basename(save_dir)}] <P1>: {[f'{z:.2f}' for z in tracker['P1_expect'][-1]]}")

    return FOM


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, required=False, default=1)
    parser.add_argument("--timestamp", type=str, default=TIMESTAMP)
    parser.add_argument("--num_threads", type=int, default=1)
    args = parser.parse_args()

    # load filename
    instance = args.instance
    filename = f"experiments/MIP/SAT/instances/uf100-430/uf100-0{instance}.cnf"
    num_channels, n_clauses, clauses = load_cnf(filename)

    # pin BLAS/OMP threads per-process to avoid oversubscription when running many in parallel
    torch.set_num_threads(args.num_threads)

    device = "cpu"
    theta_start = 1 * np.pi / 2 
    eps_theta = 0. * np.pi / 2 # fluctuations on start initial y-axis rotation (theta)
    hx_eta = 0. # disorder in magnetic field
    
    PD = 1 # purity dimension
    BD = 5 # bond dimension

    seed = 42
    options_ADAM = { # solver options
        'lr': 2e-2, 'lr_min': None, 'max_epochs': 200, 
        }
    verbose = True
    
    # directory to store data
    SAVE_DIR = os.path.join(FILE_DIR, "data", args.timestamp, f"instance_{instance}")

    # initialize input state as vacuum
    state_creator = StateCreator(Nmax=1, num_batch=1)

    # initiate starting state
    np.random.seed(seed)

    coeffs = [] # fix initial spin => redundancy / inversion symmetry
    for _ in range(num_channels):
        theta = theta_start + eps_theta * (0.5 - np.random.rand())
        coeffs += [[np.cos(theta/2), - np.sin(theta/2)]]
    list_ten = state_creator.product_state(coeffs, device=device)

    # fix bond dimension (to be added to MPDOtorch class)
    for il in range(num_channels):
        Bl = list_ten[il]

        dim_l = min(
            BD, 
            2 ** il,
            2 ** (num_channels - il),  
            )
        
        dim_r = min(
            BD, 
            2 ** (il + 1), 
            2 ** (num_channels - il - 1), 
            )

        pad = (
            0, dim_r - 1,
            0, 0,
            0, dim_l - 1,
            0, PD - 1,
        )

        Bl = F.pad(Bl, pad)
        list_ten[il] = Bl

    rho = MPDOtorch(list_ten, options={
            'max_BD': BD, 'max_PD': PD,
            'cutoff_BD': -1, 'cutoff_PD': -1, 
            'lowrank': False
        })

    # add gradient tracking on tensors (to be added on MPDOtorch class)
    for B in rho._B:
        B.requires_grad=True

    # the number of epochs
    max_epochs = options_ADAM.pop('max_epochs')

    # set torch seed (for random magnetic fields to break symmetry)
    torch.manual_seed(seed)
    hx = (1. + hx_eta * (0.5 - torch.rand(num_channels))) * n_clauses / num_channels / 4 # transverse field (normalize with num_edges)
    #schedule_coeff_h = lambda it: np.maximum(0.1, 1. - (it + 1) / (0.75 * max_epochs))
    schedule_coeff_h = lambda it: np.maximum(0., 1. - (it + 1) / (max_epochs))

    # the scheduler
    
    # define the objective from the iter_func
    tracker = Tracker() 
    objective = lambda it, rho, obj_params: iter_func(
        rho=rho, clauses=clauses, hx=hx,
        tracker=tracker, it=it,
        obj_params=obj_params,
        save_dir=SAVE_DIR,
        verbose=True,
        schedule_coeff_h=schedule_coeff_h
        )
    
    # set up optimizer (ADAM)
    lr_min = options_ADAM.pop('lr_min')
    optimizer = torch.optim.Adam(
            rho._B, **options_ADAM
        )
    if lr_min is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr_min)
    else:
        scheduler = None
    
    # create write dir
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # optimize (using iterative, epoch solver)
    epoch_optimize(
        objective, 
        options_ADAM=options_ADAM, 
        rho=rho, 
        max_epochs=max_epochs,
        scheduler=scheduler,
        epoch_optim_clear=None
    )