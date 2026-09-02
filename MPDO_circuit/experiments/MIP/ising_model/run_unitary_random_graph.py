import numpy as np
from datetime import datetime
from time import time
import os
import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Callable

from src.mpdo_circuit import MPDOCircuit
from src.mpdo_torch import StateCreator, MPDOtorch
from src.utils import BosonOperatorsTorch, eye_like
from src.create_circuit import create_qubit_circuit
from src.tracker import Tracker
from src.mpdo_optimizer import epoch_optimize
from experiments.ising_solver.utils import (
    iter_func, generate_graph
)
import numpy as np
import argparse

from src.svd_trunc import svd_trunc
from src.utils import irescale, iregroup, sqrtm


# script path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# timestamp for data storage
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
TIMESTAMP = 'test'

 
def right_canonical_gradients(rho: MPDOtorch):

    for B in rho._B:
        s = B.shape

        Bm = B.view(s[0], s[1], s[2] * s[3]) # right-unitary matrix
        Gm = B.grad.view(s[0], s[1], s[2] * s[3]) # the reshaped grad

        # project the gradient on manifold tangent space 
        G_TU = Gm @ (torch.eye(s[2] * s[3], device=Bm.device, dtype=Bm.dtype) - Bm.mH @ Bm)

        # reshape and insert back 
        B.grad = G_TU.view(s[0], s[1], s[2], s[3])

    return None


def epoch_optimize(
    objective:          Callable,
    options_ADAM:       Dict[Any, Any],
    rho:                MPDOtorch,
    max_epochs:         int                                 = 500,
    epoch_print:        int                                 = 1,
    obj_params:         "List[torch.Tensor] | None"         = None,
    param_lims:         "List[float] | None"                = None,
    epoch_optim_clear:  "int | None"                        = None,
    scheduler:          "torch.optim.lr_scheduler._LRScheduler | None" = None,
) -> None:
    """
    Run a fixed-epoch optimisation loop.
    """
    start    = time()

    for epoch in range(max_epochs):
        epoch_start = time()

        # restate optimizer, with updated params (tensors)
        optimizer_run = torch.optim.Adam(rho._B, **options_ADAM)

        if epoch_print is not None:
            print(f'\nRunning opt epoch {epoch + 1}...')

        # --- First-order optimisers (Adam, SGD, …) ---
        optimizer_run.zero_grad()

        loss = objective(epoch, rho, obj_params)

        if torch.isnan(loss):
            print("Objective is NaN — stopping.")
            break

        # compute gradients
        loss.backward()

        # project on Grassmann tangent space 
        right_canonical_gradients(rho)

        # perform update step
        optimizer_run.step()

        # reinitialize MPDO/MPS tensors
        list_ten = [B.detach().clone().contiguous() for B in rho._B]

        rho = MPDOtorch(list_ten, options={
            'max_BD': BD, 'max_PD': PD,
            'cutoff_BD': -1, 'cutoff_PD': -1,
        })

        for B in rho._B:
            B.requires_grad=True

        if epoch_print is not None and epoch % epoch_print == 0:
            print(f"Epoch {epoch + 1} finished in {time() - epoch_start:.2f}s, "
                  f"objective: {loss.item():.6e}")
            print(f"Total runtime: {time() - start:.2f}s")


def iter_func(
    rho: MPDOtorch,
    J_mat: torch.Tensor,
    hx: torch.Tensor,
    it: int,
    tracker: Tracker,
    obj_params: List[Any],
    save_dir: str,
    target: float | None = None,
    entropy_weight: float = 0.,
    Z_abs_weight: float = 0.,
    Z_lim: float = 1.,
    verbose: bool = False,
    schedule_coeffs: Callable | None = None,
    z_thres: float = 0.0,
    K_samples: int = 10
    ):

    ##################################
    # Compute objective
    ##################################

    # the Pauli X and Z operator
    X_op = torch.tensor([[0., 1.], [1., 0.]], device=rho.device, dtype=rho.dtype)
    Z_op = torch.tensor([[1., 0.], [0., -1.]], device=rho.device, dtype=rho.dtype)

    # compute correlation matrix
    C = rho.correlation_matrix(op1=Z_op, is_hermitian=True)

    # compute transverse magnetic field
    X_expect = rho.local_expectations(X_op).real

    # energy contributions
    E_zz = (J_mat * C).sum().real # ising
    E_x = (hx * X_expect).sum() # transverse field

    # total energy for iteration
    if schedule_coeffs is not None:
        E = schedule_coeffs(it)[0] * E_zz + schedule_coeffs(it)[1] * E_x
    else:
        E = E_zz

    # compute local_expect Z
    Z_expect = rho.local_expectations(Z_op).real

    # correlation corr = <Z_i Z_j> - <Z_i><Z_j>
    corrs = C - torch.outer(Z_expect, Z_expect)
    corrs.fill_diagonal_(0.)

    # discretized <Z_i> -> s_i
    spins_disc = Z_expect.sign() * (Z_expect.abs() > z_thres)
    E_disc = spins_disc @ J_mat @ spins_disc

    # compute entanglement penalty (mean square entropy profile)
    entropy_profile = rho.entropy_profile()
    entropy_penalty = entropy_profile.pow(2).sum()
    FOM = (
        E
        + entropy_weight * entropy_penalty 
        + Z_abs_weight * torch.relu(torch.abs(Z_expect) - Z_lim).pow(2).sum()
    ) / (1. + entropy_weight + Z_abs_weight)

    # sample bit strings
    bs_top = rho.sample_outcome(max=True)
    spins_top = 1. - 2 * bs_top
    E_top = spins_top @ J_mat @ spins_top

    E_sample = 0.
    spins_sample = torch.zeros_like(spins_top)
    for _ in range(K_samples):
        bs = rho.sample_outcome(max=False)
        spins_test_sample = 1. - 2 * bs # convert bits to spins!
        E_case = spins_test_sample @ J_mat @ spins_test_sample
        if E_case < E_sample:
            E_sample = E_case if E_case < E_sample else E_sample
            spins_sample = spins_test_sample


    # store information of epochs
    tracker.add('FOM', FOM)
    tracker.add('E', E)
    tracker.add('E_zz', E_zz)
    tracker.add('E_x', E_x)
    tracker.add('E_disc', E_disc)
    tracker.add('E_sample', E_sample)
    tracker.add('E_top', E_top)
    tracker.add('spins_disc', spins_disc)
    tracker.add('spins_sample', spins_sample)
    tracker.add('spins_top', spins_top)
    tracker.add('target', target)
    tracker.add('entropy_penalty', entropy_penalty)
    tracker.add('Z_expect', Z_expect)
    tracker.add('X_expect', X_expect)
    tracker.add('J_mat', J_mat)
    tracker.add('entropy_profile', entropy_profile)
    tracker.add('correlation', corrs)

    # visuals
    tracker.visualize(
        ['FOM'],
        filename=os.path.join(save_dir, 'FOM.png'))

    tracker.visualize(
        ['E_zz', 'E_disc', 'E_top', 'E_sample', 'target'],
        filename=os.path.join(save_dir, 'E.png'),
        title=f"target: {target:.0f}, E_zz: {E_zz:.2f}, E_disc: {E_disc:.0f}, E_top: {E_top:.0f}, E_sample: {E_sample:.0f}",
        ylim=[None, 1.])

    tracker.visualize(
        ['E', 'E_zz', 'E_x'],
        filename=os.path.join(save_dir, 'E_contr.png'),
        title=f"target: {target:.0f}, E: {E:.2f}, E_zz: {E_zz:.2f}, , E_x: {E_x:.2f}",
        ylim=[None, 1.])

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
    plt.bar(range(rho.num_channels), tracker['Z_expect'][-1])
    plt.xlabel("site")
    plt.ylabel(r"$\langle \hat{Z}_i \rangle$")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'expect_Z.png'))
    plt.close()

    plt.figure(figsize=(5, 3))
    dat_sz = np.array(tracker['Z_expect'])
    plt.plot(range(it + 1), dat_sz)
    plt.xlabel("iter")
    plt.ylabel(r"$\langle \hat{Z}_i \rangle$")
    plt.axhline(y=0., linestyle="--", color="k")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'expect_Z_iter.png'))
    plt.close()

    plt.figure(figsize=(3,3))
    plt.matshow(np.abs(tracker['correlation'][-1]))
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'corrs.png'))
    plt.close()
    

    # store data until last run
    tracker.save(os.path.join(save_dir, 'data.pkl'))

    # print info
    if verbose:
        print(f"[{os.path.basename(save_dir)}] FOM: {FOM:.2f}, E_zz: {E_zz:.2f}, E_disc: {E_disc:.2f}, E: {E:.2f}")
        print(f"[{os.path.basename(save_dir)}] max. BD: {rho.get_BDs()}")
        print(f"[{os.path.basename(save_dir)}] <Z>: {[f'{z:.2f}' for z in tracker['Z_expect'][-1]]}")

    return FOM


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=False, default=100)
    parser.add_argument("--timestamp", type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument("--num_threads", type=int, default=1)
    args = parser.parse_args()

    # pin BLAS/OMP threads per-process to avoid oversubscription when running many in parallel
    torch.set_num_threads(args.num_threads)

    device = "cpu"
    num_channels = args.n # number of channels
    theta_start = 1 * np.pi / 2 
    eps_theta = 0. * np.pi / 2 # fluctuations on start initial y-axis rotation (theta)
    hx_eta = 0. # disorder in magnetic field
    
    PD = 1 # purity dimension
    BD = 2 # bond dimension

    seed = 42
    options_ADAM = { # solver options
        'lr': 5e-3, 'lr_min': None, 'max_epochs': 200, 
        }
    verbose = True
    

    # directory to store data
    SAVE_DIR = os.path.join(FILE_DIR, "data", args.timestamp)



    # initialize input state as vacuum
    state_creator = StateCreator(Nmax=1, num_batch=1)


    # load ising instance
    J_mat = generate_graph(n=num_channels, vertex_degree=3)
    num_edges = np.sum(np.abs(J_mat) > 0.) // 2

    # the graph matrix
    J_mat = torch.tensor(J_mat, device=device, dtype=torch.float64)

    # the target energy
    # target = 2. * (num_edge - 2 * MC_val)

    # initiate starting state
    np.random.seed(seed)

    coeffs = [] # fix initial spin => redundancy / inversion symmetry
    for _ in range(num_channels):
        theta = theta_start + eps_theta * (0.5 - np.random.rand())
        coeffs += [[np.cos(theta/2), - np.sin(theta/2)]]
    list_ten = state_creator.product_state(coeffs, device=device)

    # add gradient tracking
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


    for B in rho._B:
        B.requires_grad=True

    # set torch seed (for random magnetic fields to break symmetry)
    torch.manual_seed(seed)
    hx = (1. + hx_eta * (0.5 - torch.rand(num_channels))) * num_edges / num_channels / 4 # transverse field (normalize with num_edges)

    # the scheduler
    max_epochs = options_ADAM.pop('max_epochs')
    schedule_coeffs = lambda it: [(it + 1) / (max_epochs ), (1. - (it + 1) / (max_epochs))]

    # define the objective from the iter_func
    tracker = Tracker() 
    objective = lambda it, rho, obj_params: iter_func(
        rho=rho, J_mat=J_mat, hx=hx,
        tracker=tracker, it=it,
        obj_params=obj_params,
        target=0.,
        save_dir=SAVE_DIR,
        verbose=True,
        schedule_coeffs=schedule_coeffs
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