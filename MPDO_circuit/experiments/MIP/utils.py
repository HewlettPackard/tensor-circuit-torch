import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from typing import Dict, List, Any, Callable
from time import time

from src.mpdo_circuit import MPDOCircuit
from src.mpdo_torch import StateCreator, MPDOtorch
from src.tracker import Tracker


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

        # reinitialize MPDO/MPS tensors for next run (avoid canonicalization step in next backward)
        list_ten = [B.detach().clone().contiguous() for B in rho._B]
        BD, PD = rho.options['max_BD'], rho.options['max_PD']

        rho = MPDOtorch(list_ten, options={
            'max_BD': BD, 'max_PD': PD,
            'cutoff_BD': -1, 'cutoff_PD': -1, # no cutoff truncation -> BD,PD fixed
        })

        for B in rho._B:
            B.requires_grad=True

        if epoch_print is not None and epoch % epoch_print == 0:
            print(f"Epoch {epoch + 1} finished in {time() - epoch_start:.2f}s, "
                  f"objective: {loss.item():.6e}")
            print(f"Total runtime: {time() - start:.2f}s")


def read_ising_instance(filename):
    """
    Read an Ising instance file and return the coupling matrix J_ij and target energy.

    File format:
        line 1: <n_spins> <target_energy>
        lines 2..: <i> <j> <J_ij>   (1-indexed spins, J symmetric: J_ij = J_ji)

    Parameters
    ----------
    filename : str
        Path to the instance file.

    Returns
    -------
    J : np.ndarray, shape (n, n)
        Symmetric coupling matrix (0-indexed internally), zero diagonal.
    target_energy : float
        The target energy given as the 2nd entry on the first line.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    n_spins, num_edge = lines[0].split()[:2]
    n_spins = int(n_spins)
    num_edge = float(num_edge)

    J = np.zeros((n_spins, n_spins))

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        i, j, val = line.split()
        i, j = int(i) - 1, int(j) - 1  # convert to 0-indexed
        val = float(val)
        J[i, j] = val
        J[j, i] = val

    return n_spins, num_edge, J


    
def get_zz_rz_params(tz: float, theta_A1: float, theta_A2: float, dtype=torch.float64, device="cuda:0"):
    def const(v=0.0):
        return torch.tensor(v, dtype=dtype, device=device)                       # requires_grad=False
    def var(v):
        return torch.tensor(v, dtype=dtype, device=device, requires_grad=True)   # trainable leaf

    params = [
        var(theta_A1),const(0.), const(0.),   # A1: alpha (trainable), theta=0, phi=0 → pure Rz
        var(theta_A2),const(0.), const(0.),           # A2: frozen identity
        const(), const(), var(tz),         # tx=0, ty=0, tz trainable
        const(), const(), const(),              # B1: frozen identity
        const(), const(), const(),              # B2: frozen identity
    ]
    return params


import networkx as nx

def generate_graph(n, m=None, vertex_degree=3):
    """
    Generate a random graph and return its symmetric adjacency matrix.

    Parameters
    ----------
    n : int
        Number of vertices.
    m : int, optional
        Number of edges. If given, generates a random G(n, m) graph
        (Erdos-Renyi with fixed edge count), ignoring vertex_degree.
    vertex_degree : int, default 3
        If m is None, generates a random d-regular graph where every
        vertex has this degree. Requires n * vertex_degree to be even.

    Returns
    -------
    np.ndarray
        n x n symmetric adjacency matrix (0/1 entries, zero diagonal).
    """
    if m is not None:
        G = nx.gnm_random_graph(n, m)
    else:
        G = nx.random_regular_graph(vertex_degree, n)

    A = nx.to_numpy_array(G)
    return A


def iter_func(
    rho: MPDOtorch,
    circuit: MPDOCircuit,
    J_mat: torch.Tensor,
    it: int,
    options_MPDO: Dict[str, Any],
    tracker: Tracker,
    obj_params: List[Any],
    save_dir: str,
    target: float | None = None,
    entropy_weight: float = 0.,
    Z_abs_weight: float = 0.,
    Z_lim: float = 1.,
    verbose: bool = False
    ):

    # update circuit couplings after previous run
    circuit.update(obj_params)

    # run circuit
    rho_run = rho.clone()
    circuit.run(rho_run, options=options_MPDO)

    ##################################
    # Compute objective
    ##################################

    Z_op = torch.tensor([[1., 0.], [0., -1.]], device=rho_run.device, dtype=rho_run.dtype)

    C = rho_run.correlation_matrix(op1=Z_op, is_hermitian=True)
    E = (J_mat * C).sum().real

    # compute local_expect Z
    Z_expect = rho_run.local_expectations(Z_op).real

    rho_run.canonical_form()

    # correlation corr = <Z_i Z_j> - <Z_i><Z_j>
    corrs = C - torch.outer(Z_expect, Z_expect)
    corrs.fill_diagonal_(0.)

    # discretized <Z_i> -> s_i
    spins = Z_expect.sign()
    E_disc = spins @ J_mat @ spins

    # compute entanglement penalty (mean square entropy profile)
    entropy_profile = rho_run.entropy_profile()
    entropy_penalty = entropy_profile.pow(2).sum()
    FOM = (
        E
        + entropy_weight * entropy_penalty / (1. + entropy_weight)
        + Z_abs_weight * torch.relu(torch.abs(Z_expect) - Z_lim).pow(2).sum()
    )

    # store outcomes
    tracker.add('FOM', FOM)
    tracker.add('E', E)
    tracker.add('E_disc', E_disc)
    tracker.add('target', target)
    tracker.add('entropy_penalty', entropy_penalty)
    tracker.add('Z_expect', Z_expect)
    tracker.add('J_mat', J_mat)
    tracker.add('entropy_profile', entropy_profile)
    tracker.add('correlation', corrs)

    # visuals
    tracker.visualize(
        ['FOM'],
        filename=os.path.join(save_dir, 'FOM.png'))

    tracker.visualize(
        ['E', 'E_disc', 'target'],
        filename=os.path.join(save_dir, 'E.png'),
        title=f"target: {target:.2f}, E: {E:.2f}, E_disc: {E_disc:.2f}",
        ylim=[target - 1, 1])

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
    plt.bar(range(rho_run.num_channels), tracker['Z_expect'][-1])
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
        print(f"[{os.path.basename(save_dir)}] FOM: {FOM:.2f}, E: {E:.2f}, E_disc: {E_disc:.2f}")
        print(f"[{os.path.basename(save_dir)}] max. BD: {rho_run.get_BDs()}")
        print(f"[{os.path.basename(save_dir)}] <Z>: {[f'{z:.2f}' for z in tracker['Z_expect'][-1]]}")

    return FOM