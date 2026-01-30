####################################################
# General utility functions for metrology
####################################################

import matplotlib.pyplot as plt
import torch
import os
import pickle
import numpy as np
from typing import List, Iterable, Tuple

from src.MPS_state_optimization import MPStorch
from src.MPS_state_optimization.analyzer import visualize_circuit, Tracker
from src.MPS_state_optimization.utils import sqrtm, mat_pow

import matplotlib.pyplot as plt
import numpy as np

def plot_rho(rho: np.ndarray, figsize: Tuple=(6, 10), save_path: str='rho', cmap: str='viridis'):
    """
    Plots the real and imaginary parts of a complex matrix with a shared colorbar.
    
    Parameters
    ----------
        rho: np.ndarray
            Complex-valued 2D array.

        figsize: Tuple
            Size of the figure (width, height).

        cmap: str (default 'viridis')
            Colormap to use.
    """

    fig, ax = plt.subplots(3, 1, figsize=figsize)

    # Shared color scale across both plots
    vmin = min(rho.real.min(), rho.imag.min())
    vmax = max(rho.real.max(), rho.imag.max())

    # Plot real and imaginary parts
    im0 = ax[0].matshow(rho.real, vmin=vmin, vmax=vmax, cmap=cmap)
    ax[0].set_title('Real Part')

    im1 = ax[1].matshow(rho.imag, vmin=vmin, vmax=vmax, cmap=cmap)
    ax[1].set_title('Imaginary Part')

    im2 = ax[2].matshow(np.abs(rho), vmin=vmin, vmax=vmax, cmap=cmap)
    ax[2].set_title('Absolute Value')

    # Create space for colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Add colorbar (shared scale)
    fig.colorbar(im1, cax=cax, label='Value')

    plt.tight_layout()
    plt.savefig(save_path)



def status_update_func(
        psi: MPStorch, 
        score: float,
        params: Iterable[float],
        runtime: float,
        iter: int, 
        tracker: Tracker,
        modes: Iterable[int]=None,
        call_print: int=1,
        write_dir: str="."
        ):
    
    """
    The status update function for metrology, called after every optimization iteration

    Parameters
    ----------
    
    psi: MPStorch
        The MPS after last iteration

    score: float
        The obtained FOM score

    params: Iterable[float]
        The parameters to store

    runtime: float
        Runtime

    iter: int
        The iteration of optimization

    tracker: Tracker
        The tracker object

    modes: Iterable[int] (default None)
        The MPS mode to optimize. If None (default) mid circuit L // 2
    call_print: int (default 1)
        Number of iterations between screen prints. Defaults to 1, every iteration.

    write_dir: str (default ".")
        The directory to write to.


    Returns
    -------
    Tracker
        The updated Tracker object
    """

    if not (iter + 1) % call_print == 0:
        return None
    
    
    # add parameters
    tracker.add('params', params)
    tracker.add('score', score)
    tracker.add('runtime', runtime)

    # plot derivative
    plt.figure(figsize=(6,4))
    plt.plot(tracker['score'])
    plt.xlabel("iter")
    plt.ylabel(r"objective")
    plt.tight_layout()
    plt.savefig(os.path.join(write_dir, f'score.png'))
    plt.close()

    # visualize circuit
    visualize_circuit(
        tracker['params'][-1], 
        save_path=os.path.join(write_dir, f'circuit_optim.png'),
        fontsize=12,
        fontsize_cb=12)
    
    # visualize density matrix, if modes specified
    if modes is not None:
        rho = psi.ptrace(modes, keep_batch=True)
        if isinstance(modes, list) and len(modes) == 2:
            rho = rho.permute((0,2,1,3)).reshape((-1, psi.d ** 2, psi.d ** 2)).squeeze()
        tracker.add('rho', rho)

        # plot mean rho
        plot_rho(rho.mean(dim=0).detach().cpu(), save_path=os.path.join(write_dir, f'rho.png'))
    
    # collect results and store in pkl
    tracker.add('objective', score)
    tracker.add('runtime', runtime)
    tracker.add('params', params)
    tracker.add('BDs', psi.get_BDs())

    # print some metrics
    print(f"score: {score}, max. BD: {np.max(psi.get_BDs())}, max. S_VN: {psi.get_entanglement_entropy_profile().max().detach().cpu():.2f}")


    # save the current metric results as pkl
    file_pkl = os.path.join(write_dir, f'results.pkl')
    with open(file_pkl, 'wb') as fp:
        pickle.dump(tracker.results, fp)
        print(f'Results saved successfully in {file_pkl}\n')

    return tracker


def classical_Fisher_information(
        psi: MPStorch, 
        d_theta: float, 
        i_check: int|None=None,  
        weight_N_max: float=100.
        ) -> torch.Tensor:

        """
        The classical Fisher information for number measurements

        Parameters
        ----------

        psi: MPStorch
            The input MPS state

        d_theta: float
            The differential for FI derivative

        i_check: int|None (default None)
            The mode to evaluate, None (default) is mid circuit L // 2

        weight_N_max: float (default 100)
            The penalty weight for occupying Nmax Fock state, to avoid over-occupation of number state

        Returns:
        --------

        torch.Tensor
            The Fisher information
        """

        # construct local density matrices
        rho_batch = psi.ptrace(i_check, keep_batch=True)
        rho_0, rho_dtheta = rho_batch[0], rho_batch[1]

        # select the diagonal of rho to get the classical number distribution
        ind = range(psi.d)
        Pn_0, Pn_dtheta = rho_0[ind,ind].real, rho_dtheta[ind,ind].real

        # compute Fisher information
        FI = torch.sum(
            ((torch.log(Pn_0) - torch.log(Pn_dtheta)) / d_theta) ** 2 * Pn_0
            )

        return FI - weight_N_max * Pn_0[-1]
        


def quantum_Fisher_information(
        psi: MPStorch, 
        d_theta: float, 
        i_check: int|None=None,  
        full_state=True
        ) -> torch.Tensor:
    
    """
    Compute the quantum Fisher information from overlap two batch states 
    (not so important, used as check, mainly)
    """

    if full_state:
        fidelity = psi.select_batch_state(0).overlap(psi.select_batch_state(1))
        QFI = 4. * (1. - fidelity) / d_theta ** 2
        return QFI

    if i_check is None:
        i_check = psi.num_channels // 2
    # get reduced rho's from two MPS
    rho_0, rho_dtheta = (
        psi.select_batch_state(0).ptrace(i_check), 
        psi.select_batch_state(1).ptrace(i_check)
    )

    sqrt_rho_0 = sqrtm(rho_0)

    arg_trace = sqrtm(
        sqrt_rho_0 @ rho_dtheta @ sqrt_rho_0,
    )

    fidelity = (arg_trace).diag().real.sum() ** 2

    QFI = 4. * (1. - fidelity) / d_theta ** 2

    return QFI
