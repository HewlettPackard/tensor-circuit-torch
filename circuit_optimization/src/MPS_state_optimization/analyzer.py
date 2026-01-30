from typing import Iterable, Tuple, Any
from qutip import Qobj, wigner
import torch
import numpy as np

from src.MPS_state_optimization import MPStorch, iregroup


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np



def get_wigner_function(rho: np.ndarray, Nx=200) -> Tuple[np.ndarray, np.ndarray]:
        
    """
    Get the Wigner function of a single-mode density matrix.

    Parameters
    ----------
    rho: np.ndarray
        State density matrix
    Nx: int
        Number of Wigner sample points in (x,p) space 

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The Wigner function sampling points and corresponding axis
    """

    # Phase space coordinates
    N = rho.shape[0]
    xvec = np.linspace(-np.sqrt(N), np.sqrt(N), Nx)

    # Wigner function
    rho_qobj = Qobj(rho, dims=[[N], [N]])
    W = wigner(rho_qobj, xvec, xvec)

    return W, xvec


def visualize_circuit(
        profile: Iterable[Iterable[float|None]], 
        dl: float=0.2, 
        dg: float=0.5, 
        frac_block: float=0.8, 
        figsize: Tuple[int]=(5,10), 
        save_path: str=None, 
        cmap: Any=None, 
        fontsize: int=10,
        fontsize_cb: int=10
        ):
    
    """
    Visualize the circuit and the corresponding J-values.

    Parameters
    ----------
    profile: Iterable[Iterable[float|None]]
        The profile of couplers, organized in list of lists of J-values 
        (per layer - per left position of two-mode gate). None means there 
        is no gate on that position.

    dl: float (default 0.2)
        Width of layer

    dg: float (default 0.5)
        Width of position

    frac_block: float (default 0.8)
         The fraction that block spans of the width and height

    figsize: Tuple[int] (default (5,10))
        Figure size

    save_path: str (default None)
        The path to save the figure. If None, no save.

    cmap: Any (default None)
        The colormap to use. Default (None) uses coolwarm

    fontsize: int (default 10)
        Fontsize of J-values in boxes.

    fontsize_cb: int (default 10)
        Fontsize of colorbar axis description
    """

    num_layer = len(profile)
    num_wg = len(profile[0]) + 1
    width, height = 2 * dg * frac_block, dl * frac_block

    min_val = np.min([np.min([g for g in layer if g is not None]) for layer in profile])
    max_val = np.max([np.max([g for g in layer if g is not None]) for layer in profile])

    if cmap is None:
        cmap = cm.coolwarm  # You can change this to another colormap like 'coolwarm', 'plasma', etc.
    lim_val = np.maximum(np.abs(min_val), np.abs(max_val))
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    for i in range(num_wg):
        ax.axvline(i * dg + width * frac_block / 4, color='k', zorder=0)

    for i_l, layer in enumerate(profile):
        for i_g, gate_val in enumerate(layer):
            
            if gate_val is None:
                continue

            y, x = dl * i_l, dg * i_g  # Bottom left
            color = cmap(norm(gate_val))  # Get color from colormap based on value
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='black', facecolor=color, alpha=0.95)
            ax.add_patch(rect)

            ax.text(x + width/2, y + height/2, f"{gate_val:.2f}", fontsize=fontsize, ha='center', va='center', fontweight='bold', color='black')



    # Show the plot
    ax.set_xlim(-dg, (num_wg + 1) * dg)
    ax.set_ylim(-dl, (num_layer + 1) * dl)
    # ax.set_aspect('equal')

    # Hide axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add a colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label(r"$J \cdot \Delta t$")
    cbar.ax.tick_params(labelsize=fontsize_cb)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_rho(rho: np.ndarray, rho_target: np.ndarray, figname: str='graphs', figsize: Tuple[int]=(8,10)):

    """
    Visualize the density matrix and compare with target density matrix.

    rho: np.ndarray
        The density matrix.

    rho_target: np.ndarray
        The target density matrix.

    figname: str='graphs'
        Default graphs (always enter path).

    figsize: Tuple[int]=(8,10)
        The figsize (matplotlib).
    
    """

    # make sure rho is in matrix form
    dims = len(rho.shape)
    rho = iregroup(rho,
        [[i for i in range(0,dims,2)], 
         [i for i in range(1,dims,2)]]
        )

    rho_target = iregroup(rho_target,
        [[i for i in range(0,dims,2)], 
         [i for i in range(1,dims,2)]]
        )

    fig, ax = plt.subplots(3, 2, figsize=figsize)

    # Titles for columns
    ax[0,0].set_title(r'$\rho_\text{optim}$', fontsize=12)
    ax[0,1].set_title(r'$\rho_\text{target}$', fontsize=12)

    # Labels for rows (rotated)
    row_labels = [r'$\text{Re}\rho$', r'$\text{Im}\rho$', r'$\text{Wigner}$']
    for i, label in enumerate(row_labels):
        ax[i,0].annotate(label, xy=(-0.5, 0.5), xycoords='axes fraction', 
                         fontsize=12, rotation=90, va='center', ha='center')

    # Real part of rho
    p = ax[0,0].matshow(rho.real, cmap='viridis')
    fig.colorbar(p, ax=ax[0,0])

    p = ax[0,1].matshow(rho_target.real, cmap='viridis')
    fig.colorbar(p, ax=ax[0,1])

    # Imaginary part of rho
    p = ax[1,0].matshow(rho.imag, cmap='viridis')
    fig.colorbar(p, ax=ax[1,0])

    p = ax[1,1].matshow(rho_target.imag, cmap='viridis')
    fig.colorbar(p, ax=ax[1,1])

    # Wigner functions
    wig, xvec1 = get_wigner_function(rho)
    wig_target, xvec2 = get_wigner_function(rho_target)

    p = ax[2,0].pcolormesh(xvec1, xvec1, wig, shading='auto', cmap='RdBu_r')
    fig.colorbar(p, ax=ax[2,0])

    p = ax[2,1].pcolormesh(xvec2, xvec2, wig_target, shading='auto', cmap='RdBu_r')
    fig.colorbar(p, ax=ax[2,1])

    plt.tight_layout()
    plt.savefig(figname)



class StateAnalyzer:

    """
    The state analyzer class, for evaluating the FOMs defined on the MPS.

    Parameters
    ----------
    psi: MPStorch
        The MPS state to be analyzed.

    modes: int|Iterable[int]
        The modes to be analyzed.
    """

    def __init__(self, psi: MPStorch, modes: int|Iterable[int]):
        self.psi = psi
        self.modes = modes

        self.device = psi.device
        self.rho = psi.ptrace(modes)


    def get_overlap(
        self, 
        rho_target: torch.Tensor, 
        mask: float|torch.Tensor=1., 
        trace_distance: bool=True
        ) -> torch.Tensor:

        """Compute the masked overlap with target DM, by default using the trace distance."""

        if trace_distance:
            diff = (self.rho - rho_target) * mask
            return torch.linalg.eigvals(diff).abs().sum()


        if not isinstance(mask, float):
            mask = mask.flatten()
    
        return ((self.rho.flatten() - rho_target.flatten()) * mask.flatten()).norm()


    def get_entanglement_measure(self, alpha: int=1, power=2) -> torch.Tensor:
        """
            Calculate entanglement profile of Renyi entropy alpha and return mean (RMSE if power is two). 
            Setting alpha=1 (default) gives the VN entanglement entropy
        """
        S_profile = self.psi.get_entanglement_entropy_profile(alpha=alpha)
        return S_profile.pow(power).sum()


    def get_g2(self) -> torch.Tensor:

        """Compute the single-mode density-density correlator."""

        # contruct diagonal number operator
        inds = range(self.rho.shape[0])
        num_vals = torch.tensor(inds, device=self.device)

        # < n * (n-1) > / <n>^2
        return (
            (self.rho[inds,inds] * num_vals * (num_vals-1)).sum().real 
                / (self.rho[inds,inds].real * num_vals).sum() ** 2
        )


class Tracker:

    """
    The class to track the results during the course of the optimization.
    """

    def __init__(self):

        # the dictionary to store the results
        self.results = {}

    def add(self, key: str, val: Any):

        """Add a key and a value to the dictionary."""

        # make sure val is in numpy format on cpu, else transfer
        val  = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val

        # append key to result
        if key in self.results:
            self.results[key].append(val)
        else:
            self.results[key] = [val]

    def __getitem__(self, key: str):

        """Get an item from the dictionary. (overloaded function)"""

        return self.results[key]



