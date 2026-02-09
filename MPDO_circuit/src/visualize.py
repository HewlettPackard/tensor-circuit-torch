import numpy as np
import torch
from typing import Dict, Tuple, Iterable, Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from src.mpdo_torch import MPDOtorch

def visualize_circuit(
        profile: Iterable[Iterable[float|None]], 
        dl: float=0.2, 
        dg: float=0.5, 
        frac_block: float=0.8, 
        figsize: Tuple[int]=(5,10), 
        filename: str=None, 
        cmap: Any=None, 
        fontsize: int=10,
        fontsize_cb: int=10,
        phase_shifts: Dict[str, Any]|None=None
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

    num_layers = len(profile)
    num_wg = len(profile[0])
    single_width, width, height = dg * frac_block, 2 * dg * frac_block, dl * frac_block

    min_val = np.min([np.min([g for g in layer if g is not None]) for layer in profile])
    max_val = np.max([np.max([g for g in layer if g is not None]) for layer in profile])

    if phase_shifts is not None:
        layer_phase_shift = phase_shifts['layer']
        phase_shift_positions = phase_shifts['positions']
        num_layers += 1
        profile = profile[:layer_phase_shift] + [None] + profile[layer_phase_shift:]

    if cmap is None:
        cmap = cm.coolwarm  # You can change this to another colormap like 'coolwarm', 'plasma', etc.
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    for i in range(num_wg):
        ax.axvline(i * dg + width * frac_block / 4, color='k', zorder=0)

    for i_l, layer in enumerate(profile):

        # visualize the phase shifts if appropriate layer
        if phase_shifts is not None and i_l == layer_phase_shift:
            for i_g, phase in enumerate(phase_shifts['positions']):
                if phase is None or phase < 1e-12:
                    continue
                else:
                    y, x = dl * i_l, dg * i_g  # Bottom left
                    rect = patches.Rectangle((x, y), single_width, height, linewidth=2, edgecolor='black', facecolor='gray', alpha=0.95)
                    ax.add_patch(rect)
                    ax.text(x + single_width/2, y + height/2, r'$\theta$', fontsize=fontsize, ha='center', va='center', fontweight='bold', color='black')
           
            continue

        # visualize the layer coupling gates
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
    ax.set_ylim(-dl, (num_layers + 1) * dl)
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

    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        
    
def visualize_number_distribution(ns: MPDOtorch, figname='numbers', figsize=(6,3)):

    ns = ns.detach().cpu().numpy()

    plt.figure(figsize=figsize)

    l = np.arange(ns.size)
    plt.bar(l, ns)
    plt.xlabel('waveguide l')
    plt.ylabel(r'intensity $\langle a^\dagger a \rangle$')
    plt.savefig(figname)
    plt.close()

def visualize_entropy(S: MPDOtorch, figname='numbers', figsize=(6,3)):

    S = S.detach().cpu().numpy()

    plt.figure(figsize=figsize)

    l = np.arange(S.size)
    plt.plot(l, S)
    plt.xlabel('waveguide l')
    plt.ylabel(r'Entropy $S_{vN}$')
    plt.savefig(figname)
    plt.close()

