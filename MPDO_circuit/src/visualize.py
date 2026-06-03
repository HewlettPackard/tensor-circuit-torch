"""
visualize.py
------------
Plotting helpers for circuit topology, photon-number distributions and
entropy profiles.
"""

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches

from src.mpdo_torch import MPDOtorch


def visualize_circuit(
    profile:      Iterable[Iterable["float | None"]],
    dl:           float                  = 0.2,
    dg:           float                  = 0.5,
    frac_block:   float                  = 0.8,
    figsize:      Tuple[int, int]        = (5, 10),
    filename:     "str | None"           = None,
    cmap:         Any                    = None,
    fontsize:     int                    = 10,
    fontsize_cb:  int                    = 10,
    phase_shifts: "Dict[str, Any] | None" = None,
) -> None:
    """
    Draw a colour-coded circuit diagram of coupler J-values.

    Each two-mode gate is a coloured rectangle; its fill colour encodes the
    J·Δt value relative to the global min/max via `cmap`.  An optional
    single-mode phase layer (grey boxes labelled θ) can be inserted at a
    specified depth.

    Parameters
    ----------
    profile      : coupling profile — list of layers, each a list of
                   J-values (float) or None (no gate at that position)
    dl           : layer height in figure units
    dg           : gate width in figure units
    frac_block   : fraction of dl/dg that each rectangle occupies
    figsize      : matplotlib figure size
    filename     : if given, save to this path and close; else call plt.show()
    cmap         : matplotlib colormap (default: coolwarm)
    fontsize     : font size for J-value labels inside boxes
    fontsize_cb  : font size for colorbar tick labels
    phase_shifts : dict with keys
                     'layer'     — depth index at which to insert phase layer
                     'positions' — per-channel phase values (None or < 1e-12 → skip)
    """
    profile    = list(profile)
    num_layers = len(profile)
    num_wg     = len(profile[0])

    single_width = dg  * frac_block
    width        = 2.0 * dg * frac_block
    height       = dl  * frac_block

    # Global colour scale
    all_vals  = [g for layer in profile for g in layer if g is not None]
    min_val, max_val = np.min(all_vals), np.max(all_vals)

    # Insert a placeholder layer for the phase-shift row
    if phase_shifts is not None:
        layer_phase_shift = phase_shifts['layer']
        num_layers += 1
        profile = profile[:layer_phase_shift] + [None] + profile[layer_phase_shift:]

    if cmap is None:
        cmap = cm.coolwarm
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    fig, ax = plt.subplots(figsize=figsize)

    # Vertical waveguide lines
    for i in range(num_wg):
        ax.axvline(i * dg + width * frac_block / 4.0, color='k', zorder=0)

    for i_l, layer in enumerate(profile):

        # Phase-shift layer
        if phase_shifts is not None and i_l == layer_phase_shift:
            for i_g, phase in enumerate(phase_shifts['positions']):
                if phase is None or phase < 1e-12:
                    continue
                y, x = dl * i_l, dg * i_g
                rect = patches.Rectangle(
                    (x, y), single_width, height,
                    linewidth=2, edgecolor='black', facecolor='gray', alpha=0.95,
                )
                ax.add_patch(rect)
                ax.text(x + single_width / 2, y + height / 2, r'$\theta$',
                        fontsize=fontsize, ha='center', va='center',
                        fontweight='bold', color='black')
            continue

        # Coupler layer
        for i_g, gate_val in enumerate(layer):
            if gate_val is None:
                continue
            y, x  = dl * i_l, dg * i_g
            color = cmap(norm(gate_val))
            rect  = patches.Rectangle(
                (x, y), width, height,
                linewidth=2, edgecolor='black', facecolor=color, alpha=0.95,
            )
            ax.add_patch(rect)
            ax.text(x + width / 2, y + height / 2, f"{gate_val:.2f}",
                    fontsize=fontsize, ha='center', va='center',
                    fontweight='bold', color='black')

    ax.set_xlim(-dg, (num_wg + 1) * dg)
    ax.set_ylim(-dl, (num_layers + 1) * dl)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label(r"$J \cdot \Delta t$")
    cbar.ax.tick_params(labelsize=fontsize_cb)

    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def visualize_number_distribution(
    ns:      MPDOtorch,
    figname: str            = 'numbers',
    figsize: Tuple[int, int] = (6, 3),
) -> None:
    """
    Bar chart of per-channel photon numbers ⟨a†a⟩.

    Parameters
    ----------
    ns      : photon-number tensor (1-D, one entry per channel)
    figname : output file path (PNG)
    figsize : figure size
    """
    ns_np = ns.detach().cpu().numpy()
    plt.figure(figsize=figsize)
    plt.bar(np.arange(ns_np.size), ns_np)
    plt.xlabel('waveguide l')
    plt.ylabel(r'intensity $\langle a^\dagger a \rangle$')
    plt.savefig(figname)
    plt.close()


def visualize_entropy(
    S:       MPDOtorch,
    figname: str            = 'entropy',
    figsize: Tuple[int, int] = (6, 3),
) -> None:
    """
    Line plot of the entropy profile S(bond) along the chain.

    Parameters
    ----------
    S       : entropy values tensor (one per bond or site)
    figname : output file path (PNG)
    figsize : figure size
    """
    S_np = S.detach().cpu().numpy()
    plt.figure(figsize=figsize)
    plt.plot(np.arange(S_np.size), S_np)
    plt.xlabel('waveguide l')
    plt.ylabel(r'Entropy $S_{vN}$')
    plt.savefig(figname)
    plt.close()
