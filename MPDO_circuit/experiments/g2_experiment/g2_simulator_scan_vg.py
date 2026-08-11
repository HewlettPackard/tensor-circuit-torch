"""
g2_simulator_scan_vg.py
-----------------------
Scans g²(0) and photon number as a function of group velocity vg, without
polariton k-dependent rescaling (no exciton-fraction weighting).

For each vg value the script:
  1. Derives the layer propagation time dt and effective nonlinearity U.
  2. Builds and runs a nonlinear photonic MPDO circuit.
  3. Collects output photon numbers and g²(0).
  4. Saves a 2-panel summary plot and a .pkl results file.

Usage
-----
    python -m experiments.g2_experiment.g2_simulator_scan_vg
           [--hg HG] [--gamma GAMMA] [--phi PHI]
           [--writedir DIR] [--device DEVICE]
"""

import os
import pickle
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from src.mpdo_torch import MPDOtorch, StateCreator
from src.create_circuit import create_nonlinear_photonic_circuit
from src.CLI_utils import get_CLI_input

# ---------------------------------------------------------------------------
# Paths / timestamps
# ---------------------------------------------------------------------------

FILE_DIR  = os.path.dirname(os.path.abspath(__file__))
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ħ in µeV·ps (converts µeV interaction energies to ps⁻¹ rates)
hbar_ueV_ps = 6.582e2


# ---------------------------------------------------------------------------
# Physical helper
# ---------------------------------------------------------------------------

def pulse_nonlinear_rate(vg: float, pulse_width: float, g: float) -> float:
    """
    Effective nonlinear interaction rate U (ps⁻¹) for a Gaussian pulse.

    Integrates the squared normalised Gaussian spatial profile to obtain the
    inverse volume element, then multiplies by the bare interaction rate g.

    Parameters
    ----------
    vg          : group velocity (µm/ps).
    pulse_width : temporal pulse width (ps).
    g           : bare nonlinear interaction rate (ps⁻¹).

    Returns
    -------
    float
        Effective nonlinear rate U (ps⁻¹).
    """
    spatial_width = pulse_width * vg
    x  = np.linspace(-5.0 * spatial_width, 5.0 * spatial_width, 5001)
    dx = x[1] - x[0]
    profile    = (np.exp(-x ** 2 / (2.0 * spatial_width ** 2))
                  / np.sqrt(2.0 * np.pi * spatial_width ** 2))
    inv_volume = np.sum(profile ** 2) * dx
    return g * inv_volume


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def CLI_overwrite(
    hg:       float,
    gamma:    float,
    phi:      float,
    writedir: str,
    device:   str,
):
    """
    Override default parameters from command-line arguments when present.

    Recognised flags: --hg, --gamma, --phi, --writedir, --device.
    Falls back to the supplied defaults when a flag is absent.

    Parameters
    ----------
    hg       : default bare exciton interaction (µeV·µm²).
    gamma    : default photon loss rate (ps⁻¹).
    phi      : default relative input phase (rad).
    writedir : default output directory path.
    device   : default Torch device string.

    Returns
    -------
    tuple
        (hg, gamma, phi, writedir, device) with CLI values substituted.
    """
    hg       = get_CLI_input('--hg',       type=float, default=hg)
    gamma    = get_CLI_input('--gamma',    type=float, default=gamma)
    phi      = get_CLI_input('--phi',      type=float, default=phi)
    writedir = get_CLI_input('--writedir', type=str,   default=writedir)
    device   = get_CLI_input('--device',   type=str,   default=device)
    return hg, gamma, phi, writedir, device


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_data(
    vgs:       np.ndarray,
    densities: np.ndarray,
    g2s:       np.ndarray,
    g2_max:    float = 2.0,
    suptitle:  str   = "results",
    figname:   str   = "results.png",
) -> None:
    """
    Save a 2-panel summary figure as a function of group velocity.

    Panels
    ------
    1. Output photon densities per channel.
    2. g²(0) per channel (clipped to g2_max if exceeded).

    Parameters
    ----------
    vgs       : group velocity array (µm/ps), shape (num_vg,).
    densities : output photon numbers, shape (num_vg, num_channels).
    g2s       : g²(0) values, shape (num_vg, num_channels).
    g2_max    : upper y-limit for the g² panel to suppress extreme outliers.
    suptitle  : figure title string.
    figname   : output file path (PNG).
    """
    num_channels = densities.shape[-1]
    labels = [f"channel {l}" for l in range(num_channels)]

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle(suptitle, fontsize=10, y=0.98)

    # Panel 1: photon densities
    ax[0].plot(vgs, densities)
    ax[0].set_ylabel("n")
    ax[0].set_title("Output densities")
    ax[0].legend(labels)

    # Panel 2: g²(0)
    ax[1].plot(vgs, g2s)
    ax[1].axhline(y=1.0, color='k', alpha=0.7, linestyle="--")
    ax[1].set_xlabel(r"$v_g$ ($\mu$m/ps)")
    ax[1].set_ylabel(r"$g^{(2)}(0)$")
    if np.max(g2s) > g2_max:
        ax[1].set_ylim(bottom=np.min(g2s), top=g2_max)
    ax[1].set_title("Density-density correlations")

    fig.tight_layout()
    fig.savefig(figname)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    device = "cuda:0"
    vgs    = np.linspace(1.0, 25.0, 50)   # group velocity scan (µm/ps)

    # --- Circuit dimensions ---
    num_channels = 6
    Nmax         = 20
    num_layers   = 5

    # --- Physical parameters (CLI-overridable) ---
    hg_ex            = 20         # bare exciton interaction (µeV·µm²)
    coupling_gauge   = np.pi / 6  # reference coupling angle (J·dt = constant)
    layer_length     = 200        # waveguide layer length (µm)
    pulse_duration   = 1          # pulse duration (ps)
    pulse_transverse = 0.5        # transverse mode size (µm)
    gamma            = 0.02       # photon loss rate (ps⁻¹)

    # --- Initial state ---
    n0, nL      = 1., 1.
    rel_phi     = np.pi / 3   # relative phase between left/right input pulses
    is_coherent = True
    n_eps_g2    = 1e-10

    writedir = os.path.join(FILE_DIR, "data", TIMESTAMP)

    # Override with CLI arguments if present
    hg_ex, gamma, rel_phi, writedir, device = CLI_overwrite(
        hg_ex, gamma, rel_phi, writedir, device
    )

    # --- Build initial MPDO state ---
    alphas = [0.0] * num_channels
    alphas[0], alphas[-1] = np.sqrt(n0), np.sqrt(nL) * np.exp(1j * rel_phi)

    state_creator = StateCreator(Nmax, num_batch=1)
    list_ten      = state_creator.product_state_coherent(alphas, device=device)
    rho           = MPDOtorch(list_ten)

    # --- Result arrays ---
    num_it         = vgs.size
    ns_track       = np.zeros((num_it, num_channels))
    g2s_track      = np.zeros((num_it, num_channels))
    U_track        = np.zeros(num_it)
    coupling_track = np.zeros(num_it)

    os.makedirs(writedir, exist_ok=True)

    # --- vg-scan loop ---
    for i, vg in enumerate(vgs):

        dt = layer_length / vg   # layer propagation time (ps)
        U  = pulse_nonlinear_rate(
            vg=vg,
            pulse_width=pulse_duration,
            g=hg_ex / pulse_transverse / hbar_ueV_ps,   # no exciton-fraction scaling
        )

        # Build circuit for this vg value (J·dt = coupling_gauge kept fixed)
        circuit = create_nonlinear_photonic_circuit(
            num_layers   = num_layers,
            U            = U,
            J            = coupling_gauge / dt,
            gamma        = gamma,
            order_kraus  = 1,
            dt           = dt,
            Nmax         = Nmax,
            num_channels = num_channels,
            requires_grad= False,
            device       = device,
        )
        options = {'max_BD': 100, 'max_PD': 100, 'cutoff_BD': 1e-9, 'cutoff_PD': 1e-9}

        # Run circuit on a fresh clone of the input state
        start  = time.time()
        rho_in = rho.clone()
        circuit.run(rho_in, options=options)

        # Collect output observables
        ns  = rho_in.number_expectations()
        g2s = rho_in.density_correlations(n_eps_g2)
        print(
            rf"vg[{i:02d}]={vg:.2f} µm/ps | {time.time()-start:.2f}s | "
            rf"t={num_layers * dt:.2f} ps | γ={gamma:.2f} ps⁻¹ | "
            rf"N_tot={ns.sum().cpu().numpy():.3f}"
        )
        print(f"  BD={rho_in.get_BDs()}, PD={rho_in.get_PDs()}\n")

        ns_track[i, :]  = ns.cpu().numpy()
        g2s_track[i, :] = g2s.cpu().numpy()
        U_track[i]       = U
        coupling_track[i] = coupling_gauge   # J·dt is constant by construction

    # --- Plot and save ---
    suptitle = (
        rf"Results: {num_channels} WGs, {num_layers} layers of "
        rf"${layer_length:.0f}\,\mu$m, "
        rf"$\hbar g={hg_ex:.1f}\,\mu$eV$\cdot\mu$m$^2$, "
        rf"$\theta_{{\rm rel}}={rel_phi/np.pi:.2f}\pi$"
    )

    plot_data(
        vgs,
        ns_track,
        g2s_track,
        suptitle= suptitle,
        g2_max  = 2.0,
        figname = os.path.join(writedir, "results.png"),
    )

    # Save all results for later analysis
    d_results = {
        'vgs':        vgs,
        'g2s':        g2s_track,
        'ns':         ns_track,
        'Uk':         U_track,
        'coupling_k': coupling_track,
        'parameters': {
            'num_channels': num_channels,
            'num_layers':   num_layers,
            'Nmax':         Nmax,
            'hg_ex':        hg_ex,
            'gamma':        gamma,
            'alphas':       alphas,
        },
    }

    file_pkl = os.path.join(writedir, 'results.pkl')
    with open(file_pkl, 'wb') as fp:
        pickle.dump(d_results, fp)
        print(f"Results saved to {file_pkl}\n")