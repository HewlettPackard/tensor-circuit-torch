"""
g2_simulator.py
---------------
Scans g²(0) and photon number across a range of in-plane wavevectors k for
a fixed nonlinear photonic circuit topology.
 
For each k value the script:
  1. Derives k-dependent physical parameters (group velocity, exciton
     fraction, effective nonlinearity U).
  2. Builds and runs a nonlinear photonic MPDO circuit.
  3. Collects output photon numbers and g²(0) values.
  4. Saves a 4-panel summary plot and a .pkl results file.
 
Polariton helpers (dispersion, group velocity, exciton fraction) are
imported from LP_utils.
 
Usage
-----
    python -m experiments.g2_experiment.g2_simulator [--hg HG] [--gamma GAMMA]
                                                      [--phi PHI] [--writedir DIR]
                                                      [--device DEVICE]
"""
 
import os
import pickle
import time
from datetime import datetime
 
import numpy as np
import matplotlib.pyplot as plt
import torch
 
from src.mpdo_torch import MPDOtorch, StateCreator
from src.create_circuit import create_nonlinear_photonic_circuit
from src.CLI_utils import get_CLI_input
from experiments.LP_utils import (
    hbar_omega_LP as omega_LP,
    vg_LP,
    exciton_fraction,
)
 
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
    ks:        np.ndarray,
    densities: np.ndarray,
    g2s:       np.ndarray,
    Us:        np.ndarray,
    wk:        np.ndarray,
    ex_frac:   np.ndarray,
    coupling:  np.ndarray,
    g2_max:    float = 2.0,
    suptitle:  str   = "results",
    figname:   str   = "results.png",
) -> None:
    """
    Save a 4-panel summary figure as a function of wavevector k.
 
    Panels
    ------
    1. Output photon densities per channel.
    2. g²(0) per channel (clipped to g2_max if exceeded).
    3. Effective nonlinearity U_k (left axis) and exciton fraction (right axis).
    4. Dimensionless gate coupling J·Δt as a function of k.
 
    Parameters
    ----------
    ks        : wavevector array (µm⁻¹), shape (num_k,).
    densities : output photon numbers, shape (num_k, num_channels).
    g2s       : g²(0) values, shape (num_k, num_channels).
    Us        : effective nonlinearity U_k (ps⁻¹), shape (num_k,).
    wk        : LP energy (eV), shape (num_k,).  Used for legend only.
    ex_frac   : excitonic Hopfield coefficient |u_k|², shape (num_k,).
    coupling  : J·Δt gate coupling, shape (num_k,).
    g2_max    : upper y-limit for the g² panel to suppress extreme outliers.
    suptitle  : figure title string.
    figname   : output file path (PNG).
    """
    num_channels = densities.shape[-1]
    labels = [f"channel {l}" for l in range(num_channels)]
 
    fig, ax = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(suptitle, fontsize=10, y=0.98)
 
    # Panel 1: photon densities
    ax[0].plot(ks, densities)
    ax[0].set_ylabel("n")
    ax[0].set_title("Output densities")
    ax[0].legend(labels)
 
    # Panel 2: g²(0)
    ax[1].plot(ks, g2s)
    ax[1].axhline(y=1.0, color='k', alpha=0.7, linestyle="--")
    ax[1].set_ylabel(r"$g^{(2)}(0)$")
    if np.max(g2s) > g2_max:
        ax[1].set_ylim(bottom=np.min(g2s), top=g2_max)
    ax[1].set_title("Density-density correlations")
 
    # Panel 3: nonlinearity U and exciton fraction (twin y-axes)
    ax3  = ax[2]
    ax3b = ax3.twinx()
    ax3.plot( ks, Us,      color='b', label=r"$U_k$",           linewidth=2)
    ax3b.plot(ks, ex_frac, color='r', label="exciton fraction", linestyle="--")
    ax3.set_ylabel(r"$U_k\,(ps^{-1})$")
    ax3b.set_ylabel("exciton fraction")
    ax3.set_title("Gate nonlinearity and exciton fraction")
    lines1, labs1 = ax3.get_legend_handles_labels()
    lines2, labs2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labs1 + labs2)
 
    # Panel 4: gate coupling
    ax[3].plot(ks, coupling)
    ax[3].set_xlabel(r"k ($\mu$m$^{-1}$)")
    ax[3].set_ylabel(r"$J\cdot\Delta t$")
    ax[3].set_title("Gate coupling")
 
    fig.tight_layout()
    fig.savefig(figname)
    plt.close(fig)
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
 
    device = "cuda:0" # CLI overridable
    ks     = np.linspace(0.0, 0.3, 51)   # wavevector scan range (µm⁻¹)
 
    # Gauge time scale from the first (most photonic) k-point
    vg_gauge = vg_LP(ks[0])
 
    # --- Circuit dimensions ---
    num_channels = 6
    Nmax         = 20
    num_layers   = 5
 
    # --- Physical parameters (CLI-overridable) ---
    hg_ex            = 50         # bare exciton interaction (µeV·µm²)
    coupling_gauge   = np.pi / 6  # reference coupling angle at k[0]
    layer_length     = 200        # waveguide layer length (µm)
    pulse_duration   = 1          # pulse duration (ps)
    pulse_transverse = 0.5        # transverse mode size (µm)
    gamma            = 0.0        # photon loss rate (ps⁻¹)
 
    # --- Initial state ---
    nL, nR      = 1., 1.
    rel_phi     = 0.2 * np.pi   # relative phase between left/right input pulses
    is_coherent = True
    placement   = 'sides'       # 'sides' or 'middle'
    n_eps_g2    = 1e-10
 
    writedir = os.path.join(FILE_DIR, "data", TIMESTAMP)
 
    # Override with CLI arguments if present
    hg_ex, gamma, rel_phi, writedir, device = CLI_overwrite(
        hg_ex, gamma, rel_phi, writedir, device
    )
 
    # --- Build initial MPDO state ---
    alphas = [0.0] * num_channels
    if placement == 'sides':
        alphas[0], alphas[-1] = np.sqrt(nL), np.sqrt(nR) * np.exp(1j * rel_phi)
    elif placement == 'middle':
        alphas[num_channels // 2 - 1] = np.sqrt(nL)
        alphas[num_channels // 2]     = np.sqrt(nR) * np.exp(1j * rel_phi)
 
    state_creator = StateCreator(Nmax, num_batch=1)
    list_ten      = state_creator.product_state_coherent(alphas, device=device)
    rho           = MPDOtorch(list_ten)
 
    # --- Result arrays ---
    num_it         = ks.size
    ns_track       = np.zeros((num_it, num_channels))
    g2s_track      = np.zeros((num_it, num_channels))
    U_track        = np.zeros(num_it)
    coupling_track = np.zeros(num_it)
 
    # Gate time scale (gauged from the first k-point)
    dt_gauge = layer_length / vg_gauge
 
    os.makedirs(writedir, exist_ok=True)
 
    # --- k-scan loop ---
    for i, k in enumerate(ks):
 
        # k-dependent physical quantities
        vg      = vg_LP(k)             # group velocity (µm/ps)
        ex_frac = exciton_fraction(k)  # excitonic Hopfield coefficient |u_k|²
        dt      = layer_length / vg    # layer propagation time (ps)
        U       = pulse_nonlinear_rate(
            vg=vg,
            pulse_width=pulse_duration,
            g=ex_frac ** 2 * (hg_ex / pulse_transverse / hbar_ueV_ps),
        ) # nonlinear coupling, evaluated from dispersion
 
        # Build circuit for this k value
        circuit = create_nonlinear_photonic_circuit(
            num_layers   = num_layers,
            U            = U,
            J            = coupling_gauge / dt_gauge,
            gamma        = gamma,
            order_kraus  = 2,
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
        ns  = rho_in.number_outcomes()
        g2s = rho_in.density_correlations(n_eps_g2)
        print(
            rf"k[{i:02d}]={k:.3f} µm⁻¹ | {time.time()-start:.2f}s | "
            rf"t={num_layers * dt:.2f} ps | γ={gamma:.2f} ps⁻¹ | "
            rf"vg={vg:.2f} µm/ps | N_tot={ns.sum().cpu().numpy():.3f}"
        )
        print(f"  BD={rho_in.get_BDs()}, PD={rho_in.get_PDs()}\n")
 
        ns_track[i, :]  = ns.cpu().numpy()
        g2s_track[i, :] = g2s.cpu().numpy()
        U_track[i]       = U
        coupling_track[i] = coupling_gauge / dt_gauge * dt
 
    # --- Plot and save ---
    suptitle = (
        rf"Results: {num_channels} WGs, {num_layers} layers of "
        rf"${layer_length:.0f}\,\mu$m, "
        rf"$\hbar g={hg_ex:.1f}\,\mu$eV$\cdot\mu$m$^2$, "
        rf"$\theta_{{\rm rel}}={rel_phi/np.pi:.2f}\pi$"
    )
 
    plot_data(
        ks,
        ns_track,
        g2s_track,
        U_track,
        wk      = omega_LP(ks),
        ex_frac = exciton_fraction(ks),
        coupling= coupling_track,
        suptitle= suptitle,
        g2_max  = 2.0,
        figname = os.path.join(writedir, "results.png"),
    )
 
    # Save all results for later analysis
    d_results = {
        'ks':         ks,
        'g2s':        g2s_track,
        'ns':         ns_track,
        'Uk':         U_track,
        'coupling_k': coupling_track,
        'ex_frac':    exciton_fraction(ks),
        'wk':         omega_LP(ks),
        'vgk':        vg_LP(ks),
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
 