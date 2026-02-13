import os
import pickle
import yaml
import numpy as np
import time
from scipy.linalg import sqrtm
from datetime import datetime
import torch
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True

from src.mpdo_circuit import MPDOCircuit
from src.mpdo_torch import StateCreator, MPDOtorch
from src.utils import BosonOperatorsTorch, eye_like
from src.create_circuit import create_nonlinear_photonic_circuit, create_phase_circuit

import numpy as np

from src.svd_trunc import svd_trunc
from src.utils import irescale, iregroup, sqrtm
from src.CLI_utils import get_CLI_input

# script path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# timestamp for data storage
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


Mc = lambda k: 1.6114 + 0.037 * k  # (photonic mode approximated as a straight line)
Exca = 1.6229  # (exciton energy)
OM = 3.4 * 1e-3 # (half the Rabi splitting)
hbar_eV_ps = 6.582 * 1e-4 # hbar, converting energy to frequency
hbar_ueV_ps = 6.582 * 1e2 # hbar, converting energy to frequency

def omega_LP(k: float):

    """
    Lower-polariton dispersion in eV and um!

    """

    LP_disp =  0.5 * (Mc(k) + Exca) - 0.5 * ((Mc(k) - Exca)**2 + 4. * OM ** 2) ** 0.5
    return LP_disp


def vg_LP(k: float, dk: float=1e-3):
    """
    Lower-polariton group velocity, in eV * um / hbar = um / ps
    """
    return (omega_LP(k + dk/2.) - omega_LP(k - dk/2.)) / dk / hbar_eV_ps


def exciton_fraction(k):
    """ 
    The exciton fraction at given k (um^(-1))
    """
    theta_k = np.arctan(OM / (omega_LP(k) - Mc(k)))
    return np.cos(theta_k) ** 2

def pulse_nonlinear_rate(vg: float, pulse_width: float, g: float):

    """
    Returns the pulse nonlinear rate in 1/ps from
    vg: group velocity (um/ps)
    pulse_width: ps
    g: nonlinear interaction rate (1/ps)
    """

    # the spatial width in WG
    spatial_width = pulse_width * vg

    # compute inverse volume element from Gaussian distribution
    x = np.linspace(-5 * spatial_width, 5 * spatial_width, 5001)
    dx = x[1] - x[0]
    profile = 1. / np.sqrt(2. * np.pi * spatial_width ** 2) \
                * np.exp( -x ** 2 / 2. / spatial_width ** 2 )
    inv_volume = np.sum(profile ** 2) * dx

    # the effective nonlinear rate
    U = g * inv_volume

    return U



def plot_data(ks, densities, g2s, Us, wk, ex_frac, coupling, g2_max=2., suptitle="results", figname='experiments/g2_experiment/test.png'):
    """
    Make a 3-panel plot:
    Panel 1: ns_track (each row is a line)
    Panel 2: g2s_track (each row is a line)
    Panel 3: wk_track (left axis) and ex_frac_track (right axis)
    x-axis for all panels is array ks
    """

    num_k = densities.shape[0]
    num_channels = densities.shape[-1]
    labels = [f"channel {l}" for l in range(num_channels)]

    fig, ax = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(suptitle, fontsize=10, y=0.98)

    # --- Panel 1: ns_track ---
    ax[0].plot(ks, densities)
    ax[0].set_xlabel(r"k ($\mu m ^{-1}$)")
    ax[0].set_ylabel("n")
    ax[0].set_title("output densities")
    ax[0].legend(labels)

    # --- Panel 2: g2s_track ---
    ax[1].plot(ks, g2s)
    ax[1].axhline(y=1., color='k', alpha=0.7, linestyle="--")
    ax[1].set_xlabel(r"k ($\mu m ^{-1}$)")
    ax[1].set_ylabel(r"$g^{(2)}(0)$")
    if np.max(g2s) > g2_max:
        ax[1].set_ylim(bottom=np.min(g2s), top=g2_max)
    ax[1].set_title("density-density correlations")
    # ax[1].legend(labels)

    # --- Panel 3: wk & exciton fraction ---
    ax3 = ax[2]
    ax3.plot(ks, Us, color='b', label="$U_k$", linewidth=2)
    ax3.set_ylabel(r"$U_k\, (ps^{-1})$")
    

    ax3b = ax3.twinx()
    ax3b.plot(ks, ex_frac, color='r', label="exciton fraction", linestyle="--")
    ax3b.set_ylabel("exciton fraction")

    ax3.set_title("Gate nonlinearity U and exciton fraction")
    ax[2].set_xlabel(r"k ($\mu m ^{-1}$)")

    # combine legends from both axes
    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_2, labels_2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_2, labels_1 + labels_2)

    # --- Panel 4: g2s_track ---
    ax[3].plot(ks, coupling)
    ax[3].set_xlabel(r"k ($\mu m ^{-1}$)")
    ax[3].set_ylabel(r"$J\cdot \Delta t$")
    ax[3].set_title("Gate coupling")

    fig.tight_layout()
    fig.savefig(figname)



def CLI_overwrite(hg, gamma, phi, writedir, device):

    """
    Overwrite the relevant parameters with CLI input
    """

    hg = get_CLI_input(
        '--hg', type=float, default=hg
        )

    gamma = get_CLI_input(
        '--gamma', type=float, default=gamma
        )
    
    phi = get_CLI_input(
        '--phi', type=float, default=phi
        )
    
    writedir = get_CLI_input(
        '--writedir', type=str, default=writedir
        )
    
    device = get_CLI_input(
        '--device', type=str, default=device
        )
    
    return hg, gamma, phi, writedir, device




if __name__ == "__main__":

    device = "cuda:0"
    ks = np.linspace(0., 0.3, 51) # k = [0,1], in um^-1
    vg_gauge = vg_LP(ks[0]) # gauge gate size, time scale and coupling with the first (photonic) k-element

    num_channels = 6
    Nmax = 20
    hg_ex = 20 # ueV * um^2: CLI overwritten
    coupling_gauge = np.pi / 6 # the reference tunneling rate (for k[0], photonic regime)
    layer_length = 200 # um
    num_layers = 5
    pulse_duration = 1 # ps
    pulse_transverse = 0.5 # um
    gamma = 0.02 # loss rate (1/ps): CLI overwritten

    # initial state
    n0, nL = 1., 1.
    rel_phi = np.pi / 3 # relative phase shift pulses: CLI overwritten
    n_Fock = [1,1,1,1,1] # for testing with single photon input
    is_coherent = True
    n_eps_g2 = 1e-10
    g2_max = None

    # to store results
    writedir = os.path.join(FILE_DIR, "data", TIMESTAMP)

    # overwrite by CLI
    hg_ex, gamma, rel_phi, writedir, device = CLI_overwrite(hg_ex, gamma, rel_phi, writedir, device)



    # PREPARE INITIAL STATE
    dict_rhos = {}
    alphas = [0.] * num_channels
    alphas[0], alphas[-1] = np.sqrt(n0), np.sqrt(nL) * np.exp(1j * rel_phi)

    # initialize input state
    state_creator = StateCreator(Nmax, num_batch=1)

    # initialise MPDO from tensors
    if is_coherent:
        list_ten = state_creator.product_state_coherent(alphas, device=device)
    else:
        list_ten = state_creator.product_state_fock(n_Fock, device=device)
    rho = MPDOtorch(list_ten)


    # data to track
    num_it = ks.size
    ns_track = np.zeros((num_it, num_channels))
    g2s_track = np.zeros((num_it, num_channels))
    wk_track = np.zeros((num_it))
    ex_frac_track = np.zeros((num_it))
    U_track = np.zeros((num_it))
    coupling_track = np.zeros((num_it))

    # gauging time scale
    dt_gauge = layer_length / vg_gauge

    # create write dir
    if not os.path.exists(writedir):
        os.makedirs(writedir)


    # loop over k-values
    for i, k in enumerate(ks):
    
        # get k-dependent values
        wk = omega_LP(k) # omega (eV)
        vg = vg_LP(k) # group velocity um / ps
        ex_frac = exciton_fraction(k) # exciton hopfield coeff |uk|^2, dimensionless
        dt = layer_length / vg # the duration of one layer
        U = pulse_nonlinear_rate(
            vg=vg, 
            pulse_width=pulse_duration, 
            g=ex_frac ** 2 * (hg_ex/pulse_transverse/hbar_ueV_ps)) # the pulse nonlinear rate


        # circuit
        d = dict(
            num_layers = num_layers,
            U = U, 
            J =  coupling_gauge / dt_gauge,
            gamma = gamma,
            order_krauss = 1,
            dt = dt,
            Nmax = Nmax,
            num_channels = num_channels,
            requires_grad = False
        )

        # options
        options = {'max_BD': 100, 'max_PD': 100, 'cutoff_BD': 1e-9, 'cutoff_PD': 1e-9}


        #######################################
        #       PREPARE
        #######################################


        # create circuits
        circuit = create_nonlinear_photonic_circuit(**d, device=device)


        # extract the circuit variables
        params_init = circuit.get_variables()

        # run the init circuit
        start = time.time()
        rho_in = rho.clone()
        circuit.run(rho_in, options=options)

        # print output
        ns = rho_in.number_outcomes()
        vars = rho_in.number_variances()
        g2s = rho_in.density_correlations(n_eps_g2)
        print(rf"iteration {i}: Finished run in {time.time()-start:.2f}s, circuit time={num_layers * dt:.2f}ps, gamma={gamma:.2f}ps^-1, vg={vg:.2f}um/ps, N_tot={ns.sum().cpu().numpy()}")
        print(f"BD={rho_in.get_BDs()}, PD={rho_in.get_PDs()}\n")

        # store data
        ns_track[i,:] = ns.cpu().numpy()
        g2s_track[i,:] = g2s.cpu().numpy()
        wk_track[i] = wk
        ex_frac_track[i] = ex_frac
        U_track[i] = U
        coupling_track[i] = coupling_gauge / dt_gauge * dt

    suptitle = (
        rf"Results: {num_channels} WGs, {num_layers} layers of "
        rf"${layer_length:.2f}\,\mu$m, "
        rf"$\hbar g={hg_ex:.2f}\,\mu$eV$\cdot\mu$m$^2$, "
        rf"$\theta_{{rel}}={rel_phi/np.pi:.2f}\pi$"
    )
 

    plot_data(
        ks,
        ns_track, 
        g2s_track, 
        U_track,
        wk = omega_LP(ks), 
        ex_frac = exciton_fraction(ks), 
        coupling = coupling_track,
        suptitle=suptitle,
        g2_max=2,
        figname=os.path.join(writedir, "results.png")
        )
    
    # save results as pkl
    d_results = {}
    d_results['ks'] = ks
    d_results['g2s'] = g2s_track
    d_results['ns'] = ns_track
    d_results['Uk'] = U_track
    d_results['coupling_k'] = coupling_track
    d_results['ex_frac'] = exciton_fraction(ks)
    d_results['wk'] = omega_LP(ks)
    d_results['vgk'] = vg_LP(ks)
    # d_results['BDs'] = rho_in.get_BDs()
    # d_results['PDs'] = rho_in.get_PDs()
    # d_results['SvN'] = rho_in.entropy_profile(entropy='entanglement')
    # d_results['SP'] = rho_in.entropy_profile(entropy='purity')
    d_results['parameters'] = {
        'num_channels': num_channels,
        'num_layers': num_layers,
        'Nmax': Nmax,
        'hg_ex': hg_ex, 
        'gamma': gamma,
        'alphas': alphas
    }


    file_pkl = os.path.join(writedir, f'results.pkl')
    with open(file_pkl, 'wb') as fp:
        pickle.dump(d_results, fp)
        print(f'Results saved successfully in {file_pkl}\n')