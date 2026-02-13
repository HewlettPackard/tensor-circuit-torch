import os
import pickle
import yaml
import numpy as np
import time
from scipy.linalg import sqrtm
from datetime import datetime
from typing import Dict, List, Any
import torch
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True

from src.mpdo_circuit import MPDOCircuit
from src.mpdo_optimizer import epoch_optimize
from src.mpdo_torch import StateCreator, MPDOtorch
from src.utils import BosonOperatorsTorch, eye_like
from src.create_circuit import create_nonlinear_photonic_circuit, create_phase_circuit
from src.tracker import Tracker

import numpy as np

from src.svd_trunc import svd_trunc
from src.utils import irescale, iregroup, sqrtm
from src.CLI_utils import get_CLI_input

# script path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# timestamp for data storage
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# directory to store data
SAVE_DIR = os.path.join(FILE_DIR,"data", TIMESTAMP)


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

def iter_func(
    rho: MPDOtorch, 
    circuit: MPDOCircuit, 
    options_MPDO: Dict[str, Any],
    tracker: Tracker, 
    it: int,
    obj_params: List[Any],
    weight_intensity: float,
    do_tracking: bool=True
    ):

    # update circuit
    circuit.update(J_matrix=obj_params[0], U=obj_params[1])

    # run circuit
    rho_run = rho.clone()
    circuit.run(rho_run, options=options_MPDO)

    # compute g2
    idx_signal = rho_run.num_channels // 2
    ns = rho_run.number_outcomes()
    g2s = rho_run.density_correlations(n_eps=1e-5)

    # store outcomes
    for il, (n, g2) in enumerate(zip(ns, g2s)):
        tracker.add(f'n_{il}', n)
        tracker.add(f'g2_{il}', g2)

    #weight = weight_intensity * (1. - g2s[idx_signal])
    weight = weight_intensity
    FOM = (1.-weight) * g2s[idx_signal] + weight * torch.abs(.1 - ns[idx_signal]) ** 2
    tracker.add('FOM', FOM)

    # visuals
    tracker.visualize(
        ['FOM', f'n_{idx_signal}', f'g2_{idx_signal}'], 
        filename=os.path.join(SAVE_DIR, 'FOM.png'))
    
    tracker.visualize(
        [f'n_{il}' for il in range(rho.num_channels)], 
        filename=os.path.join(SAVE_DIR, 'ns.png'))
    
    tracker.visualize(
        [f'g2_{il}' for il in range(rho.num_channels)], 
        filename=os.path.join(SAVE_DIR, 'g2s.png'))
    
    # store data until last run
    tracker.save(os.path.join(SAVE_DIR, 'data.pkl'))

    print(f"g2: {g2s[idx_signal]:.4e}, n: {ns[idx_signal]:.4e}")

    return FOM



if __name__ == "__main__":

    device = "cuda:0"
    k = 0.3 # um^-1
    
    num_channels = 6
    Nmax = 20
    hg_ex = 20 # ueV * um^2: CLI overwritten
    coupling_gauge = np.pi / 4 # the reference tunneling rate (for k[0], photonic regime)
    layer_length = 200 # um
    num_layers = 5
    pulse_duration = 1 # ps
    pulse_transverse = 0.5 # um
    gamma = 0.0 # loss rate (1/ps): CLI overwritten

    # initial state
    n0, nL = 1., 1.
    rel_phi = np.pi / 4 # relative phase shift pulses: CLI overwritten
    n_Fock = [1,1,1,1,1] # for testing with single photon input
    is_coherent = True
    n_eps_g2 = 1e-10
    g2_max = None

    # options
    options_ADAM = {
        'lr': 5e-3, 'lr_min': 1e-4, 'max_epochs': 100, 'param_lims': [0, np.pi], 'weight_intensity': .1
        }
    options_MPDO = {'max_BD': 100, 'max_PD': 100, 'cutoff_BD': 1e-9, 'cutoff_PD': 1e-9}

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


    # create write dir
    if not os.path.exists(writedir):
        os.makedirs(writedir)

    
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
        J =  coupling_gauge / dt,
        gamma = gamma,
        order_krauss = 1,
        dt = dt,
        Nmax = Nmax,
        num_channels = num_channels,
        requires_grad = True
    )


    #######################################
    #       PREPARE
    #######################################


    # create circuits
    circuit = create_nonlinear_photonic_circuit(**d, device=device)


    # get the circuit variables
    params_init = circuit.get_variables()

    # run the init circuit
    start = time.time()
    rho_in = rho.clone()

    # objective
    tracker = Tracker() 
    weight_intensity = options_ADAM.pop('weight_intensity') 
    objective = lambda it, rho, obj_params, do_tracking: iter_func(
        rho, 
        circuit, options_MPDO=options_MPDO,
        tracker=tracker, it=it,
        obj_params=obj_params,
        weight_intensity=weight_intensity,
        do_tracking=do_tracking
        )
    
    # the optimizer (ADAM)
    max_epochs = options_ADAM.pop('max_epochs')
    param_lims = options_ADAM.pop('param_lims')
    lr_min = options_ADAM.pop('lr_min')
    optimizer = torch.optim.Adam(
            circuit.get_variables(), **options_ADAM
        )
    if lr_min is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr_min)
    else:
        scheduler = None
    
    # train the circuit for QFI
    obj_params = [circuit.get_coupling_matrix(float_vals=False), U]
    epoch_optimize(
        objective, 
        optimizer=optimizer, 
        rho=rho, 
        max_epochs=max_epochs,
        obj_params=obj_params, 
        param_lims=param_lims,
        scheduler=scheduler,
        epoch_optim_clear=None
    )
