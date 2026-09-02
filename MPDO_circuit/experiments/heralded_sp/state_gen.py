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
import torch.nn.functional as F
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True

from src.mpdo_circuit import CouplerCircuit, PhaseCircuit
from src.mpdo_optimizer import epoch_optimize
from src.mpdo_torch import StateCreator, MPDOtorch
from src.circuit_gates import CircuitGateOneSite
from src.create_circuit import create_nonlinear_photonic_circuit, create_phase_circuit, create_nonlinear_mzi_circuit
from src.tracker import Tracker
from src.visualize import visualize_circuit, visualize_MZI_circuit

import numpy as np

from src.CLI_utils import get_CLI_input

# script path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# timestamp for data storage
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# directory to store data
# SAVE_DIR = os.path.join(FILE_DIR,"data", TIMESTAMP)
SAVE_DIR = os.path.join(FILE_DIR,"data", TIMESTAMP)

torch.autograd.set_detect_anomaly(True)

def iter_func(
    rho: MPDOtorch, 
    circuit: CouplerCircuit, 
    it: int,
    condition_number: int,
    options_MPDO: Dict[str, Any],
    tracker: Tracker, 
    obj_params: List[Any],
    entropy_weight: float = 0.,
    ):

    d = rho.Nmax + 1
    L = rho.num_channels

    site_herald = L // 2 - 1
    site_sp = site_herald + 1

    # update circuit couplings after previous run
    circuit.update(J_matrix=obj_params[0], phase_matrix=obj_params[1])

    # run circuit
    rho_run = rho.clone()
    circuit.run(rho_run, options=options_MPDO)

    ##################################
    # Compute objective
    ##################################

    # the entanglement entropy
    entropy_profile = rho_run.entropy_profile()
    entropy_penalty = entropy_profile.pow(2).sum()

    # compute two-site DM
    rho_2site = rho_run.ptrace([site_herald,site_sp])

    # compute prob. of heralding
    rho_herald = torch.einsum("jlii->jl",rho_2site)
    prob_herald = rho_herald[condition_number,condition_number].real

    # compute cond. SP probability
    rho_cond = rho_2site[condition_number, condition_number] / prob_herald
    prob_cond_sp = rho_cond[1,1].real

    FOM = - prob_cond_sp # + entropy_weight * entropy_penalty / (1. + entropy_weight)
    
    # store outcomes
    tracker.add('FOM', FOM)
    tracker.add('prob_cond_sp', prob_cond_sp)
    tracker.add('prob_herald', prob_herald)
    tracker.add('rho_cond', rho_cond)
    tracker.add('rho_2site', rho_2site)


    # visuals
    tracker.visualize(
        ['FOM'], 
        filename=os.path.join(SAVE_DIR, 'FOM.png'),
        title=f"FOM = {tracker['FOM'][-1]:4f}")
    
    tracker.visualize(
        ['prob_cond_sp', 'prob_herald'], 
        filename=os.path.join(SAVE_DIR, 'probs.png'),
        title=f"prob. SP = {tracker['prob_cond_sp'][-1]:2f}, prob. herald = {tracker['prob_herald'][-1]:2f}")
    
    
    # visualization of circuit
    visualize_MZI_circuit(
        circuit.get_coupling_matrix(), 
        circuit.get_phaseshift_matrix(), 
        filename=os.path.join(SAVE_DIR, "circuit")
    )

    # plot cond. number distribution
    fig, ax = plt.subplots(figsize=(4,4))
    ax.bar(range(d), rho_cond.diag().real.detach().cpu().numpy())
    ax.set_yscale('log')
    ax.grid(True)
    ax.set_title("Cond. number distribution")
    ax.set_xlabel('number')
    ax.set_ylabel("cond. probability")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "cond_number_distr"))
    plt.close(fig)

    # plot 2-site DM
    rho_plot = tracker['rho_2site'][-1].transpose(1,3,0,2).reshape(d**2, d**2)
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.matshow(rho_plot.real)
    fig.colorbar(im, ax=ax)

    n = tracker['rho_2site'][-1].shape[0]
    major_ticks = range(0, n + 1, d)

    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)

    ax.set_xticks([t - 0.5 for t in major_ticks], minor=True)
    ax.set_yticks([t - 0.5 for t in major_ticks], minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.75)

    ax.set_title("2-site DM")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "2site_dm"))
    plt.close(fig)

    # return FOM
    return FOM



if __name__ == "__main__":

    device = "cpu"
    
    Nmax = 5 # Fock dim
    num_channels = 4 # num channels
    num_layers = 50 # num circuit layers
    coupling_init = 0.01 # initial value for starting the convergence
    phi_init = np.pi / 2
    U = 0.125 # gate nonlinearity
    gamma_dt = 0.0 # loss rate (percentage): CLI overwritten
    condition_number = 0 # the heralded photon number (or vacuum 0) for heralding single photons

    # initial state
    alpha = 1.
    entropy_weight = 0.00

    # options
    options_ADAM = {
        'lr': 2e-2, 'max_epochs': 500, 'param_lims': (0., np.pi)
        }
    options_MPDO = {'max_BD': 5000, 'max_PD': 100, 'cutoff_BD': 1e-12, 'cutoff_PD': 1e-5}




    # PREPARE INITIAL STATE

    alphas = [torch.Tensor([alpha])] * num_channels
    state_creator = StateCreator(Nmax, num_batch=1)
    list_ten = [ten for ten in state_creator.product_state_coherent(alphas, device=device)]
    rho = MPDOtorch(list_ten)


    #######################################
    #       PREPARE CIRCUIT
    #######################################

    # circuit
    num_layers_total = 2 * num_layers
    J = [[coupling_init if (ic + il) % 2 == 0 else None for ic in range(num_channels)] 
         for il in range(num_layers)]
    phase = [[phi_init for _ in range(num_channels)] for __ in range(num_layers)]

    # Interleave: couplers at even depths, phase shifters at odd depths
    J_full = [None] * num_layers_total
    J_full[0::2] = J

    phase_full = [None] * num_layers_total
    phase_full[1::2] = phase

    d = dict(
        num_layers = num_layers_total,
        U = U, 
        J = J_full,
        phase = phase_full,
        gamma = gamma_dt,
        order_kraus = 2,
        dt = 1.,
        Nmax = Nmax,
        num_channels = num_channels,
        requires_grad_J = True,
        requires_grad_phase = True
    )

    # create circuits
    #vcircuit = create_nonlinear_photonic_circuit(**d, device=device)
    circuit = create_nonlinear_mzi_circuit(**d, device=device)
    J_matrix =  circuit.get_J_matrix(float_vals=False)
    phi_matrix = circuit.get_phase_matrix(float_vals=False)

    ##################################
    #      The optimization run
    ##################################


    # run the init circuit
    start = time.time()
    rho_in = rho.clone()


    # initiate result tracker
    tracker = Tracker() 

    objective = lambda it, rho, obj_params: iter_func(
        rho, 
        circuit,  condition_number=condition_number,
        options_MPDO=options_MPDO,
        tracker=tracker,
        it=it, 
        obj_params=obj_params,
        entropy_weight=entropy_weight
        )
    
    # the optimizer (ADAM)
    max_epochs = options_ADAM.pop('max_epochs')
    param_lims = options_ADAM.pop('param_lims')
  
    J_vars = [J for J_layer in J_matrix for J in J_layer if J is not None]
    optimizer = torch.optim.Adam(
            circuit.get_variables(), **options_ADAM
        )
    # if lr_min is not None:
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr_min)
    # else:
    #     scheduler = None

    # create write dir
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # train the circuit
    obj_params = [J_matrix, phi_matrix]
    epoch_optimize(
        objective, 
        optimizer=optimizer, 
        rho=rho, 
        max_epochs=max_epochs,
        obj_params=obj_params, 
        param_lims=param_lims,
        #scheduler=scheduler,
        epoch_optim_clear=None
    )
