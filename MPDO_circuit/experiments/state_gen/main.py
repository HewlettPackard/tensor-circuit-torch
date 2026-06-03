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
from src.create_circuit import create_nonlinear_photonic_circuit, create_phase_circuit
from src.tracker import Tracker
from src.visualize import visualize_circuit

import numpy as np

from src.CLI_utils import get_CLI_input

# script path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# timestamp for data storage
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# directory to store data
SAVE_DIR = os.path.join(FILE_DIR,"data", TIMESTAMP)



def create_target_SP(device):

    """Generate the single-photon DM for state generation"""

    state_creator = StateCreator(Nmax, num_batch=None)
    psi_target = state_creator.fock(1, device=device, direct=True)
    rho_target = torch.outer(psi_target.conj(), psi_target)

    return rho_target
    

def create_target_cat(alpha=1., phase=np.pi, device='cuda:0'):

    """Generate the cat DM for state generation"""
    
    state_creator = StateCreator(Nmax, num_batch=None)
    
    psi_target = state_creator.single_mode_cat(
            alpha, 
            phase=phase, 
            device=device,
            direct=True
            )
    rho_target = torch.outer(psi_target.conj(), psi_target)
    
    return rho_target


def iter_func(
    rho: MPDOtorch, 
    circuit: CouplerCircuit, 
    it: int,
    rho_target: torch.Tensor,
    options_MPDO: Dict[str, Any],
    tracker: Tracker, 
    obj_params: List[Any],
    mask: torch.Tensor|float = 1.,
    entropy_weight: float = 0.
    ):

    # update circuit couplings after previous run
    circuit.update(J_matrix=obj_params[0])

    # run circuit
    rho_run = rho.clone()
    circuit.run(rho_run, options=options_MPDO)

    ##################################
    # Compute objective
    ##################################

    # compute trace distance
    rho_try = rho_run.ptrace(rho_run.num_channels//2)
    trace_dist = torch.linalg.eigvals(
        (rho_try - rho_target) * mask
    ).abs().sum()

    # compute entanglement penalty (mean square entropy profile)
    entropy_penalty = rho_run.entropy_profile().pow(2).sum()

    FOM = trace_dist + entropy_weight * entropy_penalty / (1. + entropy_weight)

    # compute fidelity (with numpy, for sqrtm)
    rho_try_np, rho_target_np = rho_try.detach().cpu().numpy(), rho_target.detach().cpu().numpy()
    fidelity = np.trace(sqrtm(sqrtm(rho_try_np) @ rho_target_np @ sqrtm(rho_try_np))).real ** 2
    rho_cond = rho_try_np
    rho_cond[0,0] = 0.
    rho_cond = rho_cond / np.trace(rho_cond)
    fidelity_cond = np.trace(sqrtm(sqrtm(rho_cond) @ rho_target_np @ sqrtm(rho_cond))).real ** 2

    # store outcomes
    tracker.add('FOM', FOM)
    tracker.add('trace_dist', trace_dist)
    tracker.add('entropy_penalty', entropy_penalty)
    tracker.add('fidelity', fidelity)
    tracker.add('fidelity_cond', fidelity_cond)


    # visuals
    tracker.visualize(
        ['FOM'], 
        filename=os.path.join(SAVE_DIR, 'FOM.png'))
    
    tracker.visualize(
        ['fidelity', 'fidelity_cond'], 
        filename=os.path.join(SAVE_DIR, 'fidelity.png'))
    
    
    # visualization of circuit
    
    visualize_circuit(
        circuit.get_coupling_matrix(), 
        filename=os.path.join(SAVE_DIR, "circuit")
        )
    
    # store data until last run
    tracker.save(os.path.join(SAVE_DIR, 'data.pkl'))

    print(
        f"FOM: {FOM:.2f}, trace dist.: {trace_dist:.2f}, entropy penalty: {entropy_penalty:.2f}, fidelity: {fidelity:.2f}, cond. fid.: {fidelity_cond:.2f}")

    return FOM



if __name__ == "__main__":

    device = "cuda:5"
    
    Nmax = 10 # Fock dim
    num_channels = 11 # num channels
    num_layers = 25 # num circuit layers
    coupling_init = 0.05 # initial value for starting the convergence
    U = 0.25 # gate nonlinearity
    gamma_dt = 0.0 # loss rate (percentage): CLI overwritten
    target = "cat"

    # initial state
    alpha = 1.

    weight_zero = .1
    entropy_weight = 0.005

    # options
    options_ADAM = {
        'lr': 1e-1, 'max_epochs': 100, 'param_lims': [0, np.pi]
        }
    options_MPDO = {'max_BD': 500, 'max_PD': 100, 'cutoff_BD': 1e-6, 'cutoff_PD': 1e-5}

    # to store results
    writedir = os.path.join(FILE_DIR, "data", TIMESTAMP)



    # PREPARE INITIAL STATE

    alphas = [torch.Tensor([alpha])] * num_channels
    state_creator = StateCreator(Nmax, num_batch=1)
    list_ten = [ten for ten in state_creator.product_state_coherent(alphas, device=device)]
    rho = MPDOtorch(list_ten)

    # create write dir
    if not os.path.exists(writedir):
        os.makedirs(writedir)

    #######################################
    #       PREPARE CIRCUIT
    #######################################

    # circuit
    d = dict(
        num_layers = num_layers,
        U = U, 
        J =  coupling_init,
        gamma = gamma_dt,
        order_kraus = 2,
        dt = 1.,
        Nmax = Nmax,
        num_channels = num_channels,
        requires_grad = True
    )

    # create circuits
    circuit = create_nonlinear_photonic_circuit(**d, device=device)
    J_matrix =  circuit.get_J_matrix(float_vals=False)

    ##################################
    #      The optimization run
    ##################################


    # run the init circuit
    start = time.time()
    rho_in = rho.clone()

    # set up iter func and objective
    rho_target = create_target_cat(device=device)

    # initiate result tracker
    tracker = Tracker() 

    # the mask for optimization
    mask = torch.ones((Nmax + 1, Nmax + 1), device=device)
    d = range(Nmax + 1)
    mask[d,d] = 1. * torch.ones(Nmax + 1, device=device)
    mask[0,0] = weight_zero
    mask[1,1] = 1
    mask = mask / mask.norm()

    objective = lambda it, rho, obj_params, do_tracking: iter_func(
        rho, 
        circuit,  rho_target=rho_target,
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
    
    # train the circuit
    obj_params = [J_matrix]
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
