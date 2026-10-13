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

from src.mpdo_circuit import CouplerCircuit, PhaseCircuit, MPDOCircuit
from src.mpdo_optimizer import epoch_optimize
from src.mpdo_torch import StateCreator, MPDOtorch
from src.create_circuit import (
    create_nonlinear_photonic_circuit, create_nonlinear_mzi_circuit, create_phase_circuit
)
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


def iter_func(
    rho: MPDOtorch, 
    circuit: MPDOCircuit,
    options_MPDO: Dict[str, Any],
    tracker: Tracker, 
    it: int,
    obj_params: List[Any],
    weight_intensity: float,
    target_intensity: float=0.01,
    do_tracking: bool=True
    ):

    # update circuit
    circuit.update(J_matrix=obj_params[0], phase_matrix=obj_params[1])

    # run circuit
    rho_run = rho.clone()
    circuit.run(rho_run, options_MPDO)
   
    # compute g2
    idx_signal = 1
    ns = rho_run.number_outcomes()
    g2s = rho_run.density_correlations(n_eps=1e-5)

    # store outcomes
    for il, (n, g2) in enumerate(zip(ns, g2s)):
        tracker.add(f'n_{il}', n.item())
        tracker.add(f'g2_{il}', g2.item())
    
    J_items = []
    for d, J_layer in enumerate(circuit.get_J_matrix(float_vals=True)):
        for l, J in enumerate(J_layer):
            if J is not None:
                tracker.add(f'coupler_{d}{l}', J)
                J_items += [f'{d}{l}']

    phi_items = []
    for d, phi_layer in enumerate(circuit.get_phase_matrix(float_vals=True)):
        for l, phi in enumerate(phi_layer):
            if phi is not None:
                tracker.add(f'phi_{d}{l}', phi)
                phi_items += [f'{d}{l}']
    

    weight = weight_intensity #* np.minimum(it/100, 1.)
    FOM = (1.-weight) * g2s[idx_signal] + weight * (F.relu(target_intensity - ns[idx_signal]) / target_intensity)
    tracker.add('FOM', FOM.item())

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
    
    tracker.visualize(
        [f'coupler_{s}' for s in J_items], 
        filename=os.path.join(SAVE_DIR, 'Js.png'))
    
    tracker.visualize(
        [f'phi_{s}' for s in phi_items], 
        filename=os.path.join(SAVE_DIR, 'phis.png'))
    
    
    # store data until last run
    tracker.save(os.path.join(SAVE_DIR, 'data.pkl'))

    print(f"| g2: {g2s[idx_signal]:.4e} | n: {ns[idx_signal]:.4e} | tot n: {ns.sum():.4e} |")

    return FOM



if __name__ == "__main__":

    device = "cuda:4"
    
    num_channels = 2
    num_MZI_layers = 4
    Nmax = 20
    U = 0.01
    J_layers = [
        None,
        [np.pi / 4., None], # first coupler
        None,
        [np.pi / 4., None], # second  coupler
        ]
    phases_init = [
        [0.05, 0.], # incoming phase
        None,
        [0., np.pi/2], # outgoing, before second coupler
        None,
        ]
    gamma_dt = 0.0 # loss rate (percentage)
    gate_time = 1.

    # initial state
    n0, nL = 3., 3. 
    n_eps_g2 = 1e-10
    target_intensity = 0.1
    is_lbfgs = True

    # options
    options_ADAM = {
        'lr': 1e-3, 'lr_min': 1e-4, 'max_epochs': 100, 'param_lims': [0, np.pi], 'weight_intensity': 0.9
        }
    options_LBFGS =  {
        'lr': 1e-4, 
        'max_iter': 10, 
        'line_search_fn': 'strong_wolfe', 
        }
    options_MPDO = {'max_BD': 100, 'max_PD': 100, 'cutoff_BD': 1e-9, 'cutoff_PD': 1e-5}

    # to store results
    writedir = os.path.join(FILE_DIR, "data", TIMESTAMP)


    # PREPARE INITIAL STATE
    dict_rhos = {}
    alphas = [0.] * num_channels
    alphas[0], alphas[-1] = np.sqrt(n0), np.sqrt(nL)

    # initialize input state
    state_creator = StateCreator(Nmax, num_batch=1)
    list_ten = state_creator.product_state_coherent(alphas, device=device)
    rho = MPDOtorch(list_ten)


    # create write dir
    if not os.path.exists(writedir):
        os.makedirs(writedir)

    
    # create circuit
    d = dict(
        num_layers = num_MZI_layers,
        num_channels = num_channels,
        U = U, 
        J =  J_layers,
        phase = phases_init,
        gamma = gamma_dt,
        order_kraus = 1,
        dt = gate_time,
        Nmax = Nmax,
        requires_grad_J = True,
        requires_grad_phase = True,
        device=device
    )
    
    MZI_circuit = create_nonlinear_mzi_circuit(**d)
    J_matrix =  MZI_circuit.get_J_matrix(float_vals=False)


    #######################################
    #       PREPARE
    #######################################


    # run the init circuit
    start = time.time()
    rho_in = rho.clone()

    # objective
    tracker = Tracker() 
    weight_intensity = options_ADAM.pop('weight_intensity') 
    objective = lambda it, rho, obj_params, do_tracking: iter_func(
        rho, 
        MZI_circuit, options_MPDO=options_MPDO,
        tracker=tracker, it=it,
        obj_params=obj_params,
        weight_intensity=weight_intensity,
        target_intensity=target_intensity,
        do_tracking=do_tracking
        )
    
    # the optimizer (ADAM)
    max_epochs = options_ADAM.pop('max_epochs')
    param_lims = options_ADAM.pop('param_lims')
    lr_min = options_ADAM.pop('lr_min')

    params = MZI_circuit.get_variables()
    
    optimizer = (torch.optim.Adam(params, **options_ADAM) if not is_lbfgs 
                 else torch.optim.LBFGS(params, **options_LBFGS))
    
    if lr_min is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr_min)
    else:
        scheduler = None
    
    obj_params = [MZI_circuit.get_J_matrix(float_vals=False), 
                  MZI_circuit.get_phase_matrix(float_vals=False)]

    
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


