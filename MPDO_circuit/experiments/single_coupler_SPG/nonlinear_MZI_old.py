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


def iter_func(
    rho: MPDOtorch, 
    MZI_in: CouplerCircuit, 
    MZI_out: CouplerCircuit, 
    circuit_phase_in: PhaseCircuit,
    circuit_phase_out: PhaseCircuit,
    options_MPDO: Dict[str, Any],
    tracker: Tracker, 
    it: int,
    obj_params: List[Any],
    weight_intensity: float,
    target_intensity: float=0.01,
    do_tracking: bool=True
    ):

    # update circuit
    MZI.update(J_matrix=obj_params[0])
    circuit_phase.update(phase_matrix=obj_params[1])

    # run circuit
    rho_run = rho.clone()
    circuit_phase.run(rho_run, options=options_MPDO)
    circuit.run(rho_run, options=options_MPDO)

    # compute g2
    idx_signal = rho_run.num_channels // 2
    ns = rho_run.number_outcomes()
    g2s = rho_run.density_correlations(n_eps=1e-5)
    rel_phi = obj_params[1][0][0] - obj_params[1][0][-1]

    # store outcomes
    for il, (n, g2) in enumerate(zip(ns, g2s)):
        tracker.add(f'n_{il}', n.item())
        tracker.add(f'g2_{il}', g2.item())
        tracker.add('couplings', circuit.get_coupling_matrix())
        tracker.add('rel_phase', rel_phi.item())

    #weight = weight_intensity * (1. - g2s[idx_signal])
    weight = weight_intensity * np.minimum(it/100, 1.)
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
    
    # visualization of circuit
    
    visualize_circuit(
        circuit.get_coupling_matrix(), 
        filename=os.path.join(SAVE_DIR, "circuit")
        )
    
    # store data until last run
    tracker.save(os.path.join(SAVE_DIR, 'data.pkl'))

    print(f"g2: {g2s[idx_signal]:.4e}, n: {ns[idx_signal]:.4e}, tot n: {ns.sum():.4e}, rel. phase: {rel_phi:.2f}")

    return FOM



if __name__ == "__main__":

    device = "cuda:4"
    
    num_channels = 2
    num_layers = 1
    Nmax = 10
    U_in, U_out = 0.01, 0.
    J_in, J_out = np.pi / 4., np.pi / 4
    phases_in, phases_out = [[0., 0.01]], [[0., np.pi/4]]
    gamma_dt = 0.0 # loss rate (percentage): CLI overwritten

    # initial state
    rel_phi = np.pi / 4
    n0, nL = 1., 1. 
    n_eps_g2 = 1e-10
    target_intensity = 0.01

    # options
    options_ADAM = {
        'lr': 1e-3, 'lr_min': 1e-4, 'max_epochs': 100, 'param_lims': [0, np.pi], 'weight_intensity': 1.
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

    
    # circuit
    d = dict(
        num_layers = num_layers,
        U = U_in, 
        J =  J_in,
        gamma = gamma_dt,
        order_kraus = 1,
        dt = 1.,
        Nmax = Nmax,
        num_channels = num_channels,
        requires_grad = True
    )

    # create circuits
    MZI_in = create_nonlinear_photonic_circuit(**d, device=device)
    J_matrix_in =  MZI_in.get_J_matrix(float_vals=False)

    d['J'] = J_out 
    d['U'] = U_out
    MZI_out = create_nonlinear_photonic_circuit(**d, device=device)
    J_matrix_out =  MZI_out.get_J_matrix(float_vals=False)

    #######################################
    #       PREPARE
    #######################################


    requires_grad = [[True for i in range(num_channels)]]

    circuit_phase_in = create_phase_circuit(
        Nmax=Nmax,
        phase = phases_in,
        device=device,
        requires_grad=requires_grad
    )

    circuit_phase_out = create_phase_circuit(
        Nmax=Nmax,
        phase = phases_out,
        device=device,
        requires_grad=requires_grad
    )

    def run_circuit(rho):
        circuit_phase_in.run(rho, options=options_MPDO)
        MZI_in.run(rho, options=options_MPDO)
        circuit_phase_out.run(rho, options=options_MPDO)
        MZI_out.run(rho, options=options_MPDO)


    # run the init circuit
    start = time.time()
    rho_in = rho.clone()

    # objective
    tracker = Tracker() 
    weight_intensity = options_ADAM.pop('weight_intensity') 
    objective = lambda it, rho, obj_params, do_tracking: iter_func(
        rho, 
        run_circuit, options_MPDO=options_MPDO,
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
    J_vars = [J for J_layer in J_matrix for J in J_layer if J is not None]
    optimizer = torch.optim.Adam(
            MZI_in.get_variables() + circuit_phase_in.get_variables() \
                + MZI_out.get_variables() + phases_out.get_variables()
            , **options_ADAM
        )
    if lr_min is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr_min)
    else:
        scheduler = None
    
    # train the circuit for QFI
    obj_params = [J_matrix, circuit_phase.get_phi_matrix(float_vals=False)]
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
