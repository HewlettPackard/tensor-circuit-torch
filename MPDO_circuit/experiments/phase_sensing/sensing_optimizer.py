import os
import pickle
import yaml
import numpy as np
import time
from scipy.linalg import sqrtm
import torch
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from datetime import datetime

from src.mpdo_circuit import MPDOCircuit
from src.mpdo_torch import StateCreator, MPDOtorch
from src.utils import BosonOperatorsTorch, eye_like
from src.create_circuit import create_nonlinear_photonic_circuit, create_phase_circuit
from src.mpdo_optimizer import epoch_optimize
from src.tracker import Tracker
from experiments.phase_sensing.__old.sensing_utils import (
    quantum_Fisher_information, Gaussian_Fisher_information, number_Fisher_information,
    homodyne_Fisher_information
)
from src.visualize import visualize_circuit, visualize_number_distribution, visualize_entropy

import numpy as np

from src.svd_trunc import svd_trunc
from src.utils import irescale, iregroup, sqrtm


# dir where script is stored
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# timestamp for storing data
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
SAVE_DIR = os.path.join(FILE_DIR,"data", TIMESTAMP)


def iter_func(
        rho: MPDOtorch, 
        circuit_init: MPDOCircuit, 
        circuit_phase_pert: MPDOCircuit, 
        circuit_phase: MPDOCircuit, 
        circuit_read: MPDOCircuit, 
        options_MPDO: Dict[str, Any],
        tracker: Tracker,
        d_theta: float,
        it: int,
        phases: List[float],
        verbose: bool=True,
        obj_params: List[Any]|None=None,
        ) -> torch.tensor:

    # update circuits with the parameters
    J_init, J_read, phases = obj_params[0], obj_params[1], obj_params[2]
    circuit_init.update(J_matrix=J_init, U=U)
    circuit_read.update(J_matrix=J_read, U=U)
    circuit_phase.update(phases)

    # run initialization 
    circuit_init.run(rho, options=options_MPDO)

    # clone the output
    rho_dtheta = rho.clone()

    # apply phase shift
    circuit_phase_pert.run(rho_dtheta, options=options_MPDO)

    # store QFI
    QFI = quantum_Fisher_information(rho, rho_dtheta, d_theta)
    ns_phase = rho.number_outcomes()
    SvN_phase = rho.entropy_profile()
    visualize_number_distribution(ns_phase, figname=os.path.join(SAVE_DIR, 'numbers_phase'))
    visualize_entropy(SvN_phase, figname=os.path.join(SAVE_DIR, 'entropy'))

    # readout circuit, for unperturbed and perturbed state
    circuit_phase.run(rho, options=options_MPDO)
    circuit_phase.run(rho_dtheta, options=options_MPDO)

    circuit_read.run(rho, options=options_MPDO)
    circuit_read.run(rho_dtheta, options=options_MPDO)

    # Gaussian FI
    ops = BosonOperatorsTorch(Nmax)
    expect_n = rho.number_outcomes()
    diff_n = (expect_n - rho_dtheta.number_outcomes()) / d_theta
    Sigma_n = rho.correlation_matrix(ops.n) - torch.outer(expect_n, expect_n)
    diff_n = diff_n.to(Sigma_n.dtype)
    GFI_intensity = (diff_n @ Sigma_n.inverse() @ diff_n).real 

    ops = BosonOperatorsTorch(Nmax)
    op = 1/np.sqrt(2) * (ops.a + ops.ad)
    expect_x = rho.local_expectations(op)
    diff_x = (expect_x - rho_dtheta.local_expectations(op)) / d_theta
    Sigma_x = rho.correlation_matrix(op) - torch.outer(expect_x, expect_x)
    diff_x = diff_x.to(Sigma_x.dtype)
    GFI_phase = diff_x @ Sigma_x.inverse() @ diff_x 

    # if tracking and visualization is needed

    # store data in tracker
    tracker.add('QFI', QFI)
    tracker.add('GFI_intensity', GFI_intensity)
    tracker.add('GFI_phase', GFI_phase)
    tracker.add('entropy_profile', SvN_phase)
    tracker.add('intensity_phase', ns_phase)
    tracker.add('intensity_out', expect_n)
    tracker.add('J_init', circuit_init.get_J_matrix())
    tracker.add('J_read', circuit_read.get_J_matrix())
    tracker.add('U', U)

    # visualization of Tracker
    tracker.visualize(
        key_plot=['QFI', 'GFI_intensity', 'GFI_phase'], 
        markers = ['k', 'b', 'r', 'g', 'y'],
        filename=os.path.join(SAVE_DIR, "FOM")
        )
    
    # visualization of circuit
    visualize_circuit(
        circuit_init.get_J_matrix() + circuit_read.get_J_matrix(), 
        phase_shifts={'layer': circuit_init.num_layers, 'positions': phases},
        filename=os.path.join(SAVE_DIR, "circuit")
        )
        
    # visualization of number outcomes
    visualize_number_distribution(expect_n, figname=os.path.join(SAVE_DIR, 'numbers_read'))

    # save last tracker
    tracker.save(os.path.join(SAVE_DIR, 'data'))

    if verbose:
        print(f'\nn_out: {expect_n}')
        print(f'phases: {circuit_phase.get_phi_matrix(float_vals=True)}')
        print(f'QFI: {QFI:.2f}\t intensity GFI {GFI_intensity:.2f}\t phase GFI {GFI_phase:.2f}')
        print(f'BDs: {rho.get_BDs()}')
        print(f'PDs: {rho.get_PDs()}')

    return -GFI_intensity

    





if __name__ == "__main__":

    device = "cuda:0"

    # circuit topology and number cutoff tensor
    num_channels = 6
    num_layers_init = 6
    num_layers_read = 6
    Nmax = 10

    J_start_init = 0.596
    J_start_read = 0.1 # ps^{-1}, linear tunnel rate
    U = 0.1
    gamma = 0.0 # dissipation rate, in ps^{-1}

    # phase shift to detect
    d_theta = 0.01

    # initial state
    input_intensity = [1.] * num_channels # average number of photons per pulse (Poissonian)

    # type Fisher information
    type_FI = 'Gaussian'

    # optimizer and MPDO options
    options_MPDO = {'max_BD': 100, 'max_PD': 100, 'cutoff_BD': 1e-9, 'cutoff_PD': 1e-9}

    # optimizer options
    optimizer_type = 'ADAM'
    max_epochs = 500
    #max_epochs_CFI = 50
    #iter_alternate = [100,100]
    epoch_optim_clear = None
    iter_g_converge = 1 # number of iterations before full g is reached
    options_ADAM = {'lr': 5e-3}
    options_LBFGS =  {
        'lr': 1e-2, 
        'max_iter': 5, 
        'line_search_fn': 'strong_wolfe', 
        }
    param_lims = [0., np.pi]


    # PREPARE INITIAL STATE

    # initialize input state
    state_creator = StateCreator(Nmax, num_batch=1)

    # create MPDO from tensors (product state coherent)
    alphas = [np.sqrt(I) for I  in input_intensity]
    list_ten = state_creator.product_state_coherent(alphas, device=device)
    rho = MPDOtorch(list_ten)


    print(f'\noptimizing {num_channels} WGs, init_layers: {num_layers_init}, read layers: {num_layers_read}, gate nonlinearity: {U}.\n')

    # CIRCUITS

    # initiation
    d = dict(
        num_layers = num_layers_init,
        U = U, 
        J =  J_start_init,
        gamma = gamma,
        order_kraus = 1,
        dt = 1.,
        Nmax = Nmax,
        num_channels = num_channels,
        requires_grad = True
    )
    circuit_init = create_nonlinear_photonic_circuit(**d, device=device)

    # readout
    d['num_layers'] = num_layers_read
    d['J'] = J_start_read
    circuit_read = create_nonlinear_photonic_circuit(**d, device=device)

    # phase shift
    phases_pert = [[d_theta if i % 2 == 0 else None for i in range(num_channels)]]
    circuit_phase_pert = create_phase_circuit(
        Nmax=Nmax,
        phase = phases_pert,
        device=device
    )

    # phase shift circuit
    phases_init = [[0.01 for i in range(num_channels)]]
    circuit_phase = create_phase_circuit(
        Nmax=Nmax,
        phase = phases_init,
        device=device,
        requires_grad=True
    )


    # set objective (return FI)
    tracker = Tracker()  
    objective = lambda it, rho, obj_params: iter_func(
        rho, 
        circuit_init, circuit_phase_pert, circuit_phase, circuit_read, options_MPDO=options_MPDO,
        tracker=tracker, d_theta=d_theta, it=it, obj_params=obj_params, phases=phases_pert
        )


    obj_params = (
        [circuit_init.get_J_matrix(float_vals=False)] +
         [circuit_read.get_J_matrix(float_vals=False)] +
         [circuit_phase.get_phi_matrix(float_vals=False)]
    )

    optimizer = (
        torch.optim.Adam(
            circuit_read.get_variables() + circuit_phase.get_variables(), 
            **options_ADAM
        ) if optimizer_type == 'ADAM' else
        torch.optim.LBFGS(
            circuit_read.get_variables() + circuit_phase.get_variables(), 
            **options_LBFGS
        )
    )

    # optimizer = (
    #     torch.optim.Adam(
    #         circuit_read.get_variables(), 
    #         **options_ADAM
    #     ) if optimizer_type == 'ADAM' else
    #     torch.optim.LBFGS(
    #         circuit_read.get_variables(), 
    #         **options_LBFGS
    #     )
    # )

    # create folder to save
    os.makedirs(SAVE_DIR, exist_ok=True)

    # train the circuit for QFI
    epoch_optimize(
        objective, 
        optimizer=optimizer, 
        rho=rho, 
        max_epochs=max_epochs, 
        obj_params=obj_params,
        param_lims=param_lims,
        epoch_optim_clear=epoch_optim_clear
        )
    
    # select maximal quantum information circuit
    it_max_QFI = np.argmax(tracker['QFI'])

    # make J values tensors
    J_init = [
        [torch.tensor(J, requires_grad=False, device=device) if J else None
         for J in layer] 
        for layer in tracker['J_init'][it_max_QFI]
        ]
    J_read = [
        [torch.tensor(J, requires_grad=True, device=device) if J else None
         for J in layer] 
        for layer in tracker['J_read'][it_max_QFI]
        ]
    
    obj_params = [
        J_init,
        J_read,
        lambda it: torch.tensor(tracker['U'][it_max_QFI])
        ]
    
    # set objective (return FI) 
    objective = lambda it, rho, obj_params, do_tracking: iter_func(
        rho, 
        circuit_init, circuit_phase_pert, circuit_read, options_MPDO=options_MPDO,
        tracker=tracker, d_theta=d_theta, it=it, type_FI=type_FI,
        phases=phases_pert[0],
        obj_params=obj_params,
        do_tracking=do_tracking
        )
    
    # # train the circuit for CFI
    # epoch_optimize(
    #     objective, 
    #     optimizer=optimizer_CFI, 
    #     rho=rho, 
    #     max_epochs=max_epochs_CFI, 
    #     obj_params=obj_params,
    #     param_lims=param_lims,
    #     epoch_optim_clear=epoch_optim_clear
    #     )
    




            
    