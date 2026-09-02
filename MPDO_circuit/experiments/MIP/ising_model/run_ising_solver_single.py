import numpy as np
from datetime import datetime
import os
import torch
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from src.mpdo_circuit import MPDOCircuit
from src.mpdo_torch import StateCreator, MPDOtorch
from src.utils import BosonOperatorsTorch, eye_like
from src.create_circuit import create_qubit_circuit
from src.tracker import Tracker
from src.mpdo_optimizer import epoch_optimize
from experiments.ising_solver.utils import (
    iter_func, read_ising_instance, get_zz_rz_params
)
import numpy as np

from src.svd_trunc import svd_trunc
from src.utils import irescale, iregroup, sqrtm


# script path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# timestamp for data storage
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# directory to store data
SAVE_DIR = os.path.join(FILE_DIR,"data", TIMESTAMP)


if __name__ == "__main__":

    device = "cpu"
    #num_channels = 101 # number of channels
    filename = 'experiments/ising_solver/ising_instances/g05_60.0'
    num_layers = 50 # layer depth
    theta_start = np.pi / 2 
    eps_theta = 0.1 * np.pi / 2
    eps_theta_gate = np.pi / 2 # maximal rotation angle theta = pi/2 +- eps
    seed = 42
    options_MPDO = { # MPDO state options
        'max_BD': 5, 'max_PD': 100, 'cutoff_BD': 1e-12, 'cutoff_PD': 1e-12
        } 
    options_ADAM = { # solver options
        'lr': 1e-2, 'lr_min': None, 'max_epochs': 200, 
        'param_lims': (0., np.pi)
        }
    verbose = True
    
    # to store results
    writedir = os.path.join(FILE_DIR, "data", TIMESTAMP)

    # initial circuit params (uniform)
    tz_init, theta_A1_init, theta_A2_init = (
        0.1, # tz's
        0. * np.pi / 2 / num_layers, # A1's
        0. * np.pi / 2 / num_layers # A2's
    )

    # initialize input state as vacuum
    state_creator = StateCreator(Nmax=1, num_batch=1)


    # load ising instance
    num_channels, num_edge, J_mat = read_ising_instance(filename)
    J_mat = torch.tensor(J_mat, device=device, dtype=torch.float64)
    target = 2. * (num_edge - 2 * 536)

    # initiate starting state
    coeffs = [[np.sqrt(0.5), np.sqrt(0.5)]] # fix initial spin => redundancy / inversion symmetry
    for _ in range(num_channels-1):
        theta = theta_start + eps_theta * (0.5 - np.random.rand())
        coeffs += [[np.cos(theta/2), np.sin(theta/2)]]
    list_ten = state_creator.product_state(coeffs, device=device)
    rho = MPDOtorch(list_ten)

    # set gate params
    params = [
        [
            get_zz_rz_params(tz_init, theta_A1_init, theta_A2_init, device=device)       
            if (d + l) % 2 == 0 else None    
            for l in range(num_channels)
        ]
    for d in range(num_layers)]

    # set up circuit
    circuit = create_qubit_circuit(
        num_layers,
        num_channels,
        params = params,
        device=device
    )

    # the (random) graph matrix
    torch.manual_seed(seed)

    # define the objective from the iter_func
    tracker = Tracker() 
    obj_params = circuit.get_params()
    objective = lambda it, rho, obj_params: iter_func(
        rho, circuit, J_mat,
        options_MPDO=options_MPDO,
        tracker=tracker, it=it,
        obj_params=obj_params,
        target=target,
        save_dir=SAVE_DIR,
        verbose=True
        )
    
    # set up optimizer (ADAM)
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
    
    # create write dir
    if not os.path.exists(writedir):
        os.makedirs(writedir)

    # see if individual bounds must be given to Ry gates
    if eps_theta_gate > 1e-4:
        param_lims_list = []
        for i, _ in enumerate(optimizer.param_groups[0]['params']):
            if i % 3 == 2:
                param_lims_list.append(param_lims)
            else:
                param_lims_list.append(
                    (np.pi / 2 - eps_theta_gate, np.pi / 2 + eps_theta_gate)
                )



    # optimize (using iterative, epoch solver)
    epoch_optimize(
        objective, 
        optimizer=optimizer, 
        rho=rho, 
        max_epochs=max_epochs,
        obj_params=obj_params, 
        param_lims=param_lims if eps_theta_gate <= 1e-4 else param_lims_list,
        scheduler=scheduler,
        epoch_optim_clear=None
    )