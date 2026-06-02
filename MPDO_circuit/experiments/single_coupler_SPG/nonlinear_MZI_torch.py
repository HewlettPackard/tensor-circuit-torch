import os
import pickle
import yaml
import numpy as np
import time
from scipy.linalg import sqrtm
from datetime import datetime
from typing import Dict, List, Any
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True

from src.mpdo_circuit import CouplerCircuit, PhaseCircuit, MPDOCircuit
from src.mpdo_optimizer import epoch_optimize
from src.mpdo_torch import StateCreator, MPDOtorch
from src.create_circuit import create_nonlinear_mzi_circuit
from src.tracker import Tracker
from src.visualize import visualize_circuit

from experiments.single_coupler_SPG.nonlinear_MZI_Gauss import solve_ODE_AS_mode, get_g2

import numpy as np


# script path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# timestamp for data storage
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# directory to store data
SAVE_DIR = os.path.join(FILE_DIR,"data", TIMESTAMP)

def run_MZI(rho: MPDOtorch, d: Dict[str, Any], options: Dict[str, Any]):

    # initiate MZI circuit
    circuit = create_nonlinear_mzi_circuit(**d)

    # run circuit
    circuit.run(rho, options)

    # compute observables
    ns = rho.number_outcomes()
    g2s = rho.density_correlations(n_eps=1e-5)

    return ns, g2s



if __name__ == "__main__":

    device = "cuda:4"
    
    num_channels = 2
    num_MZI_layers = 4
    Nmax = 20
    U_in, U_out = 0., 0.
    U_coupler = 0.005
    Js = np.linspace(0., np.pi/2, 20)
    phi = 0.1
    n0, n1 = 1., 1. # initial state

    res = {
        "n_tensor":[],
        "g2_tensor": [],
        "n_gauss": [],
        "g2_gauss": []
        }

    for J in Js:
        J_layers = [
            None
            [J, None], # first coupler
            None,
            [np.pi / 4., None], # second  coupler
            ]
        
        phases_init = [
            [phi, 0.],
            None,
            [0., np.pi/2], # outgoing, before second coupler
            None,
            ]
        gamma_dt = 0.0 # loss rate (percentage)
        gate_time = 1.


        options_MPDO = {'max_BD': 100, 'max_PD': 100, 'cutoff_BD': 1e-9, 'cutoff_PD': 1e-5}

        # to store results
        writedir = os.path.join(FILE_DIR, "plot_comparison")


        # PREPARE INITIAL STATE
        dict_rhos = {}
        alphas = [0.] * num_channels
        alphas[0], alphas[-1] = np.sqrt(n0), np.sqrt(n1)

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
            U = U_coupler, 
            J =  J_layers,
            phase = phases_init,
            gamma = gamma_dt,
            order_kraus = 1,
            dt = gate_time,
            Nmax = Nmax,
            requires_grad_J = False,
            requires_grad_phase = False,
            device=device
        )

        # run the MZI and get access to n's and g2's
        ns, g2s = run_MZI(rho.clone(), d, options_MPDO)

        # convert intial state for ODEs
        n_symm = np.sum(alphas) ** 2 / 2
        alpha_0 = 1./ np.sqrt(2) * (alphas[0] * np.exp(1j * phi) - alphas[1])

        # run ODEs
        alphas_gauss, N_gauss, M_gauss = solve_ODE_AS_mode(J, U_coupler * n_symm, alpha_0, tlist = [0., 1.], gamma=gamma_dt)
        
        # compute numbers
        n_gauss = np.abs(alphas_gauss[-1]) ** 2
        g2_gauss = get_g2(alphas_gauss[-1], N_gauss[-1], M_gauss[-1])

        res['n_tensor'].append(ns[-1])
        res['g2_tensor'].append(g2s[-1].cpu().float())
        res['n_gauss'].append(n_gauss)
        res['g2_gauss'].append(g2_gauss)

    # figure
    plt.figure(figsize=(5,2))

    plt.plot(Js, res["g2_tensor"], label='tensor')
    plt.plot(Js, res["g2_gauss"], label='gauss')
    plt.legend()
    plt.xlabel("coupling Jdt")
    plt.savefig(os.path.join(writedir, 'g2s.png'))