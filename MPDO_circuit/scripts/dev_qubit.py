import numpy as np
import time
import torch
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt

import sys
from pathlib import Path

from src_dev.mpdo_circuit import MPDOCircuit
from src_dev.mpdo_torch import StateCreator, MPDOtorch
from src_dev.circuit_gates import CircuitGateTwoSite
from src_dev.utils import BosonOperatorsTorch, eye_like
from src_dev.create_circuit import create_haar_random_circuit, dephasing_krauss_ops

from src_dev.svd_trunc import svd_trunc

if __name__ == "__main__":

    torch.cuda.manual_seed(75)


    device = "cuda:1"
    num_channels = 10 # number of channels
    num_layers = 10 # layer depth
    options = {'max_BD': 500, 'max_PD': 100, 'cutoff_BD': 1e-6, 'cutoff_PD': 1e-4} # options
    verbose = True

    gamma = 0.1 # error rate

    ####################################
    # INITIAL STATE
    ####################################

    dict_rhos = {}

    # initialize input state
    state_creator = StateCreator(Nmax=1, num_batch=1)
    list_ten = state_creator.product_state_fock([0] * num_channels, device=device)
    rho = MPDOtorch(list_ten)

    ####################################
    # Create circuit and run
    ####################################

    circuit = create_haar_random_circuit(
        num_layers = num_layers,
        num_channels = num_channels,
        gamma = 0.,
        device=device
    )

    circuit.run(rho, options=options, verbose=verbose)

    print(f"BDs: {rho.get_BDs()}")
    print(f"PDs: {rho.get_PDs()}")

    S_vN = rho.entropy_profile(entropy='entanglement')
    S_P = rho.SP

    print(f"circuit out:\nS_vN: {S_vN}")
    print(f"S_purity: {S_P}")


    

    Ks = dephasing_krauss_ops(gamma=gamma, device=rho.device)

    # two-mode Ks
    Ks_2mode = torch.stack(
        [torch.einsum('ij,kl->iklj', K1, K2) for K2 in Ks for K1 in Ks],
        dim=0
        )

    K_expects = []

    for ik, K in enumerate(Ks_2mode):

        # clone rho
        rhoc = rho.clone()

        # make a circuit gate
        K_gate = CircuitGateTwoSite(1, gate_tensor=K)

        # apply to state, restore canonical form
        K_gate.apply_to(rhoc, site=num_channels // 2, normalize=False)
    
        rhoc.canonical_form()

        


        print(rhoc)


    e = 0
    # for it in range(2):

    #     print(f"\nkraus {it}:\nS_vN: {S_vN}")

    #     rhoc = rho.clone()
    #     rhoc.krauss_dissipation(Ks[:], num_select=1)

    #     S_vN = rhoc.entropy_profile(entropy='entanglement')
    #     S_P = rhoc.entropy_profile(entropy='purity')

        
    #     print(f"S_purity: {S_P}")
    #     print(rhoc.norm())
    #     print(rhoc.is_canonical())

    # e = 0
