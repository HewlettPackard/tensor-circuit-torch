import numpy as np
import time
import torch
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt

from src.mpdo_circuit import MPDOCircuit
from src.mpdo_torch import StateCreator, MPDOtorch
from src.utils import BosonOperatorsTorch, eye_like
from src.create_circuit import create_haar_random_circuit

import numpy as np

from src.svd_trunc import svd_trunc
from src.utils import irescale, iregroup, sqrtm


if __name__ == "__main__":

    device = "cuda:1"
    num_channels = 20 # number of channels
    num_layers = 40 # layer depth
    options = {'max_BD': 500, 'max_PD': 100, 'cutoff_BD': 1e-6, 'cutoff_PD': 1e-4} # options
    verbose = True

    gamma = 0.01 # error rate

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
        gamma = gamma,
        device=device
    )

    circuit.run(rho, options=options, verbose=verbose)

    print(f"BDs: {rho.get_BDs()}")
    print(f"PDs: {rho.get_PDs()}")
    
    S_vN = rho.entropy_profile(entropy='entanglement')
    S_P = rho.entropy_profile(entropy='purity')
    plt.plot(range(num_channels + 1), S_vN.cpu().numpy(), label='EE')
    plt.plot(range(num_channels), S_P.cpu().numpy(), label='PE')
    plt.legend()
    plt.ylabel(r"S")
    plt.savefig('entropy.png')