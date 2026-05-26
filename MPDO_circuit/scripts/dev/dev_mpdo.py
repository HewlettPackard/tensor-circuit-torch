import os
import pickle
import yaml
import numpy as np
import time
from scipy.linalg import sqrtm
import torch
import torch.autograd.profiler as profiler

from src.mpdo_circuit import MPDOCircuit
from src.mpdo_torch import StateCreator, MPDOtorch
from src.utils import BosonOperatorsTorch, eye_like
from src.create_circuit import create_nonlinear_photonic_circuit

import numpy as np

from src.svd_trunc import svd_trunc
from src.utils import irescale, iregroup, sqrtm


if __name__ == "__main__":

    device = "cuda:0"

    # circuit
    num_channels = 10
    num_layers = 20
    U = 0.1
    J_init = .2
    dt = 1.
    gamma = .01
    batchdim = 400

    # initial state
    Nmax = 10
    d = Nmax + 1
    alpha = 1.
    N_fock = 1
    is_coherent = True


    # options
    options = {'max_BD': 100, 'max_PD': 100, 'cutoff_BD': 1e-6, 'cutoff_PD': 1e-6}


    dict_rhos = {}

    # initialize input state
    state_creator = StateCreator(Nmax, num_batch=1)

    # coherent state
    if is_coherent:
        list_ten = state_creator.product_state_coherent(alpha * np.ones(num_channels), device=device)
    # Fock number state
    else:
        list_ten = state_creator.product_state_fock(nums = [N_fock] * num_channels, device=device)

    # initialise MPDO from tensors
    rho = MPDOtorch(list_ten)

    # the Krauss operators
    ops = BosonOperatorsTorch(Nmax, device=device)
    loss = np.sqrt(gamma) * ops.a
    no_loss = sqrtm(eye_like(loss) - loss.conj().T @ loss)
    K_ops = [loss, no_loss]
    #K_ops = None

    # create circuit
    circuit = create_nonlinear_photonic_circuit(
        num_layers=num_layers,
        num_channels=num_channels, 
        J=J_init, 
        U=U, 
        Nmax=Nmax, 
        K_ops=K_ops, 
        options=options, 
        device=device
    )

    # extract the circuit variables
    params = circuit.get_variables()

    # run the circuit
    start = time.time()
    circuit.run(rho, options=options)
    print(rf"Finished run in {time.time()-start:.2f}s")

    # print output
    ns = rho.number_outcomes()
    tot_ns = ns.sum()

    print("\nCircuit output:\n")
    print(f"ns: {ns}")
    print(f"tot n: {tot_ns}")
    print(f"BDs: {rho.get_BDs()}")

    e=0
    # ops = BosonOperatorsTorch(Nmax, to=device)

    # loss = np.sqrt(gamma) * ops.a
    # no_loss = sqrtm(eye_like(loss) - loss.conj().T @ loss)
    # # no_loss = torch.linalg.matrix_exp(- 0.5 * gamma * ops.n)
    # rho.krauss_dissipation([loss, no_loss])

    # # print output
    # ns = rho.number_outcomes()
    # tot_ns = ns.sum()

    # print("Krauss output:\n")
    # print(f"ns: {ns}")
    # print(f"tot n: {tot_ns}")
    # print(f"BDs: {rho.get_BDs()}")
    # e=0