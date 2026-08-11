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
from src.create_circuit import create_nonlinear_photonic_circuit, create_phase_circuit

import numpy as np

from src.svd_trunc import svd_trunc
from src.utils import irescale, iregroup, sqrtm


if __name__ == "__main__":

    device = "cuda:0"

    # circuit
    num_channels = 5
    Nmax = 15
    U = 0.2
    J = np.pi / 4
    gamma = 0.1
    d_theta = 0. * np.pi / 2

    # initial state
    alpha = 1
    N_fock = 1
    is_coherent = False


    # init circuit
    d_init = dict(
        num_layers = 2,
        U = U, 
        J = J,
        gamma = gamma,
        dt = 1.,
        Nmax = Nmax,
        num_channels = num_channels
    )

    # phase circuit
    d_phase = dict(
        num_layers = 1,
        phase = [[d_theta if il==0 else 0. for il in range(num_channels) ]],
        dt = 1.,
        Nmax = Nmax,
        num_channels = num_channels
    )

    # read circuit
    d_read = dict(
        num_layers = 3,
        U = U, 
        J = J,
        gamma = gamma,
        dt = 1.,
        Nmax = Nmax,
        num_channels = num_channels
    )

    # options
    options = {'max_BD': 100, 'max_PD': 100, 'cutoff_BD': 1e-9, 'cutoff_PD': 1e-9}


    #######################################
    #       PREPARE
    #######################################

    # prepare initial state
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


    # create circuits
    circuit_init = create_nonlinear_photonic_circuit(**d_init, device=device)
    circuit_phase = create_phase_circuit(**d_phase, device=device)
    circuit_read = create_nonlinear_photonic_circuit(**d_read, device=device)


    # extract the circuit variables
    params_init = circuit_init.get_variables()
    params_phase = circuit_phase.get_variables()
    params_read = circuit_read.get_variables()

    # run the init circuit
    start = time.time()
    circuit_init.run(rho, options=options)
    print(rf"Finished init run in {time.time()-start:.2f}s")

    rho_phase = rho.clone()

    # run the phase circuit
    start = time.time()
    circuit_phase.run(rho_phase, options=options)
    print(rf"Finished init run in {time.time()-start:.2f}s")

    # run the readout circuit unperturbed state
    start = time.time()
    circuit_read.run(rho, options=options)
    print(rf"Finished init run in {time.time()-start:.2f}s")

    # run the readout circuit phase-shifted state
    start = time.time()
    circuit_read.run(rho_phase, options=options)
    print(rf"Finished init run in {time.time()-start:.2f}s")

    # print output
    ns = rho.number_expectations()
    vars = rho.number_variances()
    g2s = rho.density_correlations(n_eps=1e-3)

    print(f"BDs: {rho.get_BDs()}")
    print(f"PDs: {rho.get_PDs()}")
    print(f"densities: {ns.detach().cpu().numpy()}, N_tot={ns.sum()}")
    print(f"g2s: {g2s.detach().cpu().numpy()}")
    # tot_ns = ns.sum()

    # print("\nCircuit output bare:\n")
    # print(f"ns: {ns}")
    # print(f"tot n: {tot_ns}")
    # print(f"BDs: {rho.get_BDs()}")

    #     # print phase diff 
    # ns_phase = rho_phase.number_outcomes()
    # print("\nCircuit output phase:\n")
    # print(f"ns_phase: {ns_phase}")
    # print(f"Diff: {(ns - ns_phase)}")

    # # FOM dummy
    # FOM = (((ns - ns_phase) / d_theta) ** 2 / vars).sum()
    # print(f"FOM: {FOM}")
    # FOM.backward()
    # #print('test2')
    # print(f"FOM grad {[v.grad for v in circuit_read.get_variables()]}")