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
from src.create_circuit import create_nonlinear_photonic_circuit, create_phase_circuit, amplitude_damping_kraus_ops
from src.mpdo_optimizer import epoch_optimize, sweeping_optimize
from src.tracker import Tracker
from experiments.phase_sensing.sensing_utils import (
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


Mc = lambda k: 1.6114 + 0.037 * k  # (photonic mode approximated as a straight line)
Exca = 1.6229  # (exciton energy)
OM = 3.4 * 1e-3 # (half the Rabi splitting)
hbar_eV_ps = 6.582 * 1e-4 # hbar, converting energy to frequency

def omega_LP(k: float, ps: bool=True):

    """
    Lower-polariton dispersion in eV or ps^{-1} (default) and um!

    """

    LP_disp =  0.5 * (Mc(k) + Exca) - 0.5 * ((Mc(k) - Exca)**2 + 4. * OM ** 2) ** 0.5
    return LP_disp / hbar_eV_ps if ps else LP_disp


def vg_LP(k: float, dk: float=1e-3):
    """
    Lower-polariton group velocity, in eV * um / hbar = um / ps
    """
    return (omega_LP(k + dk/2., ps=True) - omega_LP(k - dk/2., ps=True)) / dk


def exciton_fraction(k):
    """ 
    The exciton fraction at given k (um^(-1))
    """
    theta_k = np.arctan(OM / (omega_LP(k, ps=False) - Mc(k)))
    return np.cos(theta_k) ** 2
    


def iter_func(
        rho: MPDOtorch, 
        circuit_init: MPDOCircuit, 
        circuit_phase: MPDOCircuit, 
        circuit_read: MPDOCircuit, 
        options_MPDO: Dict[str, Any],
        tracker: Tracker,
        d_theta: float,
        it: int,
        phases: List[float],
        type_FI: str='Gaussian',
        target_intensity: float=1.,
        verbose: bool=True,
        obj_params: List[Any]|None=None,
        do_tracking=False,
        ) -> torch.tensor:
    
    type_FI = ("quantum" if iter_alternate is not None and it % (iter_alternate[0] + iter_alternate[1]) >= iter_alternate[0]
               else type_FI
               )

    # update circuits with the parameters
    J_init, J_read = obj_params[0], obj_params[1]
    U_it = obj_params[2](it)
    circuit_init.update(J_matrix=J_init, U=U_it)
    circuit_read.update(J_matrix=J_read, U=U_it)

    # run initialization 
    circuit_init.run(rho, options=options_MPDO)

    # clone the output
    rho_dtheta = rho.clone()

    # amplitude Krauss op
    K_ops = amplitude_damping_kraus_ops(d_theta * 2., Nmax=rho.Nmax, device=rho.device)
    rho_dtheta[-1] = torch.einsum("ij, bkjl->bkil", K_ops[0], rho_dtheta[-1])
    rho_dtheta.canonical_form(options=options_MPDO)

    # apply phase shift
    # circuit_phase.run(rho_dtheta, options=options_MPDO)

    # store QFI
    QFI = quantum_Fisher_information(rho, rho_dtheta, d_theta)
    ns_phase = rho.number_outcomes()
    SvN_phase = rho.entropy_profile()
    SNL = 4. * ns_phase[
        np.where([circuit_phase.circuit_topology[0, i] for i in range(circuit_phase.num_channels)])[0][0]
        ]
    visualize_number_distribution(ns_phase, figname=os.path.join(SAVE_DIR, 'numbers_phase'))
    visualize_entropy(SvN_phase, figname=os.path.join(SAVE_DIR, 'entropy'))

    # readout circuit, for unperturbed and perturbed state
    circuit_read.run(rho, options=options_MPDO)
    circuit_read.run(rho_dtheta, options=options_MPDO)

    # evaluate FIs
    GFI = Gaussian_Fisher_information(rho, rho_dtheta, d_theta, i_exclude=num_channels-1) 
    HFI = homodyne_Fisher_information(rho, rho_dtheta, d_theta, i_exclude=num_channels-1)
    PFI = number_Fisher_information(rho, rho_dtheta, d_theta, i_exclude=num_channels-1)

    # if tracking and visualization is needed
    if do_tracking:

        # store data in tracker
        tracker.add('QFI', QFI)
        tracker.add('SNL', SNL)
        tracker.add('PFI', PFI)
        tracker.add('HFI', HFI)
        tracker.add('GFI', GFI)
        tracker.add('N_signal', ns_phase[-1])
        tracker.add('J_init', J_init)
        tracker.add('J_read', J_read)
        tracker.add('U', U_it)

        # visualization of Tracker
        tracker.visualize(
            key_plot=['QFI','PFI', 'GFI', 'N_signal'], 
            labels=['Quantum FI','Number FI', 'Gaussian FI', 'signal strength'],
            markers = ['k-', 'b-', 'r-', ':k'],
            filename=os.path.join(SAVE_DIR, "FOM")
            )
        
        # visualization of circuit
        visualize_circuit(
            circuit_init.get_coupling_matrix() + circuit_read.get_coupling_matrix(), 
            phase_shifts={'layer': circuit_init.num_layers, 'positions': phases},
            filename=os.path.join(SAVE_DIR, "circuit")
            )
        
        # visualization of number outcomes
        ns_out = rho.number_outcomes()
        visualize_number_distribution(ns_out, figname=os.path.join(SAVE_DIR, 'numbers_read'))

        # save last tracker
        tracker.save(os.path.join(SAVE_DIR, 'data'))

        if verbose:
            print(f'n_i: {rho.number_outcomes().detach().cpu().numpy()}')
            print(f'QFI: {QFI}, PFI: {PFI}, GFI: {GFI}, HFI: {HFI}')
            print(f"BDs: {rho.get_BDs()}")
            #print(f"thetas: {[circuit_phase.circuit_topology[0][l].phi.float() for l in range(num_channels)]}")
            # print(f'parameters init: {circuit_init.get_variables()}')
            # print(f'parameters read: {circuit_read.get_variables()}')

    # select and return figure of merit
    if type_FI == 'Gaussian':
        target = GFI
    elif type_FI == 'homodyne':
        target = HFI
    elif type_FI == 'quantum':
        target = QFI
    elif type_FI == 'number':
        target = PFI
    else:
        TypeError('Invalid FI type, choose from Gaussian, homodyne, number or quantum.')

    # compute L1 penalty for violating signal intensity
    alpha = 0.8
    signal_penalty =  (ns_phase[-1] - target_intensity).abs() ** 2

    return_target = - target + signal_penalty
    print(f"FOM={return_target}\n")
    return return_target

    





if __name__ == "__main__":

    device = "cuda:1"

    # circuit topology and number cutoff tensor
    num_channels = 4
    num_layers_init = 10
    num_layers_read = 0
    Nmax = 10

    # circuit parameters
    g_ex = 10. # ueV * um^2, for excitons, gets scaled with exciton fraction of polaritons
    layer_length = 100 # um
    J_init_init = 0.1
    J_init_read = 0.01 # ps^{-1}, linear tunnel rate
    pulse_duration = 1 # ps
    pulse_transverse = 0.5 # um
    gamma = 0.0 # dissipation rate, in ps^{-1}
    k_ref = 0.3 # reference wavevecor
    d_theta = 1e-3 # infinitesimal phase shift

    # effective group velocity -> rescales pulse width and duration
    group_velocity = vg_LP(k_ref) 
    ex_frac = exciton_fraction(k_ref)

    # initial state
    signal_intensity = 2.
    probe_intensity = 1.
    target_signal = 1.
    # type Fisher information
    type_FI = 'Gaussian'

    # optimizer and MPDO options
    options_MPDO = {'max_BD': 100, 'max_PD': 100, 'cutoff_BD': 1e-6, 'cutoff_PD': 1e-6}

    # optimizer options
    optimizer_type = 'ADAM'
    max_epochs = 20
    max_sweeps = 50
    iter_alternate = None
    epoch_optim_clear = None
    iter_g_converge = 1 # number of iterations before full g is reached
    options_ADAM = {'lr': 1e-3}
    options_LBFGS =  {
        'lr': 1e-2, 
        'max_iter': 5, 
        'line_search_fn': 'strong_wolfe', 
        }
    param_lims = None # [0., np.pi]


    # PREPARE INITIAL STATE

    # initialize input state
    state_creator = StateCreator(Nmax, num_batch=1)

    # create MPDO from tensors (product state coherent)
    input_intensity = [signal_intensity] * num_channels
    input_intensity[-1] = probe_intensity
    alphas = [np.sqrt(I) for I  in input_intensity]
    list_ten = state_creator.product_state_coherent(alphas, device=device)
    rho = MPDOtorch(list_ten)


    # gate time scale
    dt_gate = layer_length / group_velocity

    # effective interaction
    U = torch.tensor(
        ex_frac ** 2 * g_ex / (pulse_duration * group_velocity * pulse_transverse)
    )
    f_U = lambda it: torch.min(U, (it+1) / iter_g_converge * U)

    print(f'\noptimizing {num_channels} WGs, init_layers: {num_layers_init}, read layers: {num_layers_read}, gate nonlinearity: {U}.\n')

    # CIRCUITS

    # initiation
    d = dict(
        num_layers = num_layers_init,
        U = U, 
        J =  J_init_init / dt_gate,
        gamma = gamma / dt_gate,
        order_krauss = 1,
        dt = dt_gate,
        Nmax = Nmax,
        num_channels = num_channels,
        requires_grad = True
    )
    circuit_init = create_nonlinear_photonic_circuit(**d, device=device)

    # readout
    d['num_layers'] = num_layers_read
    d['J'] = J_init_read
    d['right_stop'] = num_channels - 1 # exclude right channel from interference
    circuit_read = create_nonlinear_photonic_circuit(**d, device=device)

    # phase shift
    phases = [[d_theta if i == num_channels-1 else 0. for i in range(num_channels)]]
    requires_grad = [[False if i == num_channels-1 else False for i in range(num_channels)]]
    circuit_phase = create_phase_circuit(
        Nmax=Nmax,
        phase = phases,
        device=device,
        requires_grad=requires_grad
    )


    # set objective (return FI)
    tracker = Tracker()  
    objective = lambda it, rho, obj_params, do_tracking: iter_func(
        rho, 
        circuit_init, circuit_phase, circuit_read, options_MPDO=options_MPDO,
        tracker=tracker, d_theta=d_theta, it=it, type_FI=type_FI, target_intensity=target_signal,
        phases=phases[0],
        obj_params=obj_params,
        do_tracking=do_tracking
        )

    # extract all parameters to optimize
    obj_params = (
        [circuit_init.get_coupling_matrix(float_vals=False)]
        + [circuit_read.get_coupling_matrix(float_vals=False)] 
        + [f_U]
    )

    # assign the right parameters to the set of optimizers
    num_layers = num_layers_init + num_layers_read
    coupling_matrix = circuit_init.get_coupling_matrix(float_vals=False)
    optimizers = []
    for d in range(num_layers_init):
        vars = [var for var in coupling_matrix[d] if var is not None]
        optimizers.append(
            torch.optim.Adam(
                vars, 
                **options_ADAM
            )
        )

    # create folder to save
    os.makedirs(SAVE_DIR, exist_ok=True)



    # train the circuit
    sweeping_optimize(
        objectives = [objective] * num_layers_init,
        optimizers=optimizers,
        final_optimizer=None,
        rho=rho,
        max_sweeps=max_sweeps,
        max_epochs=max_epochs,
        obj_params=obj_params,
        param_lims=param_lims
    )

