#######################################################
# The main file for running circuit optimization for 
# state generations. the config yaml files contain all
# the parameters used during the optimization.
########################################################


import torch
from scipy.linalg import sqrtm, logm
import numpy as np
from typing import List, Iterable
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pickle
import yaml

from src.MPS_state_optimization import BosonOperatorsTorch, iregroup
from src.MPS_state_optimization import MPStorch, StateCreator
from src.MPS_state_optimization import NonlinearPhotonicCircuit
from src.MPS_state_optimization import MPSCircuitOptimizer
from src.MPS_state_optimization import visualize_circuit, StateAnalyzer, visualize_rho, Tracker
from src.utils import get_CLI_input

# script path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# timestamp for data storage
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')



def status_update_func(
        psi: MPStorch, 
        score: float,
        params: Iterable[float],
        runtime: float,
        iter: int, 
        tracker: Tracker,
        rho_target: np.ndarray,
        modes: List[int]=None,
        call_print: int=1,
        ):
    
    """
    The update func, run after each optimization iteration.

    Parameters
    ---------
    psi: MPStorch
        The incoming MPS 
    score: float
        The score of FOM last optimization iteration
    params: Iterable[float]
        The list of parameters
    runtime: float
        The runtime
    iter: int
        The iteration of optimization 
    tracker: Tracker
        The tracker object
    rho_target: np.ndarray
        The target density matrix
    modes: List[int] (default None)
        The list of modes for optimization. Default None gives mid circuit, L // 2
    call_print: int (default 1)
        Number of iterations after which to print output. Default is 1, print after every iteration.

    Returns
    -------
    Tracker
        The updated Tracker object
    """

    # check if print is needed
    if not (iter + 1) % call_print == 0:
        return None
    
    # default mode is mid circuit
    if modes is None:
        modes = [psi.num_channels//2]

    # set up state analyzer
    analyzer = StateAnalyzer(psi, modes=modes)

    # get the reduced density matrix
    rho = analyzer.rho.detach().cpu().numpy()

    # make sure rho is in matrix form
    dims = len(rho.shape)
    rho = iregroup(rho,
        [[i for i in range(0,dims,2)], 
         [i for i in range(1,dims,2)]]
        )

    rho_target = iregroup(rho_target,
        [[i for i in range(0,dims,2)], 
         [i for i in range(1,dims,2)]]
        )
    
    # compute fidelity
    fidelity = np.trace(sqrtm(sqrtm(rho) @ rho_target @ sqrtm(rho))).real ** 2
    
    # vacuum filtered conditional fidelity
    rho_cond = rho.copy()
    rho_cond[:,0] = 0
    rho_cond[0,:] = 0
    rho_cond = rho_cond / rho_cond.trace()
    fidelity_cond = np.trace(sqrtm(sqrtm(rho_cond) @ rho_target @ sqrtm(rho_cond))).real ** 2

    # KL divergence
    KL_div = np.trace(rho_target @ (logm(rho_target) - logm(rho))).real

    # create figure of iteration for rho comparison
    figname = os.path.join(writedir, f'rho_optim.png')
    visualize_rho(rho, rho_target, figname)

    # print some metrics
    print(f"Fidelity: {fidelity:.4f}, KL div.: {KL_div:.4f}, max. BD: {np.max(psi.get_BDs())}")

    # collect results and store in Tracker
    tracker.add('objective', score)
    tracker.add('runtime', runtime)
    tracker.add('params', params)
    tracker.add('fidelity', fidelity)
    tracker.add('rho', rho)
    tracker.add('g2', analyzer.get_g2())
    tracker.add('P1', rho[1,1])
    tracker.add('BDs', psi.get_BDs())


    # save the current metric results as pkl file (overwrite previous run)
    file_pkl = os.path.join(writedir, f'results.pkl')
    with open(file_pkl, 'wb') as fp:
        pickle.dump(tracker.results, fp)
        print(f'Results saved successfully in {file_pkl}')

    # update plot of results
    keys = ['objective', 'g2', 'P1', 'fidelity']
    fig, axs = plt.subplots(len(keys),1, figsize=(5, len(keys) * 2))
    for i, key in enumerate(keys):
        axs[i].plot(tracker[key], "-")
        axs[i].set_ylabel(key)

    axs[-1].set_xlabel("iter")
    fig.tight_layout()
    fig.savefig(os.path.join(writedir, f'convergence.png')) 
    plt.close()

    # visualize circuit
    visualize_circuit(tracker['params'][-1], save_path=os.path.join(writedir, f'circuit_optim.png'))

    return tracker


def create_target_rho(dict_state, to, state_creator:StateCreator):

    """Generate the target DM for state generation"""

    if dict_state['state'] == 'single_photon':
        psi_target = state_creator.fock(1, to=to, direct=True)
        rho_target = torch.outer(psi_target.conj(), psi_target)
    
    elif dict_state['state'] == 'cat':
        psi_target = state_creator.single_mode_cat(
            dict_state['cat_target']['alpha'], 
            phase=dict_state['cat_target']['phase'], 
            to=to,
            direct=True
            )
        rho_target = torch.outer(psi_target.conj(), psi_target)
    
    # two-mode N00N state was tested but not included in results
    elif dict_state['state'] == 'NOON':
        psi_target = state_creator.two_mode_NOON(
            n=dict_state['NOON_target']['N'],
            phase=dict_state['NOON_target']['phase'],
            to=to,
            direct=True
        )
        rho_target = torch.einsum('ij,kl->ikjl', psi_target, psi_target.conj())

    return rho_target


if __name__ == "__main__":

    # the directory to store results
    writedir = os.path.join(FILE_DIR, "data", TIMESTAMP)

    # start timer
    start = time()

    # INPUTS:
    np.random.seed(4567)
    torch.autograd.set_detect_anomaly(True)

    # get config file for simulation
    config_yaml = get_CLI_input('--config', type=str, default='config_cat.yaml')

    # Load YAML of configuration
    with open(os.path.join(FILE_DIR, config_yaml), "r") as file:
        config = yaml.safe_load(file)

    # overwrite params with command-line input if given
    config['circuit']['num_channels'] = get_CLI_input(
        '--num_channels', type=int, default=config['circuit']['num_channels']
        )
    # num_layers
    config['circuit']['num_layers'] = get_CLI_input(
        '--num_layers', type=int, default=config['circuit']['num_layers']
        )
    # U 
    config['circuit']['U'] = get_CLI_input(
        '--par', type=float, default=config['circuit']['U']
        )
    # layer depth (dt)
    config['circuit']['layer_depth'] = get_CLI_input(
        '--layer_depth', type=float, default=config['circuit']['layer_depth']
        )
    # device to run on
    config['optimization']['to'] = get_CLI_input(
        '--device', type=str, default=config['optimization']['to']
        )
    # the directory to store
    writedir = get_CLI_input(
        '--writedir', type=str, default=writedir
        )
    
    # Extract parameters
    circuit = config["circuit"]
    Nmax = circuit["Nmax"]
    num_channels = circuit["num_channels"]
    num_layers = circuit["num_layers"]
    layer_depths = [circuit["layer_depth"]] * num_layers
    J_init = circuit["J_init"]
    U = circuit["U"]
    gamma = circuit["gamma"]

    lom_weights = config["lom_weights"]
    no_mask = lom_weights["no_mask"]
    weight_zero = lom_weights["weight_zero"]
    weight_one = lom_weights["weight_one"]
    weight_high = lom_weights["weight_high"]
    weight_off_diag = lom_weights["weight_off_diag"]
    entropy_weight = lom_weights["entropy_weight"]
    g2_weight = lom_weights["g2_weight"]
    norm_weight = lom_weights["norm_weight"]

    optimization = config["optimization"]
    lr = optimization["lr"]
    num_epoch = optimization["num_epoch"]
    epoch_print = optimization["epoch_print"]
    to = optimization["to"]
    J_lims = [l / layer_depths[0] for l in optimization["ratio_lims"]]

    tebd_mps_options = config["tebd_mps_options"]
    options = {"cutoff": float(tebd_mps_options["cutoff"]), "max_BD": tebd_mps_options["max_BD"]}

    # configure initial coherent state
    initial_state = config["initial_state"]
    num_batch = initial_state["num_batch"]
    alphas = [torch.Tensor([initial_state["alpha"]])] * num_channels
    phases = np.pi * torch.rand(num_channels) if initial_state["random_phase"] else torch.zeros(num_channels)
    alphas = [a * torch.exp(1j*ph) for (a, ph) in zip(alphas, phases)]

    # update config with generated phases (if random)
    config["initial_state"]['phases'] = [float(x) for x in phases.cpu().numpy()]
    

    # the output mode for optimization
    modes = [num_channels // 2] if optimization["modes"] == 'None' else optimization["modes"]

    # print some stuff
    print(f"Nmax: {Nmax}, num_channels: {num_channels}, num_layers: {num_layers}")
    print(f"Modes: {modes}, Optimization LR: {lr}")
    print(f"TEBD/MPS options: {options}")

    # create write dir
    if not os.path.exists(writedir):
        os.makedirs(writedir)

    # Save to YAML file
    yaml_file = os.path.join(writedir, "run_config.yaml")
    with open(yaml_file, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    #################################################
    # create initial coherent product state MPS
    #################################################

    state_creator = StateCreator(Nmax, num_batch=num_batch)
    list_ten = [ten for ten in state_creator.product_state_coherent(alphas, to=to)]
    psi_mps = MPStorch(list_ten)

    print(f"Initial state: {num_channels} WGs, {num_layers} layers, {num_batch} batches")
    print(f"Cuda available: {torch.cuda.device_count()}")
    print(f"MPS init device: {psi_mps.device}")

    # generate target and mask for optimization (lower weight to vacuum (0,0) DM matrix element)
    rho_target = create_target_rho(config['target_state'], to=to, state_creator=state_creator)

    if no_mask:
        mask = 1.
    else:
        mask = weight_off_diag * torch.ones((Nmax + 1, Nmax + 1), device=to)
        d = range(Nmax + 1)
        mask[d,d] = weight_high * torch.ones(Nmax + 1, device=to)
        mask[0,0] = weight_zero
        mask[1,1] = weight_one
        mask = mask / mask.norm()

    #################################################
    # Circuit simulation and optimization
    #################################################

    # set up circuit
    circuit = NonlinearPhotonicCircuit(
        num_channels=num_channels, 
        num_layers=num_layers, 
        Nmax=Nmax, 
        layer_depth=layer_depths,
        J_init=J_init,
        U_init=U,
        gamma=gamma,
        to=to
        )
    
    # the optimizer
    optimizer = MPSCircuitOptimizer(circuit)

    # define the FOM called for optimization objective
    ops = BosonOperatorsTorch(Nmax) # the boson operators

    # the function definition, passed as input for optimization
    def FOM(psi: MPStorch, circuit):

        # set up analyzer
        analyzer = StateAnalyzer(psi, modes=modes)

        # the fom value
        fom = norm_weight * analyzer.get_overlap(
            rho_target=rho_target, mask=mask, trace_distance=True
            )
        
        # add entropy regularization
        if entropy_weight > 1e-4:
            fom = fom + entropy_weight * analyzer.get_entanglement_measure()

        # add g2 regularization
        if g2_weight > 1e-4:
            fom = fom + g2_weight * analyzer.get_g2()

        # return fom, normalized with the weights
        return fom / (norm_weight + entropy_weight + g2_weight)

    # define the function to call after every iteration of optimization
    tracker = Tracker()
    call_func = lambda psi, score, params, iter, runtime, tracker: status_update_func(
        psi, score=score, params=params, iter=iter, runtime=runtime, tracker=tracker,
        modes=modes, rho_target=rho_target.detach().cpu().numpy())
    
    # run the optimizer
    res = optimizer.epoch_train(
        psi=psi_mps, 
        objective=FOM, 
        call_func=call_func, 
        lr=lr, 
        max_epochs=num_epoch, 
        epoch_print=epoch_print,
        options=options,
        param_lims=J_lims
        )



