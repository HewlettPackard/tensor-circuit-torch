############################################################
# The main file for optimizing circuit readout for metrology 
# from coherent input states. The parameters for the optimization
# are defined in ./config.yaml file
#
# Some general utility function for metrology are defined in
# experiments/metrology/metrology_utils.py
#############################################################


import torch
import numpy as np
from time import time
from datetime import datetime
import os
import yaml
import torch

from src.MPS_state_optimization import BosonOperatorsTorch
from src.MPS_state_optimization import MPStorch, StateCreator, concat_ensemble_from_list
from src.MPS_state_optimization import NonlinearPhotonicCircuit
from src.MPS_state_optimization import MPSCircuitOptimizer
from src.MPS_state_optimization import StateAnalyzer, Tracker
from src.utils import get_CLI_input
from experiments.metrology.metrology_utils import classical_Fisher_information, status_update_func

# dir where script is stored
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# timestamp for storing data
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')



if __name__ == "__main__":

    # the directory to store results
    writedir = os.path.join(FILE_DIR, "data", TIMESTAMP)

    # for runtime tracking
    start = time()

    # INPUTS:
    np.random.seed(4567)
    torch.autograd.set_detect_anomaly(True)

   # get config file for simulation
    config_yaml = get_CLI_input('--config', type=str, default='config.yaml')
    print(config_yaml)

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
    
    # layer_depth
    config['circuit']['layer_depth'] = get_CLI_input(
        '--layer_depth', type=float, default=config['circuit']['layer_depth']
        )
    # U 
    config['circuit']['U'] = get_CLI_input(
        '--par', type=float, default=config['circuit']['U']
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
    initial_state = config["initial_state"]
    alphas = [torch.Tensor([initial_state["alpha"]])] * num_channels

    num_batch = initial_state["num_batch"]

    optimization = config["optimization"]

    i_check = [num_channels // 2] if optimization["i_check"] == 'None' else optimization["i_check"]
    lr = optimization["lr"]
    entropy_weight = optimization["entropy_weight"]
    num_epoch = optimization["num_epoch"]
    epoch_print = optimization["epoch_print"]
    to = optimization["to"]
    modes = [num_channels // 2] if optimization["modes"] == 'None' else optimization["modes"]
    J_lims = [l / layer_depths[0] for l in optimization["ratio_lims"]]

    tebd_mps_options = config["tebd_mps_options"]
    options = {"cutoff": float(tebd_mps_options["cutoff"]), "max_BD": tebd_mps_options["max_BD"]}

    # Print some info of run
    print(f"Nmax: {Nmax}, num_channels: {num_channels}, num_layers: {num_layers}")
    print(f"Modes: {modes}, Optimization LR: {lr}")
    print(f"MPS options: {options}")

    # create write dir
    if not os.path.exists(writedir):
        os.makedirs(writedir)

    # Save to YAML file
    yaml_file = os.path.join(writedir, "run_config.yaml")
    with open(yaml_file, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    #############################################
    # generate initial coherent product state MPS
    #############################################

    # state creator object
    state_creator = StateCreator(Nmax, num_batch=num_batch // 2)

    # unperturbed state (zero phase diff)
    list_ten_0 = [ten for ten in state_creator.product_state_coherent(alphas, to=to)]

    # perturbed state (small phase shift)
    d_theta = 1e-4 
    alphas_1 = [a * np.exp(1j * d_theta) if ind == modes else a for ind, a in enumerate(alphas)]
    list_ten_1 = [ten for ten in state_creator.product_state_coherent(alphas_1, to=to)]
    psi_mps = MPStorch(concat_ensemble_from_list([list_ten_0, list_ten_1]))
    
    # print some info
    print(f"Initial state: {num_channels} WGs, {num_layers} layers, {num_batch} batches")
    print(f"Cuda available: {torch.cuda.device_count()}")
    print(f"MPS init device: {psi_mps.device}")

    #########################################
    # The circuit simulation and optimization
    #########################################

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

    # define FOM function for optimization objective
    ops = BosonOperatorsTorch(Nmax)

    # figure of merit, defined as local function for computing the gradient of output
    def FOM(psi: MPStorch, circuit: NonlinearPhotonicCircuit):

        # negative FI (to maximize FI under minimization)
        FOM = -classical_Fisher_information(
            psi, i_check=i_check, d_theta=d_theta
            )
        
        # If entropy weight is given, add to FOM
        if entropy_weight > 1e-4:
            analyzer = StateAnalyzer(psi, modes=modes)
            FOM = FOM + entropy_weight * analyzer.get_entanglement_measure()
        return FOM


    # define the function to call after every iteration 
    tracker = Tracker()
    call_func = lambda psi, score, params, iter, runtime, tracker: status_update_func(
        psi, score=score, params=params, iter=iter, runtime=runtime, tracker=tracker,
        modes=i_check, write_dir=writedir)
    
    
    # run optimization of circuit
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


