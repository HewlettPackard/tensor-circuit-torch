###########################################################
# The main file for investigating the stability of 
# optimal circuits, by adding noise to the optimal couplers.
############################################################


import os
import pickle
import yaml
import numpy as np
import time
from scipy.linalg import sqrtm
from src.MPS_state_optimization import NonlinearPhotonicCircuit, StateCreator, MPStorch, StateAnalyzer

def get_simulation_data(
        root_dir: str, 
        num_channels: int, 
        num_layers: int, 
        U: float, 
        pickle_filename: str="results.pkl", 
        config_filename: str="run_config.yaml"
        ):


    for dirpath, dirnames, filenames in os.walk(root_dir):
        if pickle_filename in filenames:

            # load pickle results
            pickle_path = os.path.join(dirpath, pickle_filename)
            try:
                with open(pickle_path, 'rb') as f:
                    results = pickle.load(f)
            except Exception as e:
                print(f"Error loading {pickle_path}: {e}")

            # load yaml config
            config_path = os.path.join(dirpath, config_filename)
            try:
                with open(config_path, 'rb') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading {config_path}: {e}")

            # append to dict
            num_channels_it = config["circuit"]["num_channels"]
            num_layers_it = config["circuit"]["num_layers"]
            U_it = config["circuit"]["U"]

            if not (num_channels == num_channels_it and num_layers == num_layers_it and np.abs(U-U_it) < 1e-3):
                continue

            return config, results
    
    Warning(f"Parameters U={U}, num_channels={num_channels} num_layers={num_layers} not found")
    return None, None

def get_cat_rho(alpha=1., phase=np.pi, ind=None):

    # cat target
    psi_target = state_creator.single_mode_cat(
        alpha=alpha, 
        phase=phase, 
        to='cpu',
        direct=True
    ).numpy()
    return np.outer(psi_target.conj(), psi_target)
    


if __name__ == "__main__":

    root_dir = "experiments/gen_state/data_loop_cat"
    filename_pkl = os.path.join(root_dir, "robustness.pkl")
    to = "cuda:4"

    list_num_channels = [3, 11]
    list_num_layers = [15, 25]
    U = 0.25
    list_sigma = np.logspace(-2,0,11)
    num_it = 100

    dict_rhos = {}

    for num_channels in list_num_channels:
        dict_rhos[num_channels] ={}

        for num_layers in list_num_layers:

            dict_rhos[num_channels][num_layers] = {}

            # extract saved data for the parameters given
            config, results = get_simulation_data(root_dir, num_channels=num_channels, num_layers=num_layers, U=U)

            # select J vals of optimal run
            ind_opt = np.argmin(results["objective"])
            score = np.min(results["objective"])
            J_vals = results["params"][ind_opt]

            # construct initial state MPS
            alphas = [config["initial_state"]["alpha"]] * config["circuit"]["num_channels"]
            state_creator = StateCreator(config["circuit"]["Nmax"], num_batch=1)

            # update config entry to have float
            config["tebd_mps_options"]['cutoff'] = float(config["tebd_mps_options"]['cutoff'])

            
            for sigma in list_sigma:

                dict_rhos[num_channels][num_layers][sigma] = []

                for it in range(num_it):
    
                    update_J_vals = [
                        [J * (1. + sigma * np.random.randn()) if J is not None else None for J in layer_J] 
                        for layer_J in J_vals
                        ]
                    
                    # reinitialize input state
                    list_ten = [ten for ten in state_creator.product_state_coherent(alphas, to=to)]
                    psi_mps = MPStorch(list_ten)

                    # contruct optimal circuit
                    circuit = NonlinearPhotonicCircuit(
                        num_channels=num_channels, 
                        num_layers=num_layers, 
                        Nmax=config["circuit"]["Nmax"], 
                        layer_depth=config["circuit"]["layer_depth"],
                        J_init=update_J_vals,
                        U_init=U,
                        gamma=config["circuit"]["gamma"],
                        to=to
                    )


                    start = time.time()
                    circuit.run(psi_mps, options=config["tebd_mps_options"])

                    # construct reduced DM and target and bring everything to numpy
                    rho = psi_mps.ptrace(config['circuit']["num_channels"] // 2).detach().cpu().numpy()

                    # update dict
                    dict_rhos[num_channels][num_layers][sigma].append(rho)

                    rho_target = get_cat_rho()

                    # compute fidelity
                    fidelity = np.trace(sqrtm(sqrtm(rho) @ rho_target @ sqrtm(rho))).real ** 2
                    print(f"iter {it}: fidelity={fidelity}")

                with open(filename_pkl, 'wb') as f:
                    pickle.dump(dict_rhos, f)

                print(f"num_channels={num_channels}, num_layers={num_layers}, sigma={sigma} finished => Results stored in {filename_pkl}.")
   