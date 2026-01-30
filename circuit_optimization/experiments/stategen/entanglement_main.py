#######################################################################
# This file lets you investigate the entanglement entropy statistics
#  of the optimized states (not so important)
#######################################################################


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
    
    """Search the data in all files stored after optimization in root dir"""

    # scan all files (walk through dir structure)
    for dirpath, dirnames, filenames in os.walk(root_dir):

        # see if pickle in directory
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

            # Check if params correspond with the ones requested, if not, continue
            if not (num_channels == num_channels_it and num_layers == num_layers_it and np.abs(U-U_it) < 1e-3):
                continue
            
            # return the config and results
            return config, results
    
    # if nothing has been found
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
    filename_pkl = os.path.join(root_dir, "entanglement.pkl")
    to = "cuda:4"

    list_num_channels = [3,11]
    list_num_layers = [25]
    U = 0.25

    dict_dat = {}

    for num_channels in list_num_channels:
        dict_dat[num_channels] = {}

        for num_layers in list_num_layers:

            dict_dat[num_channels][num_layers] = {'S': [], 'BD': []}

            # extract saved data for the parameters given
            config, results = get_simulation_data(root_dir, num_channels=num_channels, num_layers=num_layers, U=U)

            # select J vals of optimal run
            ind_opt = np.argmin(results["objective"])
            score = np.min(results["objective"])
            J_vals = results["params"][-1]

            # construct initial state MPS
            alphas = [config["initial_state"]["alpha"]] * config["circuit"]["num_channels"]
            state_creator = StateCreator(config["circuit"]["Nmax"], num_batch=1)

            # update config entry to have float
            config["tebd_mps_options"]['cutoff'] = float(config["tebd_mps_options"]['cutoff'])
                    
            # reinitialize input state
            list_ten = [ten for ten in state_creator.product_state_coherent(alphas, to=to)]
            psi_mps = MPStorch(list_ten)

            for id, J_vals_layer in enumerate(J_vals):

                # contruct optimal circuit
                circuit = NonlinearPhotonicCircuit(
                    num_channels=num_channels, 
                    num_layers=1, 
                    Nmax=config["circuit"]["Nmax"], 
                    layer_depth=config["circuit"]["layer_depth"],
                    J_init=[J_vals_layer],
                    U_init=U,
                    gamma=config["circuit"]["gamma"],
                    to=to,
                    start_ind=id%2
                )


                circuit.run(psi_mps, options=config["tebd_mps_options"])

                S = psi_mps.get_entanglement_entropy_profile()
                BD = psi_mps.get_BDs()

                # update dict
                dict_dat[num_channels][num_layers]['S'].append(S.detach().cpu().numpy())
                dict_dat[num_channels][num_layers]['BD'].append(BD)

            with open(filename_pkl, 'wb') as f:
                pickle.dump(dict_dat, f)

            print(f"num_channels={num_channels}, num_layers={num_layers} finished => Results stored in {filename_pkl}.")
    