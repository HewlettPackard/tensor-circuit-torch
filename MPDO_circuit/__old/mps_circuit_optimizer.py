import numpy as np
import torch
from time import time
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Iterable

from .mps_torch import MPStorch
from ..src.mpdo_circuit import MPDOCircuit
from .analyzer import Tracker

import itertools

class MPSCircuitOptimizer(ABC):

    """
    The MPS Circuit optimizer to tune the coupling gates of the circuit using gradient descent.
    A Tracker object is initialized to track the state of the optimization through the ensuing 
    iterations

    Parameters
    ----------
    circuit: NonlinearPhotonicCircuit
        The circuit to optimize.


    """

    def __init__(
            self, 
            circuit: MPDOCircuit
            ):

        self.circuit = circuit
        self.tracker = Tracker()


    def epoch_train(
            self,
            objective: Callable[[MPStorch], float],
            call_func: Callable[[MPStorch], None],
            psi: MPStorch|None=None,
            max_epochs: int=500, 
            lr: float=0.01, 
            track_params: bool=True,
            verbose: bool=True,
            epoch_print: int=10,
            options: Dict[str, Any]={'max_BD': 100, 'cutoff': 1e-8},
            param_lims: None|List[float]=None
            ) -> Dict[str, Any]:

        """
        Epoch-based optimization of the circuit, using torch's Adam optimizer.

        Parameters
        ----------
        objective: Callable[[MPStorch], float]
            The function FOM for optimization. Takes MPS as input and returns float value for the score.

        call_func: Callable[[MPStorch], None]
            The function to call after every iteration. To store results and print information.

        psi: MPStorch
            The initial MPS state, entering the circuit.

        max_epochs: intv(default 500)
            Maximum epochs for optimization.

        lr: float (default 0.01)
            The learning rate of Adam optimizer. Constant for now, variable learning rate can be added in future.

        track_params: bool (default True)
            Track the parameter updates in when stored.

        verbose: bool (default True)
            Verbose print output.

        epoch_print: int (default 10)
            The number of epochs after which to print.

        options: Dict[str, Any] (default {'max_BD': 100, 'cutoff': 1e-8})
            The MPS max. BD and SV cutoff options.

        param_lims: None|List[float] (default None)
            Hard constraints to parameter updates. E.g., setting lower limit param_lims[0]=0
            ensures the J-values remain positive -- for experiments [0, pi] was used.


        Returns
        -------
        Tracker
            The tracker object with all information of the optimization run.
            
        """
        
        # starting time
        start = time()

        # optimizer
        optimizer = optim.Adam(self.circuit.get_variables(), lr=lr)


        epochs_start = time()
        for epoch in range(max_epochs):

            # set grads to zeros
            optimizer.zero_grad()

            # clone the intit MPS for starting run
            psi_run = psi.clone()

            # update circuit with the new obtained parameters from previous run
            self.circuit.update()

            # run the circuit
            self.circuit.run(psi_run, options=options)

            # get defined objective
            score = objective(psi_run, self.circuit)

            # check if not nan
            if torch.isnan(score):
                print("Objective is NaN...")
                break
            
            # evaluate backward pass
            score.backward()

            # step in optimization
            optimizer.step()

            # clamp model parameters to constraints
            if param_lims is not None:
                for p in self.circuit.get_variables():
                    p.data.clamp_(param_lims[0], param_lims[1])

            # append result
            params = self.circuit.get_circuit_Js() if track_params else None

            # call func to follow updates
            self.tracker = call_func(
                psi_run,
                score=score, 
                params=params, 
                iter=epoch,
                runtime=time()-start,
                tracker=self.tracker
                )

            # Print the information, if requested
            if verbose and (epoch + 1) % epoch_print == 0:  
                print(f'Epoch {epoch + 1} in {time() - epochs_start:.2f}s, objective: {score}')
                epochs_start = time()

            if verbose:
                print(f'Runtime: {time() - start:.2f}s')

        return self.tracker
    

    def LBFGS_train(
            self, 
            psi: MPStorch,
            objective: Callable[[MPStorch], float],
            call_func: Callable[[MPStorch], None],
            num_steps: int=30, 
            lr: float=0.01, 
            track_params: bool=False,
            verbose: bool=True,
            tol: float=1.e-5,
            solver_args: Dict[str,Any]={
                'line_search_fn': 'strong_wolfe', 'max_iter': 5
            },
            options: Dict[str, Any]={'max_BD': 100, 'cutoff': 1e-8}
            ) -> Dict[str, Any]:
        
        """
            OBSOLETE. 
            
            LBFGS train method. Similar as above, but need to be reviewed. 
            Use Adam optimizer epoch train.
        """
        
        # starting time
        start = time()

        # init optimizer
        optimizer = optim.LBFGS(self.circuit.get_variables(), lr=lr, **solver_args)

        # track objectives and parameters
        score_list = []
        if track_params:
            params_list = []
        

        # inline closure function
        def closure(get_state=False):

            cl_time = time()
            optimizer.zero_grad()

            psi_run = psi.clone()
            self.circuit.update()
            self.circuit.run(psi_run, options=options)
            
            score = objective(psi_run)

            if not get_state:
                score.backward()
                
                print(f"iter finished in {time()-cl_time:.2f}s: FOM={score:.4f}, max. BD: {np.max(psi_run.get_BDs())}")
                return score
            
            return score, psi_run
            

        # start iterations
        epochs_start = time()
        for step in range(num_steps):

            
            optimizer.step(closure)

            # append result
            score, psi_run = closure(True)

            # check if not nan
            if torch.isnan(score):
                print("Objective is NaN...")
                break

            score_list.append(score.to('cpu').detach().numpy())

            if track_params:
                params_list.append(self.circuit.get_variables())

            # call func to follow updates
            call_func(psi_run, score_list, None, iter=step)

            # Print the loss every 10 epochs
            if verbose:  
                print(f'Iteration {step + 1} in {time() - epochs_start:.2f}s, objective: {score}')
                epochs_start = time()

            # rel_error = 
            # if rel_error < tol

        if verbose:
            print(f'Runtime: {time() - start:.2f}s')

        dict_result = {
            'objective': np.array(score_list),
            'runtime': time() - start
        }

        if track_params:
            dict_result['params'] = params_list

        self.last_result = dict_result

        return dict_result
