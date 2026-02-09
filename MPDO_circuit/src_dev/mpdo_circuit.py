import torch
import numpy as np
import math
from typing import List, Iterable, Dict, Any, Callable, Tuple
from abc import ABC, abstractmethod

from .utils import BosonOperatorsTorch, irescale, iregroup
from .mpdo_torch import MPDOtorch
from .svd_trunc import svd_trunc
from .circuit_gates import CircuitGate


class CircuitTopology:

    def __init__(self, gates: List[List[CircuitGate|None]]):
        self.gates = gates
        self.num_layers = len(gates)
        self.num_channels = len(gates[0])

        # readout gate
        for gate in gates[0]:
            if gate is not None:
                g = gate 
                break
            
        self.Nmax = g.Nmax
        self.d = self.Nmax + 1
        self.device = g.device

    def __getitem__(self, dl: Tuple[int]|int) -> CircuitGate|List[CircuitGate]:
        if isinstance(dl, int):
            return self.gates[dl]
        if isinstance(dl, tuple) and len(dl) == 2:
            d, l = dl
            return self.gates[d][l]
        else:
            raise TypeError("Index must be a tuple of two integers (d, l) or integer.")
        

class MPDOCircuit:

    """


    """

    def __init__(
            self, 
            circuit_topology: CircuitTopology, 
            K_ops: Iterable[torch.Tensor]=None, 
            #options={'max_BD': 100, 'max_PD': 100, 'cutoff_BD': 1e-6, 'cutoff_PD': 1e-6}, 
            ):
        
        """Initializer for circuit simulation class NonlinearPhotonicCircuit."""

        # copy parameters into class variables
        self.circuit_topology = circuit_topology
        self.K_ops = K_ops
        self.num_channels = circuit_topology.num_channels
        self.num_layers = circuit_topology.num_layers
        self.Nmax = circuit_topology.Nmax
        self.d = self.Nmax + 1
        self.device = self.circuit_topology.device
        # self.options = options

    
    def update(self):

        """Perform circuit update by updating all gate (after gradient update J-values)"""

        for gates_layer in self.circuit_topology:
            for gate in gates_layer:
                if gate is not None: 
                    gate.update()


    def run(self, 
            rho: MPDOtorch, 
            options: Dict[str, Any]={'max_BD': 500, 'max_PD': 500, 'cutoff_BD': 1e-5, 'cutoff_PD': 1e-5},
            verbose: bool=False
            ):
        
        """
        Run the circuit on an MPS

        Parameters
        ----------
        rho: MPDOtorch
            The MPS initial state

        options: Dict[str, Any] default {'max_BD': 100, 'cutoff': 1e-8})
            The options to run with -- max_BD and cutoff for SVs
        """

        
        # loop over circuit layers
        for d, gates_layer in enumerate(self.circuit_topology):

            # apply gates
            for l, gate in enumerate(gates_layer):
                if gate is not None:

                    gate.apply_to(rho, site=l, options=options)
                    if verbose:
                        print(f"\n iter ({d},{l})")
                        print(rho.number_outcomes().sum())
                        print([rt.shape for rt in rho])
 

            # apply Krauss operators, if requested
            if self.K_ops is not None:
                rho.krauss_dissipation(self.K_ops, options=options)
                        

    def get_variables(self, flatten: bool=True):

        """Get the differentiable parameters (J couplings) to pass to optimizer (Adam)"""

        params = [
            self.circuit_topology[d,l].get_params() for l in range(self.num_channels) for d in range(self.num_layers) 
            if self.circuit_topology[d,l] is not None
        ]
        
        if flatten:
             return [p for psub in params for p in psub]
        
        return params

