import torch
import numpy as np
import math
from typing import List, Iterable, Dict, Any, Callable, Tuple
from abc import ABC, abstractmethod

from .utils import BosonOperatorsTorch, irescale, iregroup
from .mpdo_torch import MPDOtorch
from .svd_trunc import svd_trunc
from .circuit_gates import CircuitGate, NonlinearCouplingGate, NonlinearLocalGate, PhaseGate


class CircuitTopology:

    def __init__(self, gates: List[List[CircuitGate|None]]):
        self.gates = gates
        self.num_layers = len(gates)
        self.num_channels = 0 if self.num_layers==0 else len(gates[0])

        if self.num_channels == 0:
            self.Nmax = 0
            self.d = self.Nmax + 1
            self.device = None
            return None

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
        The circuit to run on MPDO

    """

    def __init__(
            self, 
            circuit_topology: CircuitTopology, 
            K_ops: List[Iterable[torch.Tensor]]=None, 
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
            if self.K_ops is not None and self.K_ops[d] is not None:
                rho.krauss_dissipation(self.K_ops[d], options=options)
                        

    def get_variables(self, flatten: bool=True):

        """Get the differentiable parameters (J couplings) to pass to optimizer (Adam)"""

        params = [
            [
                self.circuit_topology[d,l].get_params() for l in range(self.num_channels)
                if self.circuit_topology[d,l] is not None
            ]
            for d in range(self.num_layers)    
        ]
        
        if flatten:
             return [p for psub in params for psubsub in psub for p in psubsub]
        
        return params
    
    
class CouplerCircuit(MPDOCircuit):

    """
    The Nonlinear optical circuit with stacked layers of couplers
    """

    def __init__(
            self, 
            circuit_topology: CircuitTopology, 
            K_ops: List[Iterable[torch.Tensor]]=None, 
            ):
        
        super().__init__(circuit_topology, K_ops)


    def update(
            self, 
            J_matrix: List[List[torch.Tensor|None]], 
            phase_matrix: List[List[torch.Tensor|None]]=None, 
            U: torch.Tensor|None=None):

        """Perform circuit update by updating all gate (after gradient update J-values)"""

        for d, gates_layer in enumerate(self.circuit_topology):
            for l, gate in enumerate(gates_layer):
                if isinstance(gate, NonlinearCouplingGate): 
                    gate.update(J_matrix[d][l], U)
                if isinstance(gate, PhaseGate): 
                    gate.update(phase_matrix[d][l])
                if isinstance(gate, NonlinearLocalGate) and U is not None: 
                    gate.update(U)


    def get_J_matrix(self, float_vals: bool=True):
        
        J_matrix = [
            [
                self.circuit_topology[d,l].J # * self.circuit_topology[d,l].dt
                if isinstance(self.circuit_topology[d,l], NonlinearCouplingGate)
                else None
                for l in range(self.num_channels)
            ]
            for d in range(self.num_layers)
        ]

        if float_vals:
            J_matrix = [
                [float(J.detach().squeeze()) if J is not None else None for J in J_layer] 
                for J_layer in J_matrix
                ]

        return J_matrix
    

    def get_phase_matrix(self, float_vals: bool=True):
        
        phi_matrix = [
            [
                self.circuit_topology[d,l].phi # * self.circuit_topology[d,l].dt
                if isinstance(self.circuit_topology[d,l], PhaseGate)
                else None
                for l in range(self.num_channels)
            ]
            for d in range(self.num_layers)
        ]

        if float_vals:
            phi_matrix = [
                [float(phi.detach().squeeze()) if phi is not None else None for phi in phi_layer] 
                for phi_layer in phi_matrix
                ]

        return phi_matrix
    

    def get_coupling_matrix(self):
        """
        The couplings J * dt, float values by default (not torch tensors)
        """

        return [
            [
                self.circuit_topology[d,l].J.item() * self.circuit_topology[d,l].dt
                if isinstance(self.circuit_topology[d,l], NonlinearCouplingGate)
                else None
                for l in range(self.num_channels)
            ]
            for d in range(self.num_layers)
        ]
    

class PhaseCircuit(MPDOCircuit):

    """
    The Nonlinear optical circuit with stacked layers of couplers
    """

    def __init__(
            self, 
            circuit_topology: CircuitTopology, 
            K_ops: List[Iterable[torch.Tensor]]=None, 
            ):
        
        super().__init__(circuit_topology, K_ops)


    def update(self, phase_matrix: List[List[torch.Tensor|None]], U: torch.Tensor|None=None):

        """Perform circuit update by updating all gate phases"""

        for d, gates_layer in enumerate(self.circuit_topology):
            for l, gate in enumerate(gates_layer):
                gate.update(phase_matrix[d][l])


    def get_phi_matrix(self, float_vals: bool=True):
        
        phi_matrix = [
            [
                self.circuit_topology[d,l].phi.item() if float_vals
                else self.circuit_topology[d,l].phi
                for l in range(self.num_channels)
            ]
            for d in range(self.num_layers)
        ]

        return phi_matrix
    