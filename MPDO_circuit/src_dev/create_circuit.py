import torch
import numpy as np
from scipy.special import factorial
from typing import List, Dict, Any
from src.mpdo_circuit import MPDOCircuit, CircuitTopology
from src.circuit_gates import NonlinearCouplingGate, NonlinearLocalGate, PhaseGate, HaarCouplingGate
from src.utils import BosonOperatorsTorch, QubitOperatorTorch, sqrtm, eye_like

def create_nonlinear_photonic_circuit(
        num_layers: int, 
        num_channels: int, 
        J: List[List[float]|None]|float, 
        U: float,
        Nmax: int,
        dt: float = 1.,
        gamma: float = 0.,
        # options: Dict[str, Any] = { 
        #     'max_BD': 100,'max_PD': 100,'cutoff_BD': 0.000001,'cutoff_PD': 0.000001
        #     },
        device: str = "cuda:0",
        order_krauss: int=1,
        requires_grad: bool=True
    ) -> MPDOCircuit:

    """
    Create a nonlinear photonic circuit with differentiable couplings. The couplings can be given as
    'float', when same coupling is assigned all couplers, or 'List[List[float|None]]', when all couplers
    recieve different coupling if not None.

    The possibility (optional) to add list of Krauss operators for dissipation.

    Per convention, the brick-like gate structure starts on the left. The side gates, with no overlapping
    two-mode gate, receive a single-mode nonlinear gate.

    Returns
    -------
    MPDOCircuit circuit
        The structured circuit with a brick-like topology as specified by the input.
    """

    # create amplitude damping Krauss operators
    K_ops = amplitude_damping_krauss_ops(
        gamma * dt, 
        Nmax=Nmax, 
        order=order_krauss, 
        device=device
        )


    circuit_topology = [[None for _ in range(num_channels)] for _ in range(num_layers)]

    # fill the bulk gates
    for d in range(num_layers):
        for l in range(num_channels):

            # check the type
            if isinstance(J, float):
                if (d + l) % 2 == 0:
                    circuit_topology[d][l] = NonlinearCouplingGate(
                        Nmax=Nmax, 
                        J=torch.tensor([J], requires_grad=requires_grad, device=device), 
                        U=torch.tensor([U], device=device), 
                        dt=dt, 
                        device=device
                    )

            else:
                J_init = J[d][l]
                if J_init is not None:
                    circuit_topology[d][l] = NonlinearCouplingGate(
                        Nmax=Nmax, 
                        J=torch.tensor([J_init], requires_grad=True, device=device), 
                        U=torch.tensor([U], device=device), 
                        dt=dt, 
                        device=device
                    )

        # left side local gate
        if d % 2 == 1:
            circuit_topology[d][0] = NonlinearLocalGate(
                Nmax=Nmax, U=torch.tensor([U], device=device), dt=dt, device=device
            )

        # right side local gate
        if ((d % 2) + num_channels) % 2 == 1:
            circuit_topology[d][-1] = NonlinearLocalGate(
                Nmax=Nmax, U=torch.tensor([U], device=device), dt=dt, device=device
            )

    return MPDOCircuit(
        circuit_topology=CircuitTopology(circuit_topology),
        K_ops = K_ops,
        # options=options,
    )


def create_phase_circuit(
        num_layers: int, 
        num_channels: int, 
        phase: List[List[float]|None]|float, 
        Nmax: int,
        dt: float = 1.,
        K_ops: List[torch.Tensor]|None = None,
        options: Dict[str, Any] = { 
            'max_BD': 100,'max_PD': 100,'cutoff_BD': 0.000001,'cutoff_PD': 0.000001
            },
        device: str = "cuda:0"
    ) -> MPDOCircuit:

    circuit_topology = [[None for _ in range(num_channels)] for _ in range(num_layers)]

    # fill the bulk gates
    for d in range(num_layers):
        for l in range(num_channels):

            # check the type
            if isinstance(phase, float):
                circuit_topology[d][l] = PhaseGate(
                    Nmax=Nmax, 
                    phi=torch.tensor([phase], device=device),
                    dt=dt, 
                    device=device
                )

            else:
                phase_init = phase[d][l]
                if phase_init is not None:
                    circuit_topology[d][l] = PhaseGate(
                        Nmax=Nmax, 
                        phi=torch.tensor([phase_init], device=device),
                        dt=dt, 
                        device=device
                    )

    return MPDOCircuit(
        circuit_topology=CircuitTopology(circuit_topology),
        K_ops = K_ops,
        options=options,
    )


def create_haar_random_circuit(
        num_layers: int, 
        num_channels: int, 
        Nmax: int=1, # default qubit
        dt: float = 1.,
        gamma: float = 0.,
        # options: Dict[str, Any] = { 
        #     'max_BD': 100,'max_PD': 100,'cutoff_BD': 0.000001,'cutoff_PD': 0.000001
        #     },
        device: str = "cuda:0"
    ) -> MPDOCircuit:

    """
    Create a Haar random circuit, with stacked random two-mode couplers.

    The possibility (optional) to add list of Krauss operators for dissipation.

    Per convention, the brick-like gate structure starts on the left. The side gates, with no overlapping
    two-mode gate, receive a single-mode nonlinear gate.

    Returns
    -------
    MPDOCircuit circuit
        The structured circuit with a brick-like topology as specified by the input.
    """

    # create amplitude damping Krauss operators
    if gamma < 1e-4 or gamma is None:
        K_ops = None
    else:
        K_ops = dephasing_krauss_ops(gamma, device=device)

    circuit_topology = [[None for _ in range(num_channels)] for _ in range(num_layers)]

    # fill the bulk gates
    for d in range(num_layers):
        for l in range(num_channels-1):
            if (d + l) % 2 == 0:
                circuit_topology[d][l] = HaarCouplingGate(Nmax=Nmax, device=device)

    return MPDOCircuit(
        circuit_topology=CircuitTopology(circuit_topology),
        K_ops = K_ops,
        # options=options,
    )


def amplitude_damping_krauss_ops(gamma: float, Nmax: int, order=1, device='cpu'):
    
    # the boson operators
    ops = BosonOperatorsTorch(Nmax, device=device)

    # no photon loss operator
    no_loss = torch.diag((1. - gamma) ** (torch.arange(0, Nmax+1, device=device) / 2.)).to(torch.complex128)
    
    # list of Krauss ops
    K_ops = [no_loss]
    for n in range(1, order + 1):
        K_op_n =  ((np.sqrt(gamma) ** n / np.sqrt(factorial(n)))) * \
            no_loss @ torch.matrix_power(ops.a, n)
        
        K_ops.append(K_op_n)

    return K_ops

def qubit_krauss_ops(gamma: float, device='cpu', is_hermitian: bool=False):
    
    # the qubit operators
    ops = QubitOperatorTorch(device=device)

    # the krauss operators
    if is_hermitian:
        K_ops = [
            np.sqrt(1. - 2 * gamma) * ops.id, # identity
            np.sqrt(gamma) * ops.sx, # spin flip (T1 error)
            np.sqrt(gamma) * ops.sz # phase flip (T2 error)
        ]
    else:
        K_ops = [
            np.sqrt(1. - 2 * gamma) * ops.id, # identity
            np.sqrt(gamma/2.) * ops.sm, # spin flip (T1 error)
            np.sqrt(gamma/2.) * ops.sp, # spin flip (T1 error)
            np.sqrt(gamma/2.) * ops.P1, # phase flip (T2 error)
            - np.sqrt(gamma/2.) * ops.P0 # phase flip (T2 error)
        ]

    return K_ops


def dephasing_krauss_ops(gamma: float, device='cpu'):
    
    ops = QubitOperatorTorch(device=device)

    return [
        np.sqrt(1. - gamma) * ops.id,
        np.sqrt(gamma) * ops.P0,
        np.sqrt(gamma) * ops.P1,
    ]



        
