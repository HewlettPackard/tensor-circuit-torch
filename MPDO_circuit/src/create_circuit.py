import torch
import numpy as np
import warnings
from scipy.special import factorial
from typing import List, Dict, Any
from src.mpdo_circuit import MPDOCircuit, CouplerCircuit, PhaseCircuit, CircuitTopology
from src.circuit_gates import NonlinearCouplingGate, NonlinearLocalGate, PhaseGate, HaarCouplingGate
from src.utils import BosonOperatorsTorch, QubitOperatorTorch, sqrtm, eye_like


def create_nonlinear_mzi_circuit(
        num_layers: int,
        num_channels: int,
        J: List[List[float | None]] | float,
        U: float,
        Nmax: int,
        phase: List[List[float | None]] | float=0.,
        Delta: float = 0.,
        dt: float | List[float] = 1.,
        gamma: float = 0.,
        dt_kraus: List[float] | None = None,
        order_kraus: int = 1,
        requires_grad_J: bool = True,
        requires_grad_phase: bool = False,
        right_stop: int | None = None,
        device: str = "cuda:0"
) -> MPDOCircuit:
    """
    Create a nonlinear MZI-structured photonic circuit:
        phase layer -> coupler layer -> phase layer -> coupler layer -> ... -> phase layer

    For num_layers coupler layers, there are num_layers + 1 phase layers.
    Coupler layers follow the same brick-like topology as create_nonlinear_photonic_circuit,
    with nonlinearity U and detuning Delta passed through to each gate.

    Parameters
    ----------
    num_layers : int
        Number of coupler layers.
    num_channels : int
        Number of modes.
    J : List[List[float | None]] | float
        Coupling strengths. If float, same J everywhere (brick pattern).
        If list of lists, shape [num_layers][num_channels], None skips that gate.
    phase : List[List[float | None]] | float
        Phase values. If float, same phase everywhere.
        If list of lists, shape [num_layers + 1][num_channels], None skips that gate.
    U : float
        On-site nonlinear interaction strength (Kerr-type).
    Nmax : int
        Fock space truncation.
    Delta : float
        On-site detuning.
    dt : float | List[float]
        Timestep(s), one per coupler layer or a single shared value.
    gamma : float
        Amplitude damping rate. 0 means no dissipation.
    dt_kraus : List[float] | None
        Timesteps for Kraus operators; defaults to dt if None.
    order_kraus : int
        Order of amplitude damping Kraus expansion.
    requires_grad_J : bool
        Whether coupler parameters are differentiable.
    requires_grad_phase : bool
        Whether phase parameters are differentiable.
    right_stop : int | None
        Rightmost channel index (exclusive) for coupler gates. Defaults to num_channels.
    device : str
        Torch device string.

    Returns
    -------
    MPDOCircuit
        The MZI-structured nonlinear circuit.
    """


    if not isinstance(dt, list):
        dt = [dt] * num_layers

    right_stop = num_channels if right_stop is None else right_stop

    # Kraus operators — one per coupler layer (None for pure layers)
    if gamma > 1e-4:
        dt_kraus = dt if dt_kraus is None else dt_kraus
        K_ops_per_layer = [
            None if dt_d is None else
            amplitude_damping_kraus_ops(gamma * dt_d, Nmax=Nmax, order=order_kraus, device=device)
            for dt_d in dt_kraus
        ]
    else:
        K_ops_per_layer = [None] * num_layers

    circuit_topology = [[None for _ in range(num_channels)] for _ in range(num_layers)]

    U_t     = torch.tensor([U],     device=device)
    Delta_t = torch.tensor([Delta], device=device)

    for flat_d in range(num_layers):

        # ---- PHASE LAYER ------------------------------------------------
        if J[flat_d] == None:

            for ch in range(num_channels):
                phi = phase[flat_d][ch]
                if phi is not None:
                    circuit_topology[flat_d][ch] = PhaseGate(
                        Nmax=Nmax,
                        phi=torch.tensor([phi], dtype=torch.float64,
                                         device=device, requires_grad=requires_grad_phase),
                        dt=dt[flat_d],
                        device=device
                    )


        # ---- COUPLER LAYER ----------------------------------------------
        else:

            for ch in range(right_stop - 1):
                occupied_channels = []
                j_val = J[flat_d][ch]
                
                if j_val is not None:
                    circuit_topology[flat_d][ch] = NonlinearCouplingGate(
                        Nmax=Nmax,
                        J=torch.tensor([j_val], dtype=torch.float64,
                                        device=device, requires_grad=requires_grad_J),
                        U=U_t,
                        Delta=Delta_t,
                        dt=dt[flat_d],
                        device=device
                    )

                    occupied_channels += [ch, ch + 1] # the connected channels, occupied by gate

            # check for duplicates (doubly occupied sites, only 2-mode connectors allowed)
            if len(set(occupied_channels)) != len(occupied_channels):
                warnings.warn(f"J-list {flat_d} contains doubly occupied modes: {occupied_channels}")
            
            # ensure non-occupied modes receive local nonlinearity
            for ch in range(num_channels):
                if ch not in occupied_channels:
                    circuit_topology[flat_d][ch] = NonlinearLocalGate(
                        Nmax=Nmax, U=U_t, Delta=Delta_t, dt=dt[flat_d], device=device
                    )

    return CouplerCircuit(
        circuit_topology=CircuitTopology(circuit_topology),
        K_ops=K_ops_per_layer,
    )


def create_nonlinear_photonic_circuit(
        num_layers: int, 
        num_channels: int, 
        J: List[List[float]|None]|float, 
        U: float,
        Nmax: int,
        Delta: float = 0.,
        dt: float|List[float] = 1.,
        gamma: float = 0.,
        device: str = "cuda:0",
        dt_kraus: List[float]|None=None,
        order_kraus: int=1,
        requires_grad: bool=True,
        right_stop: int|None = None
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

    if not isinstance(dt, List):
        dt = [dt] * num_layers

    # create amplitude damping Kraus operators, up to the order specified
    if gamma > 0.0001:
        dt_kraus = dt if dt_kraus is None else dt_kraus
        K_ops = [
            None if dt_d is None else
            amplitude_damping_kraus_ops(
                gamma * dt_d, 
                Nmax=Nmax, 
                order=order_kraus, 
                device=device
            ) for dt_d in dt_kraus]
    else:
        K_ops = None


    circuit_topology = [[None for _ in range(num_channels)] for _ in range(num_layers)]

    # fill the bulk gates
    right_stop = num_channels if right_stop is None else right_stop
    for d in range(num_layers):

        for l in range(right_stop-1):

            # check the type
            if isinstance(J, float):
                if (d + l) % 2 == 0:
                    circuit_topology[d][l] = NonlinearCouplingGate(
                        Nmax=Nmax, 
                        J=torch.tensor([J], requires_grad=requires_grad, device=device), 
                        U=torch.tensor([U], device=device), 
                        Delta=torch.tensor([Delta], device=device), 
                        dt=dt[d], 
                        device=device
                    )

            else:
                J_init = J[d][l]
                if J_init is not None:
                    circuit_topology[d][l] = NonlinearCouplingGate(
                        Nmax=Nmax, 
                        J=torch.tensor([J_init], requires_grad=True, device=device), 
                        U=torch.tensor([U], device=device), 
                        Delta=torch.tensor([Delta], device=device), 
                        dt=dt[d], 
                        device=device
                    )

        # left side local gate
        if d % 2 == 1:
            circuit_topology[d][0] = NonlinearLocalGate(
                Nmax=Nmax, U=torch.tensor([U], device=device), Delta=torch.tensor([Delta], device=device), dt=dt[d], device=device
            )

        # right side local gate
        if ((d % 2) + right_stop) % 2 == 1:
            circuit_topology[d][right_stop-1] = NonlinearLocalGate(
                Nmax=Nmax, U=torch.tensor([U], device=device), Delta=torch.tensor([Delta], device=device), dt=dt[d], device=device
            )

    return CouplerCircuit(
        circuit_topology=CircuitTopology(circuit_topology),
        K_ops = K_ops,
    )


def create_nonlinear_bosonic_TEBD_step(
        num_channels: int, 
        J: List[List[float]|None]|float, 
        U: float,
        Delta: float,
        Nmax: int,
        dt: float|List[float] = 1.,
        gamma: float = 0.,
        order_kraus: int=1,
        device: str = "cuda:0",
        requires_grad: bool=False
    ) -> MPDOCircuit:

    return create_nonlinear_photonic_circuit(
        num_layers=3,
        num_channels=num_channels,
        J=J, U=U, Delta=Delta, gamma=gamma, Nmax=Nmax,
        device=device, 
        dt=[dt/2., dt, dt/2.],
        dt_kraus=[None, None, dt],
        order_kraus=order_kraus,
        requires_grad=requires_grad
    )


def create_phase_circuit(
        Nmax: int,
        phase: List[List[float]|None]|float, 
        num_layers: int|None=None, 
        num_channels: int|None=None,
        dt: float = 1.,
        K_ops: List[torch.Tensor]|None = None,
        options: Dict[str, Any] = { 
            'max_BD': 100,'max_PD': 100,'cutoff_BD': 0.000001,'cutoff_PD': 0.000001
            },
        requires_grad: List[List[bool]]|bool|None=None,
        device: str = "cuda:0"
    ) -> MPDOCircuit:

    if num_layers is None and num_channels is None:
        num_channels = len(phase[0])
        num_layers = len(phase)

    circuit_topology = [[None for _ in range(num_channels)] for _ in range(num_layers)]

    # fill the bulk gates
    for d in range(num_layers):
        for l in range(num_channels):

            # check the type
            if isinstance(phase, float):
                requires_grad
                circuit_topology[d][l] = PhaseGate(
                    Nmax=Nmax, 
                    phi=torch.tensor([phase], device=device,requires_grad=requires_grad[d][l]),
                    dt=dt, 
                    device=device
                )

            else:
                phase_init = phase[d][l]
                if phase_init is not None:
                    circuit_topology[d][l] = PhaseGate(
                        Nmax=Nmax, 
                        phi=torch.tensor(phase_init, device=device, requires_grad=requires_grad[d][l]),
                        dt=dt, 
                        device=device
                    )

    return PhaseCircuit(
        circuit_topology=CircuitTopology(circuit_topology),
        K_ops = K_ops,
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
    )




def amplitude_damping_kraus_ops(kappa: float, Nmax: int, order=1, device='cpu'):
    
    # the boson operators
    ops = BosonOperatorsTorch(Nmax, device=device)

    # no photon loss operator
    no_loss = torch.diag((1. - kappa) ** (torch.arange(0, Nmax+1, device=device) / 2.)).to(torch.complex128)
    
    # list of Krauss ops
    K_ops = [no_loss]
    for n in range(1, order + 1):
        K_op_n =  ((np.sqrt(kappa) ** n / np.sqrt(factorial(n)))) * \
            no_loss @ torch.matrix_power(ops.a, n)
        
        K_ops.append(K_op_n)

    return K_ops

def qubit_kraus_ops(gamma: float, device='cpu', is_hermitian: bool=False):
    
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



        
