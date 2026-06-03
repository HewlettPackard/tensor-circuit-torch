"""
create_circuit.py
-----------------
Factory functions for constructing common MPDO circuit configurations.

Public API
----------
create_nonlinear_mzi_circuit        : alternating phase / coupler MZI topology
create_nonlinear_photonic_circuit   : brick-like nonlinear photonic circuit
create_nonlinear_bosonic_TEBD_step  : second-order Suzuki–Trotter TEBD step
create_phase_circuit                : pure phase-shift circuit
create_haar_random_circuit          : Haar-random benchmark circuit
amplitude_damping_kraus_ops         : amplitude-damping Kraus operators
qubit_kraus_ops                     : qubit depolarising Kraus operators
dephasing_krauss_ops                : qubit dephasing Kraus operators
"""

import warnings
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.special import factorial

from src.mpdo_circuit import CouplerCircuit, CircuitTopology, MPDOCircuit, PhaseCircuit
from src.circuit_gates import (
    HaarCouplingGate,
    NonlinearCouplingGate,
    NonlinearLocalGate,
    PhaseGate,
)
from src.utils import BosonOperatorsTorch, QubitOperatorTorch


# ---------------------------------------------------------------------------
# MZI-structured circuit
# ---------------------------------------------------------------------------

def create_nonlinear_mzi_circuit(
    num_layers:         int,
    num_channels:       int,
    J:                  "List[List[float | None]] | float",
    U:                  float,
    Nmax:               int,
    phase:              "List[List[float | None]] | float" = 0.,
    Delta:              float                              = 0.,
    dt:                 "float | List[float]"              = 1.,
    gamma:              float                              = 0.,
    dt_kraus:           "List[float] | None"               = None,
    order_kraus:        int                                = 1,
    requires_grad_J:    bool                               = True,
    requires_grad_phase: bool                              = False,
    right_stop:         "int | None"                       = None,
    device:             str                                = "cuda:0",
) -> CouplerCircuit:
    """
    Create a nonlinear MZI-structured photonic circuit.

    Layout (per coupler layer d):
        phase layer d  →  coupler layer d  →  …  →  (final phase layer)

    For `num_layers` coupler layers there are `num_layers + 1` phase layers.
    Non-covered boundary modes receive local (Kerr + detuning) gates.

    Parameters
    ----------
    num_layers         : number of coupler layers
    num_channels       : number of waveguide modes
    J                  : coupling strengths — float (uniform) or
                         List[List[float|None]] with shape [num_layers][num_channels]
    U                  : Kerr nonlinear rate (same for all gates)
    Nmax               : Fock-space truncation
    phase              : phase values — float or List[List[float|None]]
                         with shape [num_layers+1][num_channels]
    Delta              : on-site detuning (same for all gates)
    dt                 : gate duration(s); scalar or per-layer list
    gamma              : amplitude-damping rate; 0 → lossless
    dt_kraus           : per-layer Kraus time steps (defaults to dt)
    order_kraus        : expansion order for amplitude-damping Kraus operators
    requires_grad_J    : make J values differentiable
    requires_grad_phase: make phase values differentiable
    right_stop         : rightmost channel (exclusive) for coupler gates
    device             : Torch device string

    Returns
    -------
    CouplerCircuit
    """
    if not isinstance(dt, list):
        dt = [dt] * num_layers

    right_stop = num_channels if right_stop is None else right_stop

    # Kraus operators per coupler layer (None for lossless layers)
    if gamma > 1e-4:
        dt_kraus = dt if dt_kraus is None else dt_kraus
        K_ops_per_layer = [
            None if dt_d is None
            else amplitude_damping_kraus_ops(gamma * dt_d, Nmax=Nmax, order=order_kraus, device=device)
            for dt_d in dt_kraus
        ]
    else:
        K_ops_per_layer = [None] * num_layers

    circuit_topology = [[None] * num_channels for _ in range(num_layers)]
    U_t     = torch.tensor([U],     device=device)
    Delta_t = torch.tensor([Delta], device=device)

    for flat_d in range(num_layers):

        if J[flat_d] is None:
            # ---- Phase layer ----
            for ch in range(num_channels):
                phi = phase[flat_d][ch]
                if phi is not None:
                    circuit_topology[flat_d][ch] = PhaseGate(
                        Nmax=Nmax,
                        phi=torch.tensor([phi], dtype=torch.float64,
                                         device=device, requires_grad=requires_grad_phase),
                        dt=dt[flat_d], device=device,
                    )
        else:
            # ---- Coupler layer ----
            occupied_channels = []
            for ch in range(right_stop - 1):
                j_val = J[flat_d][ch]
                if j_val is not None:
                    circuit_topology[flat_d][ch] = NonlinearCouplingGate(
                        Nmax=Nmax,
                        J=torch.tensor([j_val], dtype=torch.float64,
                                        device=device, requires_grad=requires_grad_J),
                        U=U_t, Delta=Delta_t,
                        dt=dt[flat_d], device=device,
                    )
                    occupied_channels += [ch, ch + 1]

            # Guard against doubly-occupied modes (only 2-mode connectors allowed)
            if len(set(occupied_channels)) != len(occupied_channels):
                warnings.warn(
                    f"Layer {flat_d}: doubly occupied modes {occupied_channels}"
                )

            # Give non-covered modes a local Kerr gate
            for ch in range(num_channels):
                if ch not in occupied_channels:
                    circuit_topology[flat_d][ch] = NonlinearLocalGate(
                        Nmax=Nmax, U=U_t, Delta=Delta_t,
                        dt=dt[flat_d], device=device,
                    )

    return CouplerCircuit(
        circuit_topology=CircuitTopology(circuit_topology),
        K_ops=K_ops_per_layer,
    )


# ---------------------------------------------------------------------------
# Brick-like nonlinear photonic circuit
# ---------------------------------------------------------------------------

def create_nonlinear_photonic_circuit(
    num_layers:   int,
    num_channels: int,
    J:            "List[List[float | None]] | float",
    U:            float,
    Nmax:         int,
    Delta:        float                = 0.,
    dt:           "float | List[float]" = 1.,
    gamma:        float                = 0.,
    device:       str                  = "cuda:0",
    dt_kraus:     "List[float] | None" = None,
    order_kraus:  int                  = 1,
    requires_grad: bool                = True,
    right_stop:   "int | None"         = None,
) -> CouplerCircuit:
    """
    Create a nonlinear photonic circuit with a brick-like coupler topology.

    Gates are placed in an alternating stagger pattern: even-index sites in
    even layers, odd-index sites in odd layers.  Boundary modes that are not
    covered by a two-mode gate receive a local ``NonlinearLocalGate``.

    Parameters
    ----------
    num_layers    : circuit depth
    num_channels  : number of waveguide modes
    J             : coupling strength — float (uniform brick pattern) or
                    List[List[float|None]] per (layer, site)
    U             : Kerr nonlinearity rate
    Nmax          : Fock-space truncation
    Delta         : on-site detuning
    dt            : gate duration(s); scalar or per-layer list
    gamma         : amplitude-damping rate per unit time; 0 → lossless
    device        : Torch device string
    dt_kraus      : per-layer Kraus time steps (defaults to dt)
    order_kraus   : expansion order for amplitude-damping Kraus operators
    requires_grad : make all J couplings differentiable
    right_stop    : rightmost channel index (exclusive); default = num_channels

    Returns
    -------
    CouplerCircuit
    """
    if not isinstance(dt, list):
        dt = [dt] * num_layers

    right_stop = num_channels if right_stop is None else right_stop

    # Build per-layer Kraus operators
    if gamma > 1e-4:
        dt_kraus = dt if dt_kraus is None else dt_kraus
        K_ops = [
            None if dt_d is None
            else amplitude_damping_kraus_ops(
                gamma * dt_d, Nmax=Nmax, order=order_kraus, device=device
            )
            for dt_d in dt_kraus
        ]
    else:
        K_ops = None

    circuit_topology = [[None] * num_channels for _ in range(num_layers)]

    for d in range(num_layers):
        for l in range(right_stop - 1):
            if isinstance(J, float):
                # Uniform J: staggered brick pattern
                if (d + l) % 2 == 0:
                    circuit_topology[d][l] = NonlinearCouplingGate(
                        Nmax=Nmax,
                        J=torch.tensor([J], requires_grad=requires_grad, device=device),
                        U=torch.tensor([U], device=device),
                        Delta=torch.tensor([Delta], device=device),
                        dt=dt[d], device=device,
                    )
            else:
                # Per-site J values
                J_init = J[d][l]
                if J_init is not None:
                    circuit_topology[d][l] = NonlinearCouplingGate(
                        Nmax=Nmax,
                        J=torch.tensor([J_init], requires_grad=True, device=device),
                        U=torch.tensor([U], device=device),
                        Delta=torch.tensor([Delta], device=device),
                        dt=dt[d], device=device,
                    )

        # Left boundary: local gate on odd layers
        if d % 2 == 1:
            circuit_topology[d][0] = NonlinearLocalGate(
                Nmax=Nmax,
                U=torch.tensor([U], device=device),
                Delta=torch.tensor([Delta], device=device),
                dt=dt[d], device=device,
            )

        # Right boundary: local gate when the pattern leaves the rightmost mode uncovered
        if ((d % 2) + right_stop) % 2 == 1:
            circuit_topology[d][right_stop - 1] = NonlinearLocalGate(
                Nmax=Nmax,
                U=torch.tensor([U], device=device),
                Delta=torch.tensor([Delta], device=device),
                dt=dt[d], device=device,
            )

    return CouplerCircuit(
        circuit_topology=CircuitTopology(circuit_topology),
        K_ops=K_ops,
    )


# ---------------------------------------------------------------------------
# TEBD step (Suzuki–Trotter, second order)
# ---------------------------------------------------------------------------

def create_nonlinear_bosonic_TEBD_step(
    num_channels: int,
    J:            "List[List[float | None]] | float",
    U:            float,
    Delta:        float,
    Nmax:         int,
    dt:           "float | List[float]" = 1.,
    gamma:        float                 = 0.,
    order_kraus:  int                   = 1,
    device:       str                   = "cuda:0",
    requires_grad: bool                 = False,
) -> CouplerCircuit:
    """
    One second-order Suzuki–Trotter TEBD time step.

    Implements the symmetric splitting dt/2 → dt → dt/2 with Kraus dissipation
    applied only at the full-dt centre layer:

        U(dt) ≈ U_free(dt/2) · U_int(dt) · U_free(dt/2) · K(dt)

    Parameters
    ----------
    num_channels  : number of spatial modes
    J             : hopping rate (uniform or per-site)
    U             : Kerr nonlinearity
    Delta         : on-site detuning
    Nmax          : Fock-space truncation
    dt            : time step size
    gamma         : single-photon loss rate
    order_kraus   : amplitude-damping expansion order
    device        : Torch device string
    requires_grad : make J differentiable (default False for TEBD)

    Returns
    -------
    CouplerCircuit representing one TEBD step
    """
    return create_nonlinear_photonic_circuit(
        num_layers   = 3,
        num_channels = num_channels,
        J=J, U=U, Delta=Delta, Nmax=Nmax,
        gamma        = gamma,
        device       = device,
        dt           = [dt / 2., dt, dt / 2.],
        dt_kraus     = [None, None, dt],   # dissipation only at full step
        order_kraus  = order_kraus,
        requires_grad= requires_grad,
    )


# ---------------------------------------------------------------------------
# Phase circuit
# ---------------------------------------------------------------------------

def create_phase_circuit(
    Nmax:          int,
    phase:         "List[List[float | None]] | float",
    num_layers:    "int | None"              = None,
    num_channels:  "int | None"              = None,
    dt:            float                     = 1.,
    K_ops:         "List[torch.Tensor] | None" = None,
    options:       Dict[str, Any]            = {
        'max_BD': 100, 'max_PD': 100,
        'cutoff_BD': 1e-6, 'cutoff_PD': 1e-6,
    },
    requires_grad: "List[List[bool]] | bool | None" = None,
    device:        str                     = "cuda:0",
) -> PhaseCircuit:
    """
    Create a circuit consisting purely of PhaseGate layers.

    Parameters
    ----------
    Nmax         : Fock-space truncation
    phase        : phase values — float (uniform) or
                   List[List[float|None]] with shape [num_layers][num_channels]
    num_layers   : inferred from `phase` if None
    num_channels : inferred from `phase` if None
    dt           : gate duration
    K_ops        : optional Kraus operators (rarely used for phase circuits)
    options      : MPDO truncation options (unused here; for API consistency)
    requires_grad: per-element differentiability flags matching `phase` shape
    device       : Torch device string

    Returns
    -------
    PhaseCircuit
    """
    if num_layers is None and num_channels is None:
        num_channels = len(phase[0])
        num_layers   = len(phase)

    circuit_topology = [[None] * num_channels for _ in range(num_layers)]

    for d in range(num_layers):
        for l in range(num_channels):
            if isinstance(phase, float):
                circuit_topology[d][l] = PhaseGate(
                    Nmax=Nmax,
                    phi=torch.tensor(
                        [phase], device=device,
                        requires_grad=requires_grad[d][l],
                    ),
                    dt=dt, device=device,
                )
            else:
                phase_init = phase[d][l]
                if phase_init is not None:
                    circuit_topology[d][l] = PhaseGate(
                        Nmax=Nmax,
                        phi=torch.tensor(
                            phase_init, device=device,
                            requires_grad=requires_grad[d][l],
                        ),
                        dt=dt, device=device,
                    )

    return PhaseCircuit(
        circuit_topology=CircuitTopology(circuit_topology),
        K_ops=K_ops,
    )


# ---------------------------------------------------------------------------
# Haar-random circuit
# ---------------------------------------------------------------------------

def create_haar_random_circuit(
    num_layers:   int,
    num_channels: int,
    Nmax:         int   = 1,   # default: qubit
    dt:           float = 1.,
    gamma:        float = 0.,
    device:       str   = "cuda:0",
) -> MPDOCircuit:
    """
    Create a Haar-random qubit (or bosonic) circuit for benchmarking.

    Gates are placed in a staggered brick pattern; dephasing Kraus operators
    are applied when gamma > 0.

    Parameters
    ----------
    num_layers   : circuit depth
    num_channels : number of modes
    Nmax         : Fock-space truncation (1 = qubit)
    dt           : gate duration (unused for Haar gates, kept for API symmetry)
    gamma        : qubit dephasing rate; 0 → unitary circuit
    device       : Torch device string

    Returns
    -------
    MPDOCircuit
    """
    K_ops = None if gamma < 1e-4 else dephasing_krauss_ops(gamma, device=device)

    circuit_topology = [[None] * num_channels for _ in range(num_layers)]
    for d in range(num_layers):
        for l in range(num_channels - 1):
            if (d + l) % 2 == 0:
                circuit_topology[d][l] = HaarCouplingGate(Nmax=Nmax, device=device)

    return MPDOCircuit(
        circuit_topology=CircuitTopology(circuit_topology),
        K_ops=K_ops,
    )


# ---------------------------------------------------------------------------
# Kraus operator constructors
# ---------------------------------------------------------------------------

def amplitude_damping_kraus_ops(
    kappa:      float,
    Nmax:       int,
    order:      int = 1,
    device:     str = 'cpu',
) -> List[torch.Tensor]:
    """
    Amplitude-damping Kraus operators up to photon-loss order `order`.

    K_0 = diag((1-κ)^{n/2}),   K_n = (√κ)^n / √n! · K_0 · a^n

    Parameters
    ----------
    kappa  : loss probability per time step (0 ≤ κ ≤ 1)
    Nmax   : Fock-space truncation
    order  : maximum number of photon-loss events per step
    device : Torch device string

    Returns
    -------
    List of (d×d) complex-128 Kraus matrices (length = order + 1)
    """
    ops = BosonOperatorsTorch(Nmax, device=device)

    # No-loss operator: diag((1-κ)^{n/2})
    no_loss = torch.diag(
        (1. - kappa) ** (torch.arange(0, Nmax + 1, device=device) / 2.)
    ).to(torch.complex128)

    K_ops = [no_loss]
    for n in range(1, order + 1):
        K_ops.append(
            (np.sqrt(kappa) ** n / np.sqrt(factorial(n)))
            * no_loss @ torch.matrix_power(ops.a, n)
        )

    return K_ops


def qubit_kraus_ops(
    gamma:        float,
    device:       str  = 'cpu',
    is_hermitian: bool = False,
) -> List[torch.Tensor]:
    """
    Qubit depolarising Kraus operators.

    Parameters
    ----------
    gamma        : error rate per step
    device       : Torch device string
    is_hermitian : if True, use the Hermitian (symmetric) decomposition
                   {id, σx, σz}; otherwise use {id, σ⁻, σ⁺, P0, P1}

    Returns
    -------
    List of (2×2) complex-128 Kraus matrices
    """
    ops = QubitOperatorTorch(device=device)

    if is_hermitian:
        return [
            np.sqrt(1. - 2. * gamma) * ops.id,
            np.sqrt(gamma) * ops.sx,
            np.sqrt(gamma) * ops.sz,
        ]
    return [
        np.sqrt(1. - 2. * gamma)  * ops.id,
        np.sqrt(gamma / 2.)       * ops.sm,
        np.sqrt(gamma / 2.)       * ops.sp,
        np.sqrt(gamma / 2.)       * ops.P1,
        -np.sqrt(gamma / 2.)      * ops.P0,
    ]


def dephasing_krauss_ops(
    gamma:  float,
    device: str = 'cpu',
) -> List[torch.Tensor]:
    """
    Qubit dephasing (Z-basis) Kraus operators: {√(1-γ) I, √γ P₀, √γ P₁}.

    Parameters
    ----------
    gamma  : dephasing probability per step
    device : Torch device string

    Returns
    -------
    List of three (2×2) complex-128 Kraus matrices
    """
    ops = QubitOperatorTorch(device=device)
    return [
        np.sqrt(1. - gamma) * ops.id,
        np.sqrt(gamma)      * ops.P0,
        np.sqrt(gamma)      * ops.P1,
    ]
