"""
mpdo_circuit.py
---------------
Circuit classes for executing sequences of gates on an MPDOtorch state.

Hierarchy
---------
CircuitTopology     : 2-D grid of CircuitGate objects indexed by (layer, site)
MPDOCircuit         : base circuit — iterates layers, applies gates and Kraus ops
├── CouplerCircuit  : extends base with J-coupling update and readout helpers
└── PhaseCircuit    : extends base with phase-matrix update and readout helper
"""

from typing import Any, Dict, Iterable, List, Tuple

import torch

from .utils import irescale, iregroup
from .mpdo_torch import MPDOtorch
from .circuit_gates import CircuitGate, NonlinearCouplingGate, NonlinearLocalGate, PhaseGate


# ---------------------------------------------------------------------------
# CircuitTopology
# ---------------------------------------------------------------------------

class CircuitTopology:
    """
    2-D grid of circuit gates indexed by (layer d, site l).

    Wraps a list-of-lists of CircuitGate (or None for empty positions).
    Metadata (Nmax, d, device) is inferred from the first non-None gate
    found in the grid.

    Parameters
    ----------
    gates : List[List[CircuitGate | None]]
        Outer index = layer; inner index = site within the layer.
        None at position (d, l) means no gate is applied there.
    """

    def __init__(self, gates: List[List["CircuitGate | None"]]):
        """Store the gate grid and infer shared metadata from the first non-None gate."""
        self.gates        = gates
        self.num_layers   = len(gates)
        self.num_channels = 0 if self.num_layers == 0 else len(gates[0])

        if self.num_channels == 0:
            self.Nmax, self.d, self.device = 0, 1, None
            return

        # Infer shared metadata from the first non-None gate
        g = next(gate for layer in gates for gate in layer if gate is not None)
        self.Nmax   = g.Nmax
        self.d      = self.Nmax + 1
        self.device = g.device

    def __getitem__(
        self,
        dl: "Tuple[int, int] | int",
    ) -> "CircuitGate | List[CircuitGate | None]":
        """
        Access gates by layer index or (layer, site) tuple.

        Parameters
        ----------
        dl : int  → returns the full list of gates in layer d.
             (int, int) → returns the gate at (layer d, site l), or None.

        Returns
        -------
        CircuitGate | None | List[CircuitGate | None]

        Raises
        ------
        TypeError
            If the index is neither an int nor a length-2 tuple.
        """
        if isinstance(dl, int):
            return self.gates[dl]
        if isinstance(dl, tuple) and len(dl) == 2:
            d, l = dl
            return self.gates[d][l]
        raise TypeError("Index must be int (layer) or 2-tuple (layer, site).")


# ---------------------------------------------------------------------------
# MPDOCircuit — base class
# ---------------------------------------------------------------------------

class MPDOCircuit:
    """
    Runs a sequence of circuit layers on an MPDOtorch state.

    Iterates over all layers in ``circuit_topology``, applies each non-None
    gate left-to-right within the layer, then applies the corresponding
    Kraus dissipation (if any) to the full state.

    Parameters
    ----------
    circuit_topology : CircuitTopology — the 2-D gate grid.
    K_ops            : per-layer Kraus operator lists.
                       K_ops[d] is a list of (d×d) Kraus matrices for layer
                       d, or None to skip dissipation in that layer.
                       Pass None to disable dissipation entirely.
    """

    def __init__(
        self,
        circuit_topology: CircuitTopology,
        K_ops:            "List[Iterable[torch.Tensor]] | None" = None,
    ):
        """Store topology and Kraus operators; copy metadata from topology."""
        self.circuit_topology = circuit_topology
        self.K_ops            = K_ops
        self.num_channels     = circuit_topology.num_channels
        self.num_layers       = circuit_topology.num_layers
        self.Nmax             = circuit_topology.Nmax
        self.d                = self.Nmax + 1
        self.device           = circuit_topology.device

    def run(
        self,
        rho:     MPDOtorch,
        options: Dict[str, Any] = {
            'max_BD': 500, 'max_PD': 500,
            'cutoff_BD': 1e-5, 'cutoff_PD': 1e-5,
        },
        verbose: bool = False,
    ) -> None:
        """
        Apply the full circuit to MPDO state ``rho`` in place.

        For each layer d, all non-None gates are applied left-to-right.
        If ``K_ops[d]`` is not None, Kraus dissipation is then applied to
        the full state.

        Parameters
        ----------
        rho     : MPDO state to evolve (modified in place).
        options : truncation options forwarded to each gate's ``apply_to``
                  and to ``krauss_dissipation``.  Keys: max_BD, max_PD,
                  cutoff_BD, cutoff_PD.
        verbose : if True, print the total photon number and tensor shapes
                  after each individual gate application.
        """
        for d, gates_layer in enumerate(self.circuit_topology):
            for l, gate in enumerate(gates_layer):
                if gate is not None:
                    gate.apply_to(rho, site=l, options=options)
                    if verbose:
                        print(f"\n iter ({d},{l}) — N_tot = {rho.number_outcomes().sum()}")
                        print([rt.shape for rt in rho._B])

            if self.K_ops is not None and self.K_ops[d] is not None:
                rho.krauss_dissipation(self.K_ops[d], options=options)

    def get_variables(self, flatten: bool = True) -> list:
        """
        Collect all differentiable parameters from the circuit topology.

        Calls ``get_params()`` on every non-None gate and aggregates the
        results.

        Parameters
        ----------
        flatten : if True (default), return a flat list suitable for
                  passing directly to a torch optimiser.
                  If False, return a nested list that mirrors the
                  (layer, site) topology structure.

        Returns
        -------
        List[torch.Tensor]
            Flat list of all trainable parameter tensors (flatten=True).
        List[List[List[torch.Tensor]]]
            Nested list mirroring topology structure (flatten=False).
        """
        params = [
            [
                self.circuit_topology[d, l].get_params()
                for l in range(self.num_channels)
                if self.circuit_topology[d, l] is not None
            ]
            for d in range(self.num_layers)
        ]
        if flatten:
            return [p for psub in params for psubsub in psub for p in psubsub]
        return params


# ---------------------------------------------------------------------------
# CouplerCircuit
# ---------------------------------------------------------------------------

class CouplerCircuit(MPDOCircuit):
    """
    Circuit of NonlinearCouplingGate (and optional NonlinearLocalGate /
    PhaseGate) layers with differentiable J couplings.

    Extends MPDOCircuit with:
    - ``update`` : push new J, phase, or U values into all gates.
    - ``get_J_matrix`` : extract J values as floats or tensors.
    - ``get_phase_matrix`` : extract phase values (if PhaseGates exist).
    - ``get_coupling_matrix`` : extract dimensionless J·Δt values.
    """

    def __init__(
        self,
        circuit_topology: CircuitTopology,
        K_ops: "List[Iterable[torch.Tensor]] | None" = None,
    ):
        """Delegate to MPDOCircuit.__init__; no additional state."""
        super().__init__(circuit_topology, K_ops)

    def update(
        self,
        J_matrix:     "List[List[torch.Tensor | None]]",
        phase_matrix: "List[List[torch.Tensor | None]] | None" = None,
        U:            "torch.Tensor | None"                    = None,
    ) -> None:
        """
        Push updated parameters into all gates.

        Called once per optimiser step to synchronise the circuit with the
        latest parameter values from the optimiser.

        Parameters
        ----------
        J_matrix     : new coupling tensors per (layer, site).
                       None entries are skipped (gate left unchanged).
        phase_matrix : new phase tensors per (layer, site).
                       None → PhaseGates are not updated this call.
        U            : new shared nonlinearity rate applied to all
                       NonlinearCouplingGate and NonlinearLocalGate
                       instances.  None → U is not updated.
        """
        for d, gates_layer in enumerate(self.circuit_topology):
            for l, gate in enumerate(gates_layer):
                if isinstance(gate, NonlinearCouplingGate):
                    gate.update(J_matrix[d][l], U)
                if isinstance(gate, PhaseGate) and phase_matrix is not None:
                    gate.update(phase_matrix[d][l])
                if isinstance(gate, NonlinearLocalGate) and U is not None:
                    gate.update(U)

    def get_J_matrix(self, float_vals: bool = True) -> list:
        """
        Extract J-coupling values from all NonlinearCouplingGate layers.

        Non-coupler positions return None.

        Parameters
        ----------
        float_vals : if True (default), detach each J tensor and return it
                     as a Python float.
                     If False, return the raw ``torch.Tensor`` with its
                     ``requires_grad`` intact (needed for optimisation).

        Returns
        -------
        List[List[float | None]]
            When ``float_vals=True``.
        List[List[torch.Tensor | None]]
            When ``float_vals=False``.
        """
        J_matrix = [
            [
                self.circuit_topology[d, l].J
                if isinstance(self.circuit_topology[d, l], NonlinearCouplingGate)
                else None
                for l in range(self.num_channels)
            ]
            for d in range(self.num_layers)
        ]
        if float_vals:
            return [
                [float(J.detach().squeeze()) if J is not None else None for J in row]
                for row in J_matrix
            ]
        return J_matrix

    def get_phase_matrix(self, float_vals: bool = True) -> list:
        """
        Extract phase values from all PhaseGate positions.

        Non-PhaseGate positions return None.

        Parameters
        ----------
        float_vals : if True (default), detach and return as Python float.
                     If False, return the raw tensor.

        Returns
        -------
        List[List[float | None]] or List[List[torch.Tensor | None]]
        """
        phi_matrix = [
            [
                self.circuit_topology[d, l].phi
                if isinstance(self.circuit_topology[d, l], PhaseGate)
                else None
                for l in range(self.num_channels)
            ]
            for d in range(self.num_layers)
        ]
        if float_vals:
            return [
                [float(phi.detach().squeeze()) if phi is not None else None for phi in row]
                for row in phi_matrix
            ]
        return phi_matrix

    def get_coupling_matrix(self) -> list:
        """
        Dimensionless gate angles J·Δt for all NonlinearCouplingGate layers.

        This is the quantity that appears in the circuit diagram and is
        directly comparable across different time-step sizes.

        Returns
        -------
        List[List[float | None]]
            J·dt at each (layer, site); None for non-coupler positions.
        """
        return [
            [
                self.circuit_topology[d, l].J.item() * self.circuit_topology[d, l].dt
                if isinstance(self.circuit_topology[d, l], NonlinearCouplingGate)
                else None
                for l in range(self.num_channels)
            ]
            for d in range(self.num_layers)
        ]


# ---------------------------------------------------------------------------
# PhaseCircuit
# ---------------------------------------------------------------------------

class PhaseCircuit(MPDOCircuit):
    """
    Circuit composed exclusively of PhaseGate layers.

    Extends MPDOCircuit with a phase-matrix update method and a readout
    helper for extracting the current phase values.
    """

    def __init__(
        self,
        circuit_topology: CircuitTopology,
        K_ops: "List[Iterable[torch.Tensor]] | None" = None,
    ):
        """Delegate to MPDOCircuit.__init__; no additional state."""
        super().__init__(circuit_topology, K_ops)

    def update(
        self,
        phase_matrix: "List[List[torch.Tensor | None]]",
        U: "torch.Tensor | None" = None,
    ) -> None:
        """
        Push updated phase values into all PhaseGate layers.

        Parameters
        ----------
        phase_matrix : new phase tensors per (layer, site).
                       None entries are skipped.
        U            : unused; present for API symmetry with
                       ``CouplerCircuit.update``.
        """
        for d, gates_layer in enumerate(self.circuit_topology):
            for l, gate in enumerate(gates_layer):
                if gate is not None:
                    gate.update(phase_matrix[d][l])

    def get_phi_matrix(self, float_vals: bool = True) -> list:
        """
        Extract phase values from all PhaseGate layers.

        Parameters
        ----------
        float_vals : if True (default), return each phase as a Python
                     float.  If False, return the raw tensor with its
                     ``requires_grad`` flag intact.

        Returns
        -------
        List[List[float]]
            When ``float_vals=True``.
        List[List[torch.Tensor]]
            When ``float_vals=False``.
        """
        return [
            [
                self.circuit_topology[d, l].phi.item()
                if float_vals
                else self.circuit_topology[d, l].phi
                for l in range(self.num_channels)
            ]
            for d in range(self.num_layers)
        ]
