"""
circuit_gates.py
----------------
Elementary gate classes for MPDO circuits.

Gate hierarchy
--------------
CircuitGate               (abstract base — shared interface)
├── CircuitGateOneSite    (single-mode gates)
│   ├── NonlinearLocalGate  — on-site Kerr nonlinearity + detuning
│   └── PhaseGate           — single-mode phase shift e^{-iφn̂}
└── CircuitGateTwoSite    (two-adjacent-mode gates with SVD bond update)
    ├── NonlinearCouplingGate — beam-splitter + Kerr interaction
    └── HaarCouplingGate      — Haar-random two-mode unitary (benchmarking)

All gate tensors are stored as rank-4 tensors of shape (d, d, d, d) for
two-site gates, or rank-2 (d, d) for single-site gates, where d = Nmax + 1.
"""

from typing import Any, Dict, Iterable, List

import torch

from src.utils import BosonOperatorsTorch, iregroup, irescale, Haar_unitary
from src.svd_trunc import svd_trunc
from src.mpdo_torch import MPDOtorch


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CircuitGate:
    """
    Abstract base class for all circuit gates.

    Subclasses must implement ``apply_to``.  The optional methods
    ``update``, ``get_params``, and ``label`` have no-op defaults.

    Parameters
    ----------
    Nmax   : Fock-space truncation; physical dimension d = Nmax + 1.
    dt     : gate duration (time step size, in ps).
    device : Torch device string.

    Class-level attributes set by subclasses
    -----------------------------------------
    H           : torch.Tensor — gate Hamiltonian H, so that
                  gate_tensor = exp(-i dt H).
    gate_tensor : torch.Tensor — precomputed gate unitary.
    is_unitary  : bool — True for all standard gates (default True).
    span        : int | None — number of modes acted on (1 or 2).
    """

    H:           torch.Tensor = None
    gate_tensor: torch.Tensor = None
    is_unitary:  bool         = True
    span:        "int | None" = None

    def __init__(self, Nmax: int, dt: float = 1., device: str = "cuda:0"):
        """Set common gate attributes and initialise bosonic operators."""
        self.Nmax   = Nmax
        self.dt     = dt
        self.device = device
        self.d      = Nmax + 1
        self.ops    = BosonOperatorsTorch(Nmax, device=device)

    def apply_to(
        self,
        rho:     MPDOtorch,
        site:    int,
        options: Dict[str, Any] = {'max_BD': 100, 'cutoff_BD': 1e-8},
    ) -> None:
        """
        Apply this gate to MPDO ``rho`` at ``site``.

        Must be implemented by every concrete subclass.

        Parameters
        ----------
        rho     : MPDO state to act on (modified in place).
        site    : left site index for the gate.  For single-site gates this
                  is the only site; for two-site gates the gate acts on
                  (site, site+1).
        options : truncation options forwarded to the SVD step (two-site
                  gates only).  Keys: max_BD, cutoff_BD.
        """
        raise NotImplementedError

    def update(self) -> None:
        """
        Recompute gate_tensor after a parameter update.

        No-op in the base class.  Override in concrete gates that hold
        differentiable parameters (J, U, φ).
        """
        return None

    def get_params(self) -> List[torch.Tensor]:
        """
        Return differentiable parameters belonging to this gate.

        Used by ``MPDOCircuit.get_variables`` to collect all optimisable
        tensors for passing to a torch optimiser.

        Returns
        -------
        List[torch.Tensor]
            Empty list in the base class; overridden in gates with
            trainable parameters.
        """
        return []

    def label(self) -> str:
        """
        Short human-readable label for circuit visualisation.

        Returns
        -------
        str
            Empty string in the base class.
        """
        return ""


# ---------------------------------------------------------------------------
# Single-mode base
# ---------------------------------------------------------------------------

class CircuitGateOneSite(CircuitGate):
    """
    Base class for single-mode gates.

    Applies the (d×d) ``gate_tensor`` to the physical index of one site
    via a tensor contraction; does not modify the bond dimension.

    span = 1
    """

    span: int = 1

    def apply_to(
        self,
        rho:     MPDOtorch,
        site:    int,
        options: Dict[str, Any] = {'max_BD': 100, 'cutoff_BD': 1e-8},
    ) -> None:
        """
        Contract gate_tensor with the physical index of ``rho[site]``.

        The contraction   B'[b,k,i,l] = Σ_j gate[i,j] B[b,k,j,l]
        is performed in place.

        Parameters
        ----------
        rho     : MPDO state (modified in place at site ``site``).
        site    : site index.
        options : unused; present for API consistency with two-site gates.
        """
        rho[site] = torch.einsum("ij,bkjl->bkil", self.gate_tensor, rho[site])


# ---------------------------------------------------------------------------
# Two-mode base
# ---------------------------------------------------------------------------

class CircuitGateTwoSite(CircuitGate):
    """
    Base class for two-adjacent-mode gates with SVD bond update.

    Applies the rank-4 ``gate_tensor`` to sites (site, site+1), then
    performs a truncated SVD to keep the bond dimension bounded.

    span = 2
    """

    span: int = 2

    def apply_to_C(
        self,
        C:       torch.Tensor,
        options: Dict[str, Any] = {'max_BD': 100, 'cutoff_BD': 1e-8},
    ) -> torch.Tensor:
        """
        Contract gate_tensor with a pre-formed two-site tensor C.

        This is a helper used internally; in most cases you should call
        ``apply_to`` directly.

        Parameters
        ----------
        C       : two-site tensor of shape
                  (batch, BD_L, d_L, d_R, BD_R, batch_R).
        options : SVD options (unused here; for API consistency).

        Returns
        -------
        torch.Tensor
            Gate-applied two-site tensor, same shape as C.
        """
        return torch.einsum("bijkl, mnkj -> bimnl", C, self.gate_tensor)

    def apply_to(
        self,
        rho:     MPDOtorch,
        site:    int,
        options: Dict[str, Any] = {'max_BD': 100, 'cutoff_BD': 1e-8},
    ) -> None:
        """
        Apply the two-site gate to ``rho`` and SVD-truncate the bond.

        Steps
        -----
        1. Form the two-site tensor C = B[site] ⊗ B[site+1].
        2. Contract C with gate_tensor to produce the updated two-site tensor.
        3. Absorb the left bond SVs SL[site] into the contracted tensor θ.
        4. Truncated SVD of θ → (U, s, Vd) with the given max_BD / cutoff.
        5. Update B[site], B[site+1], and SL[site+1] with the SVD factors.

        Parameters
        ----------
        rho     : MPDO state (modified in place at sites ``site`` and
                  ``site+1``).
        site    : left site index; the gate acts on (site, site+1).
        options : dict with keys:
                    max_BD    — maximum bond dimension after truncation.
                    cutoff_BD — relative SV cutoff for bond truncation.
        """
        i0, i1 = site, site + 1

        # Contract two-site tensor and apply gate
        C     = rho.get_C(i0)
        C     = torch.einsum("bijklc, mnkj -> bimnlc", C, self.gate_tensor)
        theta = irescale(C, rho.get_SL(i0), ind=-5)   # absorb left SVs

        # Truncated SVD along the bond
        U, s, Vd, renormalize_factor = svd_trunc(
            iregroup(theta, [[0, 1, 2], [3, 4, 5]]),
            cutoff=options['cutoff_BD'],
            max_num=options['max_BD'],
            lowrank=False,
        )

        # Left tensor: project C onto the new basis defined by Vd
        B_L = irescale(
            torch.einsum(
                'bijk,lk->bijl',
                iregroup(C, [[0], [1], [2], [3, 4, 5]]),
                Vd.conj(),
            ),
            1.0 / renormalize_factor,
            ind=0,
        )

        # Right tensor: reshape Vd to (new_BD, d, old_BD_R, batch_R)
        B_R = Vd.reshape([s.numel(), self.d, C.shape[-2], -1])

        rho[i0]    = B_L
        rho[i1]    = iregroup(B_R, [[3], [0], [1], [2]])
        rho.SL[i1] = s


# ---------------------------------------------------------------------------
# Concrete single-mode gates
# ---------------------------------------------------------------------------

class NonlinearLocalGate(CircuitGateOneSite):
    """
    On-site Kerr nonlinearity and detuning gate.

    Hamiltonian:
        H = U/2 · n̂(n̂-1) - Δ · n̂

    Used for waveguide modes that are not covered by an adjacent two-mode
    coupler gate (e.g. boundary sites in a brick-layer circuit).

    Parameters
    ----------
    Nmax   : Fock-space truncation.
    U      : Kerr nonlinear rate (ps⁻¹).
    Delta  : on-site energy detuning (ps⁻¹).
    dt     : gate duration (ps).
    device : Torch device string.
    """

    def __init__(
        self,
        Nmax:   int,
        U:      torch.Tensor,
        Delta:  torch.Tensor,
        dt:     float = 1.,
        device: str   = "cuda:0",
    ):
        """Initialise the gate and compute the initial gate tensor."""
        super().__init__(Nmax, dt, device)
        self.update(U, Delta)

    def update(
        self,
        U:     "torch.Tensor | None" = None,
        Delta: "torch.Tensor | None" = None,
    ) -> None:
        """
        Recompute the gate tensor after updating U or Delta.

        Parameters that are ``None`` are left unchanged.

        Parameters
        ----------
        U     : new Kerr rate (ps⁻¹); None → keep current value.
        Delta : new detuning (ps⁻¹); None → keep current value.
        """
        if U     is not None: self.U     = U
        if Delta is not None: self.Delta = Delta

        self.H           = 0.5 * self.U * self.ops.n_nm1 - self.Delta * self.ops.n
        self.gate_tensor = torch.matrix_exp(-1j * self.dt * self.H)


class PhaseGate(CircuitGateOneSite):
    """
    Single-mode phase gate: exp(-i φ n̂).

    Applies a Fock-number-dependent phase shift to the mode.  The phase φ
    may be a differentiable parameter (requires_grad=True) for use in
    phase-circuit optimisation.

    Parameters
    ----------
    Nmax   : Fock-space truncation.
    phi    : phase angle φ (rad).  May be a differentiable tensor.
    dt     : gate duration (default 1; φ already encodes the full angle).
    device : Torch device string.
    """

    def __init__(
        self,
        Nmax:   int,
        phi:    torch.Tensor,
        dt:     float = 1.,
        device: str   = "cuda:0",
    ):
        """Initialise the gate with phase ``phi`` and compute the gate tensor."""
        super().__init__(Nmax, dt, device)
        self.update(phi)

    def get_params(self) -> List[torch.Tensor]:
        """
        Return ``[phi]`` if ``phi.requires_grad``, else ``[]``.

        Returns
        -------
        List[torch.Tensor]
            List containing the phase tensor if it is differentiable.
        """
        return [self.phi] if self.phi.requires_grad else []

    def update(self, phi: torch.Tensor) -> None:
        """
        Set a new phase value and recompute the gate tensor.

        Parameters
        ----------
        phi : new phase angle (rad); replaces the current value entirely.
        """
        self.phi         = phi
        self.H           = phi * self.ops.n
        self.gate_tensor = torch.matrix_exp(-1j * self.dt * self.H)


# ---------------------------------------------------------------------------
# Concrete two-site gates
# ---------------------------------------------------------------------------

class NonlinearCouplingGate(CircuitGateTwoSite):
    """
    Two-mode beam-splitter + Kerr interaction gate.

    Hamiltonian:
        H = J (a†_L a_R + h.c.) + U/2 (n_L(n_L-1) + n_R(n_R-1)) + Δ(n_L + n_R)

    where L and R denote the left and right modes of the gate.

    The rank-4 Hamiltonian components (tunneling, interaction, detuning)
    are precomputed at construction and are parameter-independent; only the
    scalar coefficients J, U, Δ change during optimisation, so ``update``
    is cheap to call.

    Parameters
    ----------
    Nmax   : Fock-space truncation.
    J      : hopping / coupling rate (ps⁻¹).  May be differentiable.
    U      : Kerr nonlinear rate (ps⁻¹).  May be differentiable.
    Delta  : on-site detuning (ps⁻¹), applied symmetrically to both modes.
    dt     : gate duration (ps).
    device : Torch device string.
    """

    def __init__(
        self,
        Nmax:   int,
        J:      torch.Tensor,
        U:      torch.Tensor,
        Delta:  torch.Tensor = 0.,
        dt:     float        = 1.,
        device: str          = "cuda:0",
    ):
        """Pre-compute Hamiltonian structure terms and call update(J, U, Delta)."""
        super().__init__(Nmax=Nmax, dt=dt, device=device)

        # Parameter-independent rank-4 Hamiltonian components
        self.tunneling_gate = -(
            torch.einsum('ij,kl->iklj', self.ops.ad, self.ops.a)
            + torch.einsum('ij,kl->iklj', self.ops.a, self.ops.ad)
        )
        self.interaction_gate = 0.5 * (
            torch.einsum('ij,kl->iklj', self.ops.n_nm1, self.ops.id)
            + torch.einsum('ij,kl->iklj', self.ops.id, self.ops.n_nm1)
        )
        self.detuning_gate = -(
            torch.einsum('ij,kl->iklj', self.ops.n, self.ops.id)
            + torch.einsum('ij,kl->iklj', self.ops.id, self.ops.n)
        )

        self.update(J, U, Delta)

    def update(
        self,
        J:     "torch.Tensor | None" = None,
        U:     "torch.Tensor | None" = None,
        Delta: "torch.Tensor | None" = None,
    ) -> None:
        """
        Recompute the gate tensor after updating J, U, or Delta.

        Any argument that is ``None`` is left at its current value, so you
        can update a single parameter without touching the others.

        The matrix exponential exp(-i dt H) is reshaped and permuted into
        the canonical rank-4 gate_tensor format (d, d, d, d).

        Parameters
        ----------
        J     : new hopping rate (ps⁻¹); None → keep current.
        U     : new Kerr rate (ps⁻¹); None → keep current.
        Delta : new detuning (ps⁻¹); None → keep current.
        """
        if J     is not None: self.J     = J
        if U     is not None: self.U     = U
        if Delta is not None: self.Delta = Delta

        self.H = (
            self.J     * self.tunneling_gate
            + self.U   * self.interaction_gate
            + self.Delta * self.detuning_gate
        )

        # Regroup indices as (L_in, R_in) × (L_out, R_out) for matrix_exp,
        # then reshape and permute back to rank-4 gate tensor.
        gate_mat = torch.matrix_exp(
            iregroup(-1j * self.dt * self.H, [[0, 1], [3, 2]])
        )
        self.gate_tensor = gate_mat.reshape(self.d, self.d, self.d, self.d).permute([0, 1, 3, 2])

    def get_params(self) -> List[torch.Tensor]:
        """
        Return differentiable gate parameters.

        Returns
        -------
        List[torch.Tensor]
            Contains J and/or U if their ``requires_grad`` flag is True.
        """
        return [var for var in [self.J, self.U] if var.requires_grad]


class HaarCouplingGate(CircuitGateTwoSite):
    """
    Two-mode Haar-random unitary gate for circuit benchmarking.

    The gate tensor is drawn once from the Haar measure at construction
    and remains fixed thereafter (no trainable parameters).

    Parameters
    ----------
    Nmax   : Fock-space truncation (default 1 for qubit benchmarks).
    device : Torch device string.
    """

    def __init__(self, Nmax: int = 1, device: str = "cuda:0"):
        """Draw a Haar-random unitary and store it as the fixed gate tensor."""
        self.Nmax        = Nmax
        self.device      = device
        self.d           = Nmax + 1

        haar             = Haar_unitary(n=self.d ** 2, device=device)
        self.gate_tensor = haar.view(*([self.d] * 4)).permute([0, 1, 3, 2])
