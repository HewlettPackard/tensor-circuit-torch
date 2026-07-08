"""
mpdo_torch.py
-------------
Matrix Product Density Operator (MPDO) simulation in PyTorch.

The MPDO is stored in a hybrid mixed-canonical form.  Each site tensor B[l]
carries a batch (purity) dimension alongside the standard MPS bond and
physical indices:

    ρ = Σ_{batch} |Ψ_{batch}⟩⟨Ψ_{batch}|

with each |Ψ_{batch}⟩ stored as a right-canonical MPS.  Left bond
singular values SL[l] (l = 0 … N) and purity singular values SP[l]
(l = 0 … N-1) are maintained explicitly.

Tensor index convention for B[l]:  (batch, BD_L, d, BD_R)
  - batch : purity / ensemble index
  - BD_L  : left bond dimension
  - d     : physical dimension  (= Nmax + 1)
  - BD_R  : right bond dimension

Classes
-------
MPDOtorch              : main MPDO class (state, observables, canonicalisation)
StateCreator           : factory for standard bosonic input states
concat_ensemble_from_list : stack a list of MPDOs along the batch dimension
"""

from time import time
from typing import Dict, Iterable, List, TypeVar, Tuple

import numpy as np
import torch

from .utils import BosonOperatorsTorch, irescale, iregroup
from .svd_trunc import svd_trunc

# Self-referential type alias used in clone() return type
T = TypeVar("T", bound="MPDOtorch")


# ---------------------------------------------------------------------------
# MPDOtorch
# ---------------------------------------------------------------------------

class MPDOtorch:
    """
    Matrix Product Density Operator in mixed-canonical form.

    Parameters
    ----------
    psis              : list of N site tensors B[0] … B[N-1], each of shape
                        (batch, BD_L, d, BD_R)
    SL                : left bond singular-value vectors, length N+1;
                        SL[0] = SL[N] = [1] by convention.
                        Computed via canonicalisation when None.
    SP                : purity singular-value vectors, length N.
                        Computed via canonicalisation when None.
    no_canonical_form : if True, skip canonicalisation on init.
                        Use only when SL and SP are already known (e.g. after
                        cloning).
    """

    def __init__(
        self,
        psis:              List[torch.Tensor],
        SL:                "List[torch.Tensor] | None" = None,
        SP:                "List[torch.Tensor] | None" = None,
        no_canonical_form: bool                        = False,
    ):
        """Construct MPDO from site tensors; canonicalise unless SL/SP are supplied."""
        self.d            = psis[1].shape[-2]   # physical dimension
        self.Nmax         = self.d - 1
        self._B           = psis
        self.num_channels = len(psis)
        self.device       = psis[0].device
        self.dtype        = psis[0].dtype

        if SL is None and SP is None:
            self.SL     = [None] * (self.num_channels + 1)
            self.SP     = [None] *  self.num_channels
            self.SL[0]  = torch.tensor([1.], device=self.device)
            self.SL[-1] = torch.tensor([1.], device=self.device)
            self.canonical_form()
        else:
            self.SL = SL
            self.SP = SP

    # ------------------------------------------------------------------
    # Canonicalisation
    # ------------------------------------------------------------------

    def canonical_form(
        self,
        options: dict = {
            'max_BD': 100, 'max_PD': 100,
            'cutoff_BD': 1e-6, 'cutoff_PD': 1e-6,
        },
    ) -> None:
        """
        Bring the MPDO into mixed-canonical form.

        Delegates to ``krauss_dissipation`` with an empty Kraus list, which
        runs the full two-pass (L→R purity SVD then R→L bond SVD) sweep.

        Parameters
        ----------
        options : truncation thresholds and maximum bond / purity dimensions:
                    max_BD    — maximum bond dimension
                    max_PD    — maximum purity dimension
                    cutoff_BD — relative SV cutoff for bond truncation
                    cutoff_PD — relative SV cutoff for purity truncation
        """
        self.krauss_dissipation(K_ops=[], options=options)

    def krauss_dissipation(
        self,
        K_ops:   Iterable[torch.Tensor],
        options: dict = {
            'max_BD': 100, 'max_PD': 100,
            'cutoff_BD': 1e-6, 'cutoff_PD': 1e-6,
        },
    ) -> None:
        """
        Apply Kraus operators to every site and re-canonicalise.

        When ``K_ops`` is empty the method acts as a pure canonicalisation
        sweep without any dissipation.

        Algorithm
        ---------
        Pass 1 — L→R (purity SVD):
            For each site l, stack all Kraus operators into the batch dimension,
            then truncate the purity index with a truncated SVD (keeping at most
            ``max_PD`` components above ``cutoff_PD``), and QR-orthogonalise
            the site tensor.  The right factor R is absorbed into site l+1.

        Pass 2 — R→L (bond SVD):
            Starting from the right boundary, SVD-truncate the bond dimension
            (keeping at most ``max_BD`` components above ``cutoff_BD``) to
            place the MPDO in right-canonical form with normalised bond SVs
            stored in ``self.SL``.

        Parameters
        ----------
        K_ops   : list of (d×d) Kraus matrices.  Empty list → canonicalise only.
        options : dict with keys max_BD, max_PD, cutoff_BD, cutoff_PD.
        """
        L        = self.num_channels
        do_kraus = len(K_ops) > 0
        K_concat = torch.stack(K_ops, dim=0) if do_kraus else None

        # Pass 1: L→R — purity SVD
        B = (self._B[0] if self.SL[0] is None
             else irescale(self._B[0], self.SL[0], ind=-3))

        for il in range(L):
            if do_kraus:
                # Broadcast all Kraus operators over the batch dimension
                B_full = torch.einsum("aij, bkjm -> abkim", K_concat, B)
                B_full = B_full.reshape(K_concat.shape[0] * B.shape[0], *B_full.shape[2:])
            else:
                B_full = B

            # Truncated SVD over the purity (batch) dimension
            _, s_purity, Vd, _ = svd_trunc(
                B_full.reshape(B_full.shape[0], -1),
                cutoff=options['cutoff_PD'],
                max_num=options['max_PD'],
                lowrank=True,
            )

            # Purity-truncated site tensor, then QR for left-orthogonality
            B_kraus = irescale(Vd, factor=s_purity, ind=0).view(
                s_purity.shape + self._B[il].shape[1:]
            )
            Q, R = torch.linalg.qr(iregroup(B_kraus, [[0, 1, 2], [3]]), mode='reduced')

            self._B[il] = Q.view(B_kraus.shape)
            self.SP[il] = s_purity

            if il != L - 1:
                B = torch.einsum("ij, bjkl->bikl", R, self._B[il + 1])

        # Pass 2: R→L — bond SVD
        B = (self._B[L - 1] if self.SL[L] is None
             else irescale(self._B[L - 1], self.SL[L], ind=-1))
        B = iregroup(B, [[1], [0], [2], [3]])

        for il in range(L - 1, -1, -1):
            U_bond, s, Vd, _ = svd_trunc(
                iregroup(B, [[0], [1, 2, 3]]),
                cutoff=options['cutoff_BD'],
                max_num=options['max_BD'],
                lowrank=True,
            )

            self.SL[il] = s
            self._B[il] = iregroup(
                Vd.view([-1, B.shape[1], B.shape[2], B.shape[3]]),
                [[1], [0], [2], [3]],
            )

            if il != 0:
                B = iregroup(
                    torch.einsum(
                        "bijk,kl->bijl",
                        self._B[il - 1],
                        irescale(U_bond, s, ind=-1),
                    ),
                    [[1], [0], [2], [3]],
                )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def is_canonical(
        self,
        full: bool  = False,
        tol:  float = 1e-6,
    ) -> "bool | Dict[str, List[bool]]":
        """
        Check whether the MPDO is in (right-)canonical form.

        Each site tensor B[l] should satisfy B B† = I (right-unitarity) and
        each SV vector SL[l] should have unit norm.

        Parameters
        ----------
        full : if True, return per-tensor and per-SV flags instead of a
               single boolean.
        tol  : numerical tolerance for unitarity and normalisation checks.

        Returns
        -------
        bool
            True if all tensors are right-unitary and all SV vectors are
            normalised (only when ``full=False``).
        dict
            ``{'B': List[bool], 'SVs': List[bool]}`` — per-site and
            per-bond flags (only when ``full=True``).
        """
        B_canonical = []
        for B in self._B:
            BBd = torch.einsum("bijk, bljk->...il", B, B.conj())
            B_canonical.append(
                ((torch.eye(BBd.shape[-1], device=BBd.device) - BBd)
                 / B.numel()).norm(dim=[-2, -1]).mean() < tol
            )

        s_canonical = [
            (s.norm(dim=-1).mean() - 1.).abs() < tol for s in self.SL
        ]

        if full:
            return {'B': B_canonical, 'SVs': s_canonical}
        return all(s_canonical + B_canonical)

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def clone(self: T) -> T:
        """
        Return an independent deep copy with all gradients detached.

        All site tensors B[l] and singular-value vectors SL[l], SP[l] are
        cloned; ``requires_grad`` flags are preserved.

        Returns
        -------
        MPDOtorch
            A new MPDO with freshly cloned tensors and no canonicalisation
            step (SL and SP are already correct).
        """
        return MPDOtorch(
            [B.clone().detach().requires_grad_(B.requires_grad) for B in self._B],
            [S.clone().detach().requires_grad_(S.requires_grad) for S in self.SL],
            [S.clone().detach().requires_grad_(S.requires_grad) for S in self.SP],
            no_canonical_form=True,
        )

    # ------------------------------------------------------------------
    # Local measurements
    # ------------------------------------------------------------------

    def ptrace(self, il: int) -> torch.Tensor:
        """
        Single-site reduced density matrix ρ_il = Tr_{≠il}[ρ].

        Computed by contracting B[il] with its conjugate over all indices
        except the physical ones, weighted by the squared bond SVs SL[il]².

        Parameters
        ----------
        il : site index (0-based).

        Returns
        -------
        torch.Tensor
            Complex (d×d) local density matrix at site ``il``.
        """
        return torch.einsum(
            "bijk,bilk->jl",
            irescale(self._B[il], self.SL[il] ** 2, ind=-3),
            self._B[il].conj(),
        )

    def diagonal_local_measurement(
        self,
        il:           int,
        diagonal_els: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expectation value of a diagonal operator D = diag(diagonal_els).

        Equivalent to Tr[ρ_il · D] but avoids constructing the full local
        density matrix explicitly.

        Parameters
        ----------
        il           : site index.
        diagonal_els : 1-D tensor of length d containing the diagonal
                       elements of the operator in the Fock basis.

        Returns
        -------
        torch.Tensor
            Real scalar expectation value ⟨D⟩_il.
        """
        return torch.einsum(
            "bijk,bijk",
            irescale(
                irescale(self._B[il], self.SL[il] ** 2, ind=-3),
                diagonal_els, ind=-2,
            ),
            self._B[il].conj(),
        ).real

    def number_outcome(self, il: int) -> torch.Tensor:
        """
        Mean photon number ⟨n̂⟩ at site ``il``.

        Parameters
        ----------
        il : site index.

        Returns
        -------
        torch.Tensor
            Real scalar ⟨a†a⟩_il.
        """
        n_els = torch.arange(0, self.d, device=self.device, dtype=torch.float64)
        return self.diagonal_local_measurement(il, n_els).real

    def number_variance(self, il: int) -> torch.Tensor:
        """
        Photon-number variance Var(n̂) = ⟨n²⟩ - ⟨n⟩² at site ``il``.

        Parameters
        ----------
        il : site index.

        Returns
        -------
        torch.Tensor
            Real scalar variance of the photon number at site ``il``.
        """
        n      = self.number_outcome(il)
        n2_els = torch.arange(0, self.d, device=self.device, dtype=torch.float64) ** 2
        n2     = self.diagonal_local_measurement(il, n2_els)
        return n2 - n ** 2

    def density_correlation(self, il: int, n_eps: float = 1e-2) -> torch.Tensor:
        """
        Second-order coherence g²(0) at site ``il``.

        g²(0) = ⟨n(n-1)⟩ / ⟨n⟩²

        Both numerator and denominator are regularised by ``n_eps²`` to
        avoid division by zero in the vacuum.

        Parameters
        ----------
        il    : site index.
        n_eps : regularisation constant (default 1e-2).

        Returns
        -------
        torch.Tensor
            Real scalar g²(0) at site ``il``.
        """
        n     = self.number_outcome(il)
        n_els = torch.arange(0, self.d, device=self.device, dtype=torch.float64)
        C2    = self.diagonal_local_measurement(il, n_els * (n_els - 1.0))
        return (C2 + n_eps ** 2) / (n ** 2 + n_eps ** 2)

    def number_outcomes(self) -> torch.Tensor:
        """
        Mean photon numbers ⟨n̂_l⟩ stacked over all sites.

        Returns
        -------
        torch.Tensor
            1-D real tensor of length ``num_channels``.
        """
        return torch.stack([self.number_outcome(il) for il in range(self.num_channels)])

    def number_variances(self) -> torch.Tensor:
        """
        Photon-number variances Var(n̂_l) stacked over all sites.

        Returns
        -------
        torch.Tensor
            1-D real tensor of length ``num_channels``.
        """
        return torch.stack([self.number_variance(il) for il in range(self.num_channels)])

    def density_correlations(self, n_eps: float) -> torch.Tensor:
        """
        g²_l(0) stacked over all sites.

        Parameters
        ----------
        n_eps : regularisation constant passed to ``density_correlation``.

        Returns
        -------
        torch.Tensor
            1-D real tensor of length ``num_channels``.
        """
        return torch.stack([
            self.density_correlation(il, n_eps=n_eps)
            for il in range(self.num_channels)
        ])

    def local_expectation(
        self,
        op: torch.Tensor,
        i:  "int | None" = None,
    ) -> torch.Tensor:
        """
        Expectation value ⟨op⟩ of a single-mode operator.

        Parameters
        ----------
        op : (d×d) operator matrix in the Fock basis.
        i  : site index.  If None, returns a stack over all sites via
             ``local_expectations``.

        Returns
        -------
        torch.Tensor
            Complex scalar ⟨op⟩_i, or a 1-D tensor of length
            ``num_channels`` when ``i`` is None.
        """
        if i is None:
            return self.local_expectations(op)

        B_resc = irescale(self._B[i], self.SL[i], ind=-3)
        return torch.einsum(
            "biqj, birj, qr",
            B_resc, B_resc.conj(), op.to(B_resc.device),
        )

    def local_expectations(self, op: torch.Tensor) -> torch.Tensor:
        """
        Expectation value ⟨op⟩_l stacked over all sites.

        Parameters
        ----------
        op : (d×d) operator matrix.

        Returns
        -------
        torch.Tensor
            1-D complex tensor of length ``num_channels``.
        """
        return torch.stack([self.local_expectation(op, il) for il in range(self.num_channels)])
    
    def correlation(
        self,
        op1: torch.Tensor,
        sites: Tuple[int, int],
        op2: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """
        Two-point correlator ⟨op1_i op2_j⟩ for a single pair of sites.

        Parameters
        ----------
        op1   : (d×d) operator acting at ``sites[0]``.
        sites : pair of site indices (i, j). Order doesn't matter; they are
                sorted internally so the sweep always goes left to right.
        op2   : (d×d) operator acting at ``sites[1]``. Defaults to ``op1``.

        Returns
        -------
        torch.Tensor
            Complex scalar ⟨op1_i op2_j⟩. If ``sites[0] == sites[1]``, this
            reduces to the single-site expectation value ⟨op1 @ op2⟩_i.
        """
        start, end = sorted(sites)
        op2 = op1 if op2 is None else op2

        # coincident sites: <op1 op2> is just a local expectation value
        if start == end:
            return self.local_expectation(op1 @ op2, start)

        # open the environment at the left site, inserting op1
        B_resc = irescale(self._B[start], self.SL[start], ind=-3)
        G = torch.einsum(
            "biqj, birk, qr -> jk",
            B_resc, B_resc.conj(), op1.to(B_resc.device),
        )

        # sweep the open bond-environment through the sites in between,
        # tracing the ancilla/Kraus leg and the physical leg (identity) at each
        for m in range(start + 1, end):
            B = self._B[m]
            G = torch.einsum(
                "jk, bjlm, bkln -> mn",
                G, B, B.conj(),
            )

        # close the environment at the right site, inserting op2
        B = self._B[end]
        corr = torch.einsum(
            "jk, bjrm, bkqm, rq -> ",
            G, B, B.conj(), op2.to(G.device),
        )

        return corr

    def correlation_matrix(
        self,
        op1: torch.Tensor,
        op2: "torch.Tensor | None" = None,
        is_hermitian: bool = True,
    ) -> torch.Tensor:
        """
        Two-point correlator matrix M_ij = ⟨op1_i op2_j⟩ over all site pairs.

        Builds one bond-environment per row (anchored at site i with op1
        inserted) and sweeps it rightward once, reading off M[i, j] at every
        j > i as the environment passes through, rather than re-sweeping from
        site i for each j independently. This brings the cost of the full
        matrix down from O(N^3) (N^2 pairs, each an O(N) sweep) to O(N^2)
        (N rows, each a single O(N) sweep).

        Parameters
        ----------
        op1, op2 : (d×d) single-mode operators in the Fock basis.
            op2 defaults to op1 if not given.
        is_hermitian : if True (default), assumes op2 = op1^† so that
            M_ji = conj(M_ij), and only sweeps the op1-anchored environment,
            deriving the lower triangle by conjugation instead of building a
            second (op2-anchored) environment. Set False if op1/op2 don't
            satisfy this (e.g. op2 is not the conjugate-transpose of op1),
            in which case both M_ij and M_ji are computed explicitly.

        Returns
        -------
        torch.Tensor
            (num_sites × num_sites) complex matrix.
        """
        if op2 is None:
            op2 = op1

        # diagonals are product expectation
        op_diag = op1 @ op2

        L = self.num_channels
        device = self._B[0].device
        dtype = self._B[0].dtype
        C = torch.zeros((L, L), dtype=dtype, device=device)

        for i in range(L):
            # diagonal element: local expectation of the operator product
            C[i, i] = self.local_expectation(op_diag, i)

            # open the environment(s) at the anchor site i
            B_resc = irescale(self._B[i], self.SL[i], ind=-3)
            G1 = torch.einsum(
                "biqj, birk, qr -> jk",
                B_resc, B_resc.conj(), op1.to(device),
            )
            if not is_hermitian:
                G2 = torch.einsum(
                    "biqj, birk, qr -> jk",
                    B_resc, B_resc.conj(), op2.to(device),
                )

            # single rightward sweep: close off M[i, j] (and M[j, i] if not
            # hermitian) at every site j, then propagate past it
            for j in range(i + 1, L):
                Bj = self._B[j]

                # shared partial contraction, computed once per site j
                X1 = torch.einsum("jk, bjrm -> bkrm", G1, Bj)  # (batch, D, d, D)
                if not is_hermitian:
                    X2 = torch.einsum("jk, bjrm -> bkrm", G2, Bj)  # (batch, D, d, D)

                C[i, j] = torch.einsum("bkrm, rq, bkqm -> ", X1, op2.to(device), Bj.conj())
                if is_hermitian:
                    C[j, i] = C[i, j].conj()
                else:
                    C[j, i] = torch.einsum("bkrm, rq, bkqm -> ", X2, op1.to(device), Bj.conj())

                if j < L - 1:
                    G1 = torch.einsum("bkrm, bkrn -> mn", X1, Bj.conj())
                    if not is_hermitian:
                        G2 = torch.einsum("bkrm, bkrn -> mn", X2, Bj.conj())

        return C


    # ------------------------------------------------------------------
    # Two-site helpers
    # ------------------------------------------------------------------

    def get_C(self, j: int) -> torch.Tensor:
        """
        Contract B[j] and B[j+1] into a two-site tensor C.

        The result has shape (batch_L, BD_LL, d_L, d_R, BD_RR, batch_R)
        and is used as the starting point for two-site gate application.

        Parameters
        ----------
        j : left site index (gate spans sites j and j+1).

        Returns
        -------
        torch.Tensor
            Two-site tensor of shape (batch_L, BD_LL, d_L, d_R, BD_RR, batch_R).
        """
        return torch.einsum("bijk, cklm -> bijlmc", self._B[j], self._B[j + 1])

    def get_theta(self, j: int) -> torch.Tensor:
        """
        Two-site tensor C[j] rescaled with the left bond SVs SL[j].

        This is the object on which a two-site gate is applied before SVD.

        Parameters
        ----------
        j : left site index.

        Returns
        -------
        torch.Tensor
            Rescaled two-site tensor, same shape as ``get_C(j)``.
        """
        return irescale(self.get_C(j), self.SL[j], ind=0)

    # ------------------------------------------------------------------
    # Global properties
    # ------------------------------------------------------------------

    def norm(self) -> float:
        """
        State norm ‖ρ‖² = Tr[ρ†ρ].

        Should return 1 when the MPDO is in canonical form.

        Returns
        -------
        float
            Squared norm of the state.
        """
        return self.overlap(self).abs() ** 2

    def overlap(self, rho: T) -> torch.Tensor:
        """
        Overlap Tr[self† · rho] computed by right-to-left contraction.

        Each batch index of ``self`` is contracted with the corresponding
        batch index of ``rho``, so the result is a sum over all ensemble
        members.

        Parameters
        ----------
        rho : second MPDO (must have the same num_channels and physical dim).

        Returns
        -------
        torch.Tensor
            Complex scalar overlap.
        """
        BBp = torch.einsum(
            "bijk, bljk->il",
            self._B[self.num_channels - 1],
            rho[self.num_channels - 1].conj(),
        )
        for i in range(self.num_channels - 2, -1, -1):
            BBp = torch.einsum(
                "lm, birl, bkrm->ik", BBp, self._B[i], rho[i].conj()
            )
        return torch.einsum("...ii", BBp)

    def entropy_profile(
        self,
        alpha:   int = 1,
        entropy: str = 'entanglement',
    ) -> torch.Tensor:
        """
        Rényi entropy profile along the chain.

        For ``alpha=1`` (default) returns the von Neumann entropy
        S = -Σ p log p, where p = s² are the squared singular values.
        For ``alpha>1`` returns the order-α Rényi entropy
        S_α = 1/(1-α) log(Σ p^α).

        Parameters
        ----------
        alpha   : Rényi order.  1 = von Neumann (default).
        entropy : which SV spectrum to use:
                    'entanglement' — bond SVs SL (length N+1).
                    'purity'       — purity SVs SP (length N).

        Returns
        -------
        torch.Tensor
            1-D tensor of entropy values; length N+1 for entanglement
            entropy, length N for purity entropy.
        """
        SVDs      = self.SL if entropy == 'entanglement' else self.SP
        S_profile = []
        for s in SVDs:
            prob = s ** 2
            if alpha == 1:
                S_profile.append(-(prob * torch.log(prob + 1e-15)).sum(axis=-1).mean())
            else:
                S_profile.append(
                    1.0 / (1.0 - alpha)
                    * torch.log((prob ** alpha).sum(axis=-1).mean())
                )
        return torch.stack(S_profile)

    # ------------------------------------------------------------------
    # Bond / purity dimension accessors
    # ------------------------------------------------------------------

    def get_SL(self, i: int) -> torch.Tensor:
        """
        Left bond singular values at bond i (between sites i-1 and i).

        Parameters
        ----------
        i : bond index (0 … num_channels).

        Returns
        -------
        torch.Tensor
            1-D SV vector at bond i.
        """
        return self.SL[i]

    def set_SL(self, i: int, S: torch.Tensor) -> None:
        """
        Overwrite the left bond singular values at bond i.

        Parameters
        ----------
        i : bond index.
        S : new SV vector.
        """
        self.SL[i] = S

    def get_BD(self, i: int) -> int:
        """
        Right bond dimension at site i  (= number of SVs at bond i+1).

        Parameters
        ----------
        i : site index.

        Returns
        -------
        int
            Bond dimension between sites i and i+1.
        """
        return self.SL[i + 1].numel()

    def get_PD(self, i: int) -> int:
        """
        Purity dimension (batch size) at site i.

        Parameters
        ----------
        i : site index.

        Returns
        -------
        int
            Number of ensemble members (batch dimension) at site i.
        """
        return self._B[i].shape[0]

    def get_BDs(self) -> List[int]:
        """
        All bond dimensions, one per bond (length num_channels + 1).

        Returns
        -------
        List[int]
            Bond dimension at each bond, including the two trivial
            boundary bonds (always 1 in canonical form).
        """
        return [int(self.SL[i].shape[-1]) for i in range(self.num_channels + 1)]

    def get_PDs(self) -> List[int]:
        """
        All purity dimensions, one per site (length num_channels).

        Returns
        -------
        List[int]
            Batch (purity) dimension at each site.
        """
        return [int(self._B[i].shape[0]) for i in range(self.num_channels)]

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, i: int) -> torch.Tensor:
        """
        Access site tensor B[i].

        Parameters
        ----------
        i : site index.

        Returns
        -------
        torch.Tensor
            Site tensor of shape (batch, BD_L, d, BD_R).
        """
        return self._B[i]

    def __setitem__(self, i: int, value: torch.Tensor) -> None:
        """
        Replace site tensor B[i].

        Parameters
        ----------
        i     : site index.
        value : new site tensor; must have compatible shape.
        """
        self._B[i] = value

    # ------------------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------------------

    def to(self, device: str) -> None:
        """
        Move all tensors to ``device``.

        This is an in-place operation.  Prefer constructing states directly
        on the target device when possible; cross-device transfers are slow.

        Parameters
        ----------
        device : target Torch device string, e.g. 'cuda:0' or 'cpu'.
        """
        for B in self._B:
            B.to(device)
        for S in self.SL:
            S.to(device)


# ---------------------------------------------------------------------------
# StateCreator
# ---------------------------------------------------------------------------

class StateCreator:
    """
    Factory for standard bosonic input states as MPS site-tensor lists.

    All returned tensors are compatible with ``MPDOtorch``.

    Parameters
    ----------
    Nmax      : Fock-space truncation; physical dimension d = Nmax + 1.
    num_batch : batch (purity) dimension size.  None → no batch axis added.
    """

    def __init__(self, Nmax: int, num_batch: "int | None" = None):
        """Set up Fock-space operators and state-generation lambdas."""
        self.Nmax      = Nmax
        self.d         = Nmax + 1
        self.num_batch = num_batch
        self.ops       = BosonOperatorsTorch(Nmax)

        # Vacuum state |0⟩ — used as reference for displacement / squeezing
        self.vac    = torch.zeros(Nmax + 1, dtype=torch.complex128)
        self.vac[0] = 1.

        # Displacement operator D(α) = exp(α a† - α* a)
        self.D = lambda alpha: torch.matrix_exp(
            alpha * self.ops.ad - np.conj(alpha) * self.ops.a
            if isinstance(alpha, float)
            else alpha * self.ops.ad - alpha.conj() * self.ops.a
        )

        # Squeezing operator S(ξ) = exp(½(ξ* a² + ξ a†²))
        self.S = lambda xi: torch.matrix_exp(
            0.5 * (np.conj(xi) * self.ops.a @ self.ops.a
                   + xi * self.ops.ad @ self.ops.ad)
        )

        # Thermal state ρ_th(n_th) — diagonal in the Fock basis
        self.rho_th = lambda n_th: (
            torch.diag(
                1.0 / (n_th + 1.) * (n_th / (n_th + 1.)) ** np.arange(Nmax + 1)
            )
            if n_th > 1e-8
            else self.vac.T @ self.vac
        )

    # ------------------------------------------------------------------
    # Single-mode states
    # ------------------------------------------------------------------

    def fock(self, n: int, to: "str | None" = None, direct: bool = False) -> torch.Tensor:
        """
        Single-mode Fock state |n⟩.

        Parameters
        ----------
        n      : photon number (0 ≤ n ≤ Nmax).
        to     : Torch device string; None → keep on default device.
        direct : if True, return the bare 1-D coefficient vector without
                 bond or batch indices (useful for outer products).

        Returns
        -------
        torch.Tensor
            If ``direct=True``: 1-D complex vector of length d.
            Otherwise: tensor of shape (num_batch, 1, d, 1) (with batch
            axis) or (1, d, 1) (without).
        """
        state = torch.tensor(
            [1 if i == n else 0 for i in range(self.d)],
            dtype=torch.complex128, device=to,
        )
        if direct:
            return state

        state = torch.unsqueeze(torch.unsqueeze(state, 0), -1)
        if self.num_batch is not None:
            state = state.repeat(self.num_batch, *([1] * len(state.shape)))

        return state if to is None else state.to(to)

    def single_mode_coherent(
        self,
        alpha: "torch.Tensor | complex",
        to:    "str | None" = None,
    ) -> torch.Tensor:
        """
        Single-mode coherent state |α⟩ = D(α)|0⟩.

        Parameters
        ----------
        alpha : complex displacement amplitude.
        to    : target device.

        Returns
        -------
        torch.Tensor
            Site tensor of shape (num_batch, 1, d, 1) or (1, d, 1).
        """
        Cn    = self.D(alpha) @ self.vac
        state = torch.unsqueeze(torch.unsqueeze(Cn, 0), -1)

        if self.num_batch is not None:
            state = state.repeat(self.num_batch, *([1] * len(state.shape)))

        return state if to is None else state.to(to)

    def single_mode_cat(
        self,
        alpha:  "torch.Tensor | complex",
        device:     "str | None"  = None,
        phase:  "float | str" = 0.,
        direct: bool          = True,
    ) -> torch.Tensor:
        """
        Normalised single-mode cat state: N (|α⟩ + e^{iφ} |-α⟩).

        Parameters
        ----------
        alpha  : coherent amplitude.
        to     : target device.
        phase  : relative phase φ (rad), or the strings 'even' (φ=0) /
                 'odd' (φ=π).
        direct : if True, return the bare 1-D coefficient vector.

        Returns
        -------
        torch.Tensor
            Normalised state vector or site tensor depending on ``direct``.
        """
        if isinstance(phase, str):
            phase = 0. if phase == 'even' else np.pi

        Cn = (self.D(alpha) @ self.vac
              + np.exp(1j * phase) * self.D(-alpha) @ self.vac).to(device)
        Cn = Cn / Cn.norm()

        if direct:
            return Cn

        state = torch.unsqueeze(torch.unsqueeze(Cn, 0), -1)
        if self.num_batch is not None:
            state = state.repeat(self.num_batch, *([1] * len(state.shape)))

        return state if to is None else state.to(to)

    def two_mode_NOON(
        self,
        n:      int           = 1,
        phase:  "float | str" = 'even',
        to:     "str | None"  = None,
        direct: bool          = True,
    ) -> torch.Tensor:
        """
        Normalised two-mode N00N state: N (|N,0⟩ + e^{iφ} |0,N⟩).

        Parameters
        ----------
        n      : photon number N.
        phase  : relative phase φ (rad), or 'even' / 'odd'.
        to     : target device.
        direct : if True, return the bare 2-D coefficient matrix
                 (shape d×d, useful for embedding into an MPS).

        Returns
        -------
        torch.Tensor
            Normalised coefficient matrix (direct=True) or site tensor.
        """
        if isinstance(phase, str):
            phase = 0. if phase == 'even' else np.pi

        Cn = (torch.outer(*self.product_state_fock([n, 0], direct=True))
              + np.exp(1j * phase) * torch.outer(*self.product_state_fock([0, n], direct=True)))
        Cn = (Cn / Cn.norm()).to(to)

        if direct:
            return Cn

        state = torch.unsqueeze(torch.unsqueeze(Cn, 0), -1)
        if self.num_batch is not None:
            state = state.repeat(self.num_batch, *([1] * len(state.shape)))

        return state if to is None else state.to(to)

    # ------------------------------------------------------------------
    # Product states
    # ------------------------------------------------------------------

    def product_state_fock(
        self,
        nums:   Iterable[int],
        device: "str | None" = None,
        direct: bool         = False,
    ) -> List[torch.Tensor]:
        """
        Product Fock state |n_0⟩ ⊗ |n_1⟩ ⊗ … as a list of site tensors.

        Parameters
        ----------
        nums   : iterable of photon numbers, one per channel.
        device : target device.
        direct : passed through to ``fock``; if True each tensor is a bare
                 1-D vector.

        Returns
        -------
        List[torch.Tensor]
            One site tensor per channel.
        """
        return [self.fock(n, to=device, direct=direct) for n in nums]

    def product_state_coherent(
        self,
        alphas: Iterable,
        device: "str | None" = None,
    ) -> List[torch.Tensor]:
        """
        Product coherent state |α_0⟩ ⊗ |α_1⟩ ⊗ … as a list of site tensors.

        Parameters
        ----------
        alphas : iterable of complex amplitudes, one per channel.
        device : target device.

        Returns
        -------
        List[torch.Tensor]
            One site tensor per channel.
        """
        return [self.single_mode_coherent(alpha, to=device) for alpha in alphas]


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def concat_ensemble_from_list(
    list_psi: List[List[torch.Tensor]],
) -> List[torch.Tensor]:
    """
    Stack a list of MPDOs into a single MPDO with an enlarged batch dimension.

    All input states must have identical tensor shapes.

    Parameters
    ----------
    list_psi : list of MPDOs; each MPDO is itself a list of N site tensors.

    Returns
    -------
    List[torch.Tensor]
        N site tensors, each with a batch dimension equal to the total
        number of input states.
    """
    num_channels = len(list_psi[0])
    num_states   = len(list_psi)
    return [
        torch.cat([list_psi[i_state][ind] for i_state in range(num_states)], dim=0)
        for ind in range(num_channels)
    ]
