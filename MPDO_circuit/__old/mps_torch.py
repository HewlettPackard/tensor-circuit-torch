import torch
import numpy as np
import math
from typing import List, Iterable, Dict, Any, Callable, TypeVar, Type
from time import time
import copy

from ..src.utils import BosonOperatorsTorch, irescale, iregroup
from ..src.svd_trunc import svd_trunc

# definition type for self-reference
T = TypeVar("T", bound="MPStorch")

class MPStorch:

    """
    A PyTorch implementation of the matrix-product state. Allows batch index for 
    the tensors (first index) for the stochastic sampling.

    Parameters
    ----------
    psis: List[torch.Tensor]
        A list of the tensors for the MPS. bond dimension should be consistent (check still to be added)
    
    SL: List[torch.Tensor]|None (default None)
        Left singular values, if known beforehand. Otherwise computed with canonicalization.

    no_canonical_form: bool (default False)
        skip canonicalization step upon initiation
    """

    def __init__(self, psis: List[torch.Tensor], SL: List[torch.Tensor]|None=None, no_canonical_form: bool=False):

        """
        Initialize MPS state
        """

        # read input parameters
        self.d = psis[1].shape[-2]
        self._B = psis
        self.num_channels = len(psis)
        self.device = psis[0].device
        self.dtype = psis[0].dtype

        # check if batch dim present in state
        self.has_batch_dim = psis[1].dim() == 4
        self.batch_dim = psis[1].shape[0] if self.has_batch_dim else None

        # locally defined boson ops
        self.ops = BosonOperatorsTorch(self.d, device=self.device)

        # see if left SVs are given
        if SL is None:

            shape = (self.batch_dim, 1) if self.has_batch_dim else (1)

            # bring mps to left canonical form (fill other SVs)
            if no_canonical_form:
                self._SL = [torch.ones(shape, device=self._B[0].device)] * (self.num_channels + 1)
            else:
                # The SVs + add trivial 1's for boundaries
                self._SL = [None] * (self.num_channels + 1)
                self._SL[0] = torch.ones(shape, device=self._B[0].device)
                self._SL[-1] = torch.ones(shape,device=self._B[0].device)
                self.canonical_form() 

        else:
            self._SL = SL

    
    def is_canonical(self, full: bool=False, tol: float=1e-6) -> bool|Dict[str,List[bool]]:

        """
        Check whether MPS in canonical form.

        Parameters
        ----------
        full: bool (default False)
            Whether to return canonical form of each SV and tensor individually.
        
        tol: float (default 1e-6)
            Numerical tolerance for determing right-unitarity and normalization

        Returns
        -------
        bool or Dict[str,List[bool]]
            Whether canonical (full=False) or Dict with canonicality of tensors and SVs (full=True)
            
        """

        B_canonical = []

        for B in self._B:
            BBd = torch.einsum(
                "...ijk, ...ljk->...il", B, B.conj()
                )
            
            B_canonical.append(
                ((torch.eye(BBd.shape[-1], device=BBd.device) - BBd) / B.numel()).norm(dim=[-2,-1]).mean() < tol
            )


        s_canonical = []
        comp = self.batch_dim if self.has_batch_dim else 1.
        for s in self._SL:
            s_canonical.append((s.norm(dim=-1).mean() - 1.) < tol)

        if full:
            return {
                'B': B_canonical,
                'SVs': s_canonical
                }

        return all(s_canonical + B_canonical)
    

    def canonical_form(
            self, 
            cutoff: float=1e-10, 
            max_num: int|None=None, 
            i_start: int|None=None, 
            only_RL: bool=False,
            eps_reg: float=1e-10,
            lowrank: bool=True
        ) -> None:

        """
        Bring MPS to right canonical form

        cutoff: float (default 1e-10)
            The cutoff for the (normalized) SVs 

        max_num: int|None (default None)
            Maximal number of singular values (bond dimension)

        i_start: int|None (default None)
            Starting index, if not 0 (default)

        only_RL: bool (default False)
            If only right to left SVD pass must be done (left-right QR can be skipped)

        eps_reg: float (default 1e-10)
            Regularization for singular values, to avoid degeneracies.

        lowrank: bool (default True)
            Whether lowrank SVD from PyTorch (more efficient) or full SVD (more accurate) is used
        """
        
        # some useful parameters
        L = self.num_channels
        i0 = i_start if i_start is not None else 0

        # L -> R: make left orthogonal with QR
        if not only_RL:
            # if we actually had a canonical form before, so we should *not* ignore the 'S' (Th form)
            M = irescale(self._B[i0], self._SL[i0], ind=-3)

            # QR decomposition
            Q, R = torch.linalg.qr(iregroup(M + eps_reg, [[-3,-2], [-1]]), mode='reduced')

            # Q = unitary, R has to be multiplied to the right
            self._B[i0] = Q.view(list(M.shape[:-3]) + [M.shape[-3], M.shape[-2], -1])

            # loop L to R for rest of chain
            for i in range(i0 + 1, L-1):
                
                # select B_i
                M = self._B[i]
                
                # multiply with R from previous run and QR decompose
                M = torch.einsum("...ij, ...jkl->...ikl", R, M)
                Q, R = torch.linalg.qr(iregroup(M + eps_reg, [[-3,-2], [-1]]), mode='reduced')
                
                # Q is unitary, i.e. left canonical, R has to be multiplied to the right
                self._B[i] = Q.view(list(M.shape[:-3]) + [M.shape[-3], M.shape[-2], -1])

            # R -> L: make right orthogonal with SVD
            M = self._B[L-1]
            M = torch.einsum("...ij, ...jkl->...ikl", R, M)
        
        # if only LR
        else:
            M = self._B[L-1]
            M = irescale(M, self._SL[L], ind=-1)

        # SV decomposition and truncate, if requested
        U, s, Vd, renormalize_factor = svd_trunc(
                iregroup(M, [[-3], [-2, -1]]), 
                cutoff=cutoff, 
                max_num=max_num
                ) # 

        # update right site
        self._SL[self.num_channels -1] = s
        self._B[self.num_channels - 1] = Vd.view(list(M.shape[:-3]) + [-1, M.shape[-2], M.shape[-1]])

        for i in range(self.num_channels - 2, i0 - 1, -1):

            M = self._B[i]
            M = torch.einsum(
                "...ijk, ...kl->...ijl", 
                M, irescale(irescale(U, s, ind=-1), 1. / renormalize_factor, ind=0) # two rescales: inner for absorbing the SVs, outer for renormalizing the batches
                )

            # SV decomposition and truncate, if requested
            U, s, Vd, renormalize_factor = svd_trunc(
                    iregroup(M, [[-3], [-2, -1]]), 
                    cutoff=cutoff, 
                    max_num=max_num,
                    eps_reg=eps_reg,
                    lowrank=lowrank
                    ) # 

            # update site i
            self._SL[i] = s
            self._B[i] = Vd.view(list(M.shape[:-3]) + [-1, M.shape[-2], M.shape[-1]])
            
    
    def clone(self: T) -> T:
        """
        Clone MPS into other MPS (hard copy, all grads are detached)
        """
        return MPStorch(
            [B.clone().detach().requires_grad_(B.requires_grad) for B in self._B],
            [S.clone().detach().requires_grad_(S.requires_grad) for S in self._SL],
            no_canonical_form=True
        )


    def select_batch_state(self: T, i: int) -> T:
        """
        Select one batch state i
        """
        return MPStorch(
            [B[i] for B in self._B],
            [S[i] for S in self._SL]
        )


    def __getitem__(self, i) -> torch.Tensor:
        """
        psi[i] labels the B tensors of MPS description
        """
        return self._B[i]
    
    
    def __setitem__(self, i, value):
        """
        Set a new B tensor on mode i
        """

        self._B[i] = value

    def to(self, device: str):
        """
        Transfer device (not efficient, always avoid if you can)
        """

        for B in self._B:
            B.to(device)

        for S in self._SL:
            S.to(device)
    

    def get_theta(self, j: int) -> torch.Tensor:

        """ Contract C with its left SV """

        return irescale(self.get_C(j), self._SL[j], ind=0)


    def get_C(self, j) -> torch.Tensor:

        """Contract B_j and B_{j+1} into new two-mode tensor C. """

        return torch.einsum("...ijk, ...klm -> ...ijlm", self._B[j], self._B[j+1] )
    
    def get_SL(self, i: int) -> torch.Tensor:
        """Get the left singular values at mode i"""

        return self._SL[i]

    def set_SL(self, i: int, S):
        """Set the left SVs at mode i"""
        self._SL[i] = S

    def get_BD(self, i: int) -> int:
        """Get the (right) bond dimension at mode i"""
        return self._SL[i+1].numel()
    
    def get_BDs(self) -> List[int]:
        """Get all BDs"""
        return [int(self._SL[i].shape[-1]) for i in range(self.num_channels + 1)]
        
    def contract_to_tensor(self, i1: int|None=None, i2: int|None=None) -> torch.Tensor:
        """
        Contract B tensors from site i1 to i2, i1 defaults to 0, i2 to last tensor 
        (don't use unless for a small system check/benchmark, very inefficient and produces the full quantum tensor, 
        exactly what you want to avoid)
        """

        i1 = 0 if i1 is None else i1
        i2 = self.num_channels if i2 is None else i2

        ten = self._B[i1]
        for i in range(i1 + 1, i2):
            ten = (torch.einsum("...ijl, ...lmn->...ijmn", ten, self._B[i]) if self._B[i].dim() == 3 else
                   torch.einsum("b...ijl, b...lmn->b...ijmn", ten, self._B[i]))

        return ten
    
    def norm(self) -> float:
        """Overlap of state with itself -> norm of the state (should be 1 if canonical)"""
        return self.overlap(self)
    

    def overlap(self, psi: T) -> torch.Tensor:

        """
        Overlap between this quantum state and some other state psi.
        This is per batch, batch 0 of one state is contracted with batch 0 from other and so on.
        """

        # start right, contract with psi
        BBp = torch.einsum(
            "...ijk, ...ljk->...il", self._B[self.num_channels - 1], psi[self.num_channels - 1].conj()
            )

        for i in range(self.num_channels-2, -1, -1):

            B_self, B_other = self._B[i], psi[i]

            BBp = torch.einsum(
                    "...lm, ...irl, ...krm->...ik", BBp, B_self, B_other.conj()
                )
            
        BBp = torch.einsum(
                "...ii", BBp
                )
        
        return torch.abs(BBp) ** 2
    

    def local_expectation(
            self, 
            i: int, 
            op: torch.Tensor, 
            ensemble_average: bool=True, 
            hermitian: bool=True
            ) -> torch.Tensor:
        """
        Compute expectation value of local (single-mode) operator

        Parameters
        ----------
        i : int
            Mode for computatio
        op : torch.Tensor
            The operator for contraction (matrix of physical dimension)
        ensemble_average : bool (default True)
            Compute average over ensemble states
        hermitian : bool (default True)
            If operator is Hermitian (only real values tracked)

        Returns
        -------
        torch.Tensor
            The expectation value, array if ensemble_average not True
        """

        # get SV rescaled B
        B_resc = irescale(self._B[i], self._SL[i], ind=-3)

        # contract with operator
        expect_op = torch.einsum(
            "...iqj, ...irj, qr->...", B_resc, B_resc.conj(), op.to(B_resc.device)
        )

        # If ensemble average, give mean over batch dimension
        if ensemble_average is True:
            expect_op = expect_op.mean()

        # If hermitian give real part
        return expect_op.real if hermitian else expect_op
    

    def number_outcomes(self, ensemble_average: bool=True) -> List[torch.Tensor]:

        """
        Get number outcomes (expectation density operator) across modes.

        Parameters
        ----------
        ensemble_average: bool (default True)
            Compute ensemble average of number outcomes (not per batch state)

        Returns
        -------
        List[torch.Tensor]
            A list of tensors with number outcomes

        """

        numbers = []
        n_scale = torch.arange(self.d, dtype=torch.float64, device=self._B[0].device)
        for i in range(self.num_channels):
            B_resc = irescale(self._B[i], self._SL[i], ind=-3)
            ns = torch.einsum(
                "...ijk, ...ijk->...", irescale(B_resc, n_scale, ind=-2), B_resc.conj()
                ).real
            
            ns = ns if B_resc.dim() == 3 else ns / B_resc.shape[0]
            numbers.append(ns)

        if ensemble_average is True:
            return [ns.mean() for ns in numbers]

        return numbers

    

    def ptrace(self, sites: int|Iterable[int], extent: int=1, keep_batch: bool=False) -> torch.Tensor:

        """
        Compute partial trace and get reduced density matrix. 
        Default is single-site reduced density matrix (can be written more clean)

        Parameters
        ----------
        sites: int|Iterable[int]
            The modes contained in reduced density matrix (the ones not traced out)
        extent: int (default 1)
            If only one mode is given, the extent of subsequent modes to contain in density matrix
        keep_batch: bool (default False)
            Keep the batch dimension, otherwise it is averaged out over batch states

        Returns
        -------
        torch.Tensor
            The density matrix or batch of density matrices for different batch state (keep_batch False or True)
        """

        if isinstance(sites, int):
            sites = list(range(sites, sites + extent))
        
        if isinstance(sites, Iterable):

            # no negative things in array
            sites = [s if s >= 0 else self.num_channels + s for s in sites]
            
            # make swure it is sorted
            sites = sorted(sites)
            i0 = sites[0]
            il = sites[-1]

            # compute rho_L for site i0
            rho_L = torch.einsum(
                "...imk, ...inl->...mnkl",
                irescale(self._B[i0], self._SL[i0]**2, ind=-3), self._B[i0].conj()
                )
            
            if self.has_batch_dim and not keep_batch:
                rho_L = rho_L.mean(dim=0)
            
            # loop over intermediate sites
            for i in range(i0 + 1, il + 1):
                if i in sites:

                    # no batch in MPS (not recommended, set to trivial batch_dim=1)
                    if not self.has_batch_dim:
                        rho_L = torch.einsum(
                            "...kl, kim, ljn->...ijmn",
                            rho_L, self._B[i], self._B[i].conj()
                            )

                    # if batches not preserved, average out 
                    elif self.has_batch_dim and not keep_batch:
                        rho_L = torch.einsum(
                            "...kl, bkim, bljn->...ijmn",
                            rho_L, self._B[i], self._B[i].conj()
                            ) / self.batch_dim

                    # else, keep batch dim in contrction    
                    else:
                        rho_L = torch.einsum(
                            "..bkl, bkim, bljn->..bijmn",
                            rho_L, self._B[i], self._B[i].conj()
                            ) / self.batch_dim


                else:
                    if self.has_batch_dim and not keep_batch:
                        rho_L = torch.einsum(
                            "...kl, bkjm, bljn->...mn",
                            rho_L, self._B[i], self._B[i].conj()
                            ) / self.batch_dim
                    else:
                        rho_L = torch.einsum(
                            "...kl, kjm, ljn->...mn",
                            rho_L, self._B[i], self._B[i].conj()
                            )
                    
            rho_L = torch.einsum("...kk", rho_L)
            return rho_L

        if sites < 0:
            sites = self.num_channels + sites

        B_resc = irescale(self._B[sites], self._SL[sites], ind=-3)
        if extent == 1:
            rho = torch.einsum(
                    "...iqj, ...irj->...qr", B_resc, B_resc.conj()
                )
            
            return rho if (rho.dim() == 2 or keep_batch) else rho.mean(axis=0)

        rho_L = torch.einsum(
                    "...iqj, ...irk->...qrjk", B_resc, B_resc.conj()
                )
        
        # add virtual batch dimension if not existing
        if rho_L.dim() == 4:
            for j in range(1,extent):
                # contract rho_L with next site B's

                rho_L = torch.einsum(
                        "...jk, jqm, krn->...qrmn",
                        rho_L, self._B[sites + j], self._B[sites + j].conj(),
                    )
                
            return torch.einsum("...ll", rho_L) # contract last index

        else:
            for j in range(1,extent):
                # contract rho_L with next site B's

                rho_L = torch.einsum(
                        "b...jk, b...jqm, b...krn->b...qrmn",
                        rho_L, self._B[sites + j], self._B[sites + j].conj(),
                    )
                
            return torch.einsum("...ll", rho_L).mean(axis=0) # contract last index
        
    def get_entanglement_entropy_profile(self, alpha: int=1):

        """
            Compute entanglement profile of MPS. Make sure it is in canonical form. 
            alpha sets Renyi order of entanglement (p=0: bond dimension, p=1: Von Neumann entropy).
        """

        S_profile = []
        for s in self._SL:

            prob = s ** 2
            
            # von neumann (default)
            if alpha == 1:
                # add epsilon for zero s-values, first sum over bond dimension (SVs), then take mean over batch dimension
                S_profile.append(
                    -(prob * torch.log(prob + 1e-15)).sum(axis=-1).mean() 
                    )
            
            # higher renyi order
            else:
                S_profile.append(
                    1. / (1. - alpha) * torch.log((prob ** alpha).sum(axis=-1).mean())
                    )

        return torch.stack(S_profile)

        



class StateCreator:

    """
    Class for state generation, including a number of bosonic states, such as
    Fock states, coherent states, cat or Bell states

    Parameters
    ----------

    Nmax: int
        The maximal number of photons. Physical dimension is d = Nmax + 1 (with vacuum)
    num_batch: int|None (default None)
        number of batch states, setting the batch dimension for MC sampling

    """

    def __init__(self, Nmax: int, num_batch: int|None=None):

        """Initialize state generator"""

        # copy parameters
        self.Nmax = Nmax
        self.d = Nmax + 1 # the array dimension (+ vacuum)
        self.num_batch = num_batch

        # boson operators
        self.ops = BosonOperatorsTorch(Nmax)

        # the vacuum state
        self.vac = torch.zeros(Nmax + 1, dtype=torch.complex128)
        self.vac[0] = 1

        # define operators for displacement, squeezing and thermal state
        self.D = lambda alpha: torch.matrix_exp(alpha * self.ops.ad - np.conj(alpha) * self.ops.a if isinstance(alpha, float)
                                else alpha * self.ops.ad - alpha.conj() * self.ops.a)
        self.S = lambda xi: torch.matrix_exp(
            0.5 * (np.conj(xi) * self.ops.a @ self.ops.a + xi * self.ops.ad @ self.ops.ad)
            )
        self.rho_th = lambda n_th: torch.diag(
            1. / (n_th + 1.) * (n_th / (n_th + 1.)) ** np.arange(Nmax + 1)
        ) if n_th > 1e-8 else self.vac.T @ self.vac


    def fock(self, n: int, to: str|None=None, direct=False) -> torch.Tensor:

        """Generate single-mode Fock state |n>"""

        state = torch.tensor([1 if i==n else 0 for i in range(self.d)], dtype=torch.complex128, device=to)
        if direct:
            return state
        
        # else make right MPS form
        state = torch.unsqueeze(
            torch.unsqueeze(state, 0), 
            -1)
        
        if self.num_batch is not None:
            state = state.repeat(self.num_batch, *[1] * len(state.shape))
        
        return state if to is None else state.to(to)
    

    def product_state_fock(
            self, nums: Iterable[int], to: str|None=None, direct=False
            ) -> List[torch.Tensor]:

        """Generate product Fock state. Numers are given as integer input."""
        return [self.fock(n, to=to, direct=direct) for n in nums]
    

    def single_mode_coherent(self, alpha: torch.Tensor|complex, to: str|None=None) -> torch.Tensor:

        """Generate single-mode coherent state |alpha>"""

        Cn = self.D(alpha) @ self.vac
        
        state = torch.unsqueeze(torch.unsqueeze(Cn, 0), -1)
        
        if self.num_batch is not None:
            state = state.repeat(self.num_batch, *[1] * len(state.shape))

        return state if to is None else state.to(to)

    
    def product_state_coherent(
            self, alphas: Iterable, to: str|None=None
    ) -> List[torch.Tensor]:
        
        """Generate product state of coherent states"""
        
        return [self.single_mode_coherent(alpha, to=to) for alpha in alphas]
    
    

    def single_mode_cat(
            self, 
            alpha: torch.Tensor|complex, 
            to: str|None=None, 
            phase: float|str=0., 
            direct: bool=True
            ) -> torch.Tensor:
        
        """
        Generate single-mode cat state N |alpha> + e^{i*phi}|-alpha>. 
        "direct" indicates wether to leave out L and R default bond indices
        """
        
        # check if even or odd state required, give right phase
        if isinstance(phase, str):
            phase = 0. if phase == 'even' else np.pi
        
        # construct coefficients and normalize
        Cn = (self.D(alpha) @ self.vac + np.exp(1j*phase) * self.D(-alpha) @ self.vac).to(to)
        Cn = Cn / Cn.norm()

        # if no batch indices are added
        if direct:
            return Cn
        
        state = torch.unsqueeze(torch.unsqueeze(Cn, 0), -1)
        
        if self.num_batch is not None:
            state = state.repeat(self.num_batch, *[1] * len(state.shape))

        return state if to is None else state.to(to)

    
    def two_mode_NOON(self, n: int=1, phase: float|str='even', to: str|None=None, direct=True) -> torch.Tensor:

        """Two-mode N00N state"""
        
        # check if even or odd state required, give right phase
        if isinstance(phase, str):
            phase = 0. if phase == 'even' else np.pi

        Cn = torch.outer(*self.product_state_fock([n,0], direct=True)) + np.exp(1j * phase) * torch.outer(*self.product_state_fock([0,n], direct=True))
        Cn = Cn / Cn.norm()
        Cn = Cn.to(to)

        # if no batch indices are added
        if direct:
            return Cn
        
        state = torch.unsqueeze(torch.unsqueeze(Cn, 0), -1)
        
        if self.num_batch is not None:
            state = state.repeat(self.num_batch, *[1] * len(state.shape))

        return state if to is None else state.to(to)
    

def concat_ensemble_from_list(list_psi: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
        List of MPS into one ensemble MPS by extending tensors with one batch index. 
        all tensor expected to be of same shape (can be generalized later if needed)
    """

    num_channels = len(list_psi[0])
    num_states = len(list_psi)

    ensemble_psi = []
    for ind in range(num_channels):
        ensemble_psi.append(
            torch.cat([list_psi[i_state][ind] for i_state in range(num_states)], dim=0)
        )

    return ensemble_psi

