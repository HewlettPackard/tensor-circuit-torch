import torch
import numpy as np
import math
from typing import List, Iterable, Dict, Any, Callable, TypeVar, Type
from time import time
import copy

from .utils import BosonOperatorsTorch, irescale, iregroup, sqrtm
from .svd_trunc import svd_trunc

# definition type for self-reference
T = TypeVar("T", bound="MPDOtorch")

class MPDOtorch:

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

    def __init__(self, psis: List[torch.Tensor], SL: List[torch.Tensor]|None=None, SP: List[torch.Tensor]|None=None, no_canonical_form: bool=False):

        """
        Initialize MPS state
        """

        # read input parameters
        self.d = psis[1].shape[-2]
        self.Nmax = self.d - 1
        self._B = psis
        self.num_channels = len(psis)
        self.device = psis[0].device
        self.dtype = psis[0].dtype


        # see if left SVs are given
        if SL is None and SP is None:
            self.SL = [None] * (self.num_channels + 1) # the entanglement entropy
            self.SP = [None] * (self.num_channels) # the classical entropy
            self.SL[0] = torch.tensor([1.], device=self.device)
            self.SL[-1] = torch.tensor([1.], device=self.device)
            self.canonical_form() 

        else:
            self.SL = SL
            self.SP = SP

    
    def canonical_form(self, options={'max_BD': 100, 'max_PD': 100, 'cutoff_BD': 1e-6, 'cutoff_PD': 1e-6}):
        self.krauss_dissipation(K_ops=[], options=options)


    def krauss_dissipation(
        self, 
        K_ops: Iterable[torch.Tensor], 
        options: Dict[str,Any] = {'max_BD': 100, 'max_PD': 100, 'cutoff_BD': 1e-6, 'cutoff_PD': 1e-6},
        num_select: int|None = None
        ):
        

        # local variables
        L = self.num_channels

        # krauss ops
        do_krauss = len(K_ops) > 0

        # concat K ops
        K_concat = torch.stack(K_ops, dim=0) if do_krauss else None

        # B tensor left index  
        B = self._B[0] if self.SL[0] is None else irescale(self._B[0], self.SL[0], ind=-3)

        # loop through chain  
        for il in range(L):

            start = time()

            if do_krauss:
                # contract concat'd Krauss operator with.
                B_full = torch.einsum("aij, bkjm -> abkim", K_concat, B)

                # Join index from Krauss operator with batch index from B into one batch index for B
                B_full = B_full.reshape(K_concat.shape[0] * B.shape[0], *B_full.shape[2:])
            else:
                # No Krauss applied, just get B tensor
                B_full = B


            # SVD of K * B tensor along batch dimension, 
            # select a set of states to keep track off
            U, s_purity, Vd, rescale = svd_trunc(
                B_full.reshape(B_full.shape[0], -1),
                cutoff = options["cutoff_PD"], 
                max_num = options['max_PD'],
                lowrank=True,
                num_select=num_select
            )

            # B is new Vd tensor, rescaled with obtained SVs
            B_krauss = irescale(Vd, factor=s_purity, ind=0).view(s_purity.shape + self._B[il].shape[1:])

            # QR across bond dimension
            Q, R = torch.linalg.qr(iregroup(B_krauss, [[0,1,2], [3]]), mode='reduced')

            # rescale Vd with SVs s and reshape to correct form
            self._B[il] = Q.view(B_krauss.shape)
            self.SP[il] = s_purity

            # B for next iteration
            if il != L-1:
                B = torch.einsum("ij, bjkl->bikl", R, self._B[il+1])


        # backward R->L: SVDs, for convenience, set left bond index first
        B = (
            self._B[L-1] if self.SL[L] is None 
            else irescale(self._B[L-1], self.SL[L], ind=-1)
        )
        B =  iregroup(B, [[1],[0],[2],[3]])

        # loop from R to L through chain
        for il in range(L - 1, -1, -1):

            # SV decomposition and truncate, if requested
            U, s, Vd, renormalize_factor = svd_trunc(
                    iregroup(B, [[0], [1,2,3]]), 
                    cutoff=options['cutoff_BD'], 
                    max_num=options['max_BD'],
                    lowrank=True
                    ) # 

            # update right site, make sure to swap indices
            self.SL[il] = s
            self._B[il] = iregroup(
                Vd.view([-1, B.shape[1], B.shape[2], B.shape[3]]), 
                [[1],[0],[2],[3]]
                )

            # for next iteration, always swap indices back (if not first one)
            if il != 0:
                B = iregroup(
                    torch.einsum("bijk,kl->bijl", self._B[il-1], irescale(U, s, ind=-1)),
                    [[1],[0],[2],[3]]
                )


    
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

            # compute B @ B^\dagger
            BBd = torch.einsum(
                "bijk, bljk->...il", B, B.conj()
                )
            
            # check if it is identity matrix
            B_canonical.append(
                ((torch.eye(BBd.shape[-1], device=BBd.device) - BBd) / B.numel()).norm(dim=[-2,-1]).mean() < tol
            )


        s_canonical = []
        for s in self.SL:
            s_canonical.append((s.norm(dim=-1).mean() - 1.) < tol)

        if full:
            return {
                'B': B_canonical,
                'SVs': s_canonical
                }

        return all(s_canonical + B_canonical)
    
    
    def clone(self: T) -> T:
        """
        Clone MPS into other MPS (hard copy, all grads are detached)
        """
        return MPDOtorch(
            [B.clone().detach().requires_grad_(B.requires_grad) for B in self._B],
            [S.clone().detach().requires_grad_(S.requires_grad) for S in self.SL],
            [S.clone().detach().requires_grad_(S.requires_grad) for S in self.SP],
            no_canonical_form=True
        )
    

    def ptrace(self, il: int):
        return torch.einsum(
            "bijk,bilk->jl", 
                irescale( 
                    self._B[il], self.SL[il] ** 2, ind=-3 # index rescaling with SVs (left tensor contraction)
                    ), 
            self._B[il].conj()
            )
    

    def diagonal_local_measurement(self, il: int, diagonal_els: torch.Tensor):
        return torch.einsum(
            "bijk,bijk", 
            irescale( 
                irescale( 
                    self._B[il], self.SL[il] ** 2, ind=-3 # index rescaling with SVs (left tensor contraction)
                    ), 
                    diagonal_els, ind=-2 # index rescaling with n's (diagonal operator)
                ), 
            self._B[il].conj()
            ).real

    
    def number_outcome(self, il: int) -> torch.Tensor:

        """
        Get single number outcome at mode/site index il

        """
        
        # diagonal n elements
        n_els = torch.arange(0, self.d, device=self.device)
        return self.diagonal_local_measurement(il, n_els)
    

    def number_variance(self, il: int) -> torch.Tensor:
        n = self.number_outcome(il)

        n2_els = torch.arange(0, self.d, device=self.device) ** 2
        n2 = self.diagonal_local_measurement(il, n2_els)

        return n2 - n ** 2
    

    def density_correlation(self, il: int, n_eps: float=1e-2) -> torch.Tensor:

        # get number expectation
        n = self.number_outcome(il)

        # get second order correlator < n * (n-1)>
        n_els = torch.arange(0, self.d, device=self.device)
        n_nm1_els = n_els * (n_els - 1.)
        C2 = self.diagonal_local_measurement(il, n_nm1_els)

        # return g2
        return (C2 + n_eps ** 2) / (n ** 2 + n_eps ** 2)


    def number_outcomes(self) -> torch.Tensor:
        return torch.stack([self.number_outcome(il) for il in range(self.num_channels)])
    

    def number_variances(self) -> torch.Tensor:
        return torch.stack([self.number_variance(il) for il in range(self.num_channels)])
    

    def density_correlations(self, n_eps) -> torch.Tensor:
        return torch.stack([self.density_correlation(il, n_eps=n_eps) for il in range(self.num_channels)])


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

        for S in self.SL:
            S.to(device)
    

    def get_theta(self, j: int) -> torch.Tensor:

        """ Contract C with its left SV """

        return irescale(self.get_C(j), self.SL[j], ind=0)


    def get_C(self, j) -> torch.Tensor:

        """Contract B_j and B_{j+1} into new two-mode tensor C. """

        return torch.einsum("bijk, cklm -> bijlmc", self._B[j], self._B[j+1] )
    

    def get_SL(self, i: int) -> torch.Tensor:
        """Get the left singular values at mode i"""

        return self.SL[i]


    def set_SL(self, i: int, S):
        """Set the left SVs at mode i"""
        self.SL[i] = S


    def get_BD(self, i: int) -> int:
        """Get the (right) bond dimension at mode i"""
        return self.SL[i+1].numel()
    

    def get_PD(self, i: int) -> int:
        """Get the (right) probability dimension at mode i"""
        return self._B[i].shape[0]
    

    def get_BDs(self) -> List[int]:
        """Get all BDs"""
        return [int(self.SL[i].shape[-1]) for i in range(self.num_channels + 1)]
    

    def get_PDs(self) -> List[int]:
        """Get all probability dimensions"""
        return [int(self._B[i].shape[0]) for i in range(self.num_channels )]
    

    def norm(self) -> float:
        """Overlap of state with itself -> norm of the state (should be 1 if canonical)"""
        return self.overlap(self).abs() ** 2
    

    def overlap(self, psi: T) -> torch.Tensor:

        """
        Overlap between this quantum state and some other state psi.
        This is per batch, batch 0 of one state is contracted with batch 0 from other and so on.
        """

        # start right, contract with psi
        BBp = torch.einsum(
            "bijk, bljk->il", self._B[self.num_channels - 1], psi[self.num_channels - 1].conj()
            )

        for i in range(self.num_channels-2, -1, -1):

            B_self, B_other = self._B[i], psi[i]

            BBp = torch.einsum(
                    "lm, birl, bkrm->ik", BBp, B_self, B_other.conj()
                )
            
        overlap = torch.einsum(
                "...ii", BBp
                )
        
        return overlap
    

    def local_expectation(
            self, 
            i: int, 
            op: torch.Tensor, 
            ensemble_average: bool=True, 
            hermitian: bool=True
            ) -> torch.Tensor:
        """
        REVIEW!!!!
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
        B_resc = irescale(self._B[i], self.SL[i], ind=-3)

        # contract with operator
        expect_op = torch.einsum(
            "...iqj, ...irj, qr->...", B_resc, B_resc.conj(), op.to(B_resc.device)
        )

        # If ensemble average, give mean over batch dimension
        if ensemble_average is True:
            expect_op = expect_op.mean()

        # If hermitian give real part
        return expect_op.real if hermitian else expect_op


    def entropy_profile(self, alpha: int=1, entropy='entanglement'):

        """
            Compute entanglement profile of MPS. Make sure it is in canonical form. 
            alpha sets Renyi order of entanglement (p=0: bond dimension, p=1: Von Neumann entropy).
        """

        S_profile = []
        SVDs = self.SL if entropy == 'entanglement' else self.SP
        for s in SVDs:

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
            self, nums: Iterable[int], device: str|None=None, direct=False
            ) -> List[torch.Tensor]:

        """Generate product Fock state. Numers are given as integer input."""
        return [self.fock(n, to=device, direct=direct) for n in nums]
    

    def single_mode_coherent(self, alpha: torch.Tensor|complex, to: str|None=None) -> torch.Tensor:

        """Generate single-mode coherent state |alpha>"""

        Cn = self.D(alpha) @ self.vac
        
        state = torch.unsqueeze(torch.unsqueeze(Cn, 0), -1)
        
        if self.num_batch is not None:
            state = state.repeat(self.num_batch, *[1] * len(state.shape))

        return state if to is None else state.to(to)

    
    def product_state_coherent(
            self, alphas: Iterable, device: str|None=None
    ) -> List[torch.Tensor]:
        
        """Generate product state of coherent states"""
        
        return [self.single_mode_coherent(alpha, to=device) for alpha in alphas]
    
    

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

