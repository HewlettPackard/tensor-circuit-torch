import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Iterable

from src.utils import BosonOperatorsTorch, iregroup, irescale, Haar_unitary
from src.svd_trunc import svd_trunc
from src.mpdo_torch import MPDOtorch

class CircuitGate:

    """
    Parent class for circuit gates.

    Parameters
    ----------
    Nmax: int
        Bosonic Fock space truncation, dimension d = Nmax+1

    dt: float
        Differential time step, gate duration, better said

    to: str (default "cuda:0")
        The device to run (first cuda GPU by default)

    Attributes
    ----------
    H: torch.Tensor
        The corresponding Hamiltonian
    
    gate_tensor: torch.Tensor
        The tensor of the gate, exp(-i * dt * H)

    is_unitary: bool (default True)
        Whether gate is unitary

    """

    H: torch.Tensor = None
    gate_tensor: torch.Tensor = None
    is_unitary = True

    def __init__(self, Nmax: int, dt: float=1., device: str="cuda:0", gate_tensor: torch.Tensor=None):

        """Initializer of CircuitGate"""

        self.Nmax = Nmax
        self.dt = dt if dt is not None else 1.
        self.device = device
        self.gate_tensor = gate_tensor

        self.d = Nmax + 1
        self.ops = BosonOperatorsTorch(Nmax, device=device)

    @abstractmethod
    def apply_to(
        self, 
        rho: MPDOtorch, 
        site: int,
        options: Dict[str, Any]={'max_BD': 100, 'cutoff_BD':1e-8}
        ):

        """Apply to state, to be implemented in child classes"""

        pass


    def update(self):

        """Class updates, e.g., after iteration in optimization."""

        return None
    

    def get_params(self) -> Iterable[torch.Tensor]:
        return []



    

class CircuitGateOneSite(CircuitGate):

    """Single-mode circuit gates"""

    def apply_to(
        self, 
        rho: MPDOtorch, 
        site: int, 
        options: Dict[str, Any]={'max_BD': 100, 'cutoff_BD':1e-8}
        ):
        """
        Apply gate to MPS psi

        Parameters
        ----------
        psi: MPStorch
            The MPS state 

        site: int
            mode to apply it to

        i_batch: int|None (default None)
            The batch state to apply (default None applies equally to all batch states)
        
        """

        rho[site] = torch.einsum("ij,bkjl->bkil", self.gate_tensor, rho[site])
        

    

class CircuitGateTwoSite(CircuitGate):

    """
    Parent class for two-mode gates (adjacent)

    """

    def apply_to_C(
        self, 
        C: torch.Tensor, 
        options: Dict[str, Any]={'max_BD': 100, 'cutoff_BD':1e-8}
        ) -> torch.Tensor:
        
        """
        Apply the gate to two-mode MPS tensor
        
        Parameters
        ---------
        theta: torch.Tensor
            The two-mode contracted tensor of the MPS to apply the gate to

        no_svd: bool=True
            Return the two-mode full tensor if True. 
            Else, perform truncated SVD and return U, s, Vd tensors

        options: Dict[str, Any] (default {'max_BD': 100, 'cutoff':1e-8})
            Options for max BD and cutoff SVD
        
        """

        # compute tensor product of gate contraction
        U_C = torch.einsum("bijkl, mnkj -> bimnl", C, self.gate)
        return U_C
    

    def apply_to(
        self, 
        rho: MPDOtorch, 
        site: int,
        options: Dict[str, Any]={'max_BD': 100, 'cutoff_BD':1e-8},
        normalize: bool=True
        ):

        # left and right idxs
        i0, i1 = site, site + 1

        # apply U to tensor C = B_L * B_R
        C = rho.get_C(i0)  # the two Bs without the S on the left
        C = torch.einsum("bijklc, mnkj -> bimnlc", C, self.gate_tensor)
        theta = irescale(C, rho.get_SL(i0), ind=-5) # multiply (i.e. rescale index) with S on the left

        U, s, Vd, renormalize_factor = svd_trunc(
            iregroup(theta, [[0,1,2], [3,4,5]]),
            cutoff=options['cutoff_BD'],
            max_num=options['max_BD'],
            lowrank=False,
            normalize=normalize
            )

        # left tensor is C contracted with Vd (see tenpy source code)
        B_L = irescale(
            torch.einsum('bijk,lk->bijl', iregroup(C,[[0], [1], [2], [3,4,5]]), Vd.conj()), 
            1. / renormalize_factor,
            ind=0)
        
        # right tensor is reshaped
        B_R = Vd.reshape([s.numel(), self.d, C.shape[-2], -1])
        # B_R = Vd.view([bdim, self.d, C.shape[-1]] if Vd.dim() == 2 else [-1, bdim, self.d, C.shape[-1]])

        # update incoming state with contracted tensors
        rho[i1] = iregroup(B_R, [[3],[0],[1],[2]])
        rho[i0] = B_L
        rho.SL[i1] = s


class NonlinearCouplingGate(CircuitGateTwoSite):

    """
    The elementary to-mode nonlinear coupling gate, of which the qPIC is composed

    Parameters
    ----------
    Nmax: int
        Fock space truncation. Physical dim d = Nmax + 1 

    J: torch.Tensor|float|None (default None)
        The coupling rate

    U: torch.Tensor|float|None (default None)
        The nonlinear rate

    dt: float (default 1.)
        Time duration of gate

    to: str (default "cuda:0")
        Device to run on

    """

    def __init__(self, Nmax: int, 
                 J: torch.Tensor, 
                 U: torch.Tensor, 
                 dt: float=1.,
                 device: str="cuda:0",
        ):

        """ Nonlinear coupling gate initializer """

        super().__init__(Nmax=Nmax, dt=dt, device=device)

        # If None, should be set to zero
        self.J = J if J is not None else 0.
        self.U = U if U is not None else 0.

        # define the two-mode tunneling gate
        self.tunneling_gate = -(torch.einsum('ij,kl->iklj', self.ops.ad, self.ops.a) 
            + torch.einsum('ij,kl->iklj', self.ops.a, self.ops.ad)
        )

        # define the two-mode (separate) interaction gate
        self.interaction_gate = 0.5 * (
            torch.einsum('ij,kl->iklj', self.ops.n_nm1, self.ops.id) 
            + torch.einsum('ij,kl->iklj', self.ops.id, self.ops.n_nm1) 
        )

        # initialize gate by updating
        if not any([J, U, dt]) is None:
            self.update()


    def update(self) -> None:

        """Update the gate with the actual values for J, U, dt"""

        # define Hamiltonian as sum tunneling and interaction
        self.H = (self.J * self.tunneling_gate + self.U * self.interaction_gate)

        # imaginary exponent for unitary gate. Reshape rank-4 tensor in matrix by joining two indices and exponentiate
        gate_tensor = torch.matrix_exp(
            iregroup(-1j * self.dt * self.H, [[0,1], [3,2]]) # watch out with regrouping inds for exponent! do no mix L and R site of first and last index of matrix...
            )

        # reshape back to rank-4 tensor and permute indices correctly
        self.gate_tensor = gate_tensor.reshape(self.d, self.d, self.d, self.d).permute([0,1,3,2])


    def get_params(self):
        return [var for var in [self.J, self.U] if var.requires_grad]


class HaarCouplingGate(CircuitGateTwoSite):

    def __init__(self, Nmax=1, device: str="cuda:0",):

        self.Nmax = Nmax
        self.device = device
        self.d = Nmax + 1

        # create unitary gate
        self.haar_unitary = Haar_unitary(n=self.d ** 2, device=self.device)

        shape = [self.d] * 4
        index_order = [0, 1, 3, 2] # list(range(num_modes)) + list(range(2*num_modes-1,num_modes-1,-1))
        self.gate_tensor = self.haar_unitary.view(*shape).permute(index_order)
    

class NonlinearLocalGate(CircuitGateOneSite):

    """
    One-site local nonlinearity (for edges circuit)
    
    Parameters
    ----------
    Nmax: int
        Fock space truncation 

    U: torch.Tensor|float|None (default None)
        Nonlinear rate

    dt: float (default 1.),
        Gate duration

    to: str="cuda:0"
        Device to run on

    """

    def __init__(self, Nmax: int, 
                 U: torch.Tensor, 
                 dt: float=1.,
                 device: str="cuda:0",
                 ):
        
        """Initializer of single-mode nonlinear gate"""

        super().__init__(Nmax, dt, device)
        
        self.U = U

        self.H = 0.5 * U * self.ops.n_nm1
        self.gate_tensor = torch.matrix_exp(-1j * dt * self.H)


class PhaseGate(CircuitGateOneSite):

    """
    One-site local phase gate
    
    Parameters
    ----------
    Nmax: int
        Fock space truncation 

    phi: torch.Tensor (default None)
        The phase 

    dt: float (default 1.),
        Gate duration

    to: str="cuda:0"
        Device to run on

    """

    def __init__(self, Nmax: int, 
                 phi: torch.Tensor, 
                 dt: float=1.,
                 device: str="cuda:0",
                 ):
        
        """Initializer of single-mode nonlinear gate"""

        super().__init__(Nmax, dt, device)
        
        self.phi = phi

        self.H = phi * self.ops.n
        self.gate_tensor = torch.matrix_exp(-1j * dt * self.H)

