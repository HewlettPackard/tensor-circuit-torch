import torch
import numpy as np
import math
from typing import List, Iterable, Dict, Any, Callable, Tuple
from abc import ABC, abstractmethod

from .utils import BosonOperatorsTorch, irescale, iregroup
from .mps_torch import MPStorch
from .svd_trunc import svd_trunc


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

    def __init__(self, Nmax: int, dt: float, to: str="cuda:0"):

        """Initializer of CircuitGate"""

        self.Nmax = Nmax
        self.dt = dt if dt is not None else 1.
        self.to = to

        self.d = Nmax + 1
        self.ops = BosonOperatorsTorch(Nmax, to=to)

    # add this function everywhere!!! (more clean than having the svd decomposition done in the .circuit evaluation loop)
    @abstractmethod
    def apply_to_state(
        self, 
        psi: MPStorch, 
        i: int, 
        set_canonical: bool=True, 
        i_batch: int|None=None,
        options: Dict[str, Any]={'max_BD': 100, 'cutoff':1e-8}
        ):

        """Apply to state, to be implemented in child classes"""

        pass


    def update(self):

        """Class updates, e.g., after iteration in optimization."""

        return None
    

class CircuitGateOneSite(CircuitGate):

    """Single-mode circuit gates"""

    @abstractmethod
    def apply_to(
        self, 
        B: torch.Tensor, 
        no_svd: bool=True, 
        i_batch: int|None=None, # default apply to all batches
        options: Dict[str, Any]={'max_BD': 100, 'cutoff':1e-8}
        ) -> torch.Tensor|List[torch.Tensor]:

        """
        Abstract method to apply gate to mode tensor B.
        
        Parameters
        ----------
        B: torch.Tensor
            The MPS mode tensor to apply the gate to
        
        no_svd: bool (default True)
            Whether to apply no SVD after application (default True, for unitary gates)

        i_batch: int|None (default None)
            Apply to a specific batch state (if None (default), applied to all batches)

        options: Dict[str, Any] (default {'max_BD': 100, 'cutoff':1e-8})
            The maximal BD and cutoff to attain (irrelevant if no SVD)

    
        """

        pass  # Abstract method: must be overridden in subclasses

    def apply_to_state(
        self, 
        psi: MPStorch, 
        i: int, 
        i_batch: int|None=None,
        ):
        """
        Apply gate to MPS psi

        Parameters
        ----------
        psi: MPStorch
            The MPS state 

        i: int
            mode to apply it to

        i_batch: int|None (default None)
            The batch state to apply (default None applies equally to all batch states)
        
        """

        Bi = psi[i]

        if i_batch is not None:
            Bi[i_batch] = torch.einsum("ij, kjl->kil", self.unitary_gate, Bi[i_batch])

        else:
            Bi = torch.einsum("ij,...kjl->kil", self.unitary_gate, Bi)

    

class CircuitGateTwoSite(CircuitGate):

    """
    Parent class for two-mode gates (adjacent)

    """

    @abstractmethod
    def apply_to(
        self, 
        theta: torch.Tensor, 
        no_svd: bool=True, 
        i_batch: int|None=None,
        options: Dict[str, Any]={'max_BD': 100, 'cutoff':1e-8}
        ) -> torch.Tensor|List[torch.Tensor]:

        """Abstract method for two-mode gate contraction"""

        pass  

    

        
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
                 J: torch.Tensor|float|None=None, 
                 U: torch.Tensor|float|None=None, 
                 dt: float=1.,
                 to: str="cuda:0",
        ):

        """ Nonlinear coupling gate initializer """

        super().__init__(Nmax=Nmax, dt=dt, to=to)

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
        gate = torch.matrix_exp(
            iregroup(-1j * self.dt * self.H, [[0,1], [3,2]]) # watch out with regrouping inds for exponent! do no mix L and R site of first and last index of matrix...
            )

        # reshape back to rank-4 tensor and permute indices correctly
        self.gate = gate.reshape(self.d, self.d, self.d, self.d).permute([0,1,3,2])


    def apply_to(
            self, 
            theta: torch.Tensor, 
            no_svd: bool=True, 
            options: Dict[str, Any]={'max_BD': 100, 'cutoff':1e-8}
            ) -> torch.Tensor|Tuple[torch.Tensor]:
        
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
        theta_U = torch.einsum("...ijkl, mnkj -> ...imnl", theta, self.gate)

        # return if no SVD required
        if no_svd:
            return theta_U
        
        # else decompose tensor for return
        theta_U = theta_U.reshape((self.d * theta_U.shape[0], self.d * theta_U.shape[-1]))


        # decompose with SVD
        U, s, Vd = svd_trunc(theta_U, cutoff=options['cutoff'], max_num=options['max_BD'])

        bdim = s.numel()
        return U.reshape(-1, self.d, bdim), s, Vd.reshape(bdim, self.d, -1)
    

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
                 U: torch.Tensor|float|None=None, 
                 dt: float=1.,
                 to: str="cuda:0",
                 ):
        
        """Initializer of single-mode nonlinear gate"""

        super().__init__(Nmax, dt, to)
        
        self.U = U

        self.H = 0.5 * U * self.ops.n_nm1
        # for cdc in cdc_ops:
        #     self.H = self.H - 1j * 0.5 * cdc 
        self.gate = torch.matrix_exp(-1j * dt * self.H)
    
    def apply_to(self, B: torch.Tensor):
        
        """Single-mode apply"""

        return torch.einsum('ij, ...kjl->...kil', self.gate, B)
    

class PhotonInjectionGate(CircuitGateOneSite):

    """Gate to inject a photon -- application of a^\dagger."""

    def __init__(self, Nmax: int, 
                 to: str="cpu", dt: float=1.
                 ):
        super().__init__(Nmax, dt, to)

    
    def apply_to(self, B: torch.Tensor):
        return torch.einsum('ij, ...kjl->...kil', self.ops.ad, B)
    
class ProjectNumberStateGate(CircuitGateOneSite):

    """Gate to project mode onto a number state. takes number as input -- application of operator P_n."""

    def __init__(self, 
                 Nmax: int, 
                 num_state: int=1,
                 to: str="cpu", 
                 dt: float=1.
                 ):
        super().__init__(Nmax, dt, to)
        self.num_state = num_state
        self.gate = self.ops.proj[self.num_state]

    
    def apply_to(self, B: torch.Tensor):
        return torch.einsum('ij, ...kjl->...kil', self.gate, B)
    

class PhotonAnnihilationGate(CircuitGateOneSite):

    """Get to annihilate a photon -- application of operator a."""

    def __init__(self, Nmax: int, 
                 to: str="cpu", dt: float=1.
                 ):
        super().__init__(Nmax, dt, to)
        
        self.Nmax = Nmax
        self.ops = BosonOperatorsTorch(Nmax, to=to)
    
    def apply_to(self, B: torch.Tensor):
        return torch.einsum('ij, ...kjl->...kil', self.ops.a, B)


class LocalPhaseShiftGate(CircuitGateOneSite):

    """Apply local phase shift to the mode."""

    def __init__(self, Nmax: int, 
                 phi: torch.Tensor|float|None=None, 
                 dt: float=1,
                 to: str="cpu",
                 ):
        super().__init__(Nmax, dt, to)
        
        self.phi = phi

        # the gate Hamiltonian and unitary matrix
        self.H = phi * self.ops.n
        self.gate = torch.matrix_exp(-1j * dt * self.H)


    def apply_to(self, B: torch.Tensor):
        return torch.einsum('ij, ...kjl->...kil', self.gate, B)
    

class BatchDependentLocalPhaseShiftGate(CircuitGateOneSite):

    """Give a different shift to all batch states. Receives list of length num_batch with phases as input."""

    def __init__(self, 
                 Nmax: int, 
                 phis: List[float], 
                 dt: float=1,
                 to: str="cpu",
                 ):
        super().__init__(Nmax, dt, to)
        
        self.phis = phis

        # create gate with batch dependent phase shifts
        self.gate = torch.zeros((len(phis), *self.ops.id.shape), device=to, dtype=torch.complex128)
        for b, phi in enumerate(phis):
            self.gate[b] = torch.matrix_exp(-1j * dt * phi * self.ops.n)


    def apply_to(self, B: torch.Tensor):
        return torch.einsum('bij, bkjl -> bkil', self.gate, B)
    

class CircuitLocalLossGate(CircuitGateOneSite):

    """
    The class for sampling the photon loss events , with loss rate gamma and time duration dt.
    gamma * dt gives the probability of photon loss.
    """

    def __init__(
        self, 
        Nmax: int, 
        gamma: 1.,
        dt: float=None,
        to: str="cpu",
    ):
        super().__init__(Nmax, dt, to)
        
        self.gamma = gamma

        self.gate = torch.matrix_exp(-0.5 * dt * gamma * self.ops.n)
        self.n_scale = torch.arange(Nmax + 1, device=to)

    def apply_no_jump_to(self, B: torch.Tensor):
        return torch.einsum('ij, ...kjl->...kil', self.gate, B)
    
    def apply_jump_to(self, B: torch.Tensor):
        return torch.einsum('ij, ...kjl->...kil', self.ops.a, B)
    
    def apply_to(self, B: torch.Tensor, SL: torch.Tensor):

        # apply non-unitary gate (stored for later)
        B_out = torch.einsum("ij, bkjl->bkil", self.gate, B)
        
        # apply correct irescale on B (not B_out!), with num_vals and SVs, before contracting with its conjugate to get the densities
        ns = torch.einsum(
            "bjkl, bjkl-> b", 
            irescale(irescale(B, self.n_scale, ind=-2), SL ** 2, ind=-3), B.conj()
            ).abs()
        
        # evaluate probilities and see which ones click
        Ps = self.gamma * ns * self.dt

        max_P = torch.max(Ps)
        if max_P > 0.1:
            Warning(f"Too high probabilities (max P = {max_P}), choose lower dt.")

        # sample where the loss clicks occur
        is_click = (Ps > torch.rand_like(ns))

        # construct operator with click operator or non-herm evolution on right indices.
        # see if I can make this better than a simple for loop over batch states
        click_ops = torch.zeros(ns.numel(), *self.ops.a.shape, device=self.to, dtype=B.dtype)
        for i, jump in enumerate(is_click):
            if jump:
                click_ops[i] = self.ops.a
            else:
                click_ops[i] = self.ops.id 

        return torch.einsum("bij, bkjl->bkil", click_ops, B_out)
    


class NonlinearPhotonicCircuit:

    """
    The class to perform the forward run of evaluating the the outcome of a nonlinear photonic circuit.

    num_channels: int
        The number of circuit channels.

    num_layers: int
        The number of circuit layers.

    Nmax: int
        The maximal number of photons. Fock dimension is d = Nmax + 1
    layer_depth: List[float]|float (default 1,)
        The time duration of the layers. Default is uniform 1. (the time unit).

    start_ind: bool (default 0)
        For the two-mode gates, whether to start at index 0 or 1.

    J_init: float|Iterable[Iterable[float]] (default 0.)
        The initial J-values. By default all set to zero.
    U_init: float=0.
        The initial value for U (remains fixed, for now, not tunable)
    gamma: float (default None)
        The photon loss dissipation rate. If None (default) no dissipation included.
    clicks_per_layer: int|None=None
        If more than one sampling per click is needed. Avoid, just alter circuit layer definition.

    to: str (default "cpu")
        Device to run (cpu default)

    """

    def __init__(self, num_channels: int, num_layers: int, Nmax: int, layer_depth: List[float]|float=1.,
                 start_ind: bool=0, J_init: float|Iterable[Iterable[float]]=0., U_init: float=0., gamma: float=None, 
                 clicks_per_layer: int|None=None, to: str="cpu",
                 ):
        
        """Initializer for circuit simulation class NonlinearPhotonicCircuit."""

        # copy parameters into class variables
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.start_left = start_ind
        self.J_init = J_init
        self.U_init = U_init
        self.Nmax = Nmax
        self.d = Nmax + 1
        self.layer_depths = layer_depth if isinstance(layer_depth, Iterable) else [layer_depth] * num_layers
        self.clicks_per_layer = clicks_per_layer
        self.device = to

        # the differential elements
        self.delta_t = [d if clicks_per_layer is None else d / clicks_per_layer
                   for d in self.layer_depths]
        
        # create circuit gates with given J
        self.gate_set = []
        self.side_gate_set = []
        self.loss_gate_set = []
        for nl in range(num_layers):
            start = (start_ind + nl) % 2
            gate_set_l = []
            for j in range(num_channels - 1):

                # IMPROVE !!! shitty coding and instance checking, initiate J_init as list of nested layers in init if J_init float
                J_np = J_init[nl][j] if isinstance(J_init, Iterable) else J_init
                if J_np is None:
                    gate = None
                else:

                    # J are tensors of which gradients are tracked for final FOM (specify requires_grad=True)
                    J = torch.tensor(J_np, requires_grad=True)

                    # construct gate
                    gate = (
                        NonlinearCouplingGate(
                            Nmax, J, 
                            U_init, 
                            self.delta_t[nl], 
                            to=to,
                            ) if (j - start) % 2 == 0
                        else None
                    )
                gate_set_l.append(gate)
            
            self.gate_set.append(gate_set_l)

            # append edge gates
            d_sides = {}
            if start == 1:
                d_sides['L'] = NonlinearLocalGate(Nmax, U_init, self.delta_t[nl], to=to)
            if (num_channels - start) % 2 == 1:
                d_sides['R'] = NonlinearLocalGate(Nmax, U_init, self.delta_t[nl], to=to)
            self.side_gate_set.append(d_sides)

        # add dissipation gate if needed (constant, for now)
        if gamma is not None and gamma > 0.:
            self.loss_gate_set.append(
                CircuitLocalLossGate(Nmax, gamma=gamma, dt=self.delta_t[nl], to=to)
                )

    
    def update(self):

        """Perform circuit update by updating all gate (after gradient update J-values)"""

        for nl in range(self.num_layers):
            for i in range(self.num_channels -1):
                gate = self.gate_set[nl][i]
                if gate is None:
                    continue
                gate.update()
    
    
    def apply_gate(self, nl: int, i: int, psi: MPStorch, options: Dict[str, Any]):

        """Apply the gate in layer nl, at position i, on state psi, using the options (Dict)."""

        # get gate set for layer
        gates = self.gate_set[nl]

        # left and right idxs
        i0, i1 = i, i + 1

        # apply U to tensor C = B_L * B_R
        C = psi.get_C(i0)  # the two Bs without the S on the left
        C = gates[i0].apply_to(C, options=options)
        theta = irescale(C, psi.get_SL(i0), ind=-4) # multiply (i.e. rescale index) with S on the left

        U, s, Vd, renormalize_factor = svd_trunc(
            iregroup(theta, [[-4,-3], [-2,-1]]),
            cutoff=options['cutoff'],
            max_num=options['max_BD']
            )

        # Split tensor and update matrices
        bdim = s.shape[-1]
        B_L = irescale(
            torch.einsum('...ijk,...lk->...ijl', iregroup(C,[[-4], [-3], [-2,-1]]), Vd.conj()), 
            1. / renormalize_factor,
            ind=0)

        B_R = Vd.view([bdim, self.d, C.shape[-1]] if Vd.dim() == 2 else [-1, bdim, self.d, C.shape[-1]])

        psi[i1] = B_R
        psi[i0] = B_L
        psi.set_SL(i1, s)


    def run(self, 
            psi: MPStorch, 
            options: Dict[str, Any]={'max_BD': 100, 'cutoff': 1e-8}
            ):
        
        """
        Run the circuit on an MPS

        Parameters
        ----------
        psi: MPStorch
            The MPS initial state

        options: Dict[str, Any] default {'max_BD': 100, 'cutoff': 1e-8})
            The options to run with -- max_BD and cutoff for SVs
        """

        # the number of quantum jump click breaks
        num_breaks = self.clicks_per_layer if self.clicks_per_layer is not None else 1
        
        # loop over circuit layers
        for d in range(self.num_layers):

            # loop over breaks for quantum MC jump evaluations
            for i_break in range(num_breaks):
            
                # apply dissipation and put in canonical form
                for i in range(self.num_channels):
                
                    for loss_gate in self.loss_gate_set:
                        psi[i] = loss_gate.apply_to(psi[i], psi.get_SL(i))

                        if torch.isnan(psi[i]).any(): # for debugging
                            e=1

                        # make sure psi is in canonical form -- THIS MUST BE IMPROVED IN FUTURE!!!
                        # -> now canonicalization is performed after EVERY jump, this can be made
                        # more efficiently by syncing with QR decompositions (canonical point is il=0)
                        psi.canonical_form(cutoff=options['cutoff'], max_num=options['max_BD'])

                # apply local unitary side gates
                if 'L' in self.side_gate_set[d]:
                    B = self.side_gate_set[d]['L'].apply_to(psi[0])
                    psi[0] = B
                if 'R' in self.side_gate_set[d]:
                    B = self.side_gate_set[d]['R'].apply_to(psi[-1])
                    psi[-1] = B

                # loop and apply gates to psi
                for i in range(self.num_channels-1):

                    # check if there is a gate
                    if self.gate_set[d][i] is None or self.layer_depths[d] < 0.:
                        continue

                    # apply the gate
                    self.apply_gate(nl=d, i=i, psi=psi, options=options)
                        

    def get_params(self):

        """Get the differentiable parameters (J couplings) to pass to optimizer (Adam)"""

        params = [gate.J for gates in self.gate_set for gate in gates if gate is not None]
        return params


    def get_circuit_Js(self, to_numpy: bool=True):

        """Get the circuit J values. Indicate if they should be converted to numpy."""

        return [
            [None if gate is None or gate.J is None else float(gate.J.detach().numpy()) for gate in gates] 
            for gates in self.gate_set
            ]
    

    def get_circuit_J_grads(self, to_numpy: bool=True):

        """Get the gradients to the J parameters. Indicate if they should be converted to numpy or not."""

        return [
            [None if gate is None or gate.J.grad is None else float(gate.J.grad.numpy()) for gate in gates] 
            for gates in self.gate_set
            ]