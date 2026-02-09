import torch
import numpy as np
import math
from typing import List, Iterable, Dict, Any, Tuple

class QubitOperatorTorch:

    d: int = 2
    Nmax: int = 1

    def __init__(self, device: str='cpu'):
        self.device = device

        self.sx = torch.tensor([[0.,1.],[1., 0.]], device=self.device, dtype=torch.complex128)
        self.sy = torch.tensor([[0.,1j],[-1j, 0.]], device=self.device, dtype=torch.complex128)
        self.sz = torch.tensor([[1.,0.],[0., -1.]], device=self.device, dtype=torch.complex128)
        self.sp = torch.tensor([[0.,1.],[0., 0.]], device=self.device, dtype=torch.complex128)
        self.sm = torch.tensor([[0.,0.],[1., 0.]], device=self.device, dtype=torch.complex128)
        self.P1 = torch.tensor([[1.,0.],[0., 0.]], device=self.device, dtype=torch.complex128)
        self.P0 = torch.tensor([[0.,0.],[0., 1.]], device=self.device, dtype=torch.complex128)
        self.id = torch.tensor([[1.,0.],[0., 1.]], device=self.device, dtype=torch.complex128)


class BosonOperatorsTorch:

    """
    The class on which the bosonic operators are initilized, for later call
    
    Parameters
    ----------
    Nmax: int
        The Fock space max. occupation. Operator dimension d = Nmax + 1
    
    to: str (default 'cpu')
        What device to initiate the object on
    """

    def __init__(self, Nmax: int, device: str='cpu'):

        self.Nmax = Nmax
        self.d = Nmax + 1
        self.device = device

        # construct operators as matrices
        self.a = torch.tensor(np.diag([np.sqrt(n) for n in range(1, Nmax+1)], k=1), dtype=torch.complex128, device=device)
        self.ad = self.a.T
        self.n = self.ad @ self.a
        self.n_nm1 = self.ad @ self.ad @ self.a @ self.a
        self.id = torch.eye(Nmax + 1, dtype=torch.complex128, device=device)

        # the projection operators, on photon state n
        self.proj = []
        for i in range(self.d):
            p = torch.zeros_like(self.a, dtype=torch.complex128, device=device)
            p[i,i] = 1.
            self.proj.append(p)


def irescale(t: torch.Tensor, factor: torch.Tensor, ind: int=0):

    """
    Rescale one tensor index with factor values, of same size. This is particularly useful for rescaling
    with SVs.

    Parameters
    ----------
    t: torch.Tensor
        The tensor of which to rescale an index
    
    factor: torch.Tensor
        The factor with which to rescale (e.g., left singular values) 

    ind: int (default 0)
        The index to rescale

    Returns
    -------
    t: torch.Tensor
        The rescaled tensor.
    """

    sh = t.shape
    lv = [1] * len(sh)
    lv[ind] = sh[ind]
    lv[:len(factor.shape)-1] = factor.shape[:-1]
    return t * factor.view(lv)


def iregroup(t: torch.Tensor|np.ndarray, regroup: List[List]):

    """
    Regroup the tensor indices. Works well for torch.Tensor, not tested properly for np.ndarray

    Paramters
    ---------
    t: torch.Tensor|np.ndarray
        The tensor to of which indices need to be regrouped
    
    regroup: List[List]
        A list of lists with the regrouped indices. E.g., [[1,2], [0]],
        would return t_{(1,2),0} if t is rank-3, so indices 1 and 2 joined 
        and swapped with index 0.

    Returns
    -------
    t: torch.Tensor
        The regrouped tensor.
    """

    # get index order and desired shape of tensor
    i_order = [ind for inds in regroup for ind in inds]

    shape = [np.prod([t.shape[ind] for ind in inds]) for inds in regroup]

    # permute indices, reshape and return
    if len(t.shape) == len(i_order):
        if isinstance(t, torch.Tensor):
            return t.permute(i_order).reshape(shape)
        else:
            return np.transpose(t, i_order).reshape(shape) # why do they change names from numpy to torch??
    
    i_order = list(range(len(t.shape) - len(i_order))) + i_order # add first indices if needed
    return t.permute(i_order).reshape(-1, *shape)


def t(x: torch.Tensor, ind1: int=-2, ind2: int=-1):
    """
        Transpose of x by swapping ind1 and ind2 (default -2 and -1)
    """
    return torch.transpose(x, ind1, ind2)


def h(x: torch.Tensor, ind1: int=-2, ind2: int=-1):
    """
        Hermitian by swapping ind1 and ind2 (default -2 and -1)
    """
    return torch.transpose(x.conj(), ind1, ind2)


def mat_pow(mat, pow=0.5):

    """Matrix power, by eigenmode decomposition. Default power 1/2, so matrix square root."""
        
    eigs, Uv = torch.linalg.eig(mat)
    return Uv @ eigs.pow(pow).diag() @ Uv.T.conj()
    
def sqrtm(mat):
    
    """The matrix square root, by eigenmode decomposition."""

    return mat_pow(mat, 0.5)


def eye_like(x: torch.Tensor, n: int|None=None, m: int|None=None) -> torch.Tensor:
    """
    Return a tensor with same batch size as x. Last two indices dimension can be set (min(m,n) will set length diagonal if chosen)
    """

    n = x.shape[-2] if n is None else n
    m = n if m is None else m
    return torch.eye(n, m, dtype=x.dtype, device=x.device).repeat(*x.shape[:-2], 1, 1)



def Haar_unitary(n, device='cpu'):

    """
    Return an n x n random unitary matrix drawn from the Haar measure.
    Method: make a random (complex) Ginibre matrix Z, do QR, then remove
    the phases of R's diagonal by multiplying columns of Q by conj(diag(R)/|diag(R)|).
    """


    # real + i * imag Gaussian entries
    real = torch.randn((n, n), dtype=torch.float64, device=device)
    imag = torch.randn((n, n), dtype=torch.float64, device=device)
    Z = (real + 1j * imag).to(torch.complex128)

    # QR decomposition
    Q, R = torch.linalg.qr(Z)

    # Extract diagonal of R and compute phase factors
    diagR = torch.diagonal(R, 0)
    # avoid division by zero just in case (extremely unlikely for continuous Gaussians)
    eps = torch.finfo(diagR.real.dtype).tiny
    phases = diagR / (torch.abs(diagR) + eps)  
    # Multiply each column i of Q by conj(phases[i]) to fix R diagonals to be positive real
    Q = Q * phases.conj().unsqueeze(0)           # broadcast across rows

    return Q
