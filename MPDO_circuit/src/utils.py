"""
utils.py
--------
Shared operator classes and tensor utility functions used throughout the
MPDO simulator.

Contents
--------
- QubitOperatorTorch   : Pauli and ladder operators for a single qubit.
- BosonOperatorsTorch  : Bosonic ladder operators up to Fock truncation Nmax.
- irescale             : Index-wise tensor rescaling (e.g. with singular values).
- iregroup             : Index regrouping / reshaping of tensors.
- t, h                 : Transpose and Hermitian conjugate helpers.
- mat_pow, sqrtm       : Matrix power / square root via eigendecomposition.
- eye_like             : Batched identity matrix matching a tensor's batch shape.
- Haar_unitary         : Random unitary drawn from the Haar measure.
"""

import numpy as np
import torch
from typing import List


# ---------------------------------------------------------------------------
# Operator classes
# ---------------------------------------------------------------------------

class QubitOperatorTorch:
    """
    Single-qubit Pauli and projection operators as complex-128 Tensors.

    Attributes
    ----------
    sx, sy, sz : Pauli matrices X, Y, Z
    sp, sm     : Raising / lowering operators σ⁺, σ⁻
    P0, P1     : Projectors onto |0⟩, |1⟩
    id         : 2×2 identity
    """

    d:    int = 2
    Nmax: int = 1

    def __init__(self, device: str = 'cpu'):
        """Initialise all qubit operators on ``device``."""
        self.device = device
        kw = dict(device=device, dtype=torch.complex128)
        self.sx = torch.tensor([[0., 1.], [1.,  0. ]], **kw)
        self.sy = torch.tensor([[0., 1j], [-1j, 0. ]], **kw)
        self.sz = torch.tensor([[1., 0.], [0., -1. ]], **kw)
        self.sp = torch.tensor([[0., 1.], [0.,  0. ]], **kw)
        self.sm = torch.tensor([[0., 0.], [1.,  0. ]], **kw)
        self.P1 = torch.tensor([[1., 0.], [0.,  0. ]], **kw)
        self.P0 = torch.tensor([[0., 0.], [0.,  1. ]], **kw)
        self.id = torch.tensor([[1., 0.], [0.,  1. ]], **kw)


class BosonOperatorsTorch:
    """
    Bosonic ladder operators for a single mode truncated at Nmax photons.

    Physical Hilbert space dimension is d = Nmax + 1 (states |0⟩ … |Nmax⟩).

    Parameters
    ----------
    Nmax   : int  — Fock-space truncation; operator dimension d = Nmax + 1.
    device : str  — Torch device string (default 'cpu').

    Attributes
    ----------
    a      : annihilation operator  a
    ad     : creation operator      a†
    n      : number operator        a†a
    n_nm1  : normal-ordered n(n-1)  a†a†aa  (used for Kerr nonlinearity)
    id     : identity matrix
    proj   : list of Fock projectors  |n⟩⟨n|  for n = 0 … Nmax
    """

    def __init__(self, Nmax: int, device: str = 'cpu'):
        """Construct ladder and number operators for Fock-space dimension d = Nmax + 1."""
        self.Nmax   = Nmax
        self.d      = Nmax + 1
        self.device = device

        self.a    = torch.tensor(
            np.diag([np.sqrt(n) for n in range(1, Nmax + 1)], k=1),
            dtype=torch.complex128, device=device,
        )
        self.ad   = self.a.T
        self.n    = self.ad @ self.a
        self.n_nm1 = self.ad @ self.ad @ self.a @ self.a
        self.id   = torch.eye(Nmax + 1, dtype=torch.complex128, device=device)

        # Fock projectors |n><n|
        self.proj = []
        for i in range(self.d):
            p = torch.zeros_like(self.a, dtype=torch.complex128, device=device)
            p[i, i] = 1.
            self.proj.append(p)


# ---------------------------------------------------------------------------
# Tensor utility functions
# ---------------------------------------------------------------------------

def irescale(t: torch.Tensor, factor: torch.Tensor, ind: int = 0) -> torch.Tensor:
    """
    Rescale one axis of a tensor element-wise with a 1-D factor vector.

    Useful for contracting singular-value diagonals into MPS tensors without
    forming an explicit diagonal matrix.

    Parameters
    ----------
    t      : tensor to rescale
    factor : 1-D (or batched) factor matching the size of axis `ind`
    ind    : axis of `t` to rescale (supports negative indexing)

    Returns
    -------
    torch.Tensor — rescaled tensor (same shape as `t`)
    """
    sh  = t.shape
    lv  = [1] * len(sh)
    lv[ind] = sh[ind]
    lv[:len(factor.shape) - 1] = factor.shape[:-1]
    return t * factor.view(lv)


def iregroup(t: "torch.Tensor | np.ndarray", regroup: List[List[int]]) -> torch.Tensor:
    """
    Regroup (permute + reshape) tensor indices.

    Parameters
    ----------
    t       : input tensor
    regroup : list of index groups defining the new axes.
              E.g. ``[[1, 2], [0]]`` on a rank-3 tensor gives shape
              ``(t.shape[1]*t.shape[2], t.shape[0])``.

    Returns
    -------
    torch.Tensor — permuted and reshaped tensor
    """
    i_order = [ind for inds in regroup for ind in inds]
    shape   = [int(np.prod([t.shape[ind] for ind in inds])) for inds in regroup]

    if len(t.shape) == len(i_order):
        if isinstance(t, torch.Tensor):
            return t.permute(i_order).reshape(shape)
        return np.transpose(t, i_order).reshape(shape)

    # prepend leading batch indices if regroup covers only the trailing ones
    i_order = list(range(len(t.shape) - len(i_order))) + i_order
    return t.permute(i_order).reshape(-1, *shape)


def t(x: torch.Tensor, ind1: int = -2, ind2: int = -1) -> torch.Tensor:
    """Transpose of `x` by swapping axes `ind1` and `ind2` (default: last two)."""
    return torch.transpose(x, ind1, ind2)


def h(x: torch.Tensor, ind1: int = -2, ind2: int = -1) -> torch.Tensor:
    """Hermitian conjugate of `x`: conjugate + transpose of axes `ind1`, `ind2`."""
    return torch.transpose(x.conj(), ind1, ind2)


def mat_pow(mat: torch.Tensor, pow: float = 0.5) -> torch.Tensor:
    """
    Matrix power via eigendecomposition: M^pow = U diag(λ^pow) U†.

    Default `pow=0.5` computes the matrix square root.
    """
    eigs, Uv = torch.linalg.eig(mat)
    return Uv @ eigs.pow(pow).diag() @ Uv.T.conj()


def sqrtm(mat: torch.Tensor) -> torch.Tensor:
    """Matrix square root via eigendecomposition (calls `mat_pow(mat, 0.5)`)."""
    return mat_pow(mat, 0.5)


def eye_like(
    x: torch.Tensor,
    n: "int | None" = None,
    m: "int | None" = None,
) -> torch.Tensor:
    """
    Return an identity matrix (or batch of them) with the same batch shape,
    dtype and device as `x`.

    Parameters
    ----------
    x : reference tensor; its leading dimensions set the batch shape
    n : number of rows    (default: ``x.shape[-2]``)
    m : number of columns (default: n)
    """
    n = x.shape[-2] if n is None else n
    m = n            if m is None else m
    return torch.eye(n, m, dtype=x.dtype, device=x.device).repeat(*x.shape[:-2], 1, 1)


def Haar_unitary(n: int, device: str = 'cpu') -> torch.Tensor:
    """
    Draw an n×n random unitary matrix from the Haar measure.

    Algorithm: Ginibre matrix → QR decomposition → phase correction so that
    the diagonal of R is positive real (making Q Haar-distributed).

    Parameters
    ----------
    n      : matrix dimension
    device : Torch device string

    Returns
    -------
    torch.Tensor — shape (n, n), dtype complex128, unitary
    """
    Z = (torch.randn(n, n, dtype=torch.float64, device=device)
         + 1j * torch.randn(n, n, dtype=torch.float64, device=device)
        ).to(torch.complex128)

    Q, R = torch.linalg.qr(Z)

    # Phase-correct so R's diagonal is positive real → Q is Haar-distributed
    diagR  = torch.diagonal(R, 0)
    eps    = torch.finfo(diagR.real.dtype).tiny       # guard against division by zero
    phases = diagR / (torch.abs(diagR) + eps)
    Q      = Q * phases.conj().unsqueeze(0)           # broadcast across rows

    return Q


# ---------------------------------------------------------------------------
# Parametrized SU(4) definition (Cartan KAK decomposition)
# ---------------------------------------------------------------------------

def su2(theta: torch.Tensor, phi: torch.Tensor, lam: torch.Tensor, device: str='cpu'):
    """Single-qubit SU(2) gate via ZYZ Euler decomposition."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([
        torch.stack([c, -torch.exp(1j * lam) * s]),
        torch.stack([torch.exp(1j * phi) * s, torch.exp(1j * (phi + lam)) * c]),
    ]).to(device)


def su2_axis(alpha: torch.Tensor, theta_n: torch.Tensor, phi_n: torch.Tensor, device: str = 'cpu'):
    """
    Single-qubit SU(2) gate via axis-angle (Bloch rotation) parametrization.

    U = exp(-i * alpha/2 * n_hat . sigma), with n_hat given in spherical
    coordinates (theta_n, phi_n) on the Bloch sphere.

    Parameters
    ----------
    alpha   : rotation angle about n_hat.
    theta_n : polar angle of the rotation axis n_hat (0 <= theta_n <= pi).
    phi_n   : azimuthal angle of the rotation axis n_hat.

    Returns
    -------
    torch.Tensor
        2x2 complex SU(2) matrix.
    """
    nx = torch.sin(theta_n) * torch.cos(phi_n)
    ny = torch.sin(theta_n) * torch.sin(phi_n)
    nz = torch.cos(theta_n)

    c = torch.cos(alpha / 2)
    s = torch.sin(alpha / 2)

    # n_hat . sigma = [[nz, nx - i ny], [nx + i ny, -nz]]
    U = torch.stack([
        torch.stack([c - 1j * s * nz,        -1j * s * (nx - 1j * ny)]),
        torch.stack([-1j * s * (nx + 1j * ny), c + 1j * s * nz]),
    ]).to(device)

    return U

def canonical_core(tx: torch.Tensor, ty: torch.Tensor, tz: torch.Tensor, device: str='cpu'):
    """N(theta_x, theta_y, theta_z) = exp[i(tx XX + ty YY + tz ZZ)]."""
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)
    XX = torch.kron(X, X)
    YY = torch.kron(Y, Y)
    ZZ = torch.kron(Z, Z)
    H = tx * XX + ty * YY + tz * ZZ
    return torch.matrix_exp(1j * H)

def general_two_qubit_gate(params: List[torch.Tensor], device: str='cpu'):
    """
    params: 15-length real tensor (with requires_grad=True for optimization).
    
    Structure
    U   =   (A1⊗A2) ⋅ N(θx,θy,θz) ⋅ (B1⊗B2) 


    A1, A2, B1, B2 ​∈ SU(2):
        arbitrary single-qubit gates before/after — 4 x 3 = 12 parameters
        
        The 3 parameters:     
            alpha   : rotation angle about n_hat.
            theta_n : polar angle of the rotation axis n_hat (0 <= theta_n <= pi).
            phi_n   : azimuthal angle of the rotation axis n_hat.

    N(θx,θy,θz) = exp[i(θx XX + θy YY + θz ZZ)]:
        the nonlocal core, exactly 3 parameters, carrying all the entangling power of the gate
    """

    A1 = su2(*params[0:3], device)
    A2 = su2(*params[3:6], device)
    tx, ty, tz = params[6:9]
    B1 = su2(*params[9:12], device)
    B2 = su2(*params[12:15], device)

    N = canonical_core(tx, ty, tz, device)
    U = torch.kron(A1, A2) @ N @ torch.kron(B1, B2)
    return U