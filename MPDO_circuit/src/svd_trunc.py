"""
svd_trunc.py
------------
The core of MPDOtorch gate contractions: a differentiable truncated SVD for complex matrices, implemented as a custom
``torch.autograd.Function``.

The key challenge is that PyTorch's built-in SVD gradient is unstable for
complex matrices and does not account for SV truncation.  ``SVDTrunc``
implements a numerically stable backward pass based on:

  - Eq. (30) of https://j-towns.github.io/papers/svd-derivative.pdf
  - The imaginary correction term (L_imag) from https://arxiv.org/pdf/1909.02659

Public API
----------
svd_trunc   : main entry point — calls SVDTrunc (or the torch fallback)
SVDTrunc    : the custom autograd Function
svd_trunc_torch : torch.linalg.svd fallback (NOT recommended)

A gradient-check ``__main__`` block is included at the bottom.
"""

from typing import Tuple

import numpy as np
import torch
from torch.autograd import Function

from .utils import irescale, h, eye_like, iregroup


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def svd_trunc(
    A:          torch.Tensor,
    cutoff:     float            = 1e-8,
    max_num:    "int | None"     = None,
    normalize:  bool             = True,
    eps_reg:    float            = 0.,
    torch_svd:  bool             = False,
    lowrank:    bool             = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable truncated SVD for a (batched) complex matrix A.

    Parameters
    ----------
    A         : input matrix (…, m, n), complex
    cutoff    : relative singular-value cutoff; SVs with s²/‖s‖² < cutoff
                are discarded
    max_num   : hard cap on the number of retained SVs
    normalize : if True (default), normalise retained SVs to unit norm
    eps_reg   : add a small random diagonal to A before decomposition to
                break near-degeneracies and stabilise gradients; should be
                strictly less than `cutoff`
    torch_svd : use ``torch.linalg.svd`` instead of ``SVDTrunc``
                (NOT recommended — see ``svd_trunc_torch`` docstring)
    lowrank   : use ``torch.svd_lowrank`` for the forward pass (faster but
                stochastic); set to False for exact, deterministic SVD

    Returns
    -------
    U_trunc         : left  unitary  (…, m, k)
    s_trunc         : singular values (…, k)
    Vd_trunc        : right unitary V† (…, k, n)
    normalize_factor: pre-normalisation ‖s‖ (scalar or batch vector)
    """
    if torch_svd:
        return svd_trunc_torch(A, cutoff=cutoff, max_num=max_num,
                                normalize=normalize, eps_reg=eps_reg)

    # Optional random diagonal regularisation to avoid degenerate gradients
    if eps_reg is not None and eps_reg > 0.:
        n, m = A.shape[-2:]
        reg  = torch.zeros(n, m, device=A.device)
        idx  = torch.arange(min(n, m))
        reg[idx, idx] = eps_reg * (0.5 + torch.rand(min(n, m), device=A.device))
    else:
        reg = 0.

    return SVDTrunc.apply(A + reg, cutoff, max_num, normalize, lowrank)


# ---------------------------------------------------------------------------
# Custom autograd Function
# ---------------------------------------------------------------------------

class SVDTrunc(Function):
    """
    Truncated SVD with stable complex-valued gradients.

    The forward pass optionally uses ``torch.svd_lowrank`` for efficiency;
    the backward pass implements the analytically derived gradient including
    the imaginary correction term needed for complex matrices.
    """

    @staticmethod
    def forward(
        ctx,
        A:         torch.Tensor,
        cutoff:    float         = 1e-8,
        max_num:   "int | None"  = None,
        normalize: bool          = True,
        lowrank:   bool          = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute and truncate the SVD of A.

        Parameters
        ----------
        A        : complex matrix of shape (m, n) or (batch, m, n)
        cutoff   : relative squared-SV cutoff threshold
        max_num  : maximum number of singular values to retain
        normalize: normalise retained SVs to unit norm
        lowrank  : use stochastic lowrank SVD (faster; may be less accurate)

        Returns
        -------
        U_trunc, s_trunc, Vd_trunc, normalize_factor
        """
        if lowrank:
            U, s, V = torch.svd_lowrank(A, q=max_num)
            Vd = iregroup(V.conj(), [[-1], [-2]])
        else:
            U, s, Vd = torch.linalg.svd(A, full_matrices=True)

        # Determine bond dimension from cutoff and max_num
        max_num = s.shape[-1] if max_num is None else max_num
        bdim = torch.maximum(
            torch.minimum(
                ((s ** 2 / (s ** 2).sum()) > cutoff).sum(axis=-1).max(),
                torch.tensor(max_num, device=A.device),
            ),
            torch.tensor([1], device=A.device),
        )

        U_trunc  = U[..., :, :bdim]
        s_trunc  = s[..., :bdim]
        Vd_trunc = Vd[..., :bdim, :]

        # Normalise retained SVs
        normalize_factor = s_trunc.norm(dim=-1)
        if normalize:
            if len(s.shape) == 1:
                s_trunc = s_trunc / normalize_factor
            else:
                s_trunc = s_trunc * (1.0 / normalize_factor.unsqueeze(1))

        ctx.save_for_backward(A, U_trunc, s_trunc, Vd_trunc)
        ctx.cutoff   = cutoff
        ctx.max_num  = max_num
        ctx.normalize = normalize
        ctx.lowrank  = lowrank

        return U_trunc, s_trunc, Vd_trunc, normalize_factor

    @staticmethod
    def backward(ctx, grad_U, grad_s, grad_Vh, grad_normalize_factor):
        """
        Stable gradient of the input matrix A given upstream gradients
        w.r.t. U, s, and Vd.

        Implements Eq. (30) from https://j-towns.github.io/papers/svd-derivative.pdf
        plus the imaginary term L_imag from https://arxiv.org/pdf/1909.02659.
        """
        A, U, s, Vh = ctx.saved_tensors
        dtype = A.dtype

        k = s.shape[-1]   # retained bond dimension (unused directly but useful for docs)

        # Regularised inverse of singular values (avoids divide-by-zero)
        s_inv = 1.0 / (s + ctx.cutoff)

        Uh, V      = h(U), h(Vh)
        grad_V     = h(grad_Vh)

        # Off-diagonal factor matrix F_{ij} = s_i² - s_j²
        F = s.unsqueeze(-2) ** 2 - s.unsqueeze(-1) ** 2
        # Replace near-zero entries (degenerate SVs) with inf → zero contribution
        F = torch.where(
            F.abs() <= ctx.cutoff,
            torch.tensor(float('inf'), device=A.device, dtype=dtype),
            F,
        )

        # Imaginary correction for complex matrices
        Vh_gV  = Vh @ grad_V
        L      = Vh_gV * eye_like(Vh_gV)
        L_imag = 0.5 * irescale(U, s_inv, ind=-1) @ (h(L) - L) @ Vh

        Ut_grad_U = Uh @ grad_U
        Vt_grad_V = Vh @ grad_V

        # Symmetric adjustment terms (anti-Hermitian parts)
        sym_U = torch.nan_to_num((Ut_grad_U - h(Ut_grad_U)) / F, nan=0.)
        sym_V = torch.nan_to_num((Vt_grad_V - h(Vt_grad_V)) / F, nan=0.)

        id_U = eye_like(U, n=U.shape[-2], m=U.shape[-2])
        id_V = eye_like(V, n=V.shape[-2], m=V.shape[-2])

        grad_A = (
            (U @ irescale(sym_U, s, ind=-1)
             + (id_U - U @ Uh) @ irescale(grad_U, s_inv, ind=-1)) @ Vh
            + irescale(U, grad_s, ind=-1) @ Vh
            + U @ (
                irescale(sym_V, s, ind=-2) @ Vh
                + irescale(grad_Vh, s_inv, ind=-2) @ (id_V - V @ Vh)
            )
            + L_imag
        )

        return grad_A, None, None, None, None


# ---------------------------------------------------------------------------
# Torch fallback (NOT recommended)
# ---------------------------------------------------------------------------

def svd_trunc_torch(
    mat:       torch.Tensor,
    cutoff:    float        = 0.,
    max_num:   "int | None" = None,
    normalize: bool         = True,
    eps_reg:   float        = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Truncated SVD using ``torch.linalg.svd`` — NOT RECOMMENDED.

    Known issues:
    - Gradient instability for complex matrices.
    - Stores all singular values in memory during the backward pass (no
      truncation-aware gradient).
    - Kept only as a fallback for debugging.
    """
    if eps_reg is not None and eps_reg > 0.:
        n, m = mat.shape[-2:]
        reg  = torch.zeros(n, m, device=mat.device)
        idx  = torch.arange(min(n, m))
        reg[idx, idx] = eps_reg * (0.5 + torch.rand(min(n, m), device=mat.device))
    else:
        reg = 0.

    U, s, Vd = torch.linalg.svd(mat + reg)

    max_num = s.shape[-1] if max_num is None else max_num
    bdim = torch.maximum(
        torch.minimum(
            (s > cutoff).sum(axis=-1).max(),
            torch.tensor(max_num, device=mat.device),
        ),
        torch.tensor([1], device=mat.device),
    )
    U_trunc  = U[..., :, :bdim]
    s_trunc  = s[..., :bdim]
    Vd_trunc = Vd[..., :bdim, :]

    normalize_factor = s_trunc.norm(dim=-1)
    if normalize:
        if len(s.shape) == 1:
            s_trunc = s_trunc / normalize_factor
        else:
            s_trunc = s_trunc * (1.0 / normalize_factor.unsqueeze(1))

    return U_trunc, s_trunc, Vd_trunc, normalize_factor


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    seed = 457
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    it, n, m = 2, 100, 200
    dtype    = torch.complex128

    Ad = torch.randn(it, n, m, dtype=dtype, requires_grad=True)
    A  = Ad + 0.3 * Ad.conj()
    A.retain_grad()

    Ac = A.clone().detach().requires_grad_(True)
    Ac.retain_grad()

    # Custom SVD
    U, S, Vh, _  = SVDTrunc.apply(A, 0, 200, True, False)
    # Torch reference SVD
    Uc, Sc, Vhc  = torch.linalg.svd(Ac, full_matrices=False)

    def fom(Ul, Sl, Vhl):
        """Dummy scalar objective for gradient verification."""
        reconstructed = Ul @ irescale(Vhl, Sl, ind=-2)
        return reconstructed[0, 0, 0].abs() - reconstructed[1, 0, 1].abs()

    fom(U, S, Vh).backward()
    fom(Uc, Sc, Vhc).backward()

    max_print = 3
    print("SVDTrunc gradient (first 3×3):")
    print(A.grad[:, :max_print, :max_print])
    print("\ntorch.linalg.svd gradient (first 3×3):")
    print(Ac.grad[:, :max_print, :max_print])
