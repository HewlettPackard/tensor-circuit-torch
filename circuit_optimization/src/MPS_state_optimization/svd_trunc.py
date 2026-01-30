import torch
from torch.autograd import Function
import numpy as np
from typing import Tuple
from .utils import irescale, t, h, eye_like, iregroup


def svd_trunc(A: torch.Tensor, 
            cutoff: float=1e-8, 
            max_num: int|None=None, 
            normalize: bool=True,
            eps_reg: float=0.,
            torch_svd: bool=False,
            lowrank: bool=True
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    """
    The differentiable implementation of the truncated SVD decomposition for complex matrices. 
    Used by default for the MPS SVDs.

    Parameters
    ---------
    A: torch.Tensor
        The matrix as torch Tensor for applying the truncated SVD.

    cutoff: float (default 1e-8)
        The cutoff value for singular values

    max_num: int|None (default None)
        The maximal number of singular values to keep. By default only 'cutoff' is used selecting SVs

    normalize: bool (default True)
        Whether the singular values should be normalized. Default is True.

    eps_reg: float (default 1e-12)
        The epsilon regularization, to avoid (near) degeneracies, which might lead to unstable gradients.
        Make sure it is always lower than cutoff!

    torch_svd: bool (default False)
        Use the standard SVD provided by torch. Not recommended, problems were found with stability in
        the case of complex matrices. Moreover, it is inefficient because it stores all SVs obtained in memory for 
        the obtaining gradients, without imposing the SV truncation.

    lowrank: bool (default True)
        Use PyTorch's lowrank() for obtaining forward SVD pass. Recommended as it is much more efficient.
        However, care has to be taken since it is a stochastic method, which could lead to numerical
        inaccuracies.


    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        The U (left-unitary), s (singular values) and Vd (right-unitary) matrices, with the normalization factor.
    """

    # by default False, if not, use torch's SVD
    if torch_svd:
        return svd_trunc_torch(A, 
            cutoff=cutoff, 
            max_num=max_num, 
            normalize=normalize,
            eps_reg=eps_reg
        )

    # add random diagonal if regularization required
    if eps_reg is not None and eps_reg > 0.:
        n, m = A.shape[-2:]
        reg = torch.zeros(n, m, device=A.device)
        idx = torch.arange(np.minimum(n, m))  # Indices for diagonal elements
        reg[idx, idx] = eps_reg * (0.5 + torch.rand(np.minimum(n,m), device=A.device))
    else:
        reg = 0.

    U_trunc, s_trunc, Vd_trunc, normalize_factor = SVDTrunc.apply(A + reg, cutoff, max_num, normalize, lowrank)

    return U_trunc, s_trunc, Vd_trunc, normalize_factor



class SVDTrunc(Function):

    """
    The truncated SVD function, derived from torch Function parent class. 
    The cutoff and max_num SVs are explicitly considered for returning the 
    gradients in the backward pass.

    Difficult find, but with combo of references https://j-towns.github.io/papers/svd-derivative.pdf, Eq (30)
    and https://arxiv.org/pdf/1909.02659, final term (L_imag), the stable SVD derivative was implemented.
    It was verified with dummy example objective in __main__() (see below)
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor, cutoff: float=1e-8, max_num: int|None=None, 
                normalize: bool=True, lowrank: bool=True
                ) -> Tuple[torch.Tensor]:
        """
        Forward pass: Computes the SVD of a complex-valued matrix A.
        Saves necessary tensors for the backward pass.
        
        Parameters
        ----------
        A: torch.Tensor
            Complex-valued input matrix of shape (m, n)

        cutoff: float=1e-8
            (Normalized) singular value cutoff

        max_num: int|None (default None)
            Max number of SVs kept. Defaults to None, only cutoff considered in truncation.

        normalize: bool (default True)
            Return normalized SVs
        
        lowrank: bool (default True)
            Perform stochastic lowrank SVD provided by PyTorch. Much more efficient, but can
            be prone to stochasticity and numerical inaccuracy.
        
        Returns
        -------
        Tuple[torch.Tensor]
            The tensors U_trunc, s_trunc, Vd_trunc, and normalize_factor used for SV normalization after truncation
        """
        # Perform SVD, lowrank or full. Make sure Vd (V^\dagger) is outcome
        if lowrank:
            U, s, V = torch.svd_lowrank(A, q=max_num)
            Vd = iregroup(V.conj(), [[-1],[-2]])
        else:
            U, s, Vd = torch.linalg.svd(A, full_matrices=True)


        # truncate with cutoff and max_num, find the bond dimension
        max_num = s.shape[-1] if max_num is None else max_num
        
        # Compute the truncated SV dimension, from cutoff and max_num
        bdim = torch.maximum(
            torch.minimum(
                ((s ** 2 / s.norm()) > cutoff).sum(axis=-1).max(), torch.tensor(max_num, device=A.device)
                ), 
                torch.tensor([1], device=A.device)
                )
        
        # truncate variable
        U_trunc, s_trunc, Vd_trunc = U[..., :, :bdim], s[..., :bdim], Vd[..., :bdim, :]

        # normalize again after truncation, if needed (default)
        normalize_factor = s_trunc.norm(dim=-1) / 1.
        if normalize:
            if len(s.shape) == 1:
                s_trunc = s_trunc / normalize_factor
            else:
                s_trunc = s_trunc * (1. / normalize_factor.unsqueeze(1))
        
        # Save for backward pass
        ctx.save_for_backward(A, U_trunc, s_trunc, Vd_trunc)
        ctx.cutoff = cutoff
        ctx.max_num = max_num 
        ctx.normalize = normalize
        ctx.lowrank = lowrank

        # return truncated tensors
        return U_trunc, s_trunc, Vd_trunc, normalize_factor

    @staticmethod
    def backward(ctx, grad_U, grad_s, grad_Vh, grad_normalize_factor):
        """
        Backward pass: Computes the gradient of the input A using the gradients
        of U, s, and Vd.
        
        Parameters
        ----------
            ctx: The saved parameters
            grad_U (torch.Tensor): Gradient w.r.t. U from the loss.
            grad_S (torch.Tensor): Gradient w.r.t. S from the loss.
            grad_Vh (torch.Tensor): Gradient w.r.t. Vh from the loss.
        
        Returns
        -------
            grad_A (torch.Tensor): Gradient w.r.t. the input matrix A.
            None
            None
            None
                -> gradients w.r.t. U_trunc, s_trunc and Vd_trunc not needed
        """
        A, U, s, Vh = ctx.saved_tensors
        dtype = A.dtype

        m, n = A.shape[-2:]
        k = s.shape[-1]  # Number of singular values

        # Compute the diagonal matrix of singular values
        s_inv = 1. / (s + ctx.cutoff) # add cutoff renormlization to avoid divide by zero

        # needed hermitian conjugates
        Uh, V = h(U), h(Vh)
        grad_Uh, grad_V = h(grad_U), h(grad_Vh)

        # Compute intermediate terms for gradients
        F = s.unsqueeze(-2) ** 2 - s.unsqueeze(-1) ** 2 
        F = torch.where(F.abs() <= ctx.cutoff, torch.tensor(float('inf'), device=A.device, dtype=dtype), F)  # Avoid division by zero

        Vh_gV = (Vh @ grad_V)
        L = Vh_gV * eye_like(Vh_gV)
        L_imag = 0.5 * irescale(U, s_inv, ind=-1) @ (h(L) - L) @ Vh

        Ut_grad_U = Uh @ grad_U
        Vt_grad_V = Vh @ grad_V

        # # Symmetric adjustment terms
        sym_U = torch.nan_to_num((Ut_grad_U - h(Ut_grad_U)) / F, nan=0.)
        sym_V = torch.nan_to_num((Vt_grad_V - h(Vt_grad_V)) / F, nan=0.)

        # Gradients w.r.t. input matrix A
        id_U = eye_like(U, n=U.shape[-2], m=U.shape[-2])
        id_V = eye_like(V, n=V.shape[-2], m=V.shape[-2])

        grad_A = (
            (U @ irescale(sym_U, s, ind=-1) + (id_U - U @ Uh) @ irescale(grad_U, s_inv, ind=-1)
                ) @ Vh
            + irescale(U, grad_s, ind=-1) @ Vh 
            + U @ (
                irescale(sym_V, s, ind=-2) @ Vh + irescale(grad_Vh, s_inv, ind=-2) @ (id_V - V @ Vh)
                )
            + L_imag
        )

        return grad_A, None, None, None, None
    

def svd_trunc_torch(mat, cutoff: float=0., max_num: int|None=None, normalize: bool=True, eps_reg=1e-8, driver='gesvdj'):

    """NOT RECOMMENDED"""

    # add random diagonal if regularization required
    if eps_reg is not None and eps_reg > 0.:
        n, m = mat.shape[-2:]
        reg = torch.zeros(n, m, device=mat.device)
        idx = torch.arange(np.minimum(n, m))  # Indices for diagonal elements
        reg[idx, idx] = eps_reg * (0.5 + torch.rand(np.minimum(n,m), device=mat.device))
    else:
        reg = 0.

    U, s, Vd = torch.linalg.svd(mat + reg) #, driver=driver, full_matrices=False)

    # truncate with cutoff and max_num
    max_num = s.shape[-1] if max_num is None else max_num
    bdim = torch.maximum(
        torch.minimum(
            (s > cutoff).sum(axis=-1).max(), torch.tensor(max_num, device=mat.device)
            ), 
            torch.tensor([1], device=mat.device)
            ) # changed .sum() -> .sum(axis=-1).max()
    U_trunc, s_trunc, Vd_trunc = U[..., :, :bdim], s[..., :bdim], Vd[..., :bdim, :]


    # normalize if needed (default)
    normalize_factor = s_trunc.norm(dim=-1)
    if normalize:
        if len(s.shape) == 1:
            s_trunc = s_trunc / normalize_factor
            # U_trunc = U_trunc * normalize_factor
        else:
            s_trunc = s_trunc * (1. / normalize_factor.unsqueeze(1))
            # U_trunc = irescale(U_trunc, normalize_factor, ind=0)

    return U_trunc, s_trunc, Vd_trunc, normalize_factor


    
if __name__ == "__main__":

    seed = 457
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    it, n, m = 2, 100, 200 # indices of input tensor shape = (it, n, m)
    dtype = torch.complex128
    Ad = torch.randn(it, n, m, dtype=dtype, requires_grad=True)

    # Define A using the formula
    A = Ad + 0.3 * Ad.conj()

    # Retain gradients for A
    A.retain_grad()

    # Clone A to create an independent copy for separate computation
    Ac = A.clone().detach().requires_grad_(True)  # Ensure it's detached and requires_grad=True
    Ac.retain_grad()


    # Compute SVD using custom function
    U, S, Vh, normalize_factor = SVDTrunc.apply(A, 0, 200)
    Uc, Sc, Vhc = torch.linalg.svd(Ac, full_matrices=False)

    # (dummy) figure of merit, returning float 
    def fom(Ul, Sl, Vhl):
        return (Ul @ irescale(Vhl, Sl, ind=-2))[0, 0, 0].abs() - (Ul @ irescale(Vhl, Sl, ind=-2))[1, 0, 1].abs()

    # define loss
    loss = fom(U, S, Vh)
    loss.backward()

    lossc =  fom(Uc, Sc, Vhc)
    lossc.backward()

    max_print = 3
    # Print gradients
    print("\nA:")
    print(A[:, :max_print, :max_print])
    print("\:")
    print(U[:, :max_print, :max_print])
    print("\nUc:")
    print(Uc[:, :max_print, :max_print])
    print("\nVh:")
    print(Vh[:, :max_print, :max_print])
    print("\nVhc:")
    print(Vhc[:, :max_print, :max_print])
    print("\nSingular values S svd_trunc:")
    print(S)
    print("\nSingular values S torch:")
    print(Sc)
    print("\nGradient w.r.t. A svd_trunc:")
    print(A.grad[:, :max_print, :max_print])
    print("\nGradient w.r.t. A torch:")
    print(Ac.grad[:, :max_print, :max_print])

    