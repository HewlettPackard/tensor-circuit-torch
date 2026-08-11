import torch
import numpy as np

from src.mpdo_torch import MPDOtorch
from src.utils import BosonOperatorsTorch


def quantum_Fisher_information(
        rho_1: MPDOtorch, 
        rho_2: MPDOtorch, 
        dtheta: float,
        ) -> torch.Tensor:
    
    """
    Compute the quantum Fisher information from overlap two batch states 
    (not so important, used as check, mainly)
    """

    fidelity = rho_1.overlap(rho_2).abs() ** 2
    QFI = 4. * (1. - fidelity) / dtheta ** 2
  
    return QFI


def Gaussian_Fisher_information(
            rho_1: MPDOtorch, rho_2: MPDOtorch, dtheta: float, 
            i_check:int|None = None,
            i_exclude: int|None = None
            ) -> torch.Tensor:

    if i_check is None:
        n1 = rho_1.number_expectations()
        n2 = rho_2.number_expectations()

        sigma1 = rho_1.number_variances()

        GFIs = ((n1 - n2) / dtheta) ** 2 / sigma1

        if i_exclude is not None:
            GFIs[i_exclude] = 0.

        return GFIs.sum()
    
    else:
        n1 = rho_1.number_expectation(i_check)
        n2 = rho_2.number_expectation(i_check)

        sigma1 = rho_1.number_variance(i_check)

        GFI = (((n1 - n2) / dtheta) ** 2 / sigma1)

        return GFI


def homodyne_Fisher_information(
            rho1: MPDOtorch, rho2: MPDOtorch, dtheta: float, phi: float=np.pi/2.,
            i_check: int|None=None,
            i_exclude: int|None = None
        ) -> torch.Tensor:
    
    # init boson operators
    Nmax = rho1.Nmax
    device = rho1.device
    ops = BosonOperatorsTorch(Nmax, device=device)
    
    # derivative compute relevant quantities, field amplitude <a>, get homodyne signal
    a1, a2 = rho1.local_expectation(ops.a, i_check), rho2.local_expectation(ops.a, i_check)
    S1, S2 = 2. * (a1 * np.exp(1j*phi)).real, 2. * (a2 * np.exp(1j*phi)).real
    dS_dmu = (S2 - S1) / dtheta
    #dS_dmu = 2. / dtheta * ((a2 * np.exp(-1j*phi)) - (a1 * np.exp(-1j*phi))).imag

    # the variance field square <a^2> and intensity <ad * a>
    op_a2 = ops.a @ ops.a
    a21 = rho1.local_expectation(op_a2, i_check)
    n1 = rho1.number_expectations() if i_check is None else rho1.number_expectation(i_check)

    # the variance
    var_S = 2. * ((a21 - a1 ** 2) * np.exp(2j * phi)).real + (2. * (n1 - a1 * a1.conj()) + 1.).real

    # Fisher information
    hom_FI = (dS_dmu) ** 2 / var_S

    # if single output monitored
    if i_check is not None:
        return hom_FI

    # if multiple outputs monitored
    if i_exclude is not None:
        hom_FI[i_exclude] = 0.
    return hom_FI.sum()


def number_Fisher_information_single_output(
            rho_1: MPDOtorch, rho_2: MPDOtorch, dtheta: float,
            i_check: int|None=None
        ) -> torch.Tensor:

        """
        The classical Fisher information for number measurements

        Parameters
        ----------

        psi: MPStorch
            The input MPS state

        d_theta: float
            The differential for FI derivative

        i_check: int|None (default None)
            The mode to evaluate, None (default) is mid circuit L // 2

        weight_N_max: float (default 100)
            The penalty weight for occupying Nmax Fock state, to avoid over-occupation of number state

        Returns:
        --------

        torch.Tensor
            The Fisher information
        """

        # construct local density matrices
        rho_red_1 = rho_1.ptrace(i_check)
        rho_red_2 = rho_2.ptrace(i_check)

        # select the diagonal of rho to get the classical number distribution
        ind = range(rho_1.d)
        Pn_1, Pn_2 = rho_red_1[ind,ind].real, rho_red_2[ind,ind].real

        # compute Fisher information
        PFI = torch.sum(
            ((torch.log(Pn_1) - torch.log(Pn_2)) / dtheta) ** 2 * Pn_1
            )

        return PFI


def number_Fisher_information(
        rho_1: MPDOtorch, 
        rho_2: MPDOtorch, 
        dtheta: float,
        i_exclude: int|None=None
        ) -> torch.Tensor:
     
    PFIs = torch.zeros(rho_1.num_channels, device=rho_1.device)
    for l in range(rho_1.num_channels):
        PFIs[l] = number_Fisher_information_single_output(
              rho_1, rho_2, dtheta, i_check=l
              )

    if i_exclude is not None:
        PFIs[i_exclude] = 0.

    return PFIs.sum()
        




