import numpy as np
from tqdm import tqdm
from typing import List, Callable, Tuple

conv_ueVps_uW = 1.602176633*1e-19 * (3600 * 1e12) # conversion ueV/ps to uW


def dpsi_dt_coupling(psi, J):
    """Compute the derivative dpsi/dt using the given psi and parameter J."""
    psi_shift = np.zeros_like(psi, dtype=np.complex128)
    psi_shift[0:-1] += psi[1:]
    psi_shift[1:] += psi[:-1]
    return 1j * J * psi_shift

def rk4_step_coupling(psi, J, dt):
    """Perform a single Runge-Kutta 4 step for the wave function evolution."""
    k1 = dpsi_dt_coupling(psi, J)
    k2 = dpsi_dt_coupling(psi + 0.5 * dt * k1, J)
    k3 = dpsi_dt_coupling(psi + 0.5 * dt * k2, J)
    k4 = dpsi_dt_coupling(psi + dt * k3, J)
    
    return (1. / 6.) * (k1 + 2*k2 + 2*k3 + k4)


def split_step_gpe_1d_multi(
        psi0: np.ndarray, 
        dt: float, 
        k: np.ndarray, 
        t_samples: np.ndarray, 
        g: float,
        J: float=0., 
        dF: Callable[[float], np.ndarray]|None=None,
        gamma: float = 0., 
        V: np.ndarray | float = 0.,
        c: float = 3e8 * 1e6 / 1e12
    ) -> Tuple[List, List[np.ndarray]]:
    """
    Evolve the 1D Gross-Pitaevskii Equation using the split-step Fourier method.
    
    Parameters:
    psi0 : array
        Initial wavefunction.

    dt : float
        Time step.

    dk : float
        FFT vector of real space simulation

    t_samples : np.ndarray
        Array of time samples at which to store the wavefunction.

    g : float
        Nonlinearity parameter.

    gamma : float, optional
        Damping parameter (default is 0).

    V : array or float, optional
        External potential (default is 0).
    
    Returns:
    t_sampled: the sampled time points
    list_psi : list of arrays
        The corresponding evolved wavefunction at selected time samples.
    """

    psi = psi0.copy()
    
    # Kinetic evolution operator in Fourier space
    kinetic_phase = np.exp(-1j * np.abs(c * k * dt))
    # kinetic_phase = np.exp(-1j * (k**2) * dt / (2 * m))
    
    # Time evolution
    t = 0
    t_max = np.max(t_samples)
    num_steps = int(t_max / dt)

    t_values = np.arange(0, t_max + dt, dt)  # Ensure we step through exact times
    sample_indices = np.searchsorted(t_values, t_samples) 
    
    list_psi = []
    # if np.isclose(t, t_samples, atol=dt/2).any():
    if 0 in sample_indices:
        list_psi.append(psi.copy())
    for i_step in tqdm(range(num_steps), desc="Evolving GPE"):

        # Nonlinear and potential evolution (diagonal in real space)
        psi *= np.exp(-1j * (V + g * np.abs(psi) ** 2 - 1j * gamma) * dt / 2)

        # waveguide couplings
        if J > 0. and psi.shape[0] > 1:
            psi += dt * rk4_step_coupling(psi, J, dt=dt)

        # add drive
        if dF is not None:
            psi += -1j * dF(t)
        
        # Kinetic evolution (diagonal in momentum space)
        psi_k = np.fft.fft(psi, axis=1)
        psi_k *= kinetic_phase
        psi = np.fft.ifft(psi_k, axis=1)
        
        # Nonlinear and potential evolution again (real space)
        psi *= np.exp(-1j * (V + g * np.abs(psi) ** 2 - 1j * gamma) * dt / 2)
        
        t += dt
        #if np.isclose(t, t_samples, atol=dt/2).any():
        if i_step + 1 in sample_indices:
            list_psi.append(psi.copy())
        
    t_sampled = t_values[sample_indices]
    return t_sampled, list_psi