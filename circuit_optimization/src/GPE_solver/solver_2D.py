import numpy as np
from tqdm import tqdm
from typing import List, Callable, Tuple
import torch



def split_step_gpe_2d(
        psi0: np.ndarray|None, 
        dt: float, 
        x: np.ndarray,
        y: np.ndarray, 
        t_samples: np.ndarray, 
        g: float,
        dF: Callable[[float], np.ndarray]|None=None,
        gamma: float = 0., 
        V: np.ndarray | float = 0.,
        dispersion: np.ndarray|None=None,
        device: str='cpu'
    ) -> Tuple[List,List[np.ndarray]]:
    """
    Evolve the 2D Gross-Pitaevskii Equation using the split-step Fourier method.
    
    Parameters
    ----------
    psi0: array
        Initial wavefunction.

    dt: float
        Time step.

    x: np.ndarray
        Spatial grid in x-direction.

    y: float
        Spatial grid in y-direction.

    t_samples: np.ndarray
        Array of time samples at which to store the wavefunction.

    g: float
        Nonlinearity parameter.

    dF: Callable[[float], np.ndarray]|None (default None)
        The time-dependent driving field. Default None, no drive

    gamma: float, optional
        Damping parameter (default is 0, no photon losses).

    V: np.ndarray or float, optional
        External potential for the field (default is 0). This is the potential landscape, 
        from refractive-index shaping in 2D plane, typically.

    dispersion: np.ndarray|None (default None)
        The 2D dispersion of the field, in Fourier space.

    device: str (default 'cpu')
        The device to run torch simulation on, default 'cpu
    
    Returns
    -------
    t_sampled: The sampled t-values
    list_psi : list of arrays
        Evolved wavefunction at selected time samples.
    """

    # get some useful parameters
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    nx = x.size
    ny = y.size

    # convert data to torch
    if psi0 is None:
        psi0 = np.zeros_like(V)
    psi = torch.tensor(psi0.copy(), dtype=torch.complex64, device=device)
    V = torch.tensor(V, dtype=torch.float32, device=device)

    # Kinetic evolution operator in Fourier space
    dispersion = torch.tensor(dispersion.copy(), dtype=torch.float32, device=device)
    kinetic_phase = torch.exp(-1j * (dispersion * dt))
    
    # Time evolution
    t = 0.
    t_max = np.max(t_samples)

    t_values = np.arange(0, t_max + dt, dt)  # Ensure we step through exact times
    sample_indices = np.searchsorted(t_values, t_samples) 
    num_steps = t_values.size
    
    list_psi = []
    # if np.isclose(t, t_samples, atol=dt/2).any():
    if 0 in sample_indices:
        list_psi.append(psi.cpu().numpy())
    for i_step in tqdm(range(num_steps), desc="Evolving GPE"):

        # Nonlinear and potential evolution (diagonal in real space)
        psi *= torch.exp(-1j * (V + g * torch.abs(psi) ** 2 - 1j * gamma) * dt / 2)

        # add drive
        if dF is not None:
            dF_field = dF(t)
            if not isinstance(dF, torch.Tensor):
                dF_field = torch.tensor(dF_field, device=device)
            psi += -1j * dF_field
        
        # Kinetic evolution (diagonal in momentum space)
        psi_k = torch.fft.fft2(psi)
        psi_k *= kinetic_phase
        psi = torch.fft.ifft2(psi_k)
        
        # Nonlinear and potential evolution again (real space)
        psi *= torch.exp(-1j * (V + g * torch.abs(psi) ** 2 - 1j * gamma) * dt / 2)
        
        t += dt

        # check if result needs to be stored for sampling times
        if i_step + 1 in sample_indices:
            list_psi.append(psi.cpu().numpy())
        
    t_sampled = t_values[sample_indices]
    return t_sampled, list_psi


def split_step_gpe_2d_exciton_coupling(
        psi0: np.ndarray|None, 
        dt: float, 
        x: np.ndarray,
        y: np.ndarray, 
        t_samples: np.ndarray, 
        g: float,
        Rabi_coupling: float,
        omega_ex: float,
        refractive_index: float,
        dF: Callable[[float], np.ndarray]|None=None,
        gamma: float = 0., 
        V: np.ndarray | float = 0.,
        dispersion: np.ndarray|None=None,
        device: str='cpu'
    ) -> List[np.ndarray]:
    """

    UNDER DEVELOPMENT -> do not use blindly

    Evolve the 2D Gross-Pitaevskii Equation using the split-step Fourier method, 
    using the explicit exciton-photon coupling. Here two coupled modes are needed, 
    the exciton field and the photonic field, with a Rabi coupling between them.
    
    Parameters:
    psi0 : array
        Initial wavefunction.
    dt : float
        Time step.
    dx : float
        Spatial grid spacing.
    t_samples : array
        Array of time samples at which to store the wavefunction.
    g : float
        Nonlinearity parameter.
    gamma : float, optional
        Damping parameter (default is 0).
    V : array or float, optional
        External potential (default is 0).
    
    Returns:
    list_psi : list of arrays
        Evolved wavefunction at selected time samples.
    """

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    nx = x.size
    ny = y.size

    c_vac = 3e8 * 1e6 / 1e12

    # convert to torch, set photon and exciton states (last one starts at zero, for now)
    if psi0 is None:
        psi0 = np.zeros_like(V)
    psi_photon = torch.tensor(psi0.copy(), dtype=torch.complex64, device=device)
    psi_exciton = torch.zeros_like(psi_photon)

    # the photonic potential
    V = torch.tensor(V, dtype=torch.float32, device=device)

    # Fourier space variables
    kx = torch.fft.fftfreq(nx, dx, device=device, dtype=torch.float32) * 2. * np.pi
    ky = torch.fft.fftfreq(ny, dy, device=device, dtype=torch.float32) * 2. * np.pi
    Kx, Ky = torch.meshgrid(kx, ky, indexing='xy') 

    # Kinetic evolution operator in Fourier space
    dispersion = (c_vac / refractive_index) * (Kx ** 2 + Ky ** 2).sqrt()
    kinetic_phase = torch.exp(-1j * (dispersion * dt))
    
    # Time evolution
    t = 0
    t_max = np.max(t_samples)
    num_steps = int(t_max / dt)

    t_values = np.arange(0, t_max + dt, dt)  # Ensure we step through exact times
    sample_indices = np.searchsorted(t_values, t_samples) 

    # local func for real space step
    def real_space_step(dt_loc, psi_ph, psi_ex):
        # linear photon evolution (diagonal in real space)
        psi_ph *= torch.exp(
            -1j * (V - 1j * gamma) * dt_loc
            )
        
        # exciton field (local)
        psi_ex *=  torch.exp(-1j * dt_loc * (omega_ex + g * torch.abs(psi_ph) ** 2))

        # rabi coupling
        psi_ph += -1j * dt_loc * Rabi_coupling * psi_ex
        psi_ex += -1j * dt_loc * Rabi_coupling * psi_ph

        return psi_ph, psi_ex
    
    list_psi = []
    list_exciton = []
    # if np.isclose(t, t_samples, atol=dt/2).any():
    if 0 in sample_indices:
        list_psi.append(psi_photon.cpu().numpy())
    for i_step in tqdm(range(num_steps), desc="Evolving GPE"):

        # real space step
        psi_photon, psi_exciton = real_space_step(dt/2., psi_photon, psi_exciton)

        # add drive
        if dF is not None:
            dF_field = dF(t)
            if not isinstance(dF, torch.Tensor):
                dF_field = torch.tensor(dF_field, device=device)
            psi_photon += -1j * dF_field
        
        # Kinetic evolution (diagonal in momentum space)
        psi_k = torch.fft.fft2(psi_photon)
        psi_k *= kinetic_phase
        psi_photon = torch.fft.ifft2(psi_k)
        
        # real space step
        psi_photon, psi_exciton = real_space_step(dt/2., psi_photon, psi_exciton)

        t += dt
        #if np.isclose(t, t_samples, atol=dt/2).any():
        if i_step + 1 in sample_indices:
            list_psi.append(psi_photon.cpu().numpy())
            list_exciton.append(psi_exciton.cpu().numpy())
        
    t_sampled = t_values[sample_indices]
    return t_sampled, list_psi, list_exciton