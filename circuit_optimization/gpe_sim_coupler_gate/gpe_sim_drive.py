###########################################################################################
# File for running 2D field simulation with driving field.
###########################################################################################

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.optimize import least_squares
from typing import Iterable
import os
from datetime import datetime

from src.GPE_solver.solver_2D import split_step_gpe_2d
from src.GPE_solver.design_utils import generate_directional_coupler, mask_to_closed_waveguides_gds


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
WRITE_DIR = os.path.join(FILE_DIR, "data", TIMESTAMP)


# needed conversions and hbar
conv_ueVps_uW = 1.602176633*1e-19 * (3600 * 1e12) # conversion ueV/ps to uW
hbar_ueV = 6.582 * 1e2 # ps * ueV

def gen_waveguide_potentials(
        x: np.ndarray,
        y: np.ndarray, 
        n1: float, 
        n2: float, 
        list_center: Iterable[float], 
        w: float, 
        ) -> np.ndarray:
    
    X, Y = np.meshgrid(x,y, indexing='ij')
    n = n2 * np.ones_like(X)

    for xc in list_center:
        n[(x > (xc - w/2.)) & (x < (xc + w/2.))] = n1

    return n1 ** 2 - n ** 2
 

def generate_linear_eigenmode(
        x: np.ndarray, 
        n1: float, 
        n2: float, 
        x0: float, 
        w: float, 
        k0: float, 
        return_beta: bool=False
        ):

    # optim func
    def f_opt(x):

        r = x[0]
        beta = x[1]

        delta = np.sqrt(beta**2 - (n2 * k0)**2)
        kappa = np.sqrt((n1 * k0)**2 - beta**2)

        #res = [r**2 * (1 + (delta/kappa)**2) - 1., np.tan(kappa * L / 2) - delta/kappa]
        res = [np.cos(w/2 * kappa) - r, kappa / delta * np.sin(w/2 * kappa) - r]
        return res

    # Initial guess
    x_init = [0., n1 * k0]

    # Define bounds for each variable (lower and upper)
    bounds = [(0., 1.), (n2 * k0, n1 * k0)]  

    # Solve using the trust-region method
    solution = least_squares(f_opt, x_init, bounds=bounds)
    A, B, beta = solution.x[0], 1., solution.x[1]

    xL = x[x < x0-w/2]
    xM = x[(x >= x0-w/2) & (x <= x0+w/2)]
    xR = x[x > x0+w/2]

    delta = np.sqrt(beta**2 - (n2 * k0)**2)
    kappa = np.sqrt((n1 * k0)**2 - beta**2)
    pL = A * np.exp(delta * (xL - x0 + w/2))
    pR = A * np.exp(-delta * (xR - x0 - w/2))
    pM = B * np.cos(kappa * (xM - x0))

    if not return_beta:
        return np.concatenate((pL, pM, pR))
    return np.concatenate((pL, pM, pR)), beta


if __name__ == "__main__":

    """
        Findings
        separation: 0.5 - 2.0
        taper_length: 25
        coupler_length: 25 - 50 - 75 - 100 (normal)
        coupler_length: 12.5 - 25 - 37.5 - 50 (reduced)


    """

    x = np.linspace(0, 100, 4001) # in um
    y = np.linspace(-5, 5, 201)
    
    num_channels = 2

    # input properties -> get to external yaml files
    wavelength = 0.78 # in um
    wg_width = .5 # width of WG in um
    wg_separation_sides = wg_width + 2
    wg_separation_middle = wg_width + 0.5 # distance between coupler WGs (from center waveguides)
    coupler_length = 15
    taper_length = 20 # length of taper to coupler
    taper_bend = 5 # bend, curvature of taper

    n_WG =  3.27897 # relative refr. index of waveguides
    n_etching = 3.185349 # relative refr. index of surrounding
    g_ueV = 0 # interaction constant, ueV * um^2
    n_eff = n_WG # effective refractive index for polariton propagation]
    c_vac = (3 * 10**8 * 10**6 / 10**12) # in um / ps: speed of light in vacuum

    # get relevant params
    group_velocity = c_vac / n_eff # um/ns
    g = g_ueV / hbar_ueV # in ps * um^2
    omega = 2. * np.pi / wavelength * c_vac # omega in ps^-1

    pulse_duration = 1. # in ps
    pulse_start = 3. # in sigma
    num_photons = 3.
    device = 'cuda:2'
    no_separate_plots = False

    # simulation
    dt = 0.001 # in ps
    x_prop = x.max()
    t_max = x_prop / group_velocity # in ps
    t_samples = np.linspace(0, t_max, 21) # in ps

    # the x and k-vector
    dx, dy = x[1] - x[0], y[1] - y[0]
    nx, ny = x.size, y.size
    X, Y = np.meshgrid(x, y, indexing='xy')
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi  # Wave numbers
    Kx, Ky = np.meshgrid(kx, ky, indexing='xy')


    # COMPUTE INITIAL STATE

    # calculate linear transverse eigenmode profile
    k0_transverse = 2. * np.pi / wavelength # um^-1
    prof_transverse, beta = generate_linear_eigenmode(
        y, n1=n_WG, n2=n_etching, x0=wg_separation_sides/2, w=wg_width, k0=k0_transverse, return_beta=True
        )
    n_beta = beta / k0_transverse
    prof_transverse /= np.sqrt(np.sum(prof_transverse ** 2)) # density to be normalized

    # calculate 2D gaussian pulse profile
    k0_longitudinal =  2 * np.pi * n_beta / wavelength # the pulse wavevector, um^-1
    k0_longitudinal = ky[np.abs(ky - k0_longitudinal).argmin()] # make sure it is commensurate
    
    Xt = torch.tensor(X, device=device)
    df = torch.zeros_like(Xt, dtype=torch.complex64, device=device)
    prof_transverse_t = torch.tensor(prof_transverse, device=device)
    def dF_pulse(t):

        # the pulse entering from left
        df_stripe = (
            (torch.exp(1j * (k0_transverse * Xt - omega * t)).T * prof_transverse_t).T  # transverse WG profile times temporal phase
            * np.sqrt((1. / (np.sqrt(2.*np.pi * pulse_duration ** 2)))) * np.exp(-0.25 * (t - pulse_start) ** 2 / pulse_duration ** 2) # normalized Gauss in time
            * dt * num_photons # normalization/discretization
        )

        #df_line = (torch.exp(1j * (k0_transverse * Xt - omega * t)))[:, :100]
        df[:,:20] = df_stripe[:,:20]
        return df


    # THE POTENTIAL
    profile_coupler = generate_directional_coupler(
        x=x, y=y,
        width=wg_width, separation_sides=wg_separation_sides, separation_middle=wg_separation_middle,
        coupler_length=coupler_length, taper_length=taper_length, bend=taper_bend
    )
    delta_n = -(n_WG - n_etching) * (profile_coupler)
    V = k0_transverse * (c_vac / n_beta) * delta_n


    # THE GPE SIMULATION
    omega_k = np.sqrt((c_vac / n_beta * Ky) ** 2 + (c_vac / n_eff * Kx) ** 2)
    list_t, list_psi = split_step_gpe_2d(
        psi0=None, dt=dt, x=x, y=y, t_samples=t_samples, g=g, V=V, group_velocity=group_velocity,
        dispersion=omega_k, dF=dF_pulse,
        device=device
    )

    # create write dir
    if not os.path.exists(WRITE_DIR):
        os.makedirs(WRITE_DIR)

    from skimage import measure
    import scipy.ndimage

    # Set the downsampling factor (e.g., 4 means reduce resolution by 4x)
    factor_x = 2
    factor_y = 10

    # Downsample X and Y to match
    X_ds = scipy.ndimage.zoom(X, (1/factor_x, 1/factor_y), order=1)
    Y_ds = scipy.ndimage.zoom(Y, (1/factor_x, 1/factor_y), order=1)

    density_up = []
    density_down = []
    for t, psi in zip(list_t, list_psi):

        # Compute |psi|^2 and downsample it with smoothing
        psi_density = np.abs(psi) ** 2

        density_up.append(psi_density[:ny//2].sum(axis=0))
        density_down.append(psi_density[ny//2:].sum(axis=0))

        if not no_separate_plots:
            plt.figure()
            psi_smoothed = scipy.ndimage.zoom(psi_density, (1/factor_x, 1/factor_y), order=1)

            #plt.pcolormesh(X_ds, Y_ds, psi_smoothed.T, shading='flat')
            plt.pcolormesh(Y_ds, X_ds, psi_smoothed)

            V_alpha = np.where(V == 0, 0, 0.2)
            V_ds = scipy.ndimage.zoom(V, (1/factor_x, 1/factor_y), order=1)
            #plt.pcolormesh(X_ds, Y_ds, V_ds.T, cmap='gray', shading='flat', alpha=0.1)
            plt.pcolormesh(Y_ds, X_ds, V_ds, cmap='gray', alpha=0.1)

            plt.title(f't={t:.2f}ps')
            filename = os.path.join(WRITE_DIR, f't_{t:.2f}.png')
            plt.savefig(filename, dpi=100)
            plt.close()

    plt.figure()
    for n_up, n_down in zip(density_up, density_down):
        plt.plot(x, n_up, 'b')
        plt.plot(x, -n_down, 'r')
        plt.plot

    filename = os.path.join(WRITE_DIR, f'densities.png')
    plt.savefig(filename, dpi=100)
    


