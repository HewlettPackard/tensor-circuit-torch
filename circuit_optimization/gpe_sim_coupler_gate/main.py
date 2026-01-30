###########################################################################################
# MAIN FILE used for running 2D field GPE simulation of lower-polariton field in appendix of 
# manuscript. Without drive field, an initial field profile must be inserted for further
# time propagation.
###########################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from typing import Iterable
import os
from datetime import datetime
import scipy.ndimage
import pickle

from src.GPE_solver.solver_2D import split_step_gpe_2d
from src.GPE_solver.design_utils import generate_directional_coupler


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
WRITE_DIR = os.path.join(FILE_DIR, "data", TIMESTAMP)


# Some useful constants
conv_ueVps_uW = 1.602176633*1e-19 * (3600 * 1e12) # conversion ueV/ps to uW
hbar_ueV = 6.582 * 1e2 # ps^{-1} * ueV

def gen_waveguide_potentials(
        x: np.ndarray,
        y: np.ndarray, 
        n1: float, 
        n2: float, 
        list_center: Iterable[float], 
        w: float, 
        ) -> np.ndarray:
    
    """Local function to generate the waveguide potentials from the refractive indices"""
    
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

    """Generate the transverse waveguide eigenmode, using the Helmholtz equation"""

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
    Run the 2D GPE field simulations, everything is dimensionful. Space units are expressed in
    micrometer (um) and time units in picoseconds (ps).
    """

    # x and y grid. x is direction of photon propagation (units are um)
    x = np.linspace(0, 700, 14001)
    y = np.linspace(-5, 5, 201)
    
    # Duh, it is a coupler...
    num_channels = 2

    # input properties -> get to external yaml files
    wavelength = 0.78 # in um
    wg_width = .5 # width of WG in um
    wg_separation_sides = wg_width + 2.
    wg_separation_middle = wg_width + .5 # distance between coupler WGs (from center waveguides)
    coupler_length = 30
    taper_length = 25 # length of taper to coupler
    taper_bend = 2 # bend, curvature of taper

    # refractive indices inside waveguide and outside (etching)
    n_WG = 3.28
    n_etching = 3.18

    g_ueV = 600. # interaction constant, ueV * um^2
    n_eff = 2 * n_WG # effective refractive index for polariton propagation]
    c_vac = (3 * 10**8 * 10**6 / 10**12) # in um / ps: speed of light in vacuum (left powers of 10 for clarity)

    # get relevant params
    group_velocity = c_vac / n_eff # um/ps
    g = g_ueV / hbar_ueV # in ps * um^2

    pulse_width = 1. * group_velocity # ps -> um
    x_start_pulse = 160 # x coordinate of middle pulse at t=0 (in um)
    num_photons = 10 # average number of photons contained in pulse
    device = 'cpu' # The device to run on (change to 'cpu' if no gpu available)

    # SIMULATION PARAMETERS: simulation time is set by propagation distance
    dt = 0.001 # differential time step in ps
    x_prop = x.max() - 2 * x_start_pulse # the propagation lenth
    t_max = x_prop / group_velocity # in ps
    t_samples = np.linspace(0, t_max, 11) # in ps

    # the x and k-vector
    dx, dy = x[1] - x[0], y[1] - y[0]
    nx, ny = x.size, y.size
    X, Y = np.meshgrid(x, y, indexing='ij')
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

    # calculate 2D gaussian pulse profile
    k0_longitudinal =  2 * np.pi * n_beta / wavelength # the pulse wavevector, um^-1
    k0_longitudinal = ky[np.abs(ky - k0_longitudinal).argmin()] # make sure it is commensurate
    psi0 = np.outer(
        prof_transverse,
        np.exp(1j * k0_longitudinal * x) * np.sqrt(np.exp(-(x - x_start_pulse) ** 2 / (4 * pulse_width ** 2))).astype(np.complex128),
    )

    # normalize and scale to correct amount of photons
    psi0 = np.sqrt(num_photons) * psi0 / np.sqrt(np.sum(np.abs(psi0) ** 2) * dx * dy)


    # THE POTENTIAL: effective potential obtained from variations of refractive index (see appendix manuscript)
    profile_coupler = generate_directional_coupler(
        x=x, y=y,
        width=wg_width, separation_sides=wg_separation_sides, separation_middle=wg_separation_middle,
        coupler_length=coupler_length, taper_length=taper_length, bend=taper_bend
    )
    delta_n = (n_WG - n_etching) * (profile_coupler) + (n_WG - n_etching)
    V = - 2 * k0_transverse * (c_vac / n_etching) * delta_n


    # RUN GPE SIMULATION

    # the dispersion
    omega_k = np.sqrt((c_vac / n_beta * Ky) ** 2 + (c_vac / n_eff * Kx) ** 2)

    # run GPE solver
    list_t, list_psi = split_step_gpe_2d(
        psi0=psi0, dt=dt, x=x, y=y, t_samples=t_samples, g=g, V=V,
        dispersion=omega_k,
        device=device
    )

    # create write dir
    if not os.path.exists(WRITE_DIR):
        os.makedirs(WRITE_DIR)

    # Set the downsampling factor to reduce time for making colorplots
    #  (e.g., 4 means reduce resolution by 4x)
    factor_x = x.max() / 10
    factor_y = 2

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

        # THE PLOT

        # figure
        plt.figure(figsize=(3,6))

        # smoothen the field
        psi_smoothed = scipy.ndimage.zoom(psi_density, (1/factor_y, 1/factor_x), order=1)

        # make colormesh
        plt.pcolormesh(Y_ds, X_ds, psi_smoothed.T)
        
        # the smoothened potential (transparency alpha low)
        V_ds = scipy.ndimage.zoom(V, (1/factor_y, 1/factor_x), order=1)
        plt.pcolormesh(Y_ds, X_ds, V_ds.T, cmap='gray', alpha=0.1)

        # titles and labels etc.
        plt.title(rf'$t={t:.2f}ps$', fontsize=14)
        plt.xlabel(r'$x\; (\mu m)$',fontsize=12)
        plt.ylabel(r'$y\; (\mu m)$',fontsize=12)
        plt.tight_layout()

        # savefig and close, for next iteration
        filename = os.path.join(WRITE_DIR, f't_{t:.2f}.png')
        plt.savefig(filename, dpi=100)
        plt.close()
    
    # save pkl file with field data
    with open(os.path.join(WRITE_DIR, 'data'), 'wb') as handle:
        d = {'t_samples': list_t, 'data': list_psi, 'X': X, 'Y': Y, 'V': V}
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # figure of density upper and lower waveguide
    plt.figure()
    for n_up, n_down in zip(density_up, density_down):
        plt.plot(x, n_up, 'b')
        plt.plot(x, -n_down, 'r')

    # save figure
    filename = os.path.join(WRITE_DIR, f'densities.png')
    plt.savefig(filename, dpi=100)
    


