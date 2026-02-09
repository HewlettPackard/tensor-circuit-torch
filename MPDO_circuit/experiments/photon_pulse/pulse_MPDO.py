import torch
import numpy as np
import os
import pickle
from time import time
import matplotlib.pyplot as plt

from src.create_circuit import create_nonlinear_bosonic_TEBD_step
from src.mpdo_torch import (
    MPDOtorch,
    StateCreator
)

from src.utils import BosonOperatorsTorch

from experiments.LP_utils import vg_LP, curvature_LP

hbar_ueV_ps = 6.582 * 1e2 # hbar, units ueV * ps, converting energy to frequency


def plot_results(z, densities, a_expect, entanglement_entropy, purity_entropy, t_samples, title: str, dir: str):
    
    # update plot
    fig, ax = plt.subplots(5,1, figsize=(5,10))

    # densities 
    ax[0].plot(z, densities)
    ax[0].set_title("Intensity")

    # quadratures
    X = 1. / np.sqrt(2) * (a_expect + a_expect.conj())
    ax[1].plot(z, X.real)
    ax[1].set_title("X quadrature")

    Y = - 1j / np.sqrt(2) * (a_expect - a_expect.conj())
    ax[2].plot(z, Y.real)
    ax[2].set_title("Y quadrature")

    ax[3].plot(z, entanglement_entropy)
    ax[3].set_title("von Neumann entropy")

    ax[4].plot(z, purity_entropy)
    ax[4].set_title("purity entropy")

    ax[0].legend([f"t={t:.2f} ps" for t in t_samples], loc="upper right")

    for a in ax:
        a.set_xlabel('z (um)')

    if title is not None:
        fig.suptitle(title)

    filename = os.path.join(dir, 'field_profile')

    fig.tight_layout()
    fig.savefig(filename)
    plt.close()

    dict_result = {
        'z': z, 
        'densities': densities, 
        'a_expect': a_expect, 
        'entanglement_entropy': entanglement_entropy, 
        'purity_entropy': purity_entropy, 
        't_samples': t_samples
        }
    
    dict_filename = os.path.join(dir, 'data.pkl')
    with open(dict_filename, "wb") as f:
        pickle.dump(dict_result, f)
    



if __name__ == "__main__":

    device = 'cuda:1'
    store_device = 'cpu'
    dir_store = 'field_nonlinear_loss'

    zmax = 200. # grid -zmax <-> + zmax, in um
    nz = 401 # number of subdivisions
    Nmax = 5

    pulse_duration = 1 # ps
    pulse_intensity = 2 # average number of photons (Poissonian)
    hg = 10. # ueV * um
    gamma = 0.05 # ps^(-1)
    x_prop = 500 # um
    ref_k = .3 # um^{-1}

    is_coherent = True

    dt = 0.05
    num_samples = 11
    options_MPDO = {'max_BD': 50, 'max_PD': 50, 'cutoff_BD': 1e-8, 'cutoff_PD': 1e-5}

    group_velocity = vg_LP(ref_k) # in um / ps
    J = 0.5 * curvature_LP(ref_k) # important: factor 2 (?) -> quite sure
    t_max = x_prop / group_velocity
    t_samples = torch.linspace(0, t_max, num_samples) # in ps

    # filename figure
    abs_dir_store = os.path.join('experiments/photon_pulse', dir_store)
    os.makedirs(abs_dir_store, exist_ok=True)


    # the grid
    z = torch.linspace(-zmax, zmax, nz, device=device, dtype=torch.float64)
    dz = float(z[1] - z[0])

    # spatial extent
    sigma_z = pulse_duration * group_velocity

    # photon profile
    profile = torch.exp(- 0.25 * z ** 2 / sigma_z ** 2) * np.sqrt(dz)
    profile = np.sqrt(pulse_intensity) * profile / torch.sqrt((torch.abs(profile) ** 2).sum())

    # initialize input state
    state_creator = StateCreator(Nmax, num_batch=1)

    # initialise MPDO from tensors
    list_ten = state_creator.product_state_coherent(profile.cpu().numpy(), device=device)
    rho = MPDOtorch(list_ten)

    # bosonic operators
    ops = BosonOperatorsTorch(Nmax, device=device)

    # rescale parameters with differential element
    J_dz = J / dz ** 2
    U_dz = hg / hbar_ueV_ps / dz

    # create circuit
    TEBD_step = create_nonlinear_bosonic_TEBD_step(
        num_channels=nz,
        J=J_dz,
        U=U_dz,
        Nmax=Nmax,
        dt=dt,
        gamma=gamma,
        device=device,
        requires_grad=False
    )

    ########################################################
    # START LOOP TIME INTEGRATION
    ########################################################

    # differential times
    t_values = torch.arange(0, t_max + dt, dt)  # Ensure we step through exact times
    num_steps = t_values.numel()

    # the collection of the data samples
    collect_inds = torch.searchsorted(t_values, t_samples) 
    t_sampled = t_values[collect_inds].to(store_device)
    num_collect = collect_inds.numel()

    # the collection of psis to be returned at the end
    store_device = rho.device if store_device=='same' else store_device
    density_collect = torch.empty((num_collect, nz), dtype=torch.float64, device=store_device)
    a_expect_collect = torch.empty((num_collect, nz), dtype=torch.complex128, device=store_device)
    S_vN_collect = torch.empty((num_collect, nz+1), dtype=torch.float64, device=store_device)
    S_p_collect = torch.empty((num_collect, nz), dtype=torch.float64, device=store_device)

    sample_idx = 0
    if 0 in collect_inds:
        density_collect[sample_idx] = rho.number_outcomes().to(store_device)
        a_expect_collect[sample_idx] = rho.local_expectations(ops.a).to(store_device)
        S_vN_collect[sample_idx] = rho.entropy_profile().to(store_device)
        S_p_collect[sample_idx] = rho.entropy_profile(entropy='purity').to(store_device)
        sample_idx += 1

        title = rf"Results: $k={ref_k:.2f}$, $U_{{eff}}={U_dz:.2f}$, $J_{{eff}}={J_dz:.2f}$"
        plot_results(
            z.cpu().numpy(), 
            density_collect[:sample_idx].T.cpu().numpy(), a_expect_collect[:sample_idx].T.cpu().numpy(),
            S_vN_collect[:sample_idx, :-1].T.cpu().numpy(), S_p_collect[:sample_idx].T.cpu().numpy(),
            t_samples[:sample_idx].cpu().numpy(), title=title, dir=abs_dir_store
            )
        
        e=0
        

    start = time()
    for i_step in range(num_steps):

        print(f'\nRunning step {i_step+1}/{num_steps}...')
        step_start = time()
        TEBD_step.run(rho, options=options_MPDO)
        print(f"Step finished: {time()-step_start:.2f}s, runtime={time()-start:.2f}s")

        print(f"BD: {rho.get_BD(nz // 2)}, \
              PD: {rho.get_PD(nz // 2)}, \
              N_ph: {rho.number_outcomes().sum()}")

        if i_step + 1 in collect_inds:
            print(f"Collecting results {sample_idx}/{num_collect}")

            density_collect[sample_idx] = rho.number_outcomes().to(store_device)
            a_expect_collect[sample_idx] = rho.local_expectations(ops.a).to(store_device)
            S_vN_collect[sample_idx] = rho.entropy_profile().to(store_device)
            S_p_collect[sample_idx] = rho.entropy_profile(entropy='purity').to(store_device)
            sample_idx += 1

            title = rf"Results: $k={ref_k:.2f}$, $U_{{eff}}={U_dz:.2f}$, $J_{{eff}}={J_dz:.2f}$"
            plot_results(
                z.cpu().numpy(), 
                density_collect[:sample_idx].T.cpu().numpy(), a_expect_collect[:sample_idx].T.cpu().numpy(),
                S_vN_collect[:sample_idx, :-1].T.cpu().numpy(), S_p_collect[:sample_idx].T.cpu().numpy(),
                t_samples[:sample_idx].cpu().numpy(), title=title, dir=abs_dir_store
                )

    
    print(f"Full run finished in {time()-start:.2f}s")






    e=0
