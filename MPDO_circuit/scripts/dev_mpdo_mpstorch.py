import os
import pickle
import yaml
import numpy as np
import time
from scipy.linalg import sqrtm
import torch
import torch.autograd.profiler as profiler

from src import MPDOCircuit, StateCreator, MPStorch
from src import BosonOperatorsTorch

import numpy as np

from src.svd_trunc import svd_trunc
from src.utils import irescale, iregroup, sqrtm

def sample_local_dissipation(psi: MPStorch, c_op: torch.Tensor, dt: float=1., no_click_op: torch.Tensor|None=None):
    
    # local variables
    L = psi.num_channels
    batch_dim = psi.batch_dim

    # contract left SVs (rescale index)
    M = irescale(psi[0], psi.get_SL(0), -3)

    # no dissipator, calculate if not given
    if no_click_op is None:
        no_click_op = torch.linalg.matrix_exp(- 0.5 * dt * c_op.conj().T @ c_op)

    # track the jump clicks
    jump_clicks = torch.zeros((L, batch_dim), dtype=torch.int)
    jump_probabilities = torch.zeros((L, batch_dim), dtype=torch.float64)

    for il in range(L):

        # sample and apply
        # M = sample_and_apply(M)
               # get the norm
        norm = torch.einsum("bijk, bijk -> b", M, M.conj()).real

        # apply jump
        M_prob = torch.einsum("ij,...kjl -> ...kil", c_op, M)

        # compute click probability
        dp = dt * torch.einsum("bijk, bijk -> b", M_prob, M_prob.conj()).real / norm

        # sample clicks
        is_click = (dp > torch.rand(batch_dim))

        # construct click operators
        click_ops = torch.zeros(batch_dim, *c_op.shape, device=psi.device, dtype=psi.dtype)
        for ij, jump in enumerate(is_click):
            if jump:
                click_ops[ij] = c_op
            else:
                click_ops[ij] = no_click_op

        # apply batch of dissipators
        M = torch.einsum("...ij,...kjl -> ...kil", click_ops, M)

        # perform QR decomposition
        Q, R = torch.linalg.qr(iregroup(M, [[-3,-2], [-1]]), mode='reduced')

        # update psi at site il and il+1
        psi[il] = Q.view(list(M.shape[:-3]) + [M.shape[-3], M.shape[-2], -1])

        # create M for next iteration (if not last one)
        if not il == L-1:
            M = torch.einsum("...ij, ...jkl->...ikl", R, psi[il+1])

        # track click probabilities
        jump_probabilities[il] = dp
        jump_clicks[il] = is_click

    # get back in canonical form
    psi.canonical_form(cutoff=1e-10, lowrank=False, only_RL=True)

    # return jumps
    return jump_clicks, jump_probabilities


def krauss_local_dissipation_old(
        psi: MPStorch, 
        c_op: torch.Tensor, 
        no_click_op: torch.Tensor|None=None,
        options = {'max_BD': 100, 'max_ED': 100, 'cutoff_BD': 1e-6, 'cutoff_ED': 1e-6}):

    # local variables
    L = psi.num_channels

    # no click op
    if no_click_op is None:
        no_click_op = sqrtm(
            torch.eye(c_op.shape[0], device=c_op.device, dtype=c_op.dtype) - c_op.conj().T @ c_op
            )

    # loop through chain  
    for il in range(L):

        start = time.time()

        # two scenarios
        click_B = torch.einsum("ij, bkjm -> bkim", c_op, psi[il])
        no_click_B = torch.einsum("ij, bkjm -> bkim", no_click_op, psi[il])

        # concat  and SVD
        B_full = torch.concat([click_B, no_click_B], dim=0)
        U, s, Vd, rescale = svd_trunc(
            B_full.view(B_full.shape[0], -1),
            #iregroup(B_full, [[0],[1,2,3]]), 
            cutoff = options["cutoff_ED"], 
            max_num = options['max_ED'],
            lowrank=True
        )

        # rescale Vd with SVs s and reshape to correct form
        psi[il] = irescale(Vd, factor=s, ind=0).view(s.shape + psi[il].shape[1:])

        print(f"il={il} finished {time.time()-start:.4f}s")

    # set back in canonical form
    start = time.time()
    psi.canonical_form(lowrank=False, cutoff=options["cutoff_BD"], max_num=options['max_BD'])
    print(f"canonical form {time.time()-start:.4f}s")


def krauss_local_dissipation(
    psi: MPStorch, 
    c_op: torch.Tensor, 
    no_click_op: torch.Tensor|None=None,
    options = {'max_BD': 100, 'max_ED': 100, 'cutoff_BD': 1e-6, 'cutoff_ED': 1e-6}):

    # local variables
    L = psi.num_channels

    # no click op
    if no_click_op is None:
        no_click_op = sqrtm(
            torch.eye(c_op.shape[0], device=c_op.device, dtype=c_op.dtype) - c_op.conj().T @ c_op
            )

    # B tensor left index  
    B = irescale(psi[0], psi.get_SL(0), ind=-3)

    # loop through chain  
    for il in range(L):

        start = time.time()

        # two scenarios
        click_B = torch.einsum("ij, bkjm -> bkim", c_op, B)
        no_click_B = torch.einsum("ij, bkjm -> bkim", no_click_op, B)

        # concat  and SVD
        B_full = torch.concat([click_B, no_click_B], dim=0)
        U, s_ED, Vd, rescale = svd_trunc(
            B_full.view(B_full.shape[0], -1),
            #iregroup(B_full, [[0],[1,2,3]]), 
            cutoff = options["cutoff_ED"], 
            max_num = options['max_ED'],
            lowrank=True
        )

        # B is new Vd tensor, rescaled with obtained SVs
        B_krauss = irescale(Vd, factor=s_ED, ind=0).view(s_ED.shape + psi[il].shape[1:])

        # QR across bond dimension
        Q, R = torch.linalg.qr(iregroup(B_krauss, [[0,1,2], [3]]), mode='reduced')

        # rescale Vd with SVs s and reshape to correct form
        psi[il] = Q.view(B_krauss.shape)

        # B for next iteration
        if not il == L-1:
            B = torch.einsum("ij, bjkl->bikl", R, psi[il+1])

        print(f"il={il} finished {time.time()-start:.4f}s")

    # backward R->L SVDs, for convenience, set left bond index first
    B = psi[L-1]
    B = iregroup(irescale(B, psi.get_SL(L), ind=-1), [[1],[0],[2],[3]])

    for il in range(L - 1, -1, -1):

        # SV decomposition and truncate, if requested
        U, s, Vd, renormalize_factor = svd_trunc(
                iregroup(B, [[0], [1,2,3]]), 
                cutoff=options['cutoff_BD'], 
                max_num=options['max_BD'],
                ) # 

        # update right site, make sure to swap indices
        psi.set_SL(il, s)
        psi[il] = iregroup(
            Vd.view([-1, B.shape[1], B.shape[2], B.shape[3]]), 
            [[1],[0],[2],[3]]
            )

        # for next iteration, always swap indices back (if not last one)
        if not il == 0:
            B = iregroup(
                torch.einsum("bijk,kl->bijl", psi[il-1], irescale(U, s, ind=-1)),
                [[1],[0],[2],[3]]
            )
            
def number_outcome(psi: MPStorch, il: int) -> torch.Tensor:

    return torch.einsum(
        "cj, bijk,bick", 
        ops.n, 
        irescale(psi[il], psi.get_SL(il)**2, ind=-3), psi[il].conj()
        ).real

def number_outcomes_MPDO(psi):
    return torch.stack([number_outcome(psi, il) for il in range(psi.num_channels)])

    
if __name__ == "__main__":

    # circuit
    num_channels = 5
    num_layers = 12
    U = 0.2
    J_init = 0.001
    dt = 1.
    gamma = .1
    batchdim = 400

    # initial state
    Nmax = 10
    alpha = 1.
    N_fock = 1
    is_coherent = False
    device = "cuda:0"

    # options
    options = { 'max_BD': 500,'cutoff': 1e-6}
    options_krauss = {'max_BD': 500, 'max_ED': 100, 'cutoff_BD': 1e-6, 'cutoff_ED': 1e-6}


    # run
    d_run = {'slow_trajectory': False, 'fast_trajectory': False, "krauss": True}


    dict_rhos = {}

    # initialize input state
    state_creator = StateCreator(Nmax, num_batch=1)

    if is_coherent:
        list_ten = state_creator.product_state_coherent(alpha * np.ones(num_channels), to=device)
    else:
        list_ten = state_creator.product_state_fock(nums = [N_fock] * num_channels, to=device)
    psi = MPStorch(list_ten)

    # set up circuit
    circuit = MPDOCircuit(
        num_channels=num_channels, 
        num_layers=num_layers, 
        Nmax=Nmax, 
        layer_depth=num_layers * [1.],
        J_init=J_init,
        U_init=U,
        gamma=0.,
        to=device
    )

    # run circuit with MPS
    start = time.time()
    circuit.run(psi, options=options)

    print(rf"Finished run in {time.time()-start:.2f}s")

    # print output
    ns = torch.stack(psi.number_outcomes())
    tot_ns = ns.sum()

    print("\nCircuit output:\n")
    print(f"ns: {ns}")
    print(f"tot n: {tot_ns}")
    print(f"BDs: {psi.get_BDs()}")

    psi_c = psi.clone()


    # apply quantum jump
    ops = BosonOperatorsTorch(Nmax, to=device)
    loss = ops.a
    nonunitary = torch.matrix_exp(-0.5 * dt * gamma * ops.n)

    if d_run['slow_trajectory']:
        psi = psi_c.clone()

        start = time.time()
        for il in range(num_channels):
            psi[il] = torch.einsum("ij,...kjl -> ...kil", loss, psi[il])
            psi.canonical_form(cutoff=1e-10, lowrank=False)
        
        # print output of canonical form
        ns = torch.stack(psi.number_outcomes())
        tot_ns = ns.sum()
        print(f"\nOutput inefficient (in {time.time()-start:.4f}s):\n")
        print(f"ns: {ns}")
        print(f"tot n: {tot_ns}")


    #### efficient way
    if d_run['fast_trajectory']:
        psi_copy = psi_c.clone()
        start = time.time()
        jump_clicks, jump_probabilities = sample_local_dissipation(psi_copy, np.sqrt(gamma) * ops.a)

        # print output of canonical form
        ns = torch.stack(psi_copy.number_outcomes())
        tot_ns = ns.sum()
        print(f"\nOutput efficient (in {time.time()-start:.4f}s):\n")
        print(f"ns: {ns}")
        print(f"tot n: {tot_ns}")
        print(f"clicks: {jump_clicks.squeeze()}")
        print(f"click probabilities: {jump_probabilities.squeeze()}")

    
    #### Krauss way
    if d_run['krauss']:
        psi_copy = psi_c.clone()
        start = time.time()
        krauss_local_dissipation(psi_copy, c_op=np.sqrt(gamma) * ops.a, options=options_krauss)

        # print output of canonical form
        ns = number_outcomes_MPDO(psi_copy)
        tot_ns = ns.sum()
        print(f"\nOutput Krauss (in {time.time()-start:.4f}s):\n")
        print(f"ns: {ns}")
        print(f"tot n: {tot_ns}")
        print(f"BDs: {psi_copy.get_BDs()}")

        # second step
        krauss_local_dissipation(psi_copy, c_op=np.sqrt(gamma) * ops.a, options=options_krauss)

        # print output of canonical form
        ns = number_outcomes_MPDO(psi_copy)
        tot_ns = ns.sum()
        print(f"\nOutput Krauss (in {time.time()-start:.4f}s):\n")
        print(f"ns: {ns}")
        print(f"tot n: {tot_ns}")
        print(f"BDs: {psi_copy.get_BDs()}")

        # compute backpopagation
        ns = number_outcomes_MPDO(psi_copy)
        FOM = ns[0]

        start = time.time()
        with profiler.profile(record_shapes=True, use_device = 'cuda') as prof:
            FOM.backward()
        print(f"backward: {time.time()-start:.4f}s")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

        # print(f"clicks: {jump_clicks.squeeze()}")
        # print(f"click probabilities: {jump_probabilities.squeeze()}")

        e=0