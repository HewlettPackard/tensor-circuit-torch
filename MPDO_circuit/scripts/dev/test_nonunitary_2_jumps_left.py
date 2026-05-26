import os
import pickle
import yaml
import numpy as np
import time
from scipy.linalg import sqrtm
import torch

from src.MPS_state_optimization import NonlinearPhotonicCircuit, StateCreator, MPStorch, StateAnalyzer
from src.MPS_state_optimization import BosonOperatorsTorch

import numpy as npz


from MPS_state_optimization.tensor_circuit import CircuitLocalLossGate
from src.MPS_state_optimization.svd_trunc import svd_trunc
from src.MPS_state_optimization.utils import irescale, iregroup


if __name__ == "__main__":

    num_channels = 5
    num_layers = 10
    U = 0.25
    Nmax = 10
    alpha = 1.
    J_init = 0.3
    options = { 'max_BD': 500,'cutoff': 1e-6}
    to = 'cpu'


    dict_rhos = {}

    J_vals = [
        [np.pi/2. * np.random.rand() if (ic + il) % 2 == 0 else None for ic in range(num_channels)] 
        for il in range(num_layers)
        ]

    # initialize input state
    state_creator = StateCreator(Nmax, num_batch=1)
    list_ten = state_creator.product_state_coherent(alpha * np.ones(num_channels), to=to)
    list_ten = state_creator.product_state_fock(nums = [1] * num_channels)
    psi = MPStorch(list_ten)

    # set up circuit
    circuit = NonlinearPhotonicCircuit(
        num_channels=num_channels, 
        num_layers=num_layers, 
        Nmax=Nmax, 
        layer_depth=num_layers * [1.],
        J_init=J_init,
        U_init=U,
        gamma=0.,
        to=to
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

    dt = 1.
    gamma = .1
    ops = BosonOperatorsTorch(Nmax, to=to)

    nonunitary = torch.matrix_exp(-0.5 * dt * gamma * ops.n)
    loss = ops.a

    psi = psi_c.clone()
    il = 0
    psi[il] = torch.einsum("ij,...kjl -> ...kil", loss, psi[il])
    psi.canonical_form(cutoff=1e-10, lowrank=False)
    
    # print output of canonical form
    ns = torch.stack(psi.number_outcomes())
    tot_ns = ns.sum()
    print("\nCanonical output 1 click:\n")
    print(f"ns: {ns}")
    print(f"tot n: {tot_ns}")

    psi[il+1] = torch.einsum("ij,...kjl -> ...kil", loss, psi[il+1])
    psi.canonical_form(cutoff=1e-10, lowrank=False)

    # print output of canonical form
    ns = torch.stack(psi.number_outcomes())
    tot_ns = ns.sum()

    print("\nCanonical output 2 clicks:\n")
    print(f"ns: {ns}")
    print(f"tot n: {tot_ns}")

    #### efficient way

    psi_copy = psi_c.clone()

    # 1st click
    M = irescale(psi_copy[il], psi_copy.get_SL(il), -3)
    M = torch.einsum("ij,...kjl -> ...kil", loss, M)

    # rescale matrix with left SVs and do new svd
    Q, R = torch.linalg.qr(iregroup(M, [[-3,-2], [-1]]), mode='reduced')

    # reshape outcome Vd and update tensor + SVs in mps
    U_mps = Q.view(list(M.shape[:-3]) + [M.shape[-3], M.shape[-2], -1])
    psi_copy[il] = U_mps

    ns = torch.stack(psi_copy.number_outcomes())
    tot_ns = ns.sum()

    print("\n\nTest one click:\n")
    print(f"ns: {ns}")
    print(f"tot n: {tot_ns}")

    # compute updated site n-1
    U_p1 = torch.einsum("...ij, ...jkl->...ikl", R, psi_copy[il+1])
    n_p1 = torch.einsum("bijk,lj,bilk", U_p1, ops.n, U_p1.conj())
    norm = torch.einsum("bijk,bijk", U_p1, U_p1.conj())

    print("\nupdated site left:")
    print(f"corr n_(l+1): {n_p1}")
    print(f"norm: {norm}")
    print(f"corr n_(l+1) / norm: {n_p1/norm}")

    psi_copy[il+1] = U_p1



    # second click
    M = torch.einsum("ij,...kjl -> ...kil", loss, psi_copy[il+1])

    # rescale matrix with left SVs and do new svd
    Q, R = torch.linalg.qr(iregroup(M, [[-3,-2], [-1]]), mode='reduced')

    # reshape outcome Vd and update tensor + SVs in mps
    U_mps = Q.view(list(M.shape[:-3]) + [M.shape[-3], M.shape[-2], -1])
    psi_copy[il+1] = U_mps

    ns = torch.stack(psi_copy.number_outcomes())
    tot_ns = ns.sum()

    print("\n\nTest one click:\n")
    print(f"ns: {ns}")
    print(f"tot n: {tot_ns}")

    # compute updated site n-1
    U_p2 = torch.einsum("...ij, ...jkl->...ikl", R, psi_copy[il+2])
    n_p2 = torch.einsum("bijk,lj,bilk", U_p2, ops.n, U_p2.conj())
    norm = torch.einsum("bijk,bijk", U_p2, U_p2.conj())

    print("\nupdated site 2 left:")
    print(f"corr n_(l+2): {n_p2}")
    print(f"norm: {norm}")
    print(f"corr n_(l+2) norm: {n_p2/norm}")

    psi_copy[il+2] = U_p2

    psi_copy.canonical_form(cutoff=1e-10, lowrank=False)

    print("\nEfficient Canonical output 2 clicks:\n")
    # print output of canonical form
    ns = torch.stack(psi_copy.number_outcomes())
    tot_ns = ns.sum()
    print(f"ns: {ns}")
    print(f"tot n: {tot_ns}")

    # rescale matrix with left SVs and do new svd
    # M = irescale(irescale(psi_copy[il-1], psi_copy.get_SL(il-1), -3), psi_copy.get_SL(il), -1)
    # U, s, Vd, renormalize = svd_trunc(
    #     iregroup(M, [[-3], [-2, -1]]), 
    #     cutoff=1e-10, 
    #     lowrank=False)

    # # reshape outcome Vd and update tensor + SVs in mps
    # Vd_mps = Vd.view(list(M.shape[:-3]) + [-1, M.shape[-2], M.shape[-1]])
    # psi_copy[il-1] = Vd_mps
    # psi_copy._SL[il-1] = s
    # psi_copy[il-2] = torch.einsum("bijk, bkl->bijl", psi_copy[il-2], U)

    # ns = torch.stack(psi_copy.number_outcomes())
    # tot_ns = ns.sum()

    # print("\n\nTest two clicks:\n")
    # print(f"ns: {ns}")
    # print(f"tot n: {tot_ns}")

    # # back to canonical
    # psi_copy.canonical_form()
    # ns = torch.stack(psi_copy.number_outcomes())
    # tot_ns = ns.sum()
    # print("\n\nTest canoncial:\n")
    # print(f"ns: {ns}")
    # print(f"tot n: {tot_ns}")




    # Vlm1_SL = irescale(psi_copy[il-1], psi_copy.get_SL(il-1), -3)
    # Vlm1 = irescale(
    #     torch.einsum("bijk, bkl->bijl", Vlm1_SL, U), 
    #     psi_copy.get_SL(il), 
    #     -1)
    # nlm1 = torch.einsum("bijk,lj,bilk", Vlm1, ops.n, Vlm1.conj())
    # norm = torch.einsum("bijk,bijk", Vlm1, Vlm1.conj())
    # print(f"corr n_(l-1): {nlm1}")
    # print(f"norm: {norm}")
    # print(f"corr n_(l-1) norm: {nlm1/norm}")

    # print(Vd.shape)
    # print(s.shape)
    # print(s.norm())
    # Vd_mps_resc = irescale(Vd_mps, s, ind=-3)
    # V_norm = torch.einsum("bijk, bkji", Vd_mps, Vd_mps)
    # print(f"norm {V_norm}")

    # M = psi_copy[il]
    # Q, R = torch.linalg.qr(iregroup(M + 1e-10, [[-3,-2], [-1]]), mode='reduced')
    # psi_copy[il] = Q.view(list(M.shape[:-3]) + [M.shape[-3], M.shape[-2], -1])

    # psi_copy[il + 1] = torch.einsum("bij, bjkl->bikl", R, psi_copy[il + 1])
    # norm_ilp1 = torch.einsum("bijk, bijk", psi_copy[il], psi_copy[il].conj())

    # ns = torch.stack(psi_copy.number_outcomes())
    # tot_ns = ns.sum()

    # print("\nCanonical output:\n")
    # print(f"ns: {ns}")
    # print(f"ns_norm: {ns[il+1] * norm_ilp1}")
    # print(f"tot n: {tot_ns}")
    # print(f"BDs: {psi.get_BDs()}")
    # print(torch.einsum("bi, bijk, bijk", psi.get_SL(il+1).type(torch.complex64)**2, psi[il+1], psi[il+1].conj()))

