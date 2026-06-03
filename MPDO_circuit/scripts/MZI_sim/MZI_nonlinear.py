from math import factorial
import numpy as np
import pickle
from itertools import product
from multiprocessing import Pool
import os
from time import time
from datetime import datetime
from qutip import (
    destroy, qeye, Qobj, tensor, coherent, ket2dm, expect
)

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

N            = 10 # Fock dim
Us           = np.logspace(-3, -1, 10) # Nonlinearities
theta_in     = 0.25 * np.pi          # BS1 coupling (fixed)
theta_outs   = np.linspace(0., 0.5, 26) * np.pi # BS2 coupling, variable
gammas       = np.linspace(0., 1., 10) # loss rates
phis         = np.linspace(0., 2 * np.pi, 100) # LO phase
coupler_type = 'symmetric'
is_integrated = True

# Input state
phi_init       = 0.
alpha1, alpha2 = np.sqrt(1.), np.sqrt(0.) * np.exp(1j * phi_init)

# run on all worker
N_WORKERS = os.cpu_count()

# for storage
# dir where script is stored
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# timestamp for storing data
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
SAVE_DIR = os.path.join(FILE_DIR, "data", TIMESTAMP)

# make dir
os.makedirs(SAVE_DIR, exist_ok=True)



def amplitude_damping_kraus(gamma: float, N: int, order: int = 1) -> list:
    """
    Generate Kraus operators for a generalized bosonic amplitude damping channel,
    supporting multi-photon loss events up to a specified order.
 
    The channel is defined as:
        K_0     = diag[(1 - gamma)^(k/2)] for k = 0, ..., N-1   (no-loss)
        K_n     = (sqrt(gamma)^n / sqrt(n!)) * K_0 @ a^n         (n-photon loss)
 
    This matches the standard bosonic amplitude damping channel and reduces
    to the 2-Kraus-op form when order=1.
 
    NOTE on completeness: sum_n K_n† K_n = I holds exactly in infinite
    dimensions. In a truncated Fock space (finite N), completeness is
    approximate; higher `order` and larger `N` improve the approximation.
 
    Args:
        gamma (float): Damping/loss probability per photon. Must be in [0, 1].
        N (int):       Fock space dimension (Hilbert space size). Modes are
                       indexed |0>, |1>, ..., |N-1>.
        order (int):   Maximum number of photon loss events to include.
                       order=1 → standard 2-Kraus amplitude damping.
                       order=k → includes k-photon loss clicks.
 
    Returns:
        list[Qobj]: Kraus operators as QuTiP Qobj matrices of shape (N, N).
                    Length = order + 1.
 
    Raises:
        ValueError: If gamma is outside [0, 1] or order < 1.
 
    Example:
        # Standard amplitude damping (1-photon loss only):
        kraus_ops = amplitude_damping_kraus(gamma=0.1, N=10, order=1)
 
        # Include up to 3-photon loss events:
        kraus_ops = amplitude_damping_kraus(gamma=0.1, N=10, order=3)
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}.")
 
    # Annihilation operator in Fock space of dimension N
    a = destroy(N)
 
    # ----------------------------------------------------------------
    # K_0: No-loss (vacuum survival) operator
    # Diagonal entries: (1 - gamma)^(k/2) for Fock state |k>
    # ----------------------------------------------------------------
    k_indices = np.arange(N, dtype=float)                    # [0, 1, 2, ..., N-1]
    no_loss_diag = (1.0 - gamma) ** (k_indices / 2.0)       # entry-wise survival amplitude
 
    # Build as a QuTiP Qobj for consistent algebra downstream
    no_loss = Qobj(
        np.diag(no_loss_diag),
        dims=[[N], [N]]
    )
 
    # ----------------------------------------------------------------
    # Kraus operators list; starts with K_0
    # ----------------------------------------------------------------
    kraus_ops = [no_loss]
 
    # Precompute powers of a (a^1, a^2, ..., a^order) iteratively
    a_power = qeye(N)     # a^0 = I, will update in loop
 
    for n in range(1, order + 1):
        a_power = a_power * a        # a^n (efficient: avoid recomputing from scratch)
 
        # Prefactor: sqrt(gamma)^n / sqrt(n!)
        prefactor = (np.sqrt(gamma) ** n) / np.sqrt(factorial(n))
 
        # K_n = prefactor * K_0 @ a^n
        # Note: no_loss @ a^n applies survival amplitude THEN loss,
        # consistent with the exact bosonic channel derivation.
        K_n = prefactor * (no_loss * a_power)
 
        kraus_ops.append(K_n)
 
    return kraus_ops
 

def nonlinear_mixer_with_losses(
    N, U1, theta2, alpha1, alpha2,
    phase_shift=0., U2=0., t1=1., t2=1., theta1=np.pi/4, coupler_type='dielectric',
    gamma_in_loss=0., gamma_out_loss=0., order_kraus=3
    ):
    """
    Simulates a Mach-Zehnder Interferometer with a nonlinear element in one arm
    and a local oscillator in the second, including amplitude damping losses.

    Args:
        N (int): Fock space dimension for each mode.
        U1 (float): Nonlinear coupling strength for mode 1 (Kerr effect).
        theta2 (float): Mixing angle for the second beam splitter.
        alpha1 (complex): Coherent state amplitude for the input to arm 1.
        alpha2 (complex): Coherent state amplitude for the input to arm 2 (LO).
        phase_shift (float, optional): Phase shift applied to mode 2 before Kerr. Defaults to 0.
        U2 (float, optional): Nonlinear coupling strength for mode 2. Defaults to 0.
        t1 (float, optional): Interaction time for Kerr gate. Defaults to 1.
        t2 (float, optional): Interaction time for beam splitter. Defaults to 1.
        theta1 (float, optional): Mixing angle for the first beam splitter. Defaults to np.pi/4.
        coupler_type (str, optional): Type of beam splitter ('dielectric' or 'symmetric'). Defaults to 'dielectric'.
        gamma_in_loss (float, optional): Amplitude damping probability for mode 2
                                        before the U_Kerr operation. Defaults to 0.
        gamma_out_loss (float, optional): Amplitude damping probability for mode 2
                                        after the U_Kerr operation. Defaults to 0.

    Returns:
        tuple: Expectation values (exp_n1, exp_n2, exp_nn1, exp_nn2, exp_nn12)
            for number operators and their products in the final state.
    """

    # Bosonic operators for a two-mode system (mode 1, mode 2)
    a1 = tensor(destroy(N), qeye(N)) # Annihilation operator for mode 1
    a2 = tensor(qeye(N), destroy(N)) # Annihilation operator for mode 2

    n1 = a1.dag() * a1 # Number operator for mode 1
    n2 = a2.dag() * a2 # Number operator for mode 2

    # -----------------------------
    # Hamiltonians and Unitary Operators for MZI components
    # -----------------------------
    # Local Kerr nonlinearities (applies to both modes if U1 and U2 are non-zero)
    H_kerr = (U1 / 2) * a1.dag() * a1.dag() * a1 * a1 \
            + (U2 / 2) * a2.dag() * a2.dag() * a2 * a2

    # The U_Kerr unitary combines the Kerr interaction with a phase shift on mode 2.
    # This represents the "nonlinear element" in the MZI arm.
    U_Kerr = (1j * phase_shift * n2).expm() @ (-1j * H_kerr * t1).expm()

    # Linear mode mixing (beam splitter) Hamiltonian and Unitary
    if coupler_type == 'dielectric':
        H_BS_gen = lambda theta: 1j * theta * (a1.dag() * a2 - a1 * a2.dag())
        U_BS_func = lambda theta: (1j * np.pi * n2).expm() @ (-1j * H_BS_gen(theta) * t2).expm()
    elif coupler_type == 'symmetric':
        H_BS_gen = lambda theta: -theta * (a1.dag() * a2 + a1 * a2.dag())
        U_BS_func = lambda theta: (-1j * H_BS_gen(theta) * t2).expm()
    else:
        raise NameError(f"Coupler type '{coupler_type}' not provided. Choose 'dielectric' or 'symmetric'.")

    # -----------------------------
    # Helper function to apply amplitude damping specifically to the second mode
    # -----------------------------
    def apply_amplitude_damping_on_mode2(input_rho, gamma_prob, dim_N):
        """
        Applies an amplitude damping channel to the second mode of a two-mode system.

        Args:
            input_rho (Qobj): The input density matrix of the two-mode system.
            gamma_prob (float): The probability parameter for amplitude damping (0 to 1).
            dim_N (int): The Fock space dimension of each mode.

        Returns:
            Qobj: The density matrix after applying the amplitude damping channel.
        """
        if gamma_prob <= 0:
            return input_rho # No damping if probability is zero or negative

        # Generate Kraus operators for a single mode of dimension N
        # These operators act on a single 'N'-dimensional space.
        kraus_ops_single_mode = amplitude_damping_kraus(gamma_prob, dim_N, order=order_kraus)

        # For our two-mode system, we want to apply damping only to the second mode.
        # We achieve this by tensoring each single-mode Kraus operator with
        # the identity operator for the first mode.
        kraus_ops_composite = [tensor(qeye(dim_N), K) for K in kraus_ops_single_mode]

        # Apply the quantum channel: Lambda(rho) = sum_k K_k * rho * K_k.dag()
        output_rho = sum(K * input_rho * K.dag() for K in kraus_ops_composite)
        return output_rho

    # -----------------------------
    # Initial state (converted to density matrix for non-unitary evolution)
    # -----------------------------
    # The initial state is a tensor product of two coherent states.
    psi0 = tensor(coherent(N, alpha1), coherent(N, alpha2))
    rho = ket2dm(psi0) # Convert pure state to density matrix

    # -----------------------------
    # MZI sequence with unitary operations and loss channels
    # -----------------------------

    # 1. First Beam Splitter (Unitary evolution)
    rho = U_BS_func(theta1) * rho * U_BS_func(theta1).dag()

    # 2. In-coupling losses on the second arm (before U_Kerr)
    # Applies an amplitude damping channel to mode 2 if gamma_in_loss > 0.
    rho = apply_amplitude_damping_on_mode2(rho, gamma_in_loss, N)

    # 3. Nonlinear Kerr gate and phase shift (Unitary evolution)
    rho = U_Kerr * rho * U_Kerr.dag()

    # 4. Out-coupling losses on the second arm (after U_Kerr)
    # Applies another amplitude damping channel to mode 2 if gamma_out_loss > 0.
    rho = apply_amplitude_damping_on_mode2(rho, gamma_out_loss, N)

    # 5. Second Beam Splitter (Unitary evolution)
    rho_final = U_BS_func(theta2) * rho * U_BS_func(theta2).dag()

    # -----------------------------
    # Compute expectation values of number operators and their products
    # -----------------------------
    exp_n1 = expect(n1, rho_final)
    exp_n2 = expect(n2, rho_final)
    exp_nn1 = expect(n1 * n1, rho_final)
    exp_nn2 = expect(n2 * n2, rho_final)
    # Cross-correlation term (n1 * n2 is already a valid operator on the composite space)
    exp_nn12 = expect(n1 * n2, rho_final)

    return (exp_n1, exp_n2, exp_nn1, exp_nn2, exp_nn12)


"""
Nonlinear MZI simulation — parallelised over (U, theta_out, gamma, phi).

Result arrays have shape (nU, nTheta, nGamma, nPhi) and are saved to a
single .npz file together with the axis coordinate vectors.
"""



# ---------------------------------------------------------------------------
# Worker function  (must be top-level for pickle-ability)
# ---------------------------------------------------------------------------

def run_one(args):
    """Run a single (iU, iT, iG, iP) point and return the five observables."""
    iU, iT, iG, iP = args
    U       = Us[iU]
    theta2  = theta_outs[iT]
    gamma   = gammas[iG]
    phi     = phis[iP]

    n1, n2, nn1, nn2, nn12 = nonlinear_mixer_with_losses(
        N              = N,
        U1             = U,
        U2             = U if is_integrated else 0.,
        theta2         = theta2,
        alpha1         = alpha1,
        alpha2         = alpha2,
        phase_shift    = phi,
        coupler_type   = coupler_type,
        gamma_in_loss  = gamma,
        gamma_out_loss = gamma,
    )

    # Guard against division by zero
    g2_11 = (nn1  - n1) / n1 ** 2  if n1  > 0 else np.nan
    g2_22 = (nn2  - n2) / n2 ** 2  if n2  > 0 else np.nan
    g2_12 = nn12  / (n1 * n2)       if (n1 > 0 and n2 > 0) else np.nan

    return iU, iT, iG, iP, n1, n2, g2_11, g2_22, g2_12

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    start_time = time()

    shape = (len(Us), len(theta_outs), len(gammas), len(phis))

    # Pre-allocate result tensors
    n1s    = np.zeros(shape)
    n2s    = np.zeros(shape)
    g2_11s = np.full(shape, np.nan)
    g2_22s = np.full(shape, np.nan)
    g2_12s = np.full(shape, np.nan)

    # Build flat list of all index combinations
    index_combos = list(product(
        range(len(Us)),
        range(len(theta_outs)),
        range(len(gammas)),
        range(len(phis)),
    ))
    print(f"Total grid points : {len(index_combos):,}")
    print(f"Workers           : {N_WORKERS}")

    # ---- parallel execution ------------------------------------------------
    print("\nStarting the loop...")
    with Pool(processes=N_WORKERS) as pool:
        for result in pool.imap_unordered(run_one, index_combos, chunksize=256):
            iU, iT, iG, iP, n1, n2, g2_11, g2_22, g2_12 = result
            n1s   [iU, iT, iG, iP] = n1
            n2s   [iU, iT, iG, iP] = n2
            g2_11s[iU, iT, iG, iP] = g2_11
            g2_22s[iU, iT, iG, iP] = g2_22
            g2_12s[iU, iT, iG, iP] = g2_12

    print(f"Finished in {time()-start_time:.2f}s\n")

    
    # # ---- save ---------------------------------------------------------------


    # Save as pickle (e.g. for quick dict-style access) ---
    results_dict = {
        "axes": {"Us": Us, "theta_outs": theta_outs, "gammas": gammas, "phis": phis},
        "n1s": n1s, "n2s": n2s,
        "g2_11s": g2_11s, "g2_22s": g2_22s, "g2_12s": g2_12s,
    }

    pkl_filename = os.path.join(SAVE_DIR, "simulation_results.pkl")
    with open(pkl_filename, "wb") as f:
        pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved in {pkl_filename}")