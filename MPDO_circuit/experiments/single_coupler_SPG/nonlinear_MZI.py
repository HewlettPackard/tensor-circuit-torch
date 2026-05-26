import os
import numpy as np
import time
import torch
import torch.nn.functional as F
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from src.mpdo_circuit import CouplerCircuit, PhaseCircuit
from src.mpdo_optimizer import epoch_optimize
from src.mpdo_torch import StateCreator, MPDOtorch
from src.create_circuit import create_nonlinear_photonic_circuit, create_phase_circuit
from src.tracker import Tracker
from src.visualize import visualize_circuit

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

FILE_DIR  = os.path.dirname(os.path.abspath(__file__))
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
SAVE_DIR  = os.path.join(FILE_DIR, "data", TIMESTAMP)


# ---------------------------------------------------------------------------
# Arm: one MZI + one phase circuit, bundled together
# ---------------------------------------------------------------------------

@dataclass
class Arm:
    """Bundles a CouplerCircuit and a PhaseCircuit for one arm (in or out)."""
    mzi:   CouplerCircuit
    phase: PhaseCircuit

    def get_variables(self):
        return self.mzi.get_variables() + self.phase.get_variables()

    def run(self, rho: MPDOtorch, options: Dict[str, Any]) -> None:
        self.phase.run(rho, options=options)
        self.mzi.run(rho, options=options)

    def get_J_matrix(self, **kwargs):
        return self.mzi.get_J_matrix(**kwargs)

    def get_phi_matrix(self, **kwargs):
        return self.phase.get_phi_matrix(**kwargs)


# ---------------------------------------------------------------------------
# Iterator: owns the two arms, tracker, options, and the per-iteration logic
# ---------------------------------------------------------------------------

class CircuitIterator:
    """
    Encapsulates all state needed for the optimization loop.
    Initialized once; called each iteration via __call__.
    """

    def __init__(
        self,
        arm_in:           Arm,
        arm_out:          Arm,
        options_mpdo:     Dict[str, Any],
        tracker:          Tracker,
        save_dir:         str,
        weight_intensity: float,
        target_intensity: float = 0.01,
    ):
        self.arm_in           = arm_in
        self.arm_out          = arm_out
        self.options_mpdo     = options_mpdo
        self.tracker          = tracker
        self.save_dir         = save_dir
        self.weight_intensity = weight_intensity
        self.target_intensity = target_intensity

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_circuit(self, rho: MPDOtorch) -> None:
        """Apply in-arm then out-arm in place on a cloned state."""
        self.arm_in.run(rho,  self.options_mpdo)
        self.arm_out.run(rho, self.options_mpdo)

    def _update_arms(self) -> None:
        """Re-read current parameter values from the arm objects (no-op if
        autograd variables are used directly; kept as an extension point)."""
        pass

    # ------------------------------------------------------------------
    # Main callable — signature matches what epoch_optimize expects
    # ------------------------------------------------------------------

    def __call__(
        self,
        it:          int,
        rho:         MPDOtorch,
        obj_params = None,
        do_tracking: bool = True,
    ) -> torch.Tensor:

        # --- forward pass ---
        rho_run = rho.clone()
        self._run_circuit(rho_run)

        # --- compute observables ---
        idx_signal = rho_run.num_channels // 2
        ns  = rho_run.number_outcomes()
        g2s = rho_run.density_correlations(n_eps=1e-5)

        J_in_mat  = self.arm_in.get_J_matrix(float_vals=False)
        phi_in    = self.arm_in.get_phi_matrix(float_vals=False)
        phi_out   = self.arm_out.get_phi_matrix(float_vals=False)
        rel_phi   = phi_in[0][0] - phi_in[0][-1]

        # --- figure of merit ---
        weight = self.weight_intensity * np.minimum(it / 100, 1.)
        FOM = (
            (1. - weight) * g2s[idx_signal]
            + weight * F.relu(self.target_intensity - ns[idx_signal]) / self.target_intensity
        )

        # --- tracking ---
        if do_tracking:
            for il, (n, g2) in enumerate(zip(ns, g2s)):
                self.tracker.add(f'n_{il}',  n.item())
                self.tracker.add(f'g2_{il}', g2.item())
            self.tracker.add('couplings_in',  self.arm_in.get_J_matrix())
            self.tracker.add('couplings_out', self.arm_out.get_J_matrix())
            self.tracker.add('rel_phase', rel_phi.item())
            self.tracker.add('FOM', FOM.item())

            num_ch = rho_run.num_channels
            self.tracker.visualize(
                ['FOM', f'n_{idx_signal}', f'g2_{idx_signal}'],
                filename=os.path.join(self.save_dir, 'FOM.png'))
            self.tracker.visualize(
                [f'n_{il}' for il in range(num_ch)],
                filename=os.path.join(self.save_dir, 'ns.png'))
            self.tracker.visualize(
                [f'g2_{il}' for il in range(num_ch)],
                filename=os.path.join(self.save_dir, 'g2s.png'))
            visualize_circuit(
                self.arm_in.get_J_matrix(),
                filename=os.path.join(self.save_dir, 'circuit_in'))
            visualize_circuit(
                self.arm_out.get_J_matrix(),
                filename=os.path.join(self.save_dir, 'circuit_out'))
            self.tracker.save(os.path.join(self.save_dir, 'data.pkl'))

        print(
            f"it {it:4d} | g2: {g2s[idx_signal]:.4e} | "
            f"n: {ns[idx_signal]:.4e} | tot n: {ns.sum():.4e} | "
            f"rel φ: {rel_phi:.2f}"
        )

        return FOM


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    device = "cuda:4"

    # --- problem dimensions ---
    num_channels = 2
    num_layers   = 1
    Nmax         = 10

    # --- circuit parameters ---
    U_in,  U_out  = 0.01, 0.
    J_in,  J_out  = np.pi / 4., np.pi / 4.
    phases_in     = [[0., 0.01]]
    phases_out    = [[0., np.pi / 4]]
    gamma_dt      = 0.0

    # --- initial state ---
    n0, nL            = 1., 1.
    target_intensity  = 0.01

    # --- optimizer options ---
    options_adam = {
        'lr':               1e-3,
        'lr_min':           1e-4,
        'max_epochs':       100,
        'param_lims':       [0, np.pi],
        'weight_intensity': 1.,
    }
    options_mpdo = {
        'max_BD':    100,
        'max_PD':    100,
        'cutoff_BD': 1e-9,
        'cutoff_PD': 1e-5,
    }

    # --- output directory ---
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- initial state ---
    alphas = [0.] * num_channels
    alphas[0], alphas[-1] = np.sqrt(n0), np.sqrt(nL)
    state_creator = StateCreator(Nmax, num_batch=1)
    list_ten = state_creator.product_state_coherent(alphas, device=device)
    rho = MPDOtorch(list_ten)

    # --- build arms ---
    base_circuit_cfg = dict(
        num_layers   = num_layers,
        gamma        = gamma_dt,
        order_kraus  = 1,
        dt           = 1.,
        Nmax         = Nmax,
        num_channels = num_channels,
        requires_grad= True,
    )
    requires_grad_phase = [[True] * num_channels]

    arm_in = Arm(
        mzi=create_nonlinear_photonic_circuit(
            **base_circuit_cfg, U=U_in, J=J_in, device=device),
        phase=create_phase_circuit(
            Nmax=Nmax, phase=phases_in,
            device=device, requires_grad=requires_grad_phase),
    )

    arm_out = Arm(
        mzi=create_nonlinear_photonic_circuit(
            **base_circuit_cfg, U=U_out, J=J_out, device=device),
        phase=create_phase_circuit(
            Nmax=Nmax, phase=phases_out,
            device=device, requires_grad=requires_grad_phase),
    )

    # --- iterator (owns all shared state) ---
    tracker  = Tracker()
    iterator = CircuitIterator(
        arm_in           = arm_in,
        arm_out          = arm_out,
        options_mpdo     = options_mpdo,
        tracker          = tracker,
        save_dir         = SAVE_DIR,
        weight_intensity = options_adam.pop('weight_intensity'),
        target_intensity = target_intensity,
    )

    # --- optimizer ---
    max_epochs = options_adam.pop('max_epochs')
    param_lims = options_adam.pop('param_lims')
    lr_min     = options_adam.pop('lr_min')

    optimizer = torch.optim.Adam(
        arm_in.get_variables() + arm_out.get_variables(),
        **options_adam,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr_min)
        if lr_min is not None else None
    )

    # --- run ---
    epoch_optimize(
        iterator,
        optimizer   = optimizer,
        rho         = rho,
        max_epochs  = max_epochs,
        param_lims  = param_lims,
        scheduler   = scheduler,
        epoch_optim_clear = None,
    )