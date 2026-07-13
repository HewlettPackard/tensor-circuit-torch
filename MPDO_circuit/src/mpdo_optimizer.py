"""
mpdo_optimizer.py
-----------------
Training loops for MPDO circuit optimisation.

Functions
---------
epoch_optimize    : standard epoch loop (Adam, LBFGS, or any torch optimizer)
sweeping_optimize : layer-by-layer sweeping optimisation followed by optional
                    global fine-tuning (useful for deep circuits)
clamp             : in-place parameter clamping helper
"""

from time import time
from typing import Any, Callable, Iterable, List

import numpy as np
import torch

from .mpdo_torch import MPDOtorch
from .mpdo_circuit import MPDOCircuit


# ---------------------------------------------------------------------------
# Epoch optimiser
# ---------------------------------------------------------------------------

def epoch_optimize(
    objective:          Callable[[int, MPDOtorch, Any, bool], torch.Tensor],
    optimizer:          torch.optim.Optimizer,
    rho:                MPDOtorch,
    max_epochs:         int                                 = 500,
    epoch_print:        int                                 = 1,
    optimizer_alt:      "torch.optim.Optimizer | None"      = None,
    iter_alternate:     "list | None"                       = None,
    obj_params:         "List[torch.Tensor] | None"         = None,
    param_lims:         "List[float] | None"                = None,
    epoch_optim_clear:  "int | None"                        = None,
    scheduler:          "torch.optim.lr_scheduler._LRScheduler | None" = None,
) -> None:
    """
    Run a fixed-epoch optimisation loop.

    Supports first-order optimisers (Adam, SGD, etc.) and LBFGS.  An
    optional alternating schedule switches between a primary and an
    alternate optimiser every ``iter_alternate[0]`` / ``iter_alternate[1]``
    epochs respectively (useful for two-stage QFI/CFI optimisation).

    Parameters
    ----------
    objective         : callable(epoch, rho, obj_params, do_tracking) → loss
                        The objective should run the circuit on a fresh clone of
                        `rho` and return a differentiable scalar loss.
    optimizer         : primary torch optimizer
    rho               : input MPDO state (never mutated; cloned each epoch)
    max_epochs        : total number of gradient steps
    epoch_print       : print diagnostics every this many epochs
    optimizer_alt     : secondary optimizer used during QFI alternation sub-epochs
    iter_alternate    : [n_primary, n_alternate] epoch counts per period;
                        None → always use primary optimizer
    obj_params        : extra parameters forwarded to `objective` unchanged
    param_lims        : [lo, hi] — clamp all optimised parameters into this range
                        after each step (e.g. [0, π] for phases)
    epoch_optim_clear : clear optimizer state every this many epochs (None → never)
    scheduler         : learning-rate scheduler; stepped once per epoch

    Returns
    -------
    None
    """
    start    = time()
    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

    for epoch in range(max_epochs):
        epoch_start = time()

        # Select primary or alternate optimizer for this epoch
        if (iter_alternate is not None
                and epoch % (iter_alternate[0] + iter_alternate[1]) >= iter_alternate[0]):
            optimizer_run = optimizer_alt
            print(f'\nRunning alt epoch {epoch + 1}...')
        else:
            optimizer_run = optimizer
            print(f'\nRunning opt epoch {epoch + 1}...')

        if is_lbfgs:
            # --- LBFGS: requires a closure ---
            closure_calls = 0

            def closure():
                """LBFGS closure: zero gradients, forward pass, backward pass."""
                nonlocal closure_calls
                optimizer_run.zero_grad()
                rho_local = rho.clone()
                loss = objective(epoch, rho_local, obj_params, do_tracking=False)
                if torch.isnan(loss):
                    raise RuntimeError("Objective is NaN")
                loss.backward()
                closure_calls += 1
                return loss

            loss = optimizer_run.step(closure)

            if param_lims is not None:
                clamp(optimizer_run, param_lims)

            # Separate evaluation pass with tracking enabled
            with torch.no_grad():
                print(f"LBFGS: {closure_calls} closure calls, "
                      f"lr={optimizer.param_groups[0]['lr']:.4e}")
                rho_eval = rho.clone()
                loss = objective(epoch, rho_eval, obj_params, do_tracking=True)

        else:
            # --- First-order optimisers (Adam, SGD, …) ---
            optimizer_run.zero_grad()

            with torch.no_grad():
                rho_local = rho.clone()

            loss = objective(epoch, rho_local, obj_params)

            if torch.isnan(loss):
                print("Objective is NaN — stopping.")
                break

            loss.backward()
            optimizer_run.step()

            if param_lims is not None:
                clamp(optimizer_run, param_lims)

        # Periodic optimizer state reset (can help escape flat regions)
        if epoch_optim_clear and epoch % epoch_optim_clear == 0:
            optimizer.state.clear()
            if optimizer_alt is not None and epoch % 3 == 0:
                optimizer_alt.state.clear()

        if scheduler is not None:
            scheduler.step()

        if epoch % epoch_print == 0:
            print(f"Epoch {epoch + 1} finished in {time() - epoch_start:.2f}s, "
                  f"objective: {loss.item():.6e}")
            print(f"Total runtime: {time() - start:.2f}s")


# ---------------------------------------------------------------------------
# Sweeping optimizer
# ---------------------------------------------------------------------------

def sweeping_optimize(
    objectives:         List[Callable[[MPDOtorch, Any], float]],
    optimizers:         List[torch.optim.Optimizer],
    final_optimizer:    "torch.optim.Optimizer | None",
    rho:                MPDOtorch,
    max_sweeps:         int                         = 50,
    max_epochs:         int                         = 20,
    max_final_epochs:   int                         = 10,
    epoch_print:        int                         = 1,
    obj_params:         "List[torch.Tensor] | None" = None,
    param_lims:         "List[float] | None"        = None,
    epoch_optim_clear:  "int | None"                = None,
) -> None:
    """
    Layer-by-layer sweeping optimisation.

    Iterates through each (objective, optimizer) pair for ``max_epochs``
    gradient steps per sweep, repeating for ``max_sweeps`` full passes.
    Optionally, a final global LBFGS pass can be run after the sweeps.

    This approach is useful for deep circuits where optimising all layers
    simultaneously is slow or diverges: each layer is trained in isolation
    before moving to the next.

    Parameters
    ----------
    objectives        : list of per-layer objective callables (same signature
                        as in ``epoch_optimize``)
    optimizers        : list of per-layer torch optimizers (same length as
                        `objectives`)
    final_optimizer   : optional LBFGS (or other) optimizer for a global
                        fine-tuning pass; None → skip
    rho               : input MPDO state
    max_sweeps        : number of full left-to-right sweeps
    max_epochs        : number of gradient steps per (layer, sweep)
    max_final_epochs  : number of steps in the optional final pass
    epoch_print       : print diagnostics every this many sweeps
    obj_params        : extra parameters forwarded to all objectives
    param_lims        : parameter clamping range [lo, hi]
    epoch_optim_clear : clear optimizer state every this many sweeps (None → never)

    Returns
    -------
    None
    """
    num_optimizers = len(optimizers)
    start          = time()

    print("Starting sweeping optimisation...")

    for sweep in range(max_sweeps):
        epoch_start = time()
        print(f'\nSweep {sweep + 1}/{max_sweeps}')

        for i, (objective, optimizer) in enumerate(zip(objectives, optimizers)):
            for epoch in range(max_epochs):
                optimizer.zero_grad()
                rho_local = rho.clone()

                loss = objective(sweep, rho_local, obj_params, do_tracking=True)

                if torch.isnan(loss):
                    print(f"  [sweep {sweep}, optim {i}, epoch {epoch}] NaN — skipping.")
                    break

                loss.backward()
                optimizer.step()

                if param_lims is not None:
                    clamp(optimizer, param_lims)

            print(f"  Optimizer {i + 1}/{num_optimizers} done.")

        if sweep % epoch_print == 0:
            print(f"Sweep {sweep + 1} finished in {time() - epoch_start:.2f}s, "
                  f"last loss: {loss.item():.6e}")
            print(f"Total runtime: {time() - start:.2f}s")

    # Optional global fine-tuning with LBFGS (or any closure-capable optimizer)
    if final_optimizer is not None:
        print("\nStarting final optimisation pass...")
        for epoch_final in range(max_final_epochs):
            closure_calls = 0

            def closure():
                """LBFGS closure for the final optimisation pass."""
                nonlocal closure_calls
                final_optimizer.zero_grad()
                with torch.no_grad():
                    rho_local = rho.clone()
                loss = objectives[-1](sweep, rho_local, obj_params, do_tracking=False)
                if torch.isnan(loss):
                    raise RuntimeError("Objective is NaN in final pass.")
                loss.backward()
                closure_calls += 1
                return loss

            loss = final_optimizer.step(closure)

            if param_lims is not None:
                clamp(final_optimizer, param_lims)

            print(f"Final epoch {epoch_final + 1}: {closure_calls} closure calls, "
                  f"lr={final_optimizer.param_groups[0]['lr']:.4e}")
            with torch.no_grad():
                rho_eval = rho.clone()
                loss = objectives[-1](sweep, rho_eval, obj_params, do_tracking=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def clamp(optimizer: torch.optim.Optimizer, param_lims: List[float]) -> None:
    """
    Clamp all parameters in `optimizer` into the range [param_lims[0], param_lims[1]].

    Called after each optimizer step when phase or coupling parameters must
    remain within a physically meaningful interval.
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            p.data.clamp_(param_lims[0], param_lims[1])
