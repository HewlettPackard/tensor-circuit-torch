import numpy as np
import torch
from time import time
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Iterable

from .mpdo_torch import MPDOtorch
from .mpdo_circuit import MPDOCircuit


def epoch_optimize(
        objective: Callable[[MPDOtorch, MPDOCircuit], float],
        optimizer: torch.optim.Optimizer,
        rho: MPDOtorch,
        max_epochs: int = 500,
        epoch_print: int = 1,
        optimizer_alt: torch.optim.Optimizer|None = None,
        iter_alternate: int|None = None,
        obj_params: List[torch.Tensor]|None=None,
        param_lims: List[float]|None=None,
        epoch_optim_clear: int = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler|None=None
    ) -> None:


    start = time()

    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

    for epoch in range(max_epochs):



        epoch_start = time()

        if iter_alternate is not None and epoch % (iter_alternate[0] + iter_alternate[1]) >= iter_alternate[0]:
            optimizer_run = optimizer_alt
            print(f'\nRunning alt epoch {epoch+1}...')
        else:
            optimizer_run = optimizer
            print(f'\nRunning opt epoch {epoch+1}...')


        if is_lbfgs:

            # track how many times closure is called
            closure_calls = 0

            def closure():

                nonlocal closure_calls 

                optimizer_run.zero_grad()

                rho_run_local = rho.clone()  # avoid in-place side-effects
                loss = objective(epoch, rho_run_local, obj_params, do_tracking=False)

                if torch.isnan(loss):
                    raise RuntimeError("Objective is NaN")

                loss.backward()
                closure_calls += 1
                return loss

            loss = optimizer_run.step(closure)

            # clamp the params
            if param_lims is not None:
                clamp(optimizer_run, param_lims)

            # evaluate the circuit
            with torch.no_grad():
                print(f'Run finished with {closure_calls} closure calls, scaling factor: {optimizer.param_groups[0]["lr"]}.')
                rho_eval = rho.clone()
                loss = objective(
                    epoch,
                    rho_eval,
                    obj_params,
                    do_tracking=True
                )

        else:
            # Adam / SGD / other first-order optimizers
            optimizer_run.zero_grad()

            with torch.no_grad():
                rho_run_local = rho.clone()  # avoid in-place side-effects

            loss = objective(epoch, rho_run_local, obj_params, do_tracking=True)

            if torch.isnan(loss):
                print("Objective is NaN...")
                break

            loss.backward()
            optimizer_run.step()

            # clamp the params
            if param_lims is not None:
                clamp(optimizer_run, param_lims)

        # clear the optizers
        if epoch_optim_clear and epoch % epoch_optim_clear == 0:
            optimizer.state.clear()
            if optimizer_alt is not None:
                if epoch % 3 == 0:
                    optimizer_alt.state.clear()

        # step scheduler
        if scheduler is not None:
            scheduler.step()



        # Print debug info
        if epoch % epoch_print == 0:
            print(f"Epoch {epoch+1} finsihed in {time() - epoch_start:.2f}s, objective: {loss.item()}")
            print(f"Total runtime: {time() - start:.2f}s")

    return None



def sweeping_optimize(
        objectives: List[Callable[[MPDOtorch, Any], float]],
        optimizers: List[torch.optim.Optimizer],
        final_optimizer: torch.optim.Optimizer,
        rho: MPDOtorch,
        max_sweeps: int = 50,
        max_epochs: int = 20,
        max_final_epochs: int=10,
        epoch_print: int = 1,
        obj_params: List[torch.Tensor]|None=None,
        param_lims: List[float]|None=None,
        epoch_optim_clear: int = None
    ) -> None:

    num_optimizers = len(optimizers)


    start = time()

    # FIRST STEP, SWEEPING EPOCHS

    print("Starting")

    for sweep in range(max_sweeps):

        epoch_start = time()

        print(f'\nRunning sweep {sweep+1}...')

        for i, (objective, optimizer) in enumerate(zip(objectives, optimizers)):
            # Adam / SGD / other first-order optimizers

            for epoch in range(max_epochs):
                optimizer.zero_grad()
                rho_run_local = rho.clone()  # avoid in-place side-effects

                loss = objective(sweep, rho_run_local, obj_params, do_tracking=True)

                if torch.isnan(loss):
                    Warning(f"Sweep {sweep}, optim. {i}, epoch {epoch}: objective is NaN...")
                    break

                loss.backward()
                optimizer.step()

                # clamp the params
                if param_lims is not None:
                    clamp(optimizer, param_lims)
            
            print(f"\tOptimizer {i}/{num_optimizers} finished.")


        # Print debug info
        if sweep % epoch_print == 0:
            print(f"Epoch {sweep+1} finsihed in {time() - epoch_start:.2f}s, objective: {loss.item()}")
            print(f"Total runtime: {time() - start:.2f}s")

    
    # SECOND STEP: FINAL OPTIMIZATION
    if final_optimizer is not None:
        for epoch_final in range(max_epochs):

            # track how many times closure is called
            closure_calls = 0

            def closure():

                nonlocal closure_calls 

                final_optimizer.zero_grad()

                with torch.no_grad():
                    rho_run_local = rho.clone()  # avoid in-place side-effects

                loss = objective(sweep, rho_run_local, obj_params, do_tracking=False)

                if torch.isnan(loss):
                    raise RuntimeError("Objective is NaN")

                loss.backward()
                closure_calls += 1
                return loss

            loss = final_optimizer.step(closure)

            # clamp the params
            if param_lims is not None:
                clamp(final_optimizer, param_lims)

            # evaluate the circuit
            with torch.no_grad():
                print(f'Run finished with {closure_calls} closure calls, scaling factor: {optimizer.param_groups[0]["lr"]}.')
                rho_eval = rho.clone()
                loss = objective(
                    sweep,
                    rho_eval,
                    obj_params,
                    do_tracking=True
                )


    return None



def clamp(optimizer: torch.optim, param_lims: List[float]):

    params = [p for g in optimizer.param_groups for p in g["params"]]
    for p in params:
        p.data.clamp_(param_lims[0], param_lims[1])



