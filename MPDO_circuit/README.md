# tensor-circuit-torch

**MPDO simulation of quantum photon statistics in polaritonic integrated circuits.**

This repository contains the source code accompanying the paper:

> **Modeling the Quantum Photon Statistics in Hybrid Light-Matter Integrated Circuits**  
> M. Van Regemortel *et al.* (2025) — [arXiv:2605.23286](https://arxiv.org/abs/2605.23286)

The framework maps pulsed nonlinear waveguide dynamics onto a bosonic quantum circuit representation and simulates the output photon statistics using Matrix Product Density Operators (MPDOs) in PyTorch, with full support for dissipation, gradient-based optimisation and GPU acceleration.

---

## Overview

Exciton-polaritons in (Al)GaAs waveguides combine strong optical nonlinearities with a guided-mode geometry, making them a leading platform for integrated quantum photonics.  This codebase provides:

- A **differentiable MPDO simulator** for bosonic circuits with Fock-space truncation, Kerr nonlinearities, and amplitude-damping dissipation.
- **Polariton dispersion utilities** (LP energy, group velocity, band curvature, exciton fraction) for (Al)GaAs waveguide parameters.
- **Circuit factories** for nonlinear photonic circuits, MZI topologies, TEBD steps, phase circuits and Haar-random benchmarks.
- **Experiment scripts** reproducing the two main results of the paper: scanning g²(0) as a function of wavevector *k* (multi-channel circuit) and as a function of group velocity *v_g* (no exciton-fraction rescaling).
- **Optimisation loops** (epoch-based and layer-by-layer sweeping) for gradient-based circuit design.

---

## Repository structure

```
tensor-circuit-torch/
├── MPDO_circuit/
│   ├── src/
│   │   ├── mpdo_torch.py          # MPDOtorch state class + StateCreator
│   │   ├── mpdo_circuit.py        # CircuitTopology, MPDOCircuit, CouplerCircuit, PhaseCircuit
│   │   ├── circuit_gates.py       # Gate classes (NonlinearCouplingGate, PhaseGate, …)
│   │   ├── create_circuit.py      # Circuit factory functions
│   │   ├── mpdo_optimizer.py      # epoch_optimize, sweeping_optimize
│   │   ├── svd_trunc.py           # Differentiable truncated SVD
│   │   ├── utils.py               # Tensor utilities + operator classes
│   │   ├── tracker.py             # Optimisation logging
│   │   ├── visualize.py           # Circuit and entropy visualisation
│   │   └── CLI_utils.py           # Command-line argument helper
│   └── experiments/
│       ├── LP_utils.py            # Polariton dispersion (ω_LP, v_g, curvature, exciton fraction)
│       ├── g2_experiment/
│       │   ├── g2_simulator.py            # Scan g²(0) vs wavevector k  [Fig. paper]
│       │   └── g2_simulator_scan_vg.py    # Scan g²(0) vs group velocity v_g  [Fig. paper]
│       ├── phase_sensing/
│       │   ├── sensing_utils.py           # Fisher information metrics (QFI, GFI, HFI, NFI)
│       │   ├── sensing_optimizer.py       # Two-stage QFI → CFI optimisation
│       │   ├── sensing_optimizer_alternate.py  # Alternating QFI/CFI optimisation
│       │   └── ghost_imaging_optimizer.py # Ghost-imaging sensing optimisation
│       ├── photon_pulse/
│       │   └── pulse_MPDO.py              # Time-domain TEBD pulse propagation
│       ├── random_qubit/
│       │   └── main.py                    # Haar-random circuit benchmark
│       └── single_coupler_SPG/
│           └── nonlinear_MZI.py           # Single MZI g² optimisation
```

---

## Installation

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, NumPy, SciPy, Matplotlib.

```bash
git clone https://github.com/HewlettPackard/tensor-circuit-torch.git
cd tensor-circuit-torch
pip install -e .
```

GPU acceleration is strongly recommended for circuits with more than ~6 channels or Nmax > 15.  The scripts default to `cuda:0`; pass `--device cpu` to run on CPU.

---

## Paper experiments

The two main simulation results from the paper are reproduced by the scripts below.  Both scan the second-order coherence g²(0) and photon number across a physical parameter range for a fixed multimode nonlinear photonic circuit.

### 1 — g²(0) vs wavevector *k*  (polariton dispersion scan)

Scans the in-plane wavevector *k* from the photonic to the excitonic regime.  At each *k*, the group velocity, exciton fraction and effective nonlinearity U are derived from the LP dispersion and fed into the circuit.  This reproduces the main result showing how slow-light engineering amplifies the effective Kerr nonlinearity and pushes g²(0) below 1.

```bash
python -m MPDO_circuit.experiments.g2_experiment.g2_simulator \
    --hg 50 \
    --gamma 0.0 \
    --phi 0.628 \
    --writedir results/g2_vs_k \
    --device cuda:0
```

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| Exciton interaction | `--hg` | `50` | ħg in µeV·µm² |
| Loss rate | `--gamma` | `0.0` | γ in ps⁻¹ |
| Relative input phase | `--phi` | `0.2π` | φ_rel in rad |
| Output directory | `--writedir` | `data/<timestamp>` | Where to save results |
| Device | `--device` | `cuda:0` | Torch device string |

**Output** (saved to `--writedir`):
- `results.png` — 4-panel figure: photon densities, g²(0), effective nonlinearity U_k and exciton fraction, gate coupling J·Δt — all as a function of *k*.
- `results.pkl` — full results dictionary with keys `ks`, `ns`, `g2s`, `Uk`, `coupling_k`, `ex_frac`, `wk`, `vgk`, `parameters`.

---

### 2 — g²(0) vs group velocity *v_g*  (without polariton rescaling)

Scans the group velocity directly, without the k-dependent exciton-fraction rescaling.  This isolates the effect of slow-light compression of the pulse spatial profile on the effective nonlinearity, and serves as a clean benchmark for the circuit model independent of the polariton band structure.

```bash
python -m MPDO_circuit.experiments.g2_experiment.g2_simulator_scan_vg \
    --hg 20 \
    --gamma 0.02 \
    --phi 1.047 \
    --writedir results/g2_vs_vg \
    --device cuda:0
```

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| Exciton interaction | `--hg` | `20` | ħg in µeV·µm² |
| Loss rate | `--gamma` | `0.02` | γ in ps⁻¹ |
| Relative input phase | `--phi` | `π/3` | φ_rel in rad |
| Output directory | `--writedir` | `data/<timestamp>` | Where to save results |
| Device | `--device` | `cuda:0` | Torch device string |

**Output** (saved to `--writedir`):
- `results.png` — 2-panel figure: photon densities and g²(0) as a function of *v_g*.
- `results.pkl` — full results dictionary with keys `vgs`, `ns`, `g2s`, `Uk`, `coupling_k`, `parameters`.

---

## Physical model

The simulation maps a pulse propagating through a nonlinear waveguide array onto a layered bosonic circuit.  Each layer consists of nearest-neighbour `NonlinearCouplingGate` elements (beam-splitter + Kerr) with amplitude-damping Kraus operators for photon loss.

The Hamiltonian of each two-mode gate is:

$$H = J(a_L^\dagger a_R + \text{h.c.}) + \frac{U}{2}\left[n_L(n_L-1) + n_R(n_R-1)\right] + \Delta(n_L + n_R)$$

The effective nonlinear rate for a Gaussian pulse of temporal width σ_t propagating at group velocity v_g is:

$$U_\text{eff} = |u_k|^4 \cdot \frac{\hbar g_\text{ex}}{w_\perp} \cdot \frac{1}{\sqrt{4\pi}\, \sigma_t v_g}$$

where |u_k|² is the excitonic Hopfield coefficient and w_⊥ is the transverse mode size.

The state at each layer is represented as an MPDO:

$$\rho = \sum_\alpha |\Psi_\alpha\rangle\langle\Psi_\alpha|$$

stored as a mixed-canonical MPS with explicit bond singular values.  Dissipation is incorporated via a Kraus-operator sweep that expands the purity (batch) dimension, followed by a truncated SVD to keep it bounded.

---

## Citing

If this code contributes to your research, please cite:

```bibtex
@article{vanregemortel2025mpdo,
  title   = {Modeling the Quantum Photon Statistics in Hybrid Light-Matter Integrated Circuits},
  author  = {Van Regemortel, Mathias and others},
  journal = {arXiv preprint arXiv:2605.23286},
  year    = {2025},
  url     = {https://arxiv.org/abs/2605.23286}
}
```

---

## License

This project is licensed under the Apache 2.0 License — see [`LICENSE`](LICENSE) for details.