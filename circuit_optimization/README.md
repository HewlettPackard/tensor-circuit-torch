# circuit-optimization
> Optimization tools for quantum circuits, built in Python, using PyTorch.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![Build](https://github.com/<user>/<repo>/actions/workflows/ci.yml/badge.svg)]()
[![PyPI version](https://badge.fury.io/py/<package>.svg)](https://pypi.org/project/<package>/)

---

## Overview

**circuit_optimization** is an open-source Python package for optimization-driven quantum simulations using tensor networks.  
It provides efficient algorithms for:
- Quantum circuit simulation (forward pass)  
- circuit optimization (using autograd from PyTorch)
- Noise-aware simulation (using the Monte-Carlo trajectory framework)  
- Classical 2D field simulations for the gate design (Gross-Pitaevskii equation)

This package is modular and can be run directly as Python modules for experimentation or integrated into larger quantum workflows.

---

## Features
- Modular architecture for reproducible experiments  
- Gradient-based optimizer
- Built-in logging and data tracking  
- Easy configuration via YAML or JSON files  

---

## Installation

Clone the repository and install dependencies:

```bash
git clone git@github.com:HewlettPackard/tensor-circuit-torch.git
cd tensor-circuit-torch/circuit_optimization
pip install -r requirements.txt
```

## Run
The scripts are run as Python modules, in order to maintain a clear project structure

### Single Python scripts

The state preparation of an odd cat state:
```bash
python -m experiments.stategen.main
```

Optimizing circuit phase-sensing readout
```bash
python -m experiments.metrology.metrology_coherent.main
```

Run the classical gate simulator
```bash
python -m gpe_sim_couper_gate.main
```

### Bash files iterating over circuit parameters
The bash scripts overwrite the parameters defined in script yaml input files by CL input. Each bash script contains nested loops over the set of parameters to scan in different iterations. 

NOTE: Please make sure that you have GPUs available if you run the bash scripts (see below), otherwise it will take a long time.

Run the bash script as follows,
```bash
./loop_stategen_cat
```

## Source Code
The source code is stored in the folder "src/" and consists of "GPE_solver" for the classical field simulations and "MPS_state_optimization" for the circuit quantum simulation and optimization. We explain the latter in more detail

- "mps_torch.py": Contains the MPS tensor representation of the quantum state, with state-defined functions (e.g., entropy, canonical form etc.)
- "tensor_circuit.py": Contains the source code for the circuit gate components and the circuit architecture
- "mps_circuit_optimizer.py": contains the source code for the circuit optimizer, defined in PyTorch.
- "svd_trunc.py": The code for defining the differentiable SVD operation for complex matrices.
- "analyzer.py": The functions needed for the analysis during optimization.

## GPU support
PyTorch provides GPU support and the result from the manuscript were run using GPU (8 were available for running the bash scripts). For this you need to install the cuda libraries.

1. **Uninstall CPU-only PyTorch**
```bash
pip uninstall torch torchvision torchaudio -y
```
2. **Install PyTorch with CUDA support**
Visit the official PyTorch install selector to choose the correct CUDA version for your system: https://pytorch.org/get-started/locally/

Example (CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


## Citation
Please, if you use this code, cite the accompanying manuscript

```
Van Regemortel, Mathias, and Thomas Van Vaerenbergh. "Optimizing Quantum Photonic Integrated Circuits using Differentiable Tensor Networks." 
arXiv preprint arXiv:2509.11861 (2025).
```





