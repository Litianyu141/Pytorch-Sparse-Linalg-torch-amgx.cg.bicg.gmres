# PyTorch Sparse Solver - Module Architecture

This document provides detailed documentation of the four main functional modules and their architectural design in the `pytorch_sparse_solver` package.

## Table of Contents

- [Overview](#overview)
- [Module A: JAX-style Iterative Solvers](#module-a-jax-style-iterative-solvers)
- [Module B: PyAMGX GPU-accelerated Solvers](#module-b-pyamgx-gpu-accelerated-solvers)
- [Module C: cuDSS Direct Solver](#module-c-cudss-direct-solver)
- [Module D: Unified Interface](#module-d-unified-interface)
- [Module Independence Design](#module-independence-design)

---

## Overview

`pytorch_sparse_solver` is a modular sparse linear system solver library with the following design goals:

1. **Modularity**: Each module can be installed and used independently
2. **Compatibility**: Supports any combination of modules
3. **Unified Interface**: Provides consistent API across all backends
4. **High Performance**: Full GPU acceleration support

### Package Structure

```
pytorch_sparse_solver/
├── __init__.py              # Main entry point
├── solver.py                # Unified solver class (Module D)
├── module_a/                # JAX-style iterative solvers
│   ├── __init__.py
│   ├── torch_sparse_linalg.py   # CG, BiCGStab, GMRES
│   └── torch_tree_util.py       # PyTree utilities
├── module_b/                # PyAMGX GPU solvers
│   ├── __init__.py
│   └── torch_amgx.py            # AMGX wrapper
├── module_c/                # cuDSS direct solver
│   ├── __init__.py
│   └── cudss_solver.py          # cuDSS wrapper
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── availability.py          # Module availability detection
│   └── matrix_utils.py          # Matrix utilities
└── tests/                   # Test scripts
    ├── test_module_a.py
    ├── test_module_b.py
    ├── test_module_c.py
    ├── test_unified.py
    └── benchmark.py
```

---

## Module A: JAX-style Iterative Solvers

### Description

Module A provides pure PyTorch implementations of iterative sparse linear solvers with an API design following JAX's `scipy.sparse.linalg` module:

- **CG (Conjugate Gradient)**: For symmetric positive definite matrices
- **BiCGStab (Bi-Conjugate Gradient Stabilized)**: For general non-symmetric matrices
- **GMRES (Generalized Minimal Residual)**: For general matrices with restart support

### Dependencies

- PyTorch >= 2.0 (core dependency)
- No other external dependencies

### Usage Example

```python
from pytorch_sparse_solver.module_a import cg, bicgstab, gmres

# Conjugate Gradient
x, info = cg(A, b, tol=1e-6, maxiter=1000)

# BiCGStab
x, info = bicgstab(A, b, tol=1e-6)

# GMRES with restart
x, info = gmres(A, b, tol=1e-6, restart=30)
```

### Features

- Pure PyTorch implementation, no additional dependencies
- Supports function-based linear operators (matrix-free methods)
- Supports PyTree structured inputs
- High precision computation (default float64)

---

## Module B: PyAMGX GPU-accelerated Solvers

### Description

Module B wraps NVIDIA's AMGX library via pyamgx Python bindings, providing:

- **AMGX CG**: GPU-accelerated conjugate gradient
- **AMGX BiCGStab**: GPU-accelerated BiCGStab
- **AMGX GMRES**: GPU-accelerated GMRES
- **AMG Preconditioner**: Algebraic multigrid preconditioning

### Dependencies

- PyTorch >= 2.0
- NVIDIA GPU + CUDA
- AMGX library (https://github.com/NVIDIA/AMGX)
- pyamgx (https://github.com/shwina/pyamgx)

### Installing AMGX

```bash
# 1. Compile AMGX
git clone --recursive https://github.com/NVIDIA/AMGX.git
cd AMGX && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=89
make -j$(nproc)
sudo make install

# 2. Install pyamgx
git clone https://github.com/shwina/pyamgx.git
cd pyamgx
export AMGX_DIR=/usr/local
pip install .
```

### Usage Example

```python
from pytorch_sparse_solver.module_b import amgx_cg, amgx_bicgstab, amgx_gmres

# AMGX CG
x = amgx_cg(A, b, tol=1e-8, maxiter=1000)

# AMGX BiCGStab
x = amgx_bicgstab(A, b, tol=1e-8)

# With automatic differentiation
b.requires_grad = True
x = amgx_cg(A, b, tol=1e-8)
loss = torch.sum(x**2)
loss.backward()  # Compute gradients
```

### Features

- GPU acceleration, suitable for large-scale problems
- Automatic differentiation support (adjoint method)
- Supports both sparse and dense matrix inputs

---

## Module C: cuDSS Direct Solver

### Description

Module C wraps NVIDIA's cuDSS library via `torch.sparse.spsolve`, providing:

- **Direct Solving**: LU factorization
- **GPU Acceleration**: Leverages cuDSS library
- **High Precision**: Direct method has no iterative error

### Dependencies

- PyTorch >= 2.7 (requires source compilation with cuDSS enabled)
- NVIDIA GPU + CUDA
- cuDSS library

### Compiling PyTorch with cuDSS

```bash
# 1. Install cuDSS
wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-repo-ubuntu2204-0.6.0_0.6.0-1_amd64.deb
sudo dpkg -i cudss-local-repo-ubuntu2204-0.6.0_0.6.0-1_amd64.deb
sudo cp /var/cudss-local-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update && sudo apt-get -y install cudss

# 2. Create conda environment
conda create -n torch_sparse2.7 python=3.11 -y
conda activate torch_sparse2.7
conda install cmake ninja -y

# 3. Compile PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch && git checkout v2.7.0
git submodule sync && git submodule update --init --recursive

export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH}"
export USE_CUDSS=1
export USE_CUDA=1
export MAX_JOBS=12
python -m pip install --no-build-isolation -v -e .
```

### Usage Example

```python
from pytorch_sparse_solver.module_c import cudss_spsolve, cudss_available

if cudss_available():
    # Requires CSR format
    A_csr = A.to_sparse_csr()
    x = cudss_spsolve(A_csr, b)
```

### Features

- Direct method, no iteration required
- Highest accuracy
- Suitable for small to medium-scale problems

---

## Module D: Unified Interface

### Description

Module D provides the unified `SparseSolver` class that integrates all backends:

- **Automatic Backend Selection**: Based on availability and matrix properties
- **Consistent API**: Unified solving interface
- **Flexible Switching**: Manual backend specification supported

### Usage Example

```python
from pytorch_sparse_solver import SparseSolver, solve

# Using SparseSolver class
solver = SparseSolver()
x, result = solver.solve(A, b, method='cg')
print(f"Backend: {result.backend}, Residual: {result.residual}")

# Using convenience function
x, result = solve(A, b, method='cg', backend='auto')

# Specify backend explicitly
x, result = solve(A, b, method='cg', backend='module_a')
x, result = solve(A, b, method='cg', backend='module_b')
x, result = solve(A, b, method='direct', backend='module_c')
```

### SolverResult Structure

```python
@dataclass
class SolverResult:
    x: torch.Tensor           # Solution vector
    converged: bool           # Convergence status
    iterations: Optional[int] # Iteration count
    residual: Optional[float] # Final residual
    backend: str              # Backend used
    method: str               # Method used
```

---

## Module Independence Design

### Design Principles

1. **Lazy Imports**: Each module is only imported when needed
2. **Availability Detection**: Runtime detection of module availability
3. **Graceful Degradation**: Unavailable modules don't affect other functionality

### Availability Detection

```python
from pytorch_sparse_solver import (
    check_module_a_available,
    check_module_b_available,
    check_module_c_available,
    get_available_backends,
    print_availability_report
)

# Check each module
print(f"Module A: {check_module_a_available()}")
print(f"Module B: {check_module_b_available()}")
print(f"Module C: {check_module_c_available()}")

# Get all available backends
backends = get_available_backends()
# {'module_a': True, 'module_b': False, 'module_c': False}

# Print detailed report
print_availability_report()
```

### Installation Combinations

| Combination | Installation | Available Features |
|-------------|--------------|-------------------|
| A only | `pip install pytorch_sparse_solver` | CG, BiCGStab, GMRES (CPU/GPU) |
| A+B | Install pyamgx | Above + AMGX acceleration |
| A+C | Compile PyTorch with cuDSS | Above + Direct method |
| A+B+C | Full installation | All features |
