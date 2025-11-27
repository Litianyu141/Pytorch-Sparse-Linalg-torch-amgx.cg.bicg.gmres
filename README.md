# PyTorch Sparse Solver

[![CI](https://github.com/Litianyu141/Pytorch-sparse-solve/actions/workflows/ci.yml/badge.svg)](https://github.com/Litianyu141/Pytorch-sparse-solve/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/pytorch-2.7+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular, high-performance sparse linear system solver library for PyTorch with GPU acceleration and automatic differentiation support.

## Latest Updates

> **Update 2025.11.27**: Major refactoring into modular architecture (Module A/B/C/D) with unified interface and comprehensive benchmark suite.
>
> **Update 2025.06.25**: Added PyAMGX support with improved differentiability.

## Overview

`pytorch_sparse_solver` is a modular sparse linear system solver library with the following design goals:

1. **Modularity**: Each module can be installed and used independently
2. **Compatibility**: Supports any combination of modules
3. **Unified Interface**: Provides consistent API across all backends
4. **High Performance**: Full GPU acceleration support

### Available Modules

| Module | Description | Dependencies |
|--------|-------------|--------------|
| **Module A** | JAX-style iterative solvers (CG, BiCGStab, GMRES) | PyTorch only |
| **Module B** | Differentionable NVIDIA AMGX GPU-accelerated solvers | PyTorch + AMGX + pyamgx |
| **Module C** | cuDSS direct solver (LU factorization) | PyTorch 2.7+ with cuDSS |
| **Module D** | Unified interface for all backends | Any combination above |

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
└── tests/                   # Test suite
    ├── test_module_a.py
    ├── test_module_b.py
    ├── test_module_c.py
    ├── test_unified.py
    └── benchmark.py
```

---

## Quick Start

### Using the Unified Interface (Recommended)

```python
from pytorch_sparse_solver import SparseSolver, solve

# Create solver instance
solver = SparseSolver()

# Solve linear system Ax = b
x, result = solver.solve(A, b, method='cg')
print(f"Backend: {result.backend}, Residual: {result.residual}")

# Or use the convenience function
x, result = solve(A, b, method='cg', backend='auto')

# Specify backend explicitly
x, result = solve(A, b, method='cg', backend='module_a')
x, result = solve(A, b, method='direct', backend='module_c')
```

### Using Individual Modules

```python
# Module A: Pure PyTorch iterative solvers
from pytorch_sparse_solver.module_a import cg, bicgstab, gmres

x, info = cg(A, b, tol=1e-6, maxiter=1000)
x, info = bicgstab(A, b, tol=1e-6)
x, info = gmres(A, b, tol=1e-6, restart=30)

# Module B: AMGX GPU-accelerated solvers (requires pyamgx)
from pytorch_sparse_solver.module_b import amgx_cg, amgx_bicgstab, amgx_gmres

x = amgx_cg(A, b, tol=1e-8, maxiter=1000)

# Module C: cuDSS direct solver (requires PyTorch with cuDSS)
from pytorch_sparse_solver.module_c import cudss_spsolve

A_csr = A.to_sparse_csr()
x = cudss_spsolve(A_csr, b)
```

### Check Module Availability

```python
from pytorch_sparse_solver import (
    check_module_a_available,
    check_module_b_available,
    check_module_c_available,
    print_availability_report
)

# Print detailed availability report
print_availability_report()
```

---

## Installation

### Quick Install from GitHub

```bash
# Install directly from GitHub (recommended)
pip install git+https://github.com/Litianyu141/Pytorch-sparse-solve.git

# Or with specific branch/tag
pip install git+https://github.com/Litianyu141/Pytorch-sparse-solve.git@main
```

### Local Development Install

```bash
# Clone the repository
git clone https://github.com/Litianyu141/Pytorch-sparse-solve.git
cd Pytorch-sparse-solve

# Install in editable mode
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

### Basic Installation (Module A only)

Module A is a pure PyTorch implementation with no additional dependencies.

```bash
# Create conda environment
conda create -n pytorch_sparse python=3.11 -y
conda activate pytorch_sparse

# Install PyTorch
pip install torch>=2.0.0

# Install the package from GitHub
pip install git+https://github.com/Litianyu141/Pytorch-sparse-solve.git
```

### Module B Installation (PyAMGX)

Module B requires the NVIDIA AMGX library and pyamgx Python bindings.

#### Step 1: Install Build Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake git
```

#### Step 2: Compile AMGX

```bash
# Clone AMGX repository
git clone --recursive https://github.com/NVIDIA/AMGX.git
cd AMGX

# Create build directory
mkdir build && cd build

# Configure CMake (adjust CUDA_ARCH for your GPU)
# RTX 4090: 89, RTX 3090: 86, RTX 2080: 75, V100: 70
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCH=89 \
    -DCMAKE_INSTALL_PREFIX=/usr/local

# Compile (use multiple cores)
make -j$(nproc)

# Install
sudo make install

# Update library path
echo '/usr/local/lib' | sudo tee -a /etc/ld.so.conf.d/amgx.conf
sudo ldconfig
```

#### Step 3: Install pyamgx

```bash
# Clone pyamgx
git clone https://github.com/shwina/pyamgx.git
cd pyamgx

# Set environment variables
export AMGX_DIR=/usr/local
export CUDA_HOME=/usr/local/cuda

# Install
pip install .
```

#### Step 4: Verify Installation

```bash
python -c "import pyamgx; print('PyAMGX installed successfully!')"
```

### Module C Installation (cuDSS)

Module C requires PyTorch compiled from source with cuDSS support.

#### Step 1: Install cuDSS Library

```bash
# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-repo-ubuntu2204-0.6.0_0.6.0-1_amd64.deb
sudo dpkg -i cudss-local-repo-ubuntu2204-0.6.0_0.6.0-1_amd64.deb
sudo cp /var/cudss-local-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudss
```

#### Step 2: Create Conda Environment

```bash
conda create -n torch_sparse2.7 python=3.11 -y
conda activate torch_sparse2.7
```

#### Step 3: Install Build Dependencies

```bash
# Install via conda
conda install cmake ninja -y

# Install Python dependencies
pip install pyyaml numpy typing-extensions sympy filelock networkx jinja2 fsspec six
pip install mkl-static mkl-include
```

#### Step 4: Download PyTorch Source

```bash
cd ~
git clone --recursive https://github.com/pytorch/pytorch pytorch_source
cd pytorch_source

# Checkout v2.7.0
git checkout v2.7.0
git submodule sync
git submodule update --init --recursive
```

#### Step 5: Compile PyTorch with cuDSS

```bash
# Set environment variables
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH}"
export USE_CUDSS=1
export USE_CUDA=1
export MAX_JOBS=12  # Use half of your CPU cores

# Compile (may take 1-2 hours)
python -m pip install --no-build-isolation -v -e .
```

#### Step 6: Verify Installation

```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

# Test cuDSS
if torch.cuda.is_available():
    try:
        device = 'cuda'
        row_indices = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)
        col_indices = torch.tensor([0, 1, 1, 2, 0, 2], dtype=torch.long, device=device)
        values = torch.tensor([2.0, 1.0, 3.0, 1.0, 1.0, 4.0], dtype=torch.float32, device=device)
        indices = torch.stack([row_indices, col_indices])
        sparse_coo = torch.sparse_coo_tensor(indices, values, (3, 3), device=device)
        sparse_csr = sparse_coo.to_sparse_csr()
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        x = torch.sparse.spsolve(sparse_csr, b)
        print('cuDSS working!')
    except Exception as e:
        print('cuDSS not working:', e)
"
```

### Full Installation

To install all modules, follow this order:

1. **First complete Module C installation** (compile PyTorch with cuDSS)
2. **Then install AMGX and pyamgx** (Module B)
3. **Finally install this package**

```bash
# In the configured environment
cd /path/to/Pytorch_Sparse_Linalg-torch.cg-bicg-gmres-
pip install -e .
```

### Verify Installation

```bash
python -c "
from pytorch_sparse_solver.utils.availability import print_availability_report
print_availability_report()
"
```

Expected output (if all modules are installed):

```
============================================================
PyTorch Sparse Solver - Module Availability Report
============================================================

Module A (JAX-style Iterative Solvers): Available
  - CG (Conjugate Gradient)
  - BiCGStab (Bi-Conjugate Gradient Stabilized)
  - GMRES (Generalized Minimal Residual)

Module B (PyAMGX GPU Solver): Available
  - AMGX CG with AMG preconditioner
  - AMGX BiCGStab with AMG preconditioner
  - AMGX GMRES with AMG preconditioner
  - Automatic differentiation support

Module C (cuDSS Direct Solver): Available
  - torch.sparse.spsolve (LU factorization)
  - GPU-accelerated direct method

============================================================
Total: 3/3 modules available
============================================================
```

---

## Module Details

### Module A: JAX-style Iterative Solvers

Pure PyTorch implementation following JAX's scipy.sparse.linalg API design.

**Solvers:**
- **CG (Conjugate Gradient)**: For symmetric positive definite matrices
- **BiCGStab (Bi-Conjugate Gradient Stabilized)**: For general non-symmetric matrices
- **GMRES (Generalized Minimal Residual)**: For general matrices with restart support

**Features:**
- Pure PyTorch implementation, no additional dependencies
- Supports function-based linear operators (matrix-free methods)
- Supports PyTree structured inputs
- High precision computation (default float64)

**Usage:**

```python
from pytorch_sparse_solver.module_a import cg, bicgstab, gmres

# Conjugate Gradient
x, info = cg(A, b, tol=1e-6, maxiter=1000)

# BiCGStab
x, info = bicgstab(A, b, tol=1e-6)

# GMRES with restart
x, info = gmres(A, b, tol=1e-6, restart=30)
```

### Module B: PyAMGX GPU-accelerated Solvers

Wraps NVIDIA's AMGX library via pyamgx Python bindings.

**Solvers:**
- **AMGX CG**: GPU-accelerated conjugate gradient
- **AMGX BiCGStab**: GPU-accelerated BiCGStab
- **AMGX GMRES**: GPU-accelerated GMRES
- **AMG Preconditioner**: Algebraic multigrid preconditioning

**Features:**
- GPU acceleration, suitable for large-scale problems
- Automatic differentiation support (adjoint method)
- Supports both sparse and dense matrix inputs

**Usage:**

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

### Module C: cuDSS Direct Solver

Wraps NVIDIA's cuDSS library via `torch.sparse.spsolve`.

**Features:**
- Direct method, no iteration required
- Highest accuracy
- Suitable for small to medium-scale problems

**Usage:**

```python
from pytorch_sparse_solver.module_c import cudss_spsolve, cudss_available

if cudss_available():
    # Requires CSR format
    A_csr = A.to_sparse_csr()
    x = cudss_spsolve(A_csr, b)
```

### Module D: Unified Interface

Provides the unified `SparseSolver` class that integrates all backends.

**Features:**
- **Automatic backend selection**: Based on availability and matrix properties
- **Consistent API**: Unified solving interface
- **Flexible switching**: Manual backend specification supported

**Usage:**

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

**SolverResult Structure:**

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

## API Reference

### Unified Interface

```python
x, result = solve(A, b, method='cg', backend='auto', tol=1e-8, maxiter=1000)
```

**Parameters:**
- `A`: PyTorch tensor or callable - coefficient matrix or matvec function
- `b`: PyTorch tensor - right-hand side vector
- `method`: str - 'cg', 'bicgstab', 'gmres', or 'direct'
- `backend`: str - 'auto', 'module_a', 'module_b', or 'module_c'
- `tol`: float - convergence tolerance (default: 1e-8)
- `maxiter`: int - maximum iterations (default: 1000)

**Returns:**
- `x`: PyTorch tensor - solution vector
- `result`: SolverResult - detailed result information

### Module A Functions

```python
x, info = cg(A, b, x0=None, tol=1e-5, atol=0.0, maxiter=None, M=None)
x, info = bicgstab(A, b, x0=None, tol=1e-5, atol=0.0, maxiter=None, M=None)
x, info = gmres(A, b, x0=None, tol=1e-5, atol=0.0, restart=20, maxiter=None, M=None, solve_method='incremental')
```

**Parameters:**
- `A`: PyTorch tensor or callable - coefficient matrix or matvec function
- `b`: PyTorch tensor - right-hand side vector
- `x0`: PyTorch tensor, optional - initial guess (default: zero vector)
- `tol`: float - relative tolerance (default: 1e-5)
- `atol`: float - absolute tolerance (default: 0.0)
- `maxiter`: int, optional - maximum iterations
- `M`: callable, optional - preconditioner function
- `restart`: int (GMRES only) - Krylov subspace size (default: 20)
- `solve_method`: str (GMRES only) - 'incremental' or 'batched'

**Returns:**
- `x`: PyTorch tensor - solution vector
- `info`: int - status (0=success, >0=iterations, <0=error)

### Module B Functions

```python
x = amgx_cg(A, b, tol=1e-8, maxiter=1000)
x = amgx_bicgstab(A, b, tol=1e-8, maxiter=1000)
x = amgx_gmres(A, b, tol=1e-8, maxiter=1000)
```

### Module C Functions

```python
x = cudss_spsolve(A_csr, b)
available = cudss_available()
```

---

## Performance Benchmarks

### Running Benchmarks

```bash
# Quick benchmark
python -m pytorch_sparse_solver.tests.benchmark --quick

# Full benchmark with custom sizes
python -m pytorch_sparse_solver.tests.benchmark --sizes 100,500,1000

# Specify backends and methods
python -m pytorch_sparse_solver.tests.benchmark --backends module_a,module_c --methods cg,direct
```

Benchmark reports are automatically saved to `Logger/` directory with timestamps.

### Example Results (RTX 4090)

| Matrix Size | Module A (CG) | Module A (GMRES) | Module C (Direct) |
|-------------|---------------|------------------|-------------------|
| 100x100     | 23.4 ms       | 344.5 ms         | 41.6 ms           |
| 200x200     | 68.3 ms       | 355.0 ms         | 30.9 ms           |
| 500x500     | 46.2 ms       | 515.7 ms         | 21.4 ms           |

---

## Examples

### Lid-Driven Cavity Flow (LDC) Solver

A complete CFD example is provided in `FVM_example/LDC_by_torchsp/`:

```bash
# Basic usage
python FVM_example/LDC_by_torchsp/ldc_solver.py

# With specific solver
python FVM_example/LDC_by_torchsp/ldc_solver.py --backend module_a --method cg

# Run benchmark
python FVM_example/LDC_by_torchsp/ldc_solver.py --benchmark
```

See `FVM_example/LDC_by_torchsp/README.md` for detailed documentation.

### Function-Based Linear Operators

```python
from pytorch_sparse_solver.module_a import cg

# Define matrix-vector product function (matrix-free)
def tridiagonal_matvec(x):
    n = x.shape[0]
    y = torch.zeros_like(x)
    y[0] = 2*x[0] - x[1]
    y[1:-1] = -x[:-2] + 2*x[1:-1] - x[2:]
    y[-1] = -x[-2] + 2*x[-1]
    return y

# Solve using function-based operator
b = torch.ones(1000, dtype=torch.float64, device='cuda')
x, info = cg(tridiagonal_matvec, b, tol=1e-5)
```

### Automatic Differentiation

```python
import torch
from pytorch_sparse_solver import solve

# Create system with gradient tracking
A = torch.randn(100, 100, dtype=torch.float64, device='cuda')
A = A @ A.T + torch.eye(100, dtype=torch.float64, device='cuda') * 100
b = torch.randn(100, dtype=torch.float64, device='cuda', requires_grad=True)

# Solve with autograd support
x, result = solve(A, b, method='cg')
loss = torch.sum(x**2)
loss.backward()
print("Gradient:", b.grad)
```

---

## Testing

### Run Complete Test Suite

```bash
python run_all_tests.py
```

### Run Individual Module Tests

```bash
# Module A tests
python -m pytest src/pytorch_sparse_solver/tests/test_module_a.py -v

# Module B tests
python -m pytest src/pytorch_sparse_solver/tests/test_module_b.py -v

# Module C tests
python -m pytest src/pytorch_sparse_solver/tests/test_module_c.py -v

# Unified interface tests
python -m pytest src/pytorch_sparse_solver/tests/test_unified.py -v
```

---

## Environment Requirements

### Tested Configuration

| Component | Version |
|-----------|---------|
| GPU | NVIDIA GeForce RTX 4090 |
| OS | Ubuntu 22.04.4 LTS |
| Python | 3.11.0 |
| PyTorch | 2.7.0+cu128 |
| CUDA Runtime | 12.8 |
| GCC | 12.3.0 |

---

## Troubleshooting

### Q: Module B shows as unavailable

A: Ensure:
- pyamgx is correctly installed
- AMGX library is in system path
- CUDA is available

### Q: Module C shows as unavailable

A: Ensure:
- Using PyTorch 2.7+ compiled from source
- `USE_CUDSS=1` was set during compilation
- cuDSS library is correctly installed

### Q: PyTorch compilation fails

A: Try:
- Reduce MAX_JOBS (memory may be insufficient)
- Ensure all submodules are properly updated
- Check CUDA version compatibility

### Q: GMRES converges slowly

A: Try:
- Increase `restart` parameter
- Use BiCGStab instead for non-symmetric matrices
- Add preconditioning

---

## Installation Combinations

| Combination | Installation | Available Features |
|-------------|--------------|-------------------|
| A only | `pip install pytorch_sparse_solver` | CG, BiCGStab, GMRES (CPU/GPU) |
| A+B | Install pyamgx | Above + AMGX acceleration |
| A+C | Compile PyTorch with cuDSS | Above + Direct method |
| A+B+C | Full installation | All features |

---

## Attribution

This implementation is **algorithmically derived** from Google JAX's scipy.sparse.linalg module:

**Original JAX Implementation:**
- Repository: https://github.com/google/jax
- File: `jax/scipy/sparse/linalg.py`
- License: Apache-2.0
- Copyright: Google LLC

---

## License

Apache License 2.0

This code is licensed under Apache-2.0, the same license as the original JAX implementation from which algorithmic concepts were derived.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For bug reports and feature requests, please open an issue on GitHub.
