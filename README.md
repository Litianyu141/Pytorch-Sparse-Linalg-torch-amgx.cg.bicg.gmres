# PyTorch Sparse Linear Algebra Solvers

A comprehensive PyTorch implementation of sparse linear algebra solvers (CG, BiCGStab, GMRES, AMG) with GPU acceleration and automatic differentiation support.

## Latest Updates

> 🆕 **Update 2025.06.25**: support pyamgx[[installation-guide]](#installation) with improved differentiability

## Overview

This library provides efficient implementations of iterative sparse linear system solvers for PyTorch tensors:

```
├── src/                           # PyTorch sparse solvers
│   └── torch_sparse_linalg.py    # CG, BiCGStab, GMRES implementations
├── src_torch_amgx/                 # AMGX integration
│   └── torch_amgx.py             # Differentiable AMGX solvers
├── test_comprehensive_solvers.py  # Comprehensive testing suite
├── examples/                      # Example applications
└── README.md                     # This file
```

### PyTorch Solvers (CPU/GPU)
- **Conjugate Gradient (CG)** - for symmetric positive definite matrices
- **BiCGStab (Bi-Conjugate Gradient Stabilized)** - for general non-symmetric matrices
- **GMRES (Generalized Minimal Residual)** - for general matrices with restart capability

### AMGX Solvers (GPU-accelerated) installation-guide:[AMGX](https://github.com/NVIDIA/AMGX); [Pyamgx](https://github.com/shwina/pyamgx)
- **AMGX CG** - GPU-accelerated conjugate gradient with automatic differentiation
- **AMGX BiCGStab** - GPU-accelerated BiCGStab with automatic differentiation
- **AMGX GMRES** - GPU-accelerated GMRES with automatic differentiation

## Key Features

- **Full Automatic Differentiation**: All solvers support PyTorch's autograd
- **GPU Acceleration**: AMGX solvers leverage NVIDIA GPU acceleration
- **Multiple Matrix Types**: Optimized for different sparse matrix structures
- **Comprehensive Testing**: Built-in testing suite for accuracy, speed, and differentiability

## Environment Requirements

### Tested Environment

This library has been tested and verified on the following configuration:

**Hardware:**
- **GPU**: 6x NVIDIA GeForce RTX 4090 (24GB VRAM each)
- **CPU**: Multi-core x86_64 architecture

**Software Environment:**
- **OS**: Ubuntu 22.04.4 LTS
- **Python**: 3.11.0
- **PyTorch**: 2.7.1+cu128
- **CUDA Runtime**: 12.8.61
- **CUDA Driver**: 12.4 (compatible with CUDA 12.8)
- **NVIDIA Driver**: 550.54.14
- **GCC**: 12.3.0
- **CMake**: 3.22.1

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/pytorch-sparse-linalg.git
cd pytorch-sparse-linalg

# Install dependencies
pip install torch numpy scipy pandas matplotlib
```

### AMGX Installation (Optional)

For GPU-accelerated AMGX solvers, you need to compile AMGX and pyamgx from source:

#### Step 1: Install Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake git

# Ensure CUDA is properly installed
nvcc --version  # Should show CUDA 12.0+
```

#### Step 2: Compile AMGX Library

```bash
# Clone AMGX repository
git clone https://github.com/NVIDIA/AMGX.git
cd AMGX

# Create build directory
mkdir build && cd build

# Configure with CMake (adjust CUDA architecture for your GPU)
# For RTX 4090: compute capability 8.9
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCH=89 \
    -DCMAKE_INSTALL_PREFIX=/usr/local

# Compile (use number of CPU cores, e.g., -j12 for 12 cores)
make -j$(nproc)

# Install
sudo make install

# Update library path
echo '/usr/local/lib' | sudo tee -a /etc/ld.so.conf.d/amgx.conf
sudo ldconfig
```

#### Step 3: Compile pyamgx

```bash
# Clone pyamgx repository
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
python -c "import pyamgx; print('AMGX successfully installed!')"
```

**Tested Configuration Summary:**
```
Hardware: 6x NVIDIA RTX 4090 (24GB each)
OS: Ubuntu 22.04.4 LTS
CUDA: 12.8 Runtime / 12.4 Driver
PyTorch: 2.7.1+cu128
Python: 3.11.0
GCC: 12.3.0
```

This configuration provides optimal performance for both PyTorch and AMGX solvers.

### Installation Verification

After installation, verify everything is working correctly:

```bash
# Run the verification script
python verify_installation.py
```

This script will check:
- ✅ PyTorch and CUDA installation
- ✅ PyTorch sparse solvers functionality
- ✅ AMGX library and pyamgx bindings
- ✅ torch_amgx integration
- ✅ Automatic differentiation support

If all checks pass, you're ready to use the library!

## Quick Start

### PyTorch Solvers

```python
import torch
from src.torch_sparse_linalg import cg, bicgstab, gmres

# Create a test system Ax = b
n = 1000
A = torch.rand(n, n, dtype=torch.float64, device='cuda')
A = A + A.T + torch.eye(n, dtype=torch.float64, device='cuda') * 2  # Make SPD
b = torch.rand(n, dtype=torch.float64, device='cuda')

# Solve using Conjugate Gradient
x_cg, info = cg(A, b, tol=1e-5, maxiter=1000)
print(f"CG converged: {info == 0}")

# Solve using BiCGStab
x_bicg, info = bicgstab(A, b, tol=1e-5, maxiter=1000)
print(f"BiCGStab converged: {info == 0}")

# Solve using GMRES with restart
x_gmres, info = gmres(A, b, tol=1e-5, restart=20, maxiter=1000)
print(f"GMRES converged: {info == 0}")
```

### AMGX Solvers (GPU-accelerated)

```python
import torch
from src_torch_amgx.torch_amgx import amgx_cg, amgx_bicgstab, amgx_gmres

# Create a sparse test system
n = 1000
A = torch.rand(n, n, dtype=torch.float64, device='cuda')
A = A + A.T + torch.eye(n, dtype=torch.float64, device='cuda') * 2  # Make SPD
b = torch.rand(n, dtype=torch.float64, device='cuda')

# Solve using AMGX CG (GPU-accelerated)
x_amgx_cg = amgx_cg(A, b, tol=1e-8, maxiter=1000)

# Solve using AMGX BiCGStab (GPU-accelerated)
x_amgx_bicg = amgx_bicgstab(A, b, tol=1e-8, maxiter=1000)

# Solve using AMGX GMRES (GPU-accelerated)
x_amgx_gmres = amgx_gmres(A, b, tol=1e-8, maxiter=1000)
```

### Automatic Differentiation

All solvers support automatic differentiation:

```python
import torch
from src.torch_sparse_linalg import cg
from src_torch_amgx.torch_amgx import amgx_cg

# Create system with gradient tracking
A = torch.rand(100, 100, dtype=torch.float64, device='cuda')
A = A + A.T + torch.eye(100, dtype=torch.float64, device='cuda') * 2
b = torch.rand(100, dtype=torch.float64, device='cuda', requires_grad=True)

# PyTorch solver with autograd
x_torch = cg(A, b, tol=1e-6)[0]
loss_torch = torch.sum(x_torch**2)
loss_torch.backward()
print("PyTorch CG gradient:", b.grad)

# Reset gradients
b.grad = None

# AMGX solver with autograd
x_amgx = amgx_cg(A, b, tol=1e-6)
loss_amgx = torch.sum(x_amgx**2)
loss_amgx.backward()
print("AMGX CG gradient:", b.grad)
```

### Using Function-Based Linear Operators

The solvers also accept functions that compute matrix-vector products:

```python
import torch
from src.torch_sparse_linalg import cg, bicgstab, gmres

# Define a function that computes Ax for a tridiagonal matrix
def tridiagonal_matvec(x):
    """Compute tridiagonal matrix-vector product without storing the matrix"""
    n = x.shape[0]
    y = torch.zeros_like(x)
    
    # Apply tridiagonal pattern: [-1, 2, -1]
    y[0] = 2*x[0] - x[1] if n > 1 else 2*x[0]
    for i in range(1, n-1):
        y[i] = -x[i-1] + 2*x[i] - x[i+1]
    if n > 1:
        y[n-1] = -x[n-2] + 2*x[n-1]
    
    return y

# Create right-hand side vector
n = 1000
b = torch.ones(n, dtype=torch.float64, device='cuda')

# Solve using function-based operator
x_cg, info = cg(tridiagonal_matvec, b, tol=1e-5, maxiter=1000)
print(f"Function-based CG converged: {info == 0}")

# Works with all solvers
x_bicg, info = bicgstab(tridiagonal_matvec, b, tol=1e-5, maxiter=1000)
x_gmres, info = gmres(tridiagonal_matvec, b, tol=1e-5, restart=20, maxiter=1000)
```

## API Reference

### Function Signatures

All solvers follow the same interface pattern:

```python
x, info = solver(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None, **kwargs)
```

**Parameters:**
- `A`: PyTorch tensor or callable - coefficient matrix or matrix-vector function
- `b`: PyTorch tensor - right-hand side vector  
- `x0`: PyTorch tensor, optional - initial guess (default: zero vector)
- `tol`: float - relative convergence tolerance (default: 1e-5)
- `atol`: float - absolute convergence tolerance (default: 0.0)
- `maxiter`: int, optional - maximum iterations (default: min(n, 1000) for CG/BiCGStab, 10*n for GMRES)
- `M`: callable, optional - preconditioner function

**Returns:**
- `x`: PyTorch tensor - solution vector
- `info`: int - convergence status (0=success, >0=iterations, <0=error)


#### Conjugate Gradient (CG)

```python
x, info = cg(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None)
```

**Requirements:**
- `A` must be symmetric positive definite
- Best for symmetric positive definite systems

#### BiCGStab

```python
x, info = bicgstab(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None)
```

**Requirements:**
- `A` can be any general (nonsymmetric) linear operator
- Good for general nonsymmetric systems

#### GMRES

```python
x, info = gmres(A, b, x0=None, *, tol=1e-5, atol=0.0, restart=20, 
                maxiter=None, M=None, solve_method='incremental')
```

**Additional Parameters:**
- `restart`: int - Krylov subspace size before restart (default: 20)
- `solve_method`: str - 'incremental' or 'batched' (default: 'incremental')

**Requirements:**
- `A` can be any general linear operator
- Memory usage scales with restart parameter

### AMGX Solvers API

AMGX solvers provide GPU-accelerated solving with automatic differentiation:

```python
from src_torch_amgx.torch_amgx import amgx_cg, amgx_bicgstab, amgx_gmres

# All AMGX solvers follow this pattern:
x = solver(A, b, tol=1e-8, maxiter=1000)
```

**Parameters:**

- `A`: PyTorch tensor - coefficient matrix (sparse or dense)
- `b`: PyTorch tensor - right-hand side vector
- `tol`: float - convergence tolerance (default: 1e-8)
- `maxiter`: int - maximum iterations (default: 1000)

**Returns:**

- `x`: PyTorch tensor - solution vector

#### AMGX CG

```python
x = amgx_cg(A, b, tol=1e-8, maxiter=1000)
```

**Best for:** Symmetric positive definite matrices

#### AMGX BiCGStab

```python
x = amgx_bicgstab(A, b, tol=1e-8, maxiter=1000)
```

**Best for:** General non-symmetric matrices

#### AMGX GMRES

```python
x = amgx_gmres(A, b, tol=1e-8, maxiter=1000)
```

**Best for:** General matrices, guaranteed convergence

## Comprehensive Testing

Run the comprehensive test suite to compare all solvers:

```bash
python test_comprehensive_solvers.py
```

This will test:

- **Accuracy**: Solution and residual errors across different matrix types
- **Speed**: Performance comparison between PyTorch and AMGX solvers
- **Differentiability**: Automatic differentiation verification
- **Matrix Types**: Diagonally dominant, non-diagonally dominant, and banded matrices

The test generates an HTML report with detailed results and recommendations.

## Examples

See the `examples/` directory for detailed usage examples:

- **Basic Usage**: Simple linear system solving
- **Sparse Matrices**: Working with sparse matrices
- **Automatic Differentiation**: Gradient computation examples
- **Performance Comparison**: Benchmarking different solvers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Attribution

This implementation is **algorithmically derived** from Google JAX's scipy.sparse.linalg module:

**Original JAX Implementation:**

- Repository: https://github.com/google/jax
- File: `jax/scipy/sparse/linalg.py`
- License: Apache-2.0
- Copyright: Google LLC

## License

Apache License 2.0

**Note**: This code is licensed under Apache-2.0, the same license as the original JAX implementation from which algorithmic concepts were derived. This ensures full license compatibility and proper attribution.
