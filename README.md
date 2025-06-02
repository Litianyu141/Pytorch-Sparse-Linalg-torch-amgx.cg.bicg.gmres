# PyTorch Sparse Linear Algebra Solvers

A PyTorch implementation (JIT support) of sparse linear algebra solvers (CG, BiCGstab, GMRES), mirroring JAX's scipy.sparse.linalg module with PyTorch-specific optimizations.

## Overview

This library provides efficient implementations of iterative sparse linear system solvers for PyTorch tensors:
- **Conjugate Gradient (CG)** - for symmetric positive definite matrices
- **BiCGStab (Bi-Conjugate Gradient Stabilized)** - for general non-symmetric matrices  
- **GMRES (Generalized Minimal Residual)** - for general matrices with restart capability


## Attribution

This implementation is **algorithmically derived** from Google JAX's scipy.sparse.linalg module:

**Original JAX Implementation:**
- Repository: https://github.com/google/jax
- File: `jax/scipy/sparse/linalg.py`
- License: Apache-2.0
- Copyright: Google LLC


## Enviroment

- Cuda-12

- Pytorch-2.7 (**Note torch.sparse.spsolve require CuDSS support, public-released-wheel pytorch are not compiled with CuDSS=1, Build from source may be necessary**)

- JAX-Cuda12

## Quick Start

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

### Input Types

**Matrix A:**
- **PyTorch Tensor**: Dense matrix of shape (n, n)
- **Callable Function**: `A(x) -> y` where `y = A @ x`

**Example Function-Based Operator:**
```python
def matrix_vector_product(x):
    # Your custom matrix-vector multiplication
    return result  # same shape and structure as x
```

## Performance Comparison

The `test_sparse_gpu.py` script provides comprehensive benchmarking against JAX:

```bash
python test_sparse_gpu.py
```

### Solver Selection Guide

**For Symmetric Positive Definite Matrices:**
- Small-medium (< 10K): `torch.linalg.solve` (direct)
- Large sparse: `cg` (Conjugate Gradient)

**For General Non-Symmetric Matrices:**
- Small-medium (< 10K): `torch.linalg.solve` (direct)  
- Large sparse: `bicgstab` or `gmres`

**For Ill-Conditioned Systems:**
- Use `gmres` with smaller restart parameter
- Consider preconditioning (future work)

### Performance Tips

1. **GPU Usage**: Ensure tensors are on GPU for large matrices
2. **Data Types**: Use `torch.float64` for best numerical stability
3. **Memory**: For very large systems, consider block methods
4. **JIT Compilation**: Enable for repeated solves of similar systems

## License

Apache License 2.0 

**Note**: This code is licensed under Apache-2.0, the same license as the original JAX implementation from which algorithmic concepts were derived. This ensures full license compatibility and proper attribution.
