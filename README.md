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

Cuda-12

Pytorch-2.7 (**Note torch.sparse.spsolve require CuDSS support, public release torch are not compiled with CuDSS=1, Maybe you need to recompile locally**)

JAX-Cuda12

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

### Detailed Function Signatures

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

### Convergence Criteria

All solvers use the convergence criterion:
```
norm(residual) <= max(tol * norm(b), atol)
```

Where:
- `tol`: Relative tolerance (default: 1e-5)
- `atol`: Absolute tolerance (default: 0.0)
- `norm(b)`: Norm of the right-hand side vector

### Best Practices

**Data Types:**
- Use `torch.float64` for best numerical precision
- Complex numbers are supported via `torch.complex128`

**Device Placement:**
- Place all tensors on the same device (CPU or GPU)
- GPU acceleration provides significant speedup for large systems

**Tolerance Settings:**
- Start with default `tol=1e-5` 
- For high precision: use `tol=1e-8` or lower
- Set `atol` for absolute error control when `norm(b)` is very small

**Matrix Conditions:**
- **CG**: Requires symmetric positive definite matrices
- **BiCGStab**: Works with general nonsymmetric matrices  
- **GMRES**: Most robust for ill-conditioned systems

**Memory Considerations:**
- Function-based operators save memory for large sparse systems
- GMRES memory usage scales with `restart` parameter
- Use smaller `restart` values for memory-constrained environments

### Convergence Status Codes

The `info` return value indicates convergence status:

- `0`: Successful convergence within tolerance
- `> 0`: Maximum iterations reached (may not have converged to tolerance)
- `-1`: Failed to converge or numerical breakdown

**Notes:**
- For `cg`: `info > 0` indicates the number of iterations when max iterations reached
- For `bicgstab`: `info = 0` for success, `info != 0` for failure  
- For `gmres`: `info = 0` for success, `info = -1` for failure
- Always check convergence: `converged = (info == 0)`

## Performance Comparison

The `test_sparse_gpu.py` script provides comprehensive benchmarking against JAX:

```bash
python test_sparse_gpu.py
```

**Test Matrices:**
- Tridiagonal matrices (200×200, 500×500)
- 2D Poisson operators (5-point stencil)
- Random sparse diagonally dominant matrices
- Ultra-large matrices (10,000×10,000+)

**Benchmark Results Summary:**
- PyTorch implementations achieve comparable accuracy to JAX
- GPU acceleration provides significant speedup for large matrices
- Direct solvers (`torch.linalg.solve`) offer highest accuracy for small-medium matrices
- Iterative solvers excel for large sparse systems

## File Structure

```
Pytorch_sparse_linalg/
├── README.md                    # This file
├── src/
│   ├── torch_sparse_linalg.py  # Main solver implementations  
│   └── torch_tree_util.py      # PyTorch tree utilities (from JAX)
├── test_sparse_gpu.py          # Comprehensive benchmarking script
└── test_results.md             # Example benchmark report
```

## Implementation Details

### Algorithmic Fidelity

Our implementations follow JAX's algorithms exactly:

1. **CG Algorithm**: Implements the same preconditioned conjugate gradient as JAX
2. **BiCGStab**: Matches JAX's bi-conjugate gradient stabilized method
3. **GMRES**: Includes both incremental and batched variants with identical restart logic

### Numerical Precision

- **Default dtype**: `torch.float64` for maximum precision
- **Convergence criteria**: Identical tolerance checking as JAX
- **Stability**: Includes JAX's numerical stability improvements

### Tree Utilities

The `torch_tree_util.py` module provides PyTorch equivalents of JAX's tree manipulation:
- `tree_map`: Apply functions to nested structures
- `tree_reduce`: Reduce operations over tree structures  
- `tree_flatten/unflatten`: Convert between nested and flat representations

These utilities enable the same functional programming patterns as JAX.

## Testing and Validation

### Test Coverage

The test suite (`test_sparse_gpu.py`) validates:
- ✅ **Numerical accuracy** vs JAX reference implementations
- ✅ **Convergence behavior** across different matrix types
- ✅ **Performance characteristics** on GPU vs CPU
- ✅ **Memory usage** and scalability
- ✅ **Error handling** and edge cases

### Matrix Types Tested

1. **Tridiagonal matrices**: Classic test case for iterative methods
2. **2D Poisson operators**: Realistic PDE discretization  
3. **Random sparse diagonally dominant**: General sparse systems
4. **Ultra-large matrices**: Scalability testing (10K+ dimensions)

### Fairness of Comparison

The benchmark ensures fair JAX vs PyTorch comparison:
- ✅ Identical solver parameters (tolerance, max iterations, restart)
- ✅ Same matrix conditioning and right-hand sides
- ✅ Comparable precision (float64)
- ✅ Proper GPU synchronization timing
- ✅ Statistical analysis over multiple test cases

## Usage Recommendations

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

## Contributing

This implementation focuses on mirroring JAX's proven algorithms in PyTorch. Contributions welcome for:

- Preconditioning strategies
- Additional iterative methods (MINRES, LSQR, etc.)
- Complex number support
- Sparse matrix format optimizations
- Performance improvements


## License

Apache License 2.0 

**Note**: This code is licensed under Apache-2.0, the same license as the original JAX implementation from which algorithmic concepts were derived. This ensures full license compatibility and proper attribution.
