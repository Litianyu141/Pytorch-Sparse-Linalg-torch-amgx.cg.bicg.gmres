# PyTorch Sparse Linear Algebra Solvers

A PyTorch implementation of sparse linear algebra solvers, mirroring JAX's scipy.sparse.linalg module with PyTorch-specific optimizations.

## Overview

This library provides efficient implementations of iterative sparse linear system solvers for PyTorch tensors:
- **Conjugate Gradient (CG)** - for symmetric positive definite matrices
- **BiCGStab (Bi-Conjugate Gradient Stabilized)** - for general non-symmetric matrices  
- **GMRES (Generalized Minimal Residual)** - for general matrices with restart capability

## Key Features

- 🚀 **GPU Acceleration**: Full CUDA support with PyTorch's native GPU operations
- ⚡ **JIT Compilation**: Optional PyTorch JIT optimization for maximum performance
- 🔧 **Sparse Matrix Support**: Compatible with dense and sparse matrix formats
- 📊 **JAX API Compatibility**: Mirrors JAX's scipy.sparse.linalg interface
- 🎯 **High Precision**: Uses float64 by default for numerical stability
- 🌳 **Tree Utilities**: Includes PyTorch adaptation of JAX's tree manipulation utilities

## Attribution and License

This implementation is **algorithmically derived** from Google JAX's scipy.sparse.linalg module:

**Original JAX Implementation:**
- Repository: https://github.com/google/jax
- File: `jax/scipy/sparse/linalg.py`
- License: Apache-2.0
- Copyright: Google LLC

**Key Differences from JAX:**
1. **Framework**: Ported from JAX/NumPy to PyTorch tensors
2. **Complex Numbers**: Removed complex number support (real numbers only)
3. **JIT Compilation**: Uses PyTorch's JIT instead of JAX's
4. **GPU Optimization**: Leverages PyTorch's CUDA operations
5. **Tree Utilities**: Adapted JAX's tree manipulation for PyTorch

**Legal Status:**
- ✅ **Algorithm Implementation**: Mathematical algorithms are not copyrightable
- ✅ **Independent Implementation**: Rewritten in PyTorch, not copied code
- ✅ **Proper Attribution**: Original JAX implementation credited
- ✅ **License Compatibility**: Our Apache-2.0 license is compatible with JAX's Apache-2.0

## Installation

```bash
# Ensure PyTorch is installed with CUDA support (if desired)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For sparse matrix support (optional but recommended)
pip install torch-sparse torch-geometric
```

## Quick Start

```python
import torch
from torch_sparse_linalg import cg, bicgstab, gmres

# Create a test system Ax = b
n = 1000
A = torch.rand(n, n, dtype=torch.float64, device='cuda')
A = A + A.T + torch.eye(n, dtype=torch.float64, device='cuda') * 2  # Make SPD
b = torch.rand(n, dtype=torch.float64, device='cuda')

# Solve using Conjugate Gradient
x_cg, info = cg(A, b, tol=1e-8, maxiter=1000)
print(f"CG converged: {info == 0}")

# Solve using BiCGStab  
x_bicg, info = bicgstab(A, b, tol=1e-8, maxiter=1000)
print(f"BiCGStab converged: {info == 0}")

# Solve using GMRES with restart
x_gmres, info = gmres(A, b, tol=1e-8, restart=30, maxiter=1000)
print(f"GMRES converged: {info == 0}")
```

## API Reference

### Function Signatures

All solvers follow the same interface pattern:

```python
x, info = solver(A, b, tol=1e-6, maxiter=None, **kwargs)
```

**Parameters:**
- `A`: PyTorch tensor, shape (n, n) - coefficient matrix
- `b`: PyTorch tensor, shape (n,) - right-hand side vector  
- `tol`: float - convergence tolerance (default: 1e-6)
- `maxiter`: int - maximum iterations (default: min(n, 1000))

**Returns:**
- `x`: PyTorch tensor - solution vector
- `info`: int - convergence status (0=success, >0=iterations, <0=error)

### Solver-Specific Parameters

**GMRES Additional Parameters:**
- `restart`: int - restart parameter (default: min(20, n))
- `solve_method`: str - 'incremental' or 'batched' (default: 'incremental')

### Convergence Status Codes

- `0`: Successful convergence
- `> 0`: Number of iterations when stopped (may not have converged)
- `-1`: Maximum iterations reached without convergence
- `-2`: Numerical error or breakdown
- `-3`: Invalid parameters

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
torch_sparse_linalg/
├── README.md                    # This file
├── torch_sparse_linalg.py      # Main solver implementations  
├── torch_tree_util.py          # PyTorch tree utilities (from JAX)
├── test_sparse_gpu.py          # Comprehensive benchmarking script
└── sparse_solver_benchmark_results.md  # Example benchmark report
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

Apache License 2.0 - see LICENSE file for details.

**Author**: Litianyu141

**Note**: This code is licensed under Apache-2.0, the same license as the original JAX implementation from which algorithmic concepts were derived. This ensures full license compatibility and proper attribution.
