# PyTorch vs JAX Sparse Matrix Solver Performance Comparison Report

**Generated at**: 2025-06-03 01:26:50

## Test Environment

- **PyTorch version**: 2.7.0+cu128
- **Device**: CUDA
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **JAX device**: cuda:0

## Overall Statistics

- **Total tests**: 56
- **Successfully converged**: 36 (64.3%)
- **Failed tests**: 20

## Detailed Test Results

### Sparse 2D Poisson 15×15 (225×225, PyTorch COO)

**Matrix size**: 225×225

| Framework | Solver | Time(s) | Solution Error | Residual Error | Status | Method Details |
|-----------|--------|---------|----------------|----------------|--------|-----------------|
| JAX | BiCGStab | 0.179 | 1.60e-06 | 5.40e-07 | ❌ Failed - Residual too large (residual=5.40e-07) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | CG | 0.611 | 3.96e-07 | 6.36e-07 | ❌ Failed - Residual too large (residual=6.36e-07) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | GMRES | 0.556 | 4.24e-06 | 3.62e-07 | ✅ Success - JAX converged (info=0) - JAX GMRES (restart=22, maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| PyTorch | BiCGStab | 0.092 | 1.30e-06 | 4.62e-07 | ✅ Success - BiCGStab (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | CG | 0.061 | 3.96e-07 | 6.36e-07 | ✅ Success - CG (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | Direct | 0.001 | 2.20e-14 | 1.65e-14 | ✅ Success (Dense fallback) - Success (Dense fallback) - Direct solve | ⚠️ Dense fallback (torch.linalg.solve) |
| PyTorch | GMRES | 0.098 | 4.24e-06 | 3.62e-07 | ✅ Success - GMRES (restart=22, maxiter=1000, tol=1e-08) | N/A |

**Best performance**:
- **Fastest**: PyTorch Direct (0.001s)
- **Most accurate**: PyTorch Direct (2.20e-14)

### Sparse 2D Poisson 22×22 (484×484, PyTorch COO)

**Matrix size**: 484×484

| Framework | Solver | Time(s) | Solution Error | Residual Error | Status | Method Details |
|-----------|--------|---------|----------------|----------------|--------|-----------------|
| JAX | BiCGStab | 0.187 | 5.75e-06 | 6.17e-07 | ❌ Failed - Residual too large (residual=6.17e-07) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | CG | 0.266 | 8.90e-07 | 9.51e-07 | ❌ Failed - Residual too large (residual=9.51e-07) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | GMRES | 0.604 | 2.12e-05 | 9.55e-07 | ✅ Success - JAX converged (info=0) - JAX GMRES (restart=30, maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| PyTorch | BiCGStab | 0.168 | 5.73e-06 | 6.15e-07 | ✅ Success - BiCGStab (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | CG | 0.077 | 8.90e-07 | 9.51e-07 | ✅ Success - CG (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | Direct | 0.007 | 3.20e-14 | 2.53e-14 | ✅ Success (Dense fallback) - Success (Dense fallback) - Direct solve | ⚠️ Dense fallback (torch.linalg.solve) |
| PyTorch | GMRES | 0.106 | 2.12e-05 | 9.55e-07 | ✅ Success - GMRES (restart=30, maxiter=1000, tol=1e-08) | N/A |

**Best performance**:
- **Fastest**: PyTorch Direct (0.007s)
- **Most accurate**: PyTorch Direct (3.20e-14)

### Sparse Diagonal Dominant 300×300 (PyTorch COO)

**Matrix size**: 300×300

| Framework | Solver | Time(s) | Solution Error | Residual Error | Status | Method Details |
|-----------|--------|---------|----------------|----------------|--------|-----------------|
| JAX | BiCGStab | 0.168 | 5.32e-08 | 1.35e-07 | ❌ Failed - Residual too large (residual=1.35e-07) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | CG | 0.405 | 3.30e-04 | 9.08e-04 | ❌ Failed - Residual too large (residual=9.08e-04) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | GMRES | 0.576 | 6.36e-15 | 1.78e-14 | ✅ Success - JAX converged (info=0) - JAX GMRES (restart=30, maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| PyTorch | BiCGStab | 0.025 | 1.26e-08 | 2.32e-08 | ✅ Success - BiCGStab (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | CG | 0.974 | 7.33e-04 | 1.87e-03 | ❌ Failed - Not converged (max iterations reached) - CG (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | Direct | 0.004 | 1.14e-14 | 2.66e-14 | ✅ Success (Dense fallback) - Success (Dense fallback) - Direct solve | ⚠️ Dense fallback (torch.linalg.solve) |
| PyTorch | GMRES | 0.038 | 5.03e-15 | 1.32e-14 | ✅ Success - GMRES (restart=30, maxiter=1000, tol=1e-08) | N/A |

**Best performance**:
- **Fastest**: PyTorch Direct (0.004s)
- **Most accurate**: PyTorch GMRES (5.03e-15)

### Sparse Diagonal Dominant 600×600 (PyTorch COO)

**Matrix size**: 600×600

| Framework | Solver | Time(s) | Solution Error | Residual Error | Status | Method Details |
|-----------|--------|---------|----------------|----------------|--------|-----------------|
| JAX | BiCGStab | 0.242 | 3.86e-07 | 6.91e-07 | ❌ Failed - Residual too large (residual=6.91e-07) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | CG | 0.487 | 3.39e-04 | 9.96e-04 | ❌ Failed - Residual too large (residual=9.96e-04) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | GMRES | 0.584 | 1.06e-14 | 3.31e-14 | ✅ Success - JAX converged (info=0) - JAX GMRES (restart=30, maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| PyTorch | BiCGStab | 0.022 | 1.75e-07 | 4.68e-07 | ✅ Success - BiCGStab (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | CG | 0.966 | 2.88e-04 | 8.66e-04 | ❌ Failed - Not converged (max iterations reached) - CG (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | Direct | 0.013 | 1.28e-14 | 3.74e-14 | ✅ Success (Dense fallback) - Success (Dense fallback) - Direct solve | ⚠️ Dense fallback (torch.linalg.solve) |
| PyTorch | GMRES | 0.033 | 9.95e-15 | 3.04e-14 | ✅ Success - GMRES (restart=30, maxiter=1000, tol=1e-08) | N/A |

**Best performance**:
- **Fastest**: PyTorch Direct (0.013s)
- **Most accurate**: PyTorch GMRES (9.95e-15)

### Sparse Diagonal Dominant Asymmetric 400×400 (PyTorch COO)

**Matrix size**: 400×400

| Framework | Solver | Time(s) | Solution Error | Residual Error | Status | Method Details |
|-----------|--------|---------|----------------|----------------|--------|-----------------|
| JAX | BiCGStab | 0.167 | 4.58e-08 | 6.81e-07 | ❌ Failed - Residual too large (residual=6.81e-07) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | CG | 0.262 | 7.46e-01 | 7.14e+00 | ❌ Failed - Residual too large (residual=7.14e+00) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | GMRES | 0.454 | 2.00e-10 | 1.28e-09 | ✅ Success - JAX converged (info=0) - JAX GMRES (restart=30, maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| PyTorch | BiCGStab | 0.028 | 1.31e-07 | 6.37e-07 | ✅ Success - BiCGStab (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | CG | 0.979 | 7.31e-01 | 7.02e+00 | ❌ Failed - Not converged (max iterations reached) - CG (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | Direct | 0.004 | 1.31e-14 | 1.21e-13 | ✅ Success (Dense fallback) - Success (Dense fallback) - Direct solve | ⚠️ Dense fallback (torch.linalg.solve) |
| PyTorch | GMRES | 0.039 | 1.49e-10 | 9.63e-10 | ✅ Success - GMRES (restart=30, maxiter=1000, tol=1e-08) | N/A |

**Best performance**:
- **Fastest**: PyTorch Direct (0.004s)
- **Most accurate**: PyTorch Direct (1.31e-14)

### Sparse Non-Diagonal Dominant Asymmetric 400×400 (PyTorch COO)

**Matrix size**: 400×400

| Framework | Solver | Time(s) | Solution Error | Residual Error | Status | Method Details |
|-----------|--------|---------|----------------|----------------|--------|-----------------|
| JAX | BiCGStab | 0.201 | 5.43e+03 | 4.51e+04 | ❌ Failed - Residual too large (residual=4.51e+04) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | CG | 0.415 | 2.45e+13 | 2.08e+14 | ❌ Failed - Residual too large (residual=2.08e+14) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | GMRES | 17.294 | 1.89e+01 | 1.47e+02 | ✅ Success - JAX converged (info=0) - JAX GMRES (restart=30, maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| PyTorch | BiCGStab | 0.145 | 4.61e+02 | 3.94e+03 | ❌ Failed - Not converged (max iterations reached) - BiCGStab (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | CG | 0.795 | 1.78e+13 | 1.50e+14 | ❌ Failed - Not converged (max iterations reached) - CG (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | Direct | 0.005 | 2.29e-12 | 5.94e-13 | ✅ Success (Dense fallback) - Success (Dense fallback) - Direct solve | ⚠️ Dense fallback (torch.linalg.solve) |
| PyTorch | GMRES | 30.000 | ∞ | ∞ | ❌ Timeout (30s) - Solver runtime exceeded limit | N/A |

**Best performance**:
- **Fastest**: PyTorch Direct (0.005s)
- **Most accurate**: PyTorch Direct (2.29e-12)

### Sparse Tridiagonal 200×200 (PyTorch COO)

**Matrix size**: 200×200

| Framework | Solver | Time(s) | Solution Error | Residual Error | Status | Method Details |
|-----------|--------|---------|----------------|----------------|--------|-----------------|
| JAX | BiCGStab | 0.238 | 7.53e-04 | 1.93e-07 | ❌ Failed - Residual too large (residual=1.93e-07) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | CG | 0.562 | 1.46e-13 | 1.80e-13 | ✅ Success - Converged by residual check (residual=1.80e-13) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | GMRES | 2.412 | 1.32e-03 | 3.27e-07 | ✅ Success - JAX converged (info=0) - JAX GMRES (restart=20, maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| PyTorch | BiCGStab | 0.645 | 2.19e-04 | 5.62e-08 | ✅ Success - BiCGStab (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | CG | 2.864 | 8.88e-13 | 5.21e-14 | ✅ Success - CG (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | Direct | 0.060 | 3.28e-13 | 2.97e-15 | ✅ Success (Dense fallback) - Success (Dense fallback) - Direct solve | ⚠️ Dense fallback (torch.linalg.solve) |
| PyTorch | GMRES | 11.066 | 1.32e-03 | 3.27e-07 | ✅ Success - GMRES (restart=20, maxiter=1000, tol=1e-08) | N/A |

**Best performance**:
- **Fastest**: PyTorch Direct (0.060s)
- **Most accurate**: JAX CG (1.46e-13)

### Sparse Tridiagonal 500×500 (PyTorch COO)

**Matrix size**: 500×500

| Framework | Solver | Time(s) | Solution Error | Residual Error | Status | Method Details |
|-----------|--------|---------|----------------|----------------|--------|-----------------|
| JAX | BiCGStab | 0.321 | 1.51e-03 | 2.34e-07 | ❌ Failed - Residual too large (residual=2.34e-07) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | CG | 0.331 | 4.95e-12 | 1.30e-12 | ✅ Success - Converged by residual check (residual=1.30e-12) - JAX CG (maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| JAX | GMRES | 4.596 | 1.35e-02 | 5.36e-07 | ✅ Success - JAX converged (info=0) - JAX GMRES (restart=30, maxiter=1000, tol=1e-08) | ℹ️ JAX sparse to dense conversion |
| PyTorch | BiCGStab | 1.358 | 3.82e-03 | 1.84e-07 | ✅ Success - BiCGStab (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | CG | 0.894 | 5.26e-12 | 1.09e-12 | ✅ Success - CG (maxiter=1000, tol=1e-08) | N/A |
| PyTorch | Direct | 0.006 | 1.06e-12 | 4.50e-15 | ✅ Success (Dense fallback) - Success (Dense fallback) - Direct solve | ⚠️ Dense fallback (torch.linalg.solve) |
| PyTorch | GMRES | 27.969 | 1.35e-02 | 5.36e-07 | ✅ Success - GMRES (restart=30, maxiter=1000, tol=1e-08) | N/A |

**Best performance**:
- **Fastest**: PyTorch Direct (0.006s)
- **Most accurate**: PyTorch Direct (1.06e-12)

## Framework Comparison Analysis

| Framework | Average Time(s) | Average Accuracy | Convergence Rate |
|-----------|-----------------|------------------|------------------|
| PyTorch | 1.757 | 7.28e-04 | 81.2% |
| JAX | 2.797 | 1.89e+00 | 41.7% |

## Algorithm Comparison

### BiCGStab

| Framework | Average Time(s) | Average Accuracy | Convergence Rate |
|-----------|-----------------|------------------|------------------|
| PyTorch | 0.334 | 5.78e-04 | 87.5% |
| JAX | ∞ | ∞ | 0.0% |

### CG

| Framework | Average Time(s) | Average Accuracy | Convergence Rate |
|-----------|-----------------|------------------|------------------|
| PyTorch | 0.974 | 3.21e-07 | 50.0% |
| JAX | 0.447 | 2.55e-12 | 25.0% |

### GMRES

| Framework | Average Time(s) | Average Accuracy | Convergence Rate |
|-----------|-----------------|------------------|------------------|
| PyTorch | 5.621 | 2.13e-03 | 87.5% |
| JAX | 3.385 | 2.36e+00 | 100.0% |

## Key Findings

### Direct Solver Methods Analysis

**Method Distribution**:
- **torch.sparse.spsolve (Sparse)**: 0 tests
- **torch.linalg.solve (Dense fallback)**: 8 tests
- **torch.linalg.solve (Dense direct)**: 0 tests

**Method Details**:
1. **✅ Sparse solve (torch.sparse.spsolve)**: Native sparse matrix solving on GPU, most memory efficient
2. **⚠️ Dense fallback (torch.linalg.solve)**: Automatic fallback when sparse solve fails, maintains GPU computation
3. **ℹ️ Dense direct (torch.linalg.solve)**: Direct dense solving for non-sparse input matrices

**Note**: Dense fallback occurred when torch.sparse.spsolve was not supported for the specific matrix configuration or GPU setup.

### Advantages of Direct Solver

The direct solver achieved extremely high accuracy across all tests (average: 4.71e-13).

**Reason analysis**:
1. **Mathematical exactness**: Direct solvers use exact algorithms like LU decomposition, theoretically limited only by floating-point precision
2. **No iterative errors**: Unlike iterative algorithms that require multiple approximations, avoiding cumulative errors
3. **Sparse matrix advantages**: For moderately-sized sparse matrices, modern direct solvers are very efficient
4. **Numerical stability**: PyTorch's implementation uses highly optimized LAPACK routines

### Characteristics of Iterative Solvers

- **CG algorithm**: Excellent performance on symmetric positive definite matrices (PyTorch average accuracy: 3.21e-07)
- **Application scenarios**: Iterative solvers are suitable for ultra-large scale matrices when direct solvers run out of memory
- **Convergence dependency**: Convergence strongly depends on matrix condition number and preconditioners

## Usage Recommendations

### Solver Selection Guide

1. **Small to medium sparse matrices** (< 10,000×10,000):
   - **First choice**: PyTorch Direct solver (automatic sparse/dense selection)
     - Tries `torch.sparse.spsolve` first (most efficient for sparse matrices)
     - Automatically falls back to `torch.linalg.solve` if needed
     - Provides highest accuracy with intelligent method selection
   - **Alternative**: PyTorch CG (for symmetric positive definite matrices)

2. **Large sparse matrices** (> 10,000×10,000):
   - **Symmetric positive definite**: PyTorch CG
   - **General matrices**: PyTorch GMRES or BiCGStab
   - **Consider using preconditioners** to improve convergence speed

3. **GPU acceleration**:
   - PyTorch implementations have better GPU support
   - All solvers (including automatic fallbacks) maintain GPU computation
   - GPU acceleration effects are more pronounced for large matrices
   - Use `solve_method='batched'` for GMRES on GPU

4. **Method reliability**:
   - The intelligent Direct solver automatically handles GPU compatibility issues
   - No manual intervention needed for sparse vs dense selection
   - Detailed method information is provided for transparency

---
*Report automatically generated by PyTorch sparse linear algebra benchmark tool*
