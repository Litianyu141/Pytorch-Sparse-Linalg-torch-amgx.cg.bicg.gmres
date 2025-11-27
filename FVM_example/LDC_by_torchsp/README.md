# Lid-Driven Cavity Flow Solver (LDC_by_torchsp)

A 2D Lid-Driven Cavity (LDC) flow solver example implemented using `pytorch_sparse_solver`.

## Table of Contents

- [Introduction](#introduction)
- [Physical Problem](#physical-problem)
- [Numerical Methods](#numerical-methods)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Solver Options](#solver-options)
- [Performance Benchmarks](#performance-benchmarks)
- [Example Results](#example-results)

---

## Introduction

This example demonstrates how to use the `pytorch_sparse_solver` package to solve linear systems in CFD problems. The LDC problem is a classic benchmark widely used to verify the correctness of CFD codes.

### Supported Solvers

| Module | Methods | Description |
|--------|---------|-------------|
| Module A | CG, BiCGStab, GMRES | JAX-style iterative solvers, pure PyTorch implementation |
| Module B | AMGX-CG, AMGX-BiCGStab, AMGX-GMRES | NVIDIA AMGX GPU-accelerated solvers |
| Module C | cuDSS Direct | cuDSS direct solver (LU factorization) |

---

## Physical Problem

### Problem Description

In a 2D square cavity, the top wall moves at constant velocity $U_t$, while the other three walls are stationary.

```
       Ut (moving wall)
    ←←←←←←←←←←←←←←←←
    ↓                ↓
    ↓                ↓
    ↓   Flow Domain  ↓
    ↓                ↓
    ↓                ↓
    ————————————————————
       (stationary wall)
```

### Governing Equations

Incompressible Navier-Stokes equations:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}$$

$$\nabla \cdot \mathbf{u} = 0$$

### Dimensionless Parameters

- **Reynolds number**: $Re = \frac{U_t L}{\nu}$
- Where $L$ is the cavity side length, $\nu$ is kinematic viscosity

---

## Numerical Methods

### Spatial Discretization

- **Finite Volume Method (FVM)**: Second-order central differencing
- **Staggered Grid**: Velocities defined at cell faces, pressure at cell centers
- **Boundary Conditions**: No-slip wall conditions

### Time Integration

Fractional Step Method:

1. **Momentum Predictor Step**: Solve for $\tilde{\mathbf{u}}$
2. **Pressure Poisson Equation**: Solve for $p$
3. **Velocity Correction Step**: Correct to $\mathbf{u}^{n+1}$

### Pressure Poisson Equation

$$\nabla^2 p = \frac{1}{\Delta t} \nabla \cdot \tilde{\mathbf{u}}$$

This is a large sparse linear system $Ax = b$, solved using `pytorch_sparse_solver`.

---

## File Structure

```
LDC_by_torchsp/
├── README.md                                    # This document
├── ldc_solver.py                               # Main solver (using pytorch_sparse_solver)
├── cavity_flow.py                              # Original configuration entry
└── FVM_Staggered_uniform_torch_optimized.py    # Original FVM implementation
```

---

## Usage

### Basic Run

```bash
# Enter the example directory
cd FVM_example/LDC_by_torchsp

# Run with default settings (CG solver, 100x100 grid, 100 steps)
python ldc_solver.py

# Run with specified parameters
python ldc_solver.py --nx 64 --steps 200 --Re 100
```

### Select Solver

```bash
# Use Module A's BiCGStab
python ldc_solver.py --backend module_a --method bicgstab

# Use Module A's GMRES
python ldc_solver.py --backend module_a --method gmres

# Use Module C's cuDSS direct solver
python ldc_solver.py --backend module_c --method direct

# Use Module B's AMGX (requires pyamgx installation)
python ldc_solver.py --backend module_b --method cg
```

### Plotting and Saving

```bash
# Plot every 10 steps
python ldc_solver.py --plot 10

# Save results to specified directory
python ldc_solver.py --plot 10 --save-dir results/Re100
```

### Performance Benchmark

```bash
# Run benchmark for all available solvers
python ldc_solver.py --benchmark

# Specify grid size and steps
python ldc_solver.py --benchmark --nx 64 --steps 100
```

---

## Solver Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--nx` | 100 | Grid resolution (nx × nx) |
| `--steps` | 100 | Number of time steps |
| `--Re` | 100.0 | Reynolds number |
| `--backend` | auto | Solver backend (auto/module_a/module_b/module_c) |
| `--method` | cg | Solver method (cg/bicgstab/gmres/direct) |
| `--plot` | 0 | Plot interval (0=no plotting) |
| `--benchmark` | False | Run benchmark |
| `--save-dir` | None | Directory to save results |

### Solver Configuration Details

#### Module A (JAX-style Iterative Solvers)

```python
from pytorch_sparse_solver import SparseSolver

solver = SparseSolver()

# CG - for symmetric positive definite matrices
x, result = solver.cg(A, b, tol=1e-10, maxiter=1000)

# BiCGStab - for non-symmetric matrices
x, result = solver.bicgstab(A, b, tol=1e-10, maxiter=1000)

# GMRES - general solver, good for difficult convergence
x, result = solver.gmres(A, b, tol=1e-10, maxiter=1000, restart=30)
```

#### Module B (PyAMGX GPU Solvers)

Requires AMGX and pyamgx installation:

```bash
# See installation docs: docs/installation.md
pip install pyamgx  # Requires pre-compiled AMGX
```

```python
from pytorch_sparse_solver.module_b import amgx_cg, amgx_bicgstab

x = amgx_cg(A, b, tol=1e-8, maxiter=1000)
```

#### Module C (cuDSS Direct Solver)

Requires PyTorch 2.7+ compiled with cuDSS:

```python
from pytorch_sparse_solver.module_c import cudss_solve

# Input must be CSR format sparse matrix
A_csr = A.to_sparse_csr()
x = cudss_solve(A_csr, b)
```

---

## Performance Benchmarks

Run benchmark command:

```bash
python ldc_solver.py --benchmark --nx 50 --steps 50
```

### Example Output

```
======================================================================
LDC Solver Benchmark
======================================================================
============================================================
PyTorch Sparse Solver - Module Availability Report
============================================================

Module A (JAX-style Iterative Solvers): Available
Module B (PyAMGX GPU Solver): Not Available
Module C (cuDSS Direct Solver): Available

======================================================================
Benchmark Summary
======================================================================

Performance Ranking (fastest first):
  1. Module C - cuDSS Direct        : 2.145s (solver: 0.892s)
  2. Module A - CG                  : 3.567s (solver: 2.124s)
  3. Module A - BICGSTAB            : 4.123s (solver: 2.680s)
  4. Module A - GMRES               : 5.234s (solver: 3.791s)
```

---

## Example Results

### Re = 100 Steady State Solution

![LDC Re=100](../../Logger/ldc_Re100_example.png)

### Flow Characteristics

- Main vortex located slightly above the cavity center
- Small secondary vortices in lower left and right corners
- Maximum velocity gradient near the top wall

### Validation

Can be compared with Ghia et al. (1982) benchmark data:

```python
# Centerline u-velocity distribution (x = 0.5)
y_ref = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
u_ref = [0.0000, -0.0372, -0.0419, -0.0477, -0.0643, -0.1015, -0.1566, -0.2109, -0.2058, -0.1364, 0.0033, 0.2315, 0.6872, 0.7372, 0.7887, 0.8412, 1.0000]
```

---

## Code Examples

### Minimal Example

```python
from ldc_solver import LDCSimulation, SolverConfig

# Create solver configuration
config = SolverConfig(
    backend='module_a',
    method='cg',
    tol=1e-10
)

# Initialize simulation
sim = LDCSimulation(
    nx=64, ny=64,
    Re=100.0,
    solver_config=config
)

# Run 500 steps
sim.run(nsteps=500, plot_interval=50)

# Get velocity field
ucc, vcc, speed = sim.get_velocities()
```

### Comparing Different Solvers

```python
import time
from ldc_solver import LDCSimulation, SolverConfig

solvers = [
    ('module_a', 'cg'),
    ('module_a', 'bicgstab'),
    ('module_c', 'direct'),
]

for backend, method in solvers:
    config = SolverConfig(backend=backend, method=method)
    sim = LDCSimulation(nx=50, ny=50, solver_config=config)

    start = time.time()
    sim.run(nsteps=100, plot_interval=0)
    elapsed = time.time() - start

    print(f"{backend}/{method}: {elapsed:.3f}s")
```

---

## Troubleshooting

### Common Issues

**Q: Module C not available**

A: You need to compile PyTorch from source with cuDSS enabled. See `docs/installation.md`.

**Q: GMRES converges slowly**

A: Try increasing the `restart` parameter or use BiCGStab:

```bash
python ldc_solver.py --method bicgstab
```

**Q: GPU out of memory**

A: Reduce grid size or use direct solver:

```bash
python ldc_solver.py --nx 64 --backend module_c
```

---

## References

1. Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411.

2. Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Taylor & Francis.

---

*This example is part of the pytorch_sparse_solver package. For more information, see the main README.md.*
