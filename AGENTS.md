# Repository Guide for Agents

## What this repository is

This repository implements a modular PyTorch library for solving sparse linear systems `Ax = b`, mainly for scientific computing and PDE workloads.

The package name is `pytorch_sparse_solver`, and the code uses a `src/` layout:

- `src/pytorch_sparse_solver/`: main library
- `src/run.py`: unified test and benchmark runner
- `FVM_example/`: application examples, not core package code
- `src/docs/`: installation and module architecture notes

The central idea is "multiple solver backends behind one interface":

- Module A: pure PyTorch iterative solvers
- Module B: NVIDIA AMGX via `pyamgx`
- Module C: cuDSS direct solve via `torch.sparse.spsolve`
- Unified interface: `SparseSolver` in `solver.py`

## Functional summary

### Module A: pure PyTorch iterative solvers

Primary file: `src/pytorch_sparse_solver/module_a/torch_sparse_linalg.py`

Implements:

- `cg`
- `bicgstab`
- `gmres`
- differentiable variants of all three solvers

Important characteristics:

- inspired by JAX `scipy.sparse.linalg`
- accepts dense tensors and callable linear operators
- includes PyTree utilities in `module_a/torch_tree_util.py`
- supports implicit differentiation through custom autograd logic
- the standard public `cg` / `bicgstab` / `gmres` APIs now auto-attach implicit-differentiation backward logic for plain 2D tensor systems when `A` or `b` requires gradients
- defaults to high precision (`float64` / `complex128`)
- has optional `torch.compile` plumbing, but `_JIT_ENABLED` is currently `False`

Important nuance:

- legacy `*_differentiable` entry points still exist for compatibility
- but normal users should prefer the standard solver names first

This is the most self-contained backend and the safest place to start debugging.

### Module B: AMGX backend

Primary file: `src/pytorch_sparse_solver/module_b/torch_amgx.py`

Implements GPU-accelerated solves through NVIDIA AMGX and `pyamgx`, with custom backward logic based on implicit differentiation.

Current public entry points:

- `amgx_cg`
- `amgx_bicgstab`
- `amgx_gmres`
- `amgx_amg`

Key constraints:

- requires CUDA
- requires AMGX installed separately
- requires `pyamgx`
- code converts tensors to CPU NumPy / SciPy CSR structures to interact with AMGX
- the public wrapper now returns gradients to the original input `A` and `b`, not only to internal CSR value buffers

Important nuance:

- `amgx_amg` is the direct AMGX AMG solver
- `amgx_cg` / `amgx_bicgstab` / `amgx_gmres` are Krylov-method frontends implemented through AMGX, not the same thing as `amgx_amg`

Treat this as an optional backend. Always gate usage with availability checks.

### Module C: cuDSS direct solver

Primary file: `src/pytorch_sparse_solver/module_c/cudss_solver.py`

Implements a direct sparse solve using `torch.sparse.spsolve`, plus a custom backward pass.

Key constraints:

- requires CUDA
- requires a PyTorch build with cuDSS support
- effectively expects CSR input for the direct path

This backend is only for `direct` solving in the unified interface.

### Unified interface

Primary file: `src/pytorch_sparse_solver/solver.py`

Core public objects:

- `SparseSolver`
- `solve`
- `cg`
- `bicgstab`
- `gmres`
- `amg`
- `direct_solve`
- `SolverResult`

Backend selection behavior:

- `method="direct"` requires Module C
- `method="amg"` requires Module B
- iterative methods prefer Module B on CUDA if available
- otherwise Module A is the normal fallback

## Important paths

- `README.md`: user-facing overview and installation notes
- `pyproject.toml`: packaging metadata and dependency declarations
- `src/pytorch_sparse_solver/__init__.py`: public exports
- `src/pytorch_sparse_solver/utils/availability.py`: authoritative backend availability checks
- `src/pytorch_sparse_solver/utils/matrix_utils.py`: sparse format conversion and matrix builders
- `src/pytorch_sparse_solver/tests/`: correctness tests and benchmark code
- `src/pytorch_sparse_solver/tests/test_gpu_validation.py`: authoritative GPU integration validation target
- `FVM_example/LDC_by_torchsp/ldc_solver.py`: compatibility wrapper to the Module A LDC solver
- `FVM_example/LDC_by_torchsp/ldc_solver_common.py`: shared LDC/FVM implementation
- `FVM_example/LDC_by_torchsp/ldc_solver_module_a.py`: Module A LDC variant
- `FVM_example/LDC_by_torchsp/ldc_solver_module_b.py`: Module B LDC variant
- `FVM_example/LDC_by_torchsp/ldc_solver_module_c.py`: Module C LDC variant
- `FVM_example/LDC_by_torchsp/ldc_solver_module_d.py`: unified-interface LDC variant

## How the repository is intended to be used

Typical usage patterns:

- import the unified solver and let it choose an available backend
- use Module A directly for pure PyTorch iterative solves
- use Module B or C only when the local environment has the required NVIDIA stack
- if you want true AMG behavior on NVIDIA AMGX, use `amgx_amg` or unified `method="amg"`
- use the LDC examples as application/demo coverage for all four backend styles

The project is aimed at sparse linear algebra inside numerical simulation workflows, especially Poisson-like systems and PDE discretizations.

## Test and benchmark entry points

Primary commands:

- `python src/run.py --test`
- `python src/run.py --benchmark`
- `python src/run.py --all`
- `pytest`
- `python src/pytorch_sparse_solver/tests/test_gpu_validation.py`

There are dedicated test scripts for each backend:

- `src/pytorch_sparse_solver/tests/test_module_a.py`
- `src/pytorch_sparse_solver/tests/test_module_b.py`
- `src/pytorch_sparse_solver/tests/test_module_c.py`
- `src/pytorch_sparse_solver/tests/test_unified.py`
- `src/pytorch_sparse_solver/tests/test_gpu_validation.py`

Benchmark code lives in:

- `src/pytorch_sparse_solver/tests/benchmark.py`

## Practical guidance for future agents

When modifying this repository:

- treat `src/pytorch_sparse_solver/` as the source of truth over README prose
- expect Module A to be the easiest path for local work
- do not assume CUDA, `pyamgx`, or cuDSS are installed
- use `check_module_a_available`, `check_module_b_available`, and `check_module_c_available` before backend-specific work
- remember Module C wants CSR matrices
- remember Module A also supports callable operators and some PyTree-oriented logic
- keep public API compatibility in mind because `__init__.py` re-exports the main interface
- prefer extending the normal public APIs with automatic differentiation behavior rather than forcing users onto separate `*_differentiable` names
- if you touch Module B, explicitly distinguish between AMG itself and Krylov methods that happen to be implemented through AMGX
- after changing solver behavior, rerun `test_gpu_validation.py` on CUDA rather than relying only on CPU-oriented scripts

## Known inconsistencies / gotchas

There are a few metadata mismatches in the repository:

- `README.md` says Python 3.11+, while `pyproject.toml` says `>=3.9`
- `pyproject.toml` version is `0.1.1`, while `src/pytorch_sparse_solver/__init__.py` reports `1.0.0`
- docs describe "Module D" as the unified interface, but there is no physical `module_d/` package; that role is implemented by `solver.py`
- some older comments / docs still imply that only special `*_differentiable` entry points are differentiable; the current code supports automatic implicit-diff wrapping in the standard Module A APIs for 2D tensor systems
- some older docs loosely describe Module B as "AMG-preconditioned CG/BiCGStab/GMRES"; the code now also exposes a real AMG entry point, `amgx_amg`

If you need authoritative behavior, prefer the actual code paths over the marketing/documentation wording.

## Environment note

This repository is not self-contained unless PyTorch is installed. Optional backends need additional system dependencies beyond Python packages.

If imports fail locally, first verify:

- `torch` is installed
- CUDA availability matches the backend being exercised
- `pyamgx` exists for Module B
- cuDSS-enabled PyTorch exists for Module C

## Verified state on this machine

The repository has already been validated in a dedicated conda environment named `psl-verify-pt27` on GPU `6`.

Important verified facts:

- source-built PyTorch `2.7.0a0+git1341794` with CUDA `12.8` and working `torch.sparse.spsolve`
- `pyamgx` installed and working in that environment
- reported backend availability there: `module_a=True`, `module_b=True`, `module_c=True`
- GPU integration validation passed there

Most recent verified outcomes:

- standard Module A `cg` / `bicgstab` / `gmres` forward solves work on GPU
- standard Module A `cg` / `bicgstab` / `gmres` also backpropagate on GPU when inputs require gradients
- Module B forward solves work on GPU for `amgx_cg`, `amgx_bicgstab`, `amgx_gmres`, and `amgx_amg`
- Module B backward passes work on GPU for RHS gradients and public matrix gradients
- Module C forward and backward work on GPU
- unified interface works for explicit backends, `backend="auto"`, `method="direct"`, and `method="amg"`
- backend-specific LDC/FVM examples for Modules A/B/C/D all complete on GPU

For exact commands and logs, see:

- `Logger/2026-03-13_psl_verify_install.md`
- `Logger/2026-03-13_psl_verify_results.md`
- `Logger/2026-03-13_psl_verify_environment.yml`
