"""
PyTorch Sparse Solver - A Modular Sparse Linear System Solver Library

This package provides multiple backends for solving sparse linear systems Ax = b:

- **Module A**: JAX-style iterative solvers (CG, BiCGStab, GMRES) - pure PyTorch
- **Module B**: PyAMGX GPU-accelerated solvers with AMG preconditioner
- **Module C**: cuDSS direct solver via torch.sparse.spsolve

Each module can be installed independently, making this library flexible for
different deployment scenarios.

Quick Start:
    >>> from pytorch_sparse_solver import SparseSolver, solve
    >>>
    >>> # Using the SparseSolver class
    >>> solver = SparseSolver()
    >>> x, result = solver.solve(A, b, method='cg')
    >>>
    >>> # Or use convenience functions
    >>> x, result = solve(A, b, method='cg')

Module-specific Usage:
    >>> # Module A (always available if PyTorch is installed)
    >>> from pytorch_sparse_solver.module_a import cg, bicgstab, gmres
    >>> x, info = cg(A, b, tol=1e-6)
    >>>
    >>> # Module B (requires pyamgx)
    >>> from pytorch_sparse_solver.module_b import amgx_cg
    >>> x = amgx_cg(A, b, tol=1e-8)
    >>>
    >>> # Module C (requires PyTorch with cuDSS)
    >>> from pytorch_sparse_solver.module_c import cudss_spsolve
    >>> x = cudss_spsolve(A_csr, b)

Check Module Availability:
    >>> from pytorch_sparse_solver.utils import get_available_backends
    >>> print(get_available_backends())
    {'module_a': True, 'module_b': False, 'module_c': False}
"""

__version__ = '1.0.0'
__author__ = 'Litianyu141'
__license__ = 'Apache-2.0'

# Import main solver interface
from .solver import (
    SparseSolver,
    SolverResult,
    SolverMethod,
    SolverBackend,
    solve,
    cg,
    bicgstab,
    gmres,
    direct_solve,
)

# Import availability checking utilities
from .utils.availability import (
    check_module_a_available,
    check_module_b_available,
    check_module_c_available,
    get_available_backends,
    print_availability_report,
)

# Import matrix utilities
from .utils.matrix_utils import (
    dense_to_sparse_csr,
    sparse_coo_to_csr,
    ensure_sparse_format,
    create_tridiagonal_sparse_coo,
    create_poisson_2d_sparse_coo,
    compute_residual,
    compute_relative_residual,
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',

    # Main solver interface
    'SparseSolver',
    'SolverResult',
    'SolverMethod',
    'SolverBackend',
    'solve',
    'cg',
    'bicgstab',
    'gmres',
    'direct_solve',

    # Availability checking
    'check_module_a_available',
    'check_module_b_available',
    'check_module_c_available',
    'get_available_backends',
    'print_availability_report',

    # Matrix utilities
    'dense_to_sparse_csr',
    'sparse_coo_to_csr',
    'ensure_sparse_format',
    'create_tridiagonal_sparse_coo',
    'create_poisson_2d_sparse_coo',
    'compute_residual',
    'compute_relative_residual',
]


def _lazy_import_module_a():
    """Lazy import for Module A."""
    from . import module_a
    return module_a


def _lazy_import_module_b():
    """Lazy import for Module B."""
    from . import module_b
    return module_b


def _lazy_import_module_c():
    """Lazy import for Module C."""
    from . import module_c
    return module_c
