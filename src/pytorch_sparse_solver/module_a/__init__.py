"""
Module A: JAX-style Iterative Sparse Solvers for PyTorch

This module provides pure PyTorch implementations of iterative linear solvers:
- CG (Conjugate Gradient) - for symmetric positive definite matrices
- BiCGStab (Bi-Conjugate Gradient Stabilized) - for general non-symmetric matrices
- GMRES (Generalized Minimal Residual) - for general matrices with restart capability

The implementation follows JAX's scipy.sparse.linalg API closely, allowing users
familiar with JAX to easily transition to PyTorch.

Features:
- Pure PyTorch implementation, no external dependencies
- GPU acceleration via PyTorch's CUDA support
- Support for function-based linear operators (matrix-free methods)
- High precision computation (float64 by default)

Example:
    >>> import torch
    >>> from pytorch_sparse_solver.module_a import cg, bicgstab, gmres
    >>>
    >>> # Create a simple SPD matrix
    >>> n = 100
    >>> A = torch.randn(n, n, dtype=torch.float64)
    >>> A = A @ A.T + torch.eye(n, dtype=torch.float64) * 10
    >>> b = torch.randn(n, dtype=torch.float64)
    >>>
    >>> # Solve using CG
    >>> x, info = cg(A, b, tol=1e-6)
    >>> print(f"Converged: {info == 0}")
"""

from .torch_sparse_linalg import cg, bicgstab, gmres
from .torch_tree_util import tree_leaves, tree_map, tree_flatten, tree_unflatten, Partial

__all__ = [
    'cg',
    'bicgstab',
    'gmres',
    'tree_leaves',
    'tree_map',
    'tree_flatten',
    'tree_unflatten',
    'Partial',
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'Litianyu141'
__license__ = 'Apache-2.0'
