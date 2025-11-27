"""
Module C: cuDSS GPU-Accelerated Direct Sparse Solver

This module provides GPU-accelerated direct sparse linear system solver using
NVIDIA's cuDSS library through PyTorch's torch.sparse.spsolve interface.

Key Features:
- Direct solver (LU factorization) - no iteration required
- GPU-accelerated via NVIDIA cuDSS library
- Supports sparse CSR format matrices
- Automatic differentiation support via custom backward pass

Requirements:
- PyTorch >= 2.7 compiled with USE_CUDSS=1
- NVIDIA GPU with CUDA support
- cuDSS library installed

Example:
    >>> import torch
    >>> from pytorch_sparse_solver.module_c import cudss_spsolve, cudss_available
    >>>
    >>> if cudss_available():
    >>>     # Create a sparse CSR matrix
    >>>     n = 1000
    >>>     A_csr = create_sparse_csr_matrix(n)  # your sparse matrix
    >>>     b = torch.rand(n, dtype=torch.float64, device='cuda')
    >>>
    >>>     # Solve using cuDSS
    >>>     x = cudss_spsolve(A_csr, b)
"""

from .cudss_solver import (
    cudss_spsolve,
    cudss_available,
    DifferentiableCuDSSSolver,
)

__all__ = [
    'cudss_spsolve',
    'cudss_available',
    'DifferentiableCuDSSSolver',
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'Litianyu141'
__license__ = 'Apache-2.0'
