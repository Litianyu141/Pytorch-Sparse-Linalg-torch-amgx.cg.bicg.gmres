"""
Module B: PyAMGX GPU-Accelerated Sparse Solvers with Automatic Differentiation

This module provides GPU-accelerated sparse linear system solvers using NVIDIA's
AMGX library with full automatic differentiation support through PyTorch's autograd.

Key Features:
- Forward pass: Uses pyamgx for high-performance GPU solving
- Backward pass: Custom gradient computation using implicit function theorem
- Multiple solvers: CG, BiCGStab, GMRES with AMG preconditioner
- Automatic configuration: Optimized settings for different matrix types

Requirements:
- NVIDIA GPU with CUDA support
- pyamgx library installed (https://github.com/shwina/pyamgx)
- AMGX library installed (https://github.com/NVIDIA/AMGX)

Example:
    >>> import torch
    >>> from pytorch_sparse_solver.module_b import amgx_cg, amgx_bicgstab, amgx_gmres
    >>>
    >>> # Create a sparse test system
    >>> n = 1000
    >>> A = torch.rand(n, n, dtype=torch.float64, device='cuda')
    >>> A = A + A.T + torch.eye(n, dtype=torch.float64, device='cuda') * 2
    >>> b = torch.rand(n, dtype=torch.float64, device='cuda')
    >>>
    >>> # Solve using AMGX CG
    >>> x = amgx_cg(A, b, tol=1e-8)
"""

# Lazy imports to avoid ImportError when pyamgx is not installed
_torch_amgx = None

def _lazy_import():
    """Lazily import torch_amgx module."""
    global _torch_amgx
    if _torch_amgx is None:
        from . import torch_amgx as _ta
        _torch_amgx = _ta
    return _torch_amgx

def amgx_cg(A, b, tol=1e-8, maxiter=1000):
    """
    Solve Ax = b using AMGX CG with AMG preconditioner.

    Parameters:
        A: Sparse or dense matrix (torch.Tensor)
        b: Right-hand side vector (torch.Tensor)
        tol: Convergence tolerance (default: 1e-8)
        maxiter: Maximum iterations (default: 1000)

    Returns:
        x: Solution vector (torch.Tensor)
    """
    mod = _lazy_import()
    return mod.amgx_cg(A, b, tol=tol, maxiter=maxiter)

def amgx_bicgstab(A, b, tol=1e-8, maxiter=1000):
    """
    Solve Ax = b using AMGX BiCGStab with AMG preconditioner.

    Parameters:
        A: Sparse or dense matrix (torch.Tensor)
        b: Right-hand side vector (torch.Tensor)
        tol: Convergence tolerance (default: 1e-8)
        maxiter: Maximum iterations (default: 1000)

    Returns:
        x: Solution vector (torch.Tensor)
    """
    mod = _lazy_import()
    return mod.amgx_bicgstab(A, b, tol=tol, maxiter=maxiter)

def amgx_gmres(A, b, tol=1e-8, maxiter=1000):
    """
    Solve Ax = b using AMGX GMRES with AMG preconditioner.

    Parameters:
        A: Sparse or dense matrix (torch.Tensor)
        b: Right-hand side vector (torch.Tensor)
        tol: Convergence tolerance (default: 1e-8)
        maxiter: Maximum iterations (default: 1000)

    Returns:
        x: Solution vector (torch.Tensor)
    """
    mod = _lazy_import()
    return mod.amgx_gmres(A, b, tol=tol, maxiter=maxiter)

def DifferentiableAMGXSolver():
    """Get the DifferentiableAMGXSolver class."""
    mod = _lazy_import()
    return mod.DifferentiableAMGXSolver

def AMGXManager():
    """Get the AMGXManager class."""
    mod = _lazy_import()
    return mod.AMGXManager

__all__ = [
    'amgx_cg',
    'amgx_bicgstab',
    'amgx_gmres',
    'DifferentiableAMGXSolver',
    'AMGXManager',
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'Litianyu141'
__license__ = 'Apache-2.0'
