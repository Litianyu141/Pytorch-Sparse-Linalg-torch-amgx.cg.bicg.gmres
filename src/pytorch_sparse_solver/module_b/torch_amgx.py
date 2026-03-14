#!/usr/bin/env python3
"""
Differentiable AMGX Solver for PyTorch

This module provides GPU-accelerated sparse linear system solvers using NVIDIA's AMGX library
with full automatic differentiation support through PyTorch's autograd system.

Key Features:
- Forward pass: Uses pyamgx for high-performance GPU solving
- Backward pass: Custom gradient computation using implicit function theorem
- Multiple solvers: CG, BiCGStab, GMRES
- Automatic configuration: Optimized settings for different matrix types

Author: Augment Agent
Date: 2025-06-24
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from scipy.sparse import coo_matrix, csr_matrix
from typing import Tuple, Optional

# Global AMGX state management
class AMGXManager:
    _initialized = False

    @classmethod
    def initialize(cls):
        if not cls._initialized:
            import pyamgx
            pyamgx.initialize()
            cls._initialized = True

    @classmethod
    def finalize(cls):
        if cls._initialized:
            try:
                import pyamgx
                pyamgx.finalize()
            except:
                pass
            cls._initialized = False


def _build_solver_config(solver_type: str, tolerance: float, max_iters: int, n: int) -> dict:
    """Create a conservative AMGX config that works across the supported solvers."""
    if solver_type.upper() == "AMG":
        return {
            "config_version": 2,
            "determinism_flag": 1,
            "exception_handling": 1,
            "solver": {
                "solver": "AMG",
                "scope": "main",
                "algorithm": "AGGREGATION",
                "selector": "SIZE_4",
                "smoother": "JACOBI_L1",
                "presweeps": 0,
                "postsweeps": 3,
                "print_grid_stats": 0,
                "monitor_residual": 1,
                "obtain_timings": 0,
                "max_iters": max_iters,
                "convergence": "RELATIVE_INI",
                "max_levels": 50,
                "cycle": "CG",
                "tolerance": tolerance,
                "norm": "L2",
                "use_scalar_norm": 1,
            },
        }

    config_dict = {
        "config_version": 2,
        "determinism_flag": 1,
        "exception_handling": 1,
        "solver": {
            "monitor_residual": 1,
            "solver": solver_type.upper(),
            "scope": "main",
            "max_iters": max_iters,
            "tolerance": tolerance,
            "convergence": "RELATIVE_INI_CORE",
            "norm": "L2",
            "use_scalar_norm": 1,
            "preconditioner": {
                "scope": "amg_solver",
                "solver": "NOSOLVER",
            },
        },
    }

    if solver_type.upper() == "GMRES":
        # Keep restart positive for small systems and bounded for larger ones.
        config_dict["solver"]["gmres_n_restart"] = max(10, min(30, n))

    return config_dict


def _solve_with_amgx_csr(
    A_csr: csr_matrix,
    b_np: np.ndarray,
    solver_type: str,
    tolerance: float,
    max_iters: int,
) -> np.ndarray:
    """Solve a CSR system with AMGX and return the NumPy solution vector."""
    import pyamgx

    n, _ = A_csr.shape
    cfg = resources = A_amgx = x_amgx = b_amgx = solver = None

    AMGXManager.initialize()

    try:
        config_dict = _build_solver_config(solver_type, tolerance, max_iters, n)

        cfg = pyamgx.Config()
        cfg.create_from_dict(config_dict)

        resources = pyamgx.Resources()
        resources.create_simple(cfg)

        A_amgx = pyamgx.Matrix()
        A_amgx.create(resources, mode='dDDI')
        A_amgx.upload_CSR(A_csr)

        x_amgx = pyamgx.Vector()
        x_amgx.create(resources, mode='dDDI')
        x_amgx.upload(np.zeros(n, dtype=np.float64))

        b_amgx = pyamgx.Vector()
        b_amgx.create(resources, mode='dDDI')
        b_amgx.upload(b_np.astype(np.float64, copy=False))

        solver = pyamgx.Solver()
        solver.create(resources, cfg)
        solver.setup(A_amgx)
        solver.solve(b_amgx, x_amgx)

        return x_amgx.download()
    finally:
        if solver is not None:
            solver.destroy()
        if A_amgx is not None:
            A_amgx.destroy()
        if x_amgx is not None:
            x_amgx.destroy()
        if b_amgx is not None:
            b_amgx.destroy()
        if resources is not None:
            resources.destroy()
        if cfg is not None:
            cfg.destroy()


def _tensor_to_scipy_csr(A: torch.Tensor) -> csr_matrix:
    """Convert dense/COO/CSR torch tensors to SciPy CSR without keeping autograd edges."""
    if getattr(A, "is_sparse_csr", False):
        crow = A.crow_indices().detach().cpu().numpy().astype(np.int32)
        col = A.col_indices().detach().cpu().numpy().astype(np.int32)
        values = A.values().detach().cpu().numpy().astype(np.float64)
        return csr_matrix((values, col, crow), shape=A.shape)

    if A.is_sparse:
        A_coo = A.coalesce()
        indices = A_coo.indices().detach().cpu().numpy()
        values = A_coo.values().detach().cpu().numpy().astype(np.float64)
        return coo_matrix((values, (indices[0], indices[1])), shape=A.shape).tocsr()

    return csr_matrix(A.detach().cpu().numpy().astype(np.float64))

class DifferentiableAMGXSolver(Function):
    """
    PyTorch autograd Function that wraps AMGX solver with custom gradient computation.
    
    Forward pass: Solves Ax = b using AMGX
    Backward pass: Computes gradients using the implicit function theorem
    
    For the linear system Ax = b, if we have a loss L(x), then:
    - dL/dA = -x * (dL/dx)^T (outer product)
    - dL/db = A^(-T) * (dL/dx) where A^(-T) is the transpose of A^(-1)
    """
    
    @staticmethod
    def forward(ctx, A_values: torch.Tensor, A_indices: torch.Tensor, 
                A_indptr: torch.Tensor, b: torch.Tensor, 
                matrix_shape: Tuple[int, int],
                solver_type: str = "BICGSTAB",
                tolerance: float = 1e-12,
                max_iters: int = 1000) -> torch.Tensor:
        """
        Forward pass: Solve Ax = b using AMGX
        
        Args:
            ctx: PyTorch context for saving tensors for backward pass
            A_values: CSR matrix values (nnz,)
            A_indices: CSR matrix column indices (nnz,)
            A_indptr: CSR matrix row pointers (n+1,)
            b: Right-hand side vector (n,)
            matrix_shape: Shape of matrix A (n, n)
            solver_type: AMGX solver type ("CG", "BICGSTAB", "GMRES")
            tolerance: Convergence tolerance
            max_iters: Maximum iterations
            
        Returns:
            x: Solution vector (n,)
        """
        import pyamgx
        
        n, _ = matrix_shape
        device = A_values.device
        
        # Convert to CPU numpy arrays for AMGX
        A_values_np = A_values.detach().cpu().numpy().astype(np.float64)
        A_indices_np = A_indices.detach().cpu().numpy().astype(np.int32)
        A_indptr_np = A_indptr.detach().cpu().numpy().astype(np.int32)
        b_np = b.detach().cpu().numpy().astype(np.float64)
        
        # Initialize AMGX
        AMGXManager.initialize()

        # Create AMGX configuration
        config_dict = {
            "config_version": 2,
            "determinism_flag": 1,
            "exception_handling": 1,
            "solver": {
                "monitor_residual": 1,
                "solver": solver_type.upper(),
                "max_iters": max_iters,
                "tolerance": tolerance,
                "convergence": "RELATIVE_INI_CORE"
            }
        }
        
        # Add restart for GMRES
        if solver_type.upper() == "GMRES":
            config_dict["solver"]["gmres_n_restart"] = min(30, n // 10, 100)

        # Create AMGX configuration
        cfg = pyamgx.Config()
        cfg.create_from_dict(config_dict)

        # Create resources and matrix
        resources = pyamgx.Resources()
        resources.create_simple(cfg)

        # Create AMGX matrix and vectors
        A_amgx = pyamgx.Matrix()
        A_amgx.create(resources, mode='dDDI')

        # Create scipy CSR matrix for upload
        A_csr = csr_matrix((A_values_np, A_indices_np, A_indptr_np), shape=matrix_shape)
        A_amgx.upload_CSR(A_csr)
        
        x_amgx = pyamgx.Vector()
        x_amgx.create(resources, mode='dDDI')
        x_amgx.upload(np.zeros(n, dtype=np.float64))
        
        b_amgx = pyamgx.Vector()
        b_amgx.create(resources, mode='dDDI')
        b_amgx.upload(b_np)
        
        # Create solver and solve
        solver = pyamgx.Solver()
        solver.create(resources, cfg)
        solver.setup(A_amgx)
        solver.solve(b_amgx, x_amgx)
        
        # Download solution
        x_np = x_amgx.download()
        
        # Cleanup AMGX objects
        solver.destroy()
        A_amgx.destroy()
        x_amgx.destroy()
        b_amgx.destroy()
        resources.destroy()
        cfg.destroy()
        
        # Convert back to PyTorch tensor
        x = torch.from_numpy(x_np).to(device=device, dtype=A_values.dtype)
        
        # Save tensors for backward pass
        ctx.save_for_backward(A_values, A_indices, A_indptr, b, x)
        ctx.matrix_shape = matrix_shape
        ctx.solver_config = (solver_type, tolerance, max_iters)
        
        return x
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass: Compute gradients using implicit function theorem
        
        For Ax = b, we have:
        - dL/dA = -x * (A^(-T) * dL/dx)^T = -x * v^T where v = A^(-T) * dL/dx
        - dL/db = A^(-T) * dL/dx = v
        
        Args:
            ctx: PyTorch context with saved tensors
            grad_output: Gradient w.r.t. output x (dL/dx)
            
        Returns:
            Gradients w.r.t. (A_values, A_indices, A_indptr, b, matrix_shape, solver_type, tolerance, max_iters)
        """
        import pyamgx
        
        A_values, A_indices, A_indptr, b, x = ctx.saved_tensors
        matrix_shape = ctx.matrix_shape
        solver_type, tolerance, max_iters = ctx.solver_config
        n, _ = matrix_shape
        device = A_values.device
        
        # Convert to numpy for AMGX (solve A^T * v = grad_output)
        A_values_np = A_values.detach().cpu().numpy().astype(np.float64)
        A_indices_np = A_indices.detach().cpu().numpy().astype(np.int32)
        A_indptr_np = A_indptr.detach().cpu().numpy().astype(np.int32)
        grad_output_np = grad_output.detach().cpu().numpy().astype(np.float64)
        
        # Initialize AMGX for transpose solve
        AMGXManager.initialize()
        
        # Create configuration for transpose solve
        config_dict = {
            "config_version": 2,
            "determinism_flag": 1,
            "exception_handling": 1,
            "solver": {
                "monitor_residual": 1,
                "solver": solver_type.upper(),
                "max_iters": max_iters,
                "tolerance": tolerance,
                "convergence": "RELATIVE_INI_CORE"
            }
        }
        
        if solver_type.upper() == "GMRES":
            config_dict["solver"]["gmres_n_restart"] = min(30, n // 10, 100)
        
        cfg = pyamgx.Config()
        cfg.create_from_dict(config_dict)
        
        resources = pyamgx.Resources()
        resources.create_simple(cfg)
        
        # Create transpose matrix A^T
        A_csr = csr_matrix((A_values_np, A_indices_np, A_indptr_np), shape=matrix_shape)
        A_T_csr = A_csr.transpose()
        
        A_T_amgx = pyamgx.Matrix()
        A_T_amgx.create(resources, mode='dDDI')
        A_T_amgx.upload_CSR(A_T_csr)
        
        v_amgx = pyamgx.Vector()
        v_amgx.create(resources, mode='dDDI')
        v_amgx.upload(np.zeros(n, dtype=np.float64))
        
        grad_amgx = pyamgx.Vector()
        grad_amgx.create(resources, mode='dDDI')
        grad_amgx.upload(grad_output_np)
        
        # Solve A^T * v = grad_output
        solver = pyamgx.Solver()
        solver.create(resources, cfg)
        solver.setup(A_T_amgx)
        solver.solve(grad_amgx, v_amgx)
        
        # Download solution
        v_np = v_amgx.download()
        
        # Cleanup
        solver.destroy()
        A_T_amgx.destroy()
        v_amgx.destroy()
        grad_amgx.destroy()
        resources.destroy()
        cfg.destroy()
        
        # Convert back to PyTorch
        v = torch.from_numpy(v_np).to(device=device, dtype=A_values.dtype)
        x_cpu = x.cpu()
        
        # Compute gradients
        # dL/dA = -x * v^T (outer product), but we only need gradients w.r.t. non-zero elements
        grad_A_values = -x_cpu[A_indices_np] * v.cpu()[A_indices_np]
        
        # dL/db = v
        grad_b = v
        
        return grad_A_values, None, None, grad_b, None, None, None, None


class DifferentiableAMGXWrapper(Function):
    """
    Public autograd wrapper that preserves gradients w.r.t. the original input matrix.

    Unlike the lower-level CSR-values function above, this wrapper returns gradients
    that match the shape/layout of the user's original `A` tensor.
    """

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        b: torch.Tensor,
        solver_type: str,
        tolerance: float,
        max_iters: int,
    ) -> torch.Tensor:
        A_csr = _tensor_to_scipy_csr(A)
        b_np = b.detach().cpu().numpy().astype(np.float64)
        x_np = _solve_with_amgx_csr(A_csr, b_np, solver_type, tolerance, max_iters)

        x = torch.from_numpy(x_np).to(device=b.device, dtype=b.dtype)

        ctx.save_for_backward(A, x)
        ctx.matrix_shape = A.shape
        ctx.solver_config = (solver_type, tolerance, max_iters)
        ctx.csr_components = (
            A_csr.data.astype(np.float64, copy=True),
            A_csr.indices.astype(np.int32, copy=True),
            A_csr.indptr.astype(np.int32, copy=True),
        )

        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        A, x = ctx.saved_tensors
        solver_type, tolerance, max_iters = ctx.solver_config
        data, indices, indptr = ctx.csr_components

        A_csr = csr_matrix((data, indices, indptr), shape=ctx.matrix_shape)
        grad_output_np = grad_output.detach().cpu().numpy().astype(np.float64)
        v_np = _solve_with_amgx_csr(A_csr.transpose().tocsr(), grad_output_np, solver_type, tolerance, max_iters)
        v = torch.from_numpy(v_np).to(device=grad_output.device, dtype=grad_output.dtype)

        grad_A = None
        if ctx.needs_input_grad[0]:
            if getattr(A, "is_sparse_csr", False):
                crow = A.crow_indices()
                col = A.col_indices()
                grad_values = torch.empty_like(A.values())
                for row in range(A.shape[0]):
                    start = crow[row].item()
                    end = crow[row + 1].item()
                    cols = col[start:end]
                    grad_values[start:end] = -v[row] * x[cols]
                grad_A = torch.sparse_csr_tensor(crow, col, grad_values, size=A.shape)
            elif A.is_sparse:
                A_coo = A.coalesce()
                indices_t = A_coo.indices()
                rows = indices_t[0]
                cols = indices_t[1]
                grad_values = -v[rows] * x[cols]
                grad_A = torch.sparse_coo_tensor(indices_t, grad_values, size=A.shape).coalesce()
            else:
                grad_A = -torch.outer(v, x)

        grad_b = v if ctx.needs_input_grad[1] else None
        return grad_A, grad_b, None, None, None

# Convenience functions for different solver types
def amgx_cg(A: torch.Tensor, b: torch.Tensor, tol: float = 1e-8, maxiter: int = 1000) -> torch.Tensor:
    """Solve Ax = b using AMGX CG solver"""
    return _solve_amgx(A, b, "CG", tol, maxiter)

def amgx_bicgstab(A: torch.Tensor, b: torch.Tensor, tol: float = 1e-8, maxiter: int = 1000) -> torch.Tensor:
    """Solve Ax = b using AMGX BiCGStab solver"""
    return _solve_amgx(A, b, "BICGSTAB", tol, maxiter)

def amgx_gmres(A: torch.Tensor, b: torch.Tensor, tol: float = 1e-8, maxiter: int = 1000) -> torch.Tensor:
    """Solve Ax = b using AMGX GMRES solver"""
    return _solve_amgx(A, b, "GMRES", tol, maxiter)


def amgx_amg(A: torch.Tensor, b: torch.Tensor, tol: float = 1e-8, maxiter: int = 1000) -> torch.Tensor:
    """Solve Ax = b using the AMGX AMG solver directly."""
    return _solve_amgx(A, b, "AMG", tol, maxiter)

def _solve_amgx(A: torch.Tensor, b: torch.Tensor, solver_type: str, tol: float, maxiter: int) -> torch.Tensor:
    """Internal function to solve using AMGX"""
    if not A.is_cuda or not b.is_cuda:
        raise ValueError("AMGX solvers require CUDA tensors for both A and b")

    if A.requires_grad or b.requires_grad:
        return DifferentiableAMGXWrapper.apply(A, b, solver_type, tol, maxiter)

    A_csr = _tensor_to_scipy_csr(A)
    b_np = b.detach().cpu().numpy().astype(np.float64)
    x_np = _solve_with_amgx_csr(A_csr, b_np, solver_type, tol, maxiter)
    return torch.from_numpy(x_np).to(device=b.device, dtype=b.dtype)
