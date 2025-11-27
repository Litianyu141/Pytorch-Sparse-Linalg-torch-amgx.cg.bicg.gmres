#!/usr/bin/env python3
# Copyright 2025 Litianyu141
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Differentiable cuDSS Direct Solver for PyTorch

This module provides GPU-accelerated direct sparse linear system solvers using
NVIDIA's cuDSS library through torch.sparse.spsolve with full automatic
differentiation support.

Requirements:
- PyTorch >= 2.7 compiled with USE_CUDSS=1
- NVIDIA GPU with CUDA support
- cuDSS library installed

Key Features:
- Forward pass: Uses torch.sparse.spsolve for direct LU factorization solving
- Backward pass: Custom gradient computation using implicit function theorem
- Automatic differentiation: Full autograd support for optimization problems
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple, Optional, Union
import warnings
from functools import lru_cache


@lru_cache(maxsize=1)
def cudss_available() -> bool:
    """
    Check if cuDSS is available for sparse direct solving.

    Returns:
        bool: True if cuDSS is available and working
    """
    try:
        if not torch.cuda.is_available():
            return False

        if not hasattr(torch.sparse, 'spsolve'):
            return False

        # Try a simple test solve
        device = 'cuda'
        row_indices = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)
        col_indices = torch.tensor([0, 1, 1, 2, 0, 2], dtype=torch.long, device=device)
        values = torch.tensor([2.0, 1.0, 3.0, 1.0, 1.0, 4.0], dtype=torch.float32, device=device)
        indices = torch.stack([row_indices, col_indices])
        sparse_coo = torch.sparse_coo_tensor(indices, values, (3, 3), dtype=torch.float32, device=device)
        sparse_csr = sparse_coo.to_sparse_csr()
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)

        _ = torch.sparse.spsolve(sparse_csr, b)
        return True

    except RuntimeError as e:
        # cuDSS not available or not working
        return False
    except Exception as e:
        warnings.warn(f"cuDSS availability check failed: {e}")
        return False


class DifferentiableCuDSSSolver(Function):
    """
    PyTorch autograd Function that wraps cuDSS solver with custom gradient computation.

    Forward pass: Solves Ax = b using torch.sparse.spsolve (cuDSS backend)
    Backward pass: Computes gradients using the implicit function theorem

    For the linear system Ax = b, if we have a loss L(x), then:
    - dL/dA = -x * (A^(-T) * dL/dx)^T = -x * v^T
    - dL/db = A^(-T) * dL/dx = v
    """

    @staticmethod
    def forward(ctx, A_csr: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Solve Ax = b using cuDSS via torch.sparse.spsolve

        Args:
            ctx: PyTorch context for saving tensors for backward pass
            A_csr: Sparse CSR matrix (n, n)
            b: Right-hand side vector (n,)

        Returns:
            x: Solution vector (n,)
        """
        if not A_csr.is_sparse_csr:
            raise ValueError("Matrix A must be in sparse CSR format")

        # Solve using torch.sparse.spsolve
        x = torch.sparse.spsolve(A_csr, b)

        # Save for backward pass
        ctx.save_for_backward(A_csr, b, x)

        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Backward pass: Compute gradients using implicit function theorem

        For Ax = b, we have:
        - dL/dA = -x * v^T where v = A^(-T) * dL/dx
        - dL/db = v = A^(-T) * dL/dx

        Args:
            ctx: PyTorch context with saved tensors
            grad_output: Gradient w.r.t. output x (dL/dx)

        Returns:
            Gradients w.r.t. (A_csr, b)
        """
        A_csr, b, x = ctx.saved_tensors

        # Solve A^T * v = grad_output
        # First, get A^T in CSR format
        A_csc = A_csr.to_sparse_csc()

        # For A^T, we need to transpose: swap crow_indices and ccol_indices
        A_T_csr = torch.sparse_csr_tensor(
            crow_indices=A_csc.ccol_indices(),
            col_indices=A_csc.row_indices(),
            values=A_csc.values(),
            size=(A_csr.shape[1], A_csr.shape[0])
        )

        # Solve A^T * v = grad_output
        v = torch.sparse.spsolve(A_T_csr, grad_output)

        # Gradient w.r.t. b: dL/db = v
        grad_b = v

        # Gradient w.r.t. A values:
        # dL/dA[i,j] = -x[j] * v[i] for non-zero entries
        # Since we're dealing with CSR, we need to compute this for each non-zero
        crow_indices = A_csr.crow_indices()
        col_indices = A_csr.col_indices()

        # Compute gradient for each non-zero element
        grad_A_values = torch.zeros_like(A_csr.values())

        for row in range(A_csr.shape[0]):
            start = crow_indices[row]
            end = crow_indices[row + 1]
            cols = col_indices[start:end]
            grad_A_values[start:end] = -v[row] * x[cols]

        # Create sparse gradient tensor (same structure as A)
        grad_A = torch.sparse_csr_tensor(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=grad_A_values,
            size=A_csr.shape
        )

        return grad_A, grad_b


def cudss_spsolve(
    A: torch.Tensor,
    b: torch.Tensor,
    check_input: bool = True
) -> torch.Tensor:
    """
    Solve sparse linear system Ax = b using cuDSS direct solver.

    This function uses NVIDIA's cuDSS library through PyTorch's torch.sparse.spsolve
    interface for GPU-accelerated direct solving via LU factorization.

    Args:
        A: Sparse matrix (CSR, COO, or dense format). Will be converted to CSR.
        b: Right-hand side vector
        check_input: Whether to validate inputs (default: True)

    Returns:
        x: Solution vector

    Raises:
        RuntimeError: If cuDSS is not available
        ValueError: If input validation fails

    Example:
        >>> import torch
        >>> from pytorch_sparse_solver.module_c import cudss_spsolve
        >>>
        >>> # Create a sparse CSR matrix
        >>> row = torch.tensor([0, 0, 1, 2, 2])
        >>> col = torch.tensor([0, 2, 1, 0, 2])
        >>> data = torch.tensor([1., 2., 3., 4., 5.])
        >>> A_csr = torch.sparse_csr_tensor(
        ...     crow_indices=torch.tensor([0, 2, 3, 5]),
        ...     col_indices=col,
        ...     values=data,
        ...     size=(3, 3)
        ... ).cuda()
        >>> b = torch.tensor([1., 2., 3.]).cuda()
        >>>
        >>> x = cudss_spsolve(A_csr, b)
    """
    if not cudss_available():
        raise RuntimeError(
            "cuDSS is not available. Please ensure:\n"
            "1. PyTorch >= 2.7 is compiled with USE_CUDSS=1\n"
            "2. NVIDIA GPU with CUDA support is available\n"
            "3. cuDSS library is installed\n"
            "See README.md for installation instructions."
        )

    if check_input:
        # Ensure tensors are on CUDA
        if not A.is_cuda:
            A = A.cuda()
        if not b.is_cuda:
            b = b.cuda()

        # Ensure A is in CSR format
        if not A.is_sparse_csr:
            if A.is_sparse:
                A = A.coalesce().to_sparse_csr()
            else:
                # Dense to sparse
                A = A.to_sparse_coo().to_sparse_csr()

        # Validate shapes
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix A must be square, got shape {A.shape}")
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has {A.shape[0]} rows, b has {b.shape[0]} elements")

    # Use differentiable solver if gradients are needed
    if A.requires_grad or b.requires_grad:
        return DifferentiableCuDSSSolver.apply(A, b)
    else:
        # Direct solve without gradient tracking
        return torch.sparse.spsolve(A, b)


class CuDSSSparseSolver(nn.Module):
    """
    PyTorch Module wrapper for cuDSS sparse solver.

    This class provides a convenient Module interface for the cuDSS solver,
    making it easy to integrate into neural network architectures.

    Example:
        >>> solver = CuDSSSparseSolver()
        >>> x = solver(A_csr, b)
    """

    def __init__(self, check_available: bool = True):
        """
        Initialize the cuDSS solver module.

        Args:
            check_available: Whether to check cuDSS availability on init
        """
        super().__init__()
        if check_available and not cudss_available():
            warnings.warn(
                "cuDSS is not available. Solver will raise RuntimeError when called."
            )

    def forward(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Solve Ax = b using cuDSS.

        Args:
            A: Sparse matrix (will be converted to CSR if needed)
            b: Right-hand side vector

        Returns:
            x: Solution vector
        """
        return cudss_spsolve(A, b)

    def extra_repr(self) -> str:
        return f"cudss_available={cudss_available()}"


def batch_cudss_spsolve(
    A_list: list,
    b_batch: torch.Tensor
) -> torch.Tensor:
    """
    Solve multiple sparse linear systems in batch.

    Note: Currently solves sequentially. Future versions may support
    true batched solving if cuDSS adds batch support.

    Args:
        A_list: List of sparse CSR matrices
        b_batch: Batch of right-hand side vectors (batch_size, n)

    Returns:
        x_batch: Batch of solution vectors (batch_size, n)
    """
    if len(A_list) != b_batch.shape[0]:
        raise ValueError(
            f"Number of matrices ({len(A_list)}) must match batch size ({b_batch.shape[0]})"
        )

    results = []
    for A, b in zip(A_list, b_batch):
        x = cudss_spsolve(A, b)
        results.append(x)

    return torch.stack(results)
