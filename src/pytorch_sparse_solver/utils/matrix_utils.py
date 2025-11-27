"""
Matrix utility functions for pytorch_sparse_solver.

This module provides utilities for converting between different sparse matrix formats.
"""

import torch
from typing import Tuple, Optional, Union
import warnings


def dense_to_sparse_csr(
    A: torch.Tensor,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Convert a dense matrix to sparse CSR format.

    Args:
        A: Dense matrix tensor of shape (n, n)
        device: Target device (default: same as input)

    Returns:
        Sparse CSR tensor
    """
    if A.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {A.ndim}D")

    if device is None:
        device = A.device

    # Convert to sparse COO first, then to CSR
    sparse_coo = A.to_sparse_coo()
    if device != A.device:
        sparse_coo = sparse_coo.to(device)

    return sparse_coo.to_sparse_csr()


def sparse_coo_to_csr(
    sparse_coo: torch.Tensor
) -> torch.Tensor:
    """
    Convert a sparse COO tensor to CSR format.

    Args:
        sparse_coo: Sparse COO tensor

    Returns:
        Sparse CSR tensor
    """
    if not sparse_coo.is_sparse:
        raise ValueError("Input must be a sparse tensor")

    return sparse_coo.coalesce().to_sparse_csr()


def ensure_sparse_format(
    A: torch.Tensor,
    format: str = 'csr'
) -> torch.Tensor:
    """
    Ensure the matrix is in the specified sparse format.

    Args:
        A: Input matrix (dense or sparse)
        format: Target format ('csr', 'coo', 'csc')

    Returns:
        Sparse tensor in the specified format
    """
    if not A.is_sparse:
        # Dense matrix - convert to sparse
        A = A.to_sparse_coo()

    if format == 'csr':
        return A.coalesce().to_sparse_csr()
    elif format == 'coo':
        return A.coalesce()
    elif format == 'csc':
        return A.coalesce().to_sparse_csc()
    else:
        raise ValueError(f"Unknown format: {format}. Use 'csr', 'coo', or 'csc'")


def get_csr_components(
    A: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract CSR components from a sparse tensor.

    Args:
        A: Sparse CSR tensor

    Returns:
        Tuple of (values, col_indices, row_ptr)
    """
    if not A.is_sparse_csr:
        A = ensure_sparse_format(A, 'csr')

    crow_indices = A.crow_indices()
    col_indices = A.col_indices()
    values = A.values()

    return values, col_indices, crow_indices


def create_sparse_csr_from_components(
    values: torch.Tensor,
    col_indices: torch.Tensor,
    row_ptr: torch.Tensor,
    shape: Tuple[int, int],
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Create a sparse CSR tensor from components.

    Args:
        values: Non-zero values
        col_indices: Column indices
        row_ptr: Row pointers
        shape: Matrix shape (n, m)
        device: Target device
        dtype: Data type

    Returns:
        Sparse CSR tensor
    """
    if device is None:
        device = values.device
    if dtype is None:
        dtype = values.dtype

    return torch.sparse_csr_tensor(
        crow_indices=row_ptr.to(device),
        col_indices=col_indices.to(device),
        values=values.to(device=device, dtype=dtype),
        size=shape
    )


def create_tridiagonal_sparse_coo(
    n: int,
    diag_val: float = 2.0,
    off_diag_val: float = -1.0,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """
    Create a tridiagonal sparse COO tensor.

    Args:
        n: Matrix dimension
        diag_val: Main diagonal value
        off_diag_val: Off-diagonal value
        device: Target device
        dtype: Data type

    Returns:
        Sparse COO tensor representing a tridiagonal matrix
    """
    indices = []
    values = []

    # Main diagonal
    main_diag_i = torch.arange(n, device=device)
    indices.append(torch.stack([main_diag_i, main_diag_i]))
    values.append(torch.full((n,), diag_val, device=device, dtype=dtype))

    # Upper diagonal
    if n > 1:
        upper_i = torch.arange(n - 1, device=device)
        indices.append(torch.stack([upper_i, upper_i + 1]))
        values.append(torch.full((n - 1,), off_diag_val, device=device, dtype=dtype))

    # Lower diagonal
    if n > 1:
        lower_i = torch.arange(n - 1, device=device)
        indices.append(torch.stack([lower_i + 1, lower_i]))
        values.append(torch.full((n - 1,), off_diag_val, device=device, dtype=dtype))

    all_indices = torch.cat(indices, dim=1)
    all_values = torch.cat(values)

    sparse_matrix = torch.sparse_coo_tensor(
        all_indices, all_values, (n, n),
        device=device, dtype=dtype
    )
    return sparse_matrix.coalesce()


def create_poisson_2d_sparse_coo(
    nx: int,
    ny: int,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """
    Create a 2D Poisson matrix using 5-point stencil.

    Args:
        nx: Number of grid points in x direction
        ny: Number of grid points in y direction
        device: Target device
        dtype: Data type

    Returns:
        Sparse COO tensor representing the Poisson operator
    """
    n = nx * ny

    def idx(i, j):
        return i * ny + j

    row_indices = []
    col_indices = []
    values = []

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)

            # Main diagonal
            row_indices.append(k)
            col_indices.append(k)
            values.append(4.0)

            # Left neighbor
            if i > 0:
                row_indices.append(k)
                col_indices.append(idx(i - 1, j))
                values.append(-1.0)

            # Right neighbor
            if i < nx - 1:
                row_indices.append(k)
                col_indices.append(idx(i + 1, j))
                values.append(-1.0)

            # Bottom neighbor
            if j > 0:
                row_indices.append(k)
                col_indices.append(idx(i, j - 1))
                values.append(-1.0)

            # Top neighbor
            if j < ny - 1:
                row_indices.append(k)
                col_indices.append(idx(i, j + 1))
                values.append(-1.0)

    indices = torch.tensor([row_indices, col_indices], device=device, dtype=torch.long)
    values = torch.tensor(values, device=device, dtype=dtype)

    sparse_matrix = torch.sparse_coo_tensor(indices, values, (n, n), device=device, dtype=dtype)
    return sparse_matrix.coalesce()


def compute_residual(
    A: Union[torch.Tensor, callable],
    x: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """
    Compute the residual r = b - Ax.

    Args:
        A: Matrix (dense, sparse, or callable)
        x: Solution vector
        b: Right-hand side vector

    Returns:
        Residual vector
    """
    if callable(A):
        Ax = A(x)
    elif A.is_sparse:
        Ax = torch.sparse.mm(A, x.unsqueeze(-1)).squeeze(-1)
    else:
        Ax = torch.mv(A, x)

    return b - Ax


def compute_relative_residual(
    A: Union[torch.Tensor, callable],
    x: torch.Tensor,
    b: torch.Tensor
) -> float:
    """
    Compute the relative residual ||b - Ax|| / ||b||.

    Args:
        A: Matrix (dense, sparse, or callable)
        x: Solution vector
        b: Right-hand side vector

    Returns:
        Relative residual (scalar)
    """
    residual = compute_residual(A, x, b)
    return (torch.norm(residual) / torch.norm(b)).item()
