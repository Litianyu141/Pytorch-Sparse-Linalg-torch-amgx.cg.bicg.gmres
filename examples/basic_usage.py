#!/usr/bin/env python3
"""
Basic Usage Examples for PyTorch Sparse Linear Algebra Solvers

This file demonstrates basic usage of both PyTorch and AMGX solvers
for different types of linear systems.

Author: Augment Agent
Date: 2025-06-24
"""

import torch
import numpy as np

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PyTorch solvers
from src.torch_sparse_linalg import cg, bicgstab, gmres

# Try to import AMGX solvers
try:
    from src_torch_amgx.torch_amgx import amgx_cg, amgx_bicgstab, amgx_gmres
    HAS_AMGX = True
    print("✅ AMGX solvers available")
except ImportError:
    HAS_AMGX = False
    print("⚠️  AMGX solvers not available")

def example_pytorch_solvers():
    """Example using PyTorch solvers"""
    print("\n🔧 PyTorch Solvers Example")
    print("-" * 40)
    
    # Create a symmetric positive definite matrix
    n = 100
    torch.manual_seed(42)
    A = torch.randn(n, n, dtype=torch.float64, device='cuda')
    A = A @ A.T + torch.eye(n, dtype=torch.float64, device='cuda')  # Make SPD
    
    # Create right-hand side
    x_true = torch.randn(n, dtype=torch.float64, device='cuda')
    b = A @ x_true
    
    print(f"Matrix size: {n}x{n}")
    print(f"Matrix condition number: {torch.linalg.cond(A).item():.2e}")
    
    # Solve using CG
    x_cg, info_cg = cg(A, b, tol=1e-8, maxiter=1000)
    error_cg = torch.norm(x_cg - x_true).item()
    print(f"CG: converged={info_cg==0}, error={error_cg:.2e}")
    
    # Solve using BiCGStab
    x_bicg, info_bicg = bicgstab(A, b, tol=1e-8, maxiter=1000)
    error_bicg = torch.norm(x_bicg - x_true).item()
    print(f"BiCGStab: converged={info_bicg==0}, error={error_bicg:.2e}")
    
    # Solve using GMRES
    x_gmres, info_gmres = gmres(A, b, tol=1e-8, maxiter=1000, restart=20)
    error_gmres = torch.norm(x_gmres - x_true).item()
    print(f"GMRES: converged={info_gmres==0}, error={error_gmres:.2e}")

def example_amgx_solvers():
    """Example using AMGX solvers"""
    if not HAS_AMGX:
        print("\n⚠️  AMGX solvers not available, skipping example")
        return
    
    print("\n🚀 AMGX Solvers Example")
    print("-" * 40)
    
    # Create a symmetric positive definite matrix
    n = 100
    torch.manual_seed(42)
    A = torch.randn(n, n, dtype=torch.float64, device='cuda')
    A = A @ A.T + torch.eye(n, dtype=torch.float64, device='cuda')  # Make SPD
    
    # Create right-hand side
    x_true = torch.randn(n, dtype=torch.float64, device='cuda')
    b = A @ x_true
    
    print(f"Matrix size: {n}x{n}")
    
    # Solve using AMGX CG
    x_amgx_cg = amgx_cg(A, b, tol=1e-8, maxiter=1000)
    error_amgx_cg = torch.norm(x_amgx_cg - x_true).item()
    print(f"AMGX CG: error={error_amgx_cg:.2e}")
    
    # Solve using AMGX BiCGStab
    x_amgx_bicg = amgx_bicgstab(A, b, tol=1e-8, maxiter=1000)
    error_amgx_bicg = torch.norm(x_amgx_bicg - x_true).item()
    print(f"AMGX BiCGStab: error={error_amgx_bicg:.2e}")

def example_automatic_differentiation():
    """Example demonstrating automatic differentiation"""
    print("\n🔬 Automatic Differentiation Example")
    print("-" * 40)
    
    # Create a small system for gradient testing
    n = 50
    torch.manual_seed(42)
    A = torch.randn(n, n, dtype=torch.float64, device='cuda')
    A = A @ A.T + torch.eye(n, dtype=torch.float64, device='cuda')  # Make SPD
    
    # Create b with gradient tracking
    b = torch.randn(n, dtype=torch.float64, device='cuda', requires_grad=True)
    
    print(f"Matrix size: {n}x{n}")
    
    # PyTorch CG with autograd
    x_torch = cg(A, b, tol=1e-6, maxiter=500)[0]
    loss_torch = torch.sum(x_torch**2)
    loss_torch.backward()
    
    print(f"PyTorch CG gradient norm: {torch.norm(b.grad).item():.2e}")
    
    # Reset gradients
    b.grad = None
    
    # AMGX CG with autograd (if available)
    if HAS_AMGX:
        x_amgx = amgx_cg(A, b, tol=1e-6, maxiter=500)
        loss_amgx = torch.sum(x_amgx**2)
        loss_amgx.backward()
        
        print(f"AMGX CG gradient norm: {torch.norm(b.grad).item():.2e}")
        
        # Compare gradients
        b.grad = None
        x_torch = cg(A, b, tol=1e-6, maxiter=500)[0]
        loss_torch = torch.sum(x_torch**2)
        loss_torch.backward()
        grad_torch = b.grad.clone()
        
        b.grad = None
        x_amgx = amgx_cg(A, b, tol=1e-6, maxiter=500)
        loss_amgx = torch.sum(x_amgx**2)
        loss_amgx.backward()
        grad_amgx = b.grad.clone()
        
        grad_diff = torch.norm(grad_torch - grad_amgx).item()
        print(f"Gradient difference: {grad_diff:.2e}")

def example_sparse_matrices():
    """Example with sparse matrices"""
    print("\n🕸️  Sparse Matrix Example")
    print("-" * 40)
    
    # Create a sparse tridiagonal matrix
    n = 1000
    
    # Create tridiagonal matrix: [-1, 2, -1]
    indices = []
    values = []
    
    # Main diagonal
    main_diag = torch.arange(n, device='cuda')
    indices.append(torch.stack([main_diag, main_diag]))
    values.append(torch.full((n,), 2.0, dtype=torch.float64, device='cuda'))
    
    # Upper diagonal
    if n > 1:
        upper_diag = torch.arange(n-1, device='cuda')
        indices.append(torch.stack([upper_diag, upper_diag + 1]))
        values.append(torch.full((n-1,), -1.0, dtype=torch.float64, device='cuda'))
    
    # Lower diagonal
    if n > 1:
        lower_diag = torch.arange(n-1, device='cuda')
        indices.append(torch.stack([lower_diag + 1, lower_diag]))
        values.append(torch.full((n-1,), -1.0, dtype=torch.float64, device='cuda'))
    
    # Combine all
    all_indices = torch.cat(indices, dim=1)
    all_values = torch.cat(values)
    
    A_sparse = torch.sparse_coo_tensor(all_indices, all_values, (n, n), dtype=torch.float64, device='cuda')
    A_sparse = A_sparse.coalesce()
    
    print(f"Sparse matrix size: {n}x{n}")
    print(f"Non-zeros: {A_sparse._nnz()}")
    print(f"Sparsity: {(1 - A_sparse._nnz() / (n*n)) * 100:.1f}%")
    
    # Create RHS
    torch.manual_seed(42)
    x_true = torch.randn(n, dtype=torch.float64, device='cuda')
    b = torch.sparse.mm(A_sparse, x_true.unsqueeze(1)).squeeze(1)
    
    # Convert to dense for PyTorch solvers
    A_dense = A_sparse.to_dense()
    
    # Solve using PyTorch BiCGStab
    import time
    start_time = time.time()
    x_bicg, info_bicg = bicgstab(A_dense, b, tol=1e-8, maxiter=1000)
    pytorch_time = time.time() - start_time
    error_bicg = torch.norm(x_bicg - x_true).item()
    print(f"PyTorch BiCGStab: time={pytorch_time:.4f}s, error={error_bicg:.2e}")
    
    # Solve using AMGX BiCGStab (if available)
    if HAS_AMGX:
        start_time = time.time()
        x_amgx_bicg = amgx_bicgstab(A_sparse, b, tol=1e-8, maxiter=1000)
        amgx_time = time.time() - start_time
        error_amgx_bicg = torch.norm(x_amgx_bicg - x_true).item()
        print(f"AMGX BiCGStab: time={amgx_time:.4f}s, error={error_amgx_bicg:.2e}")
        
        speedup = pytorch_time / amgx_time if amgx_time > 0 else float('inf')
        print(f"AMGX speedup: {speedup:.2f}x")

def main():
    """Run all examples"""
    print("🚀 PyTorch Sparse Linear Algebra Solvers - Examples")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📍 Using device: {device}")
    
    if device == 'cuda':
        print(f"📍 GPU: {torch.cuda.get_device_name(0)}")
    
    # Run examples
    example_pytorch_solvers()
    example_amgx_solvers()
    example_automatic_differentiation()
    example_sparse_matrices()
    
    print("\n✅ All examples completed!")

if __name__ == "__main__":
    main()
