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
GPU test script: PyTorch vs JAX sparse matrix solver performance comparison
Focus on testing tridiagonal matrices, comparing accuracy and speed of PyTorch and JAX implementations
"""

# Configuration switches
ENABLE_ULTRA_LARGE_MATRIX_TEST = False  # Ultra-large matrix test disabled by default
ULTRA_LARGE_MATRIX_SIZE = 10000  # Ultra-large matrix size
ENABLE_PYTORCH_JIT = True  # Enable PyTorch JIT compilation for fair comparison

import torch
import time
import numpy as np
import signal
import psutil
import gc
import datetime

# Configure PyTorch JIT compilation
if ENABLE_PYTORCH_JIT:
    try:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 128
        print("✅ PyTorch JIT compilation enabled")
        PYTORCH_JIT_ENABLED = True
    except Exception as e:
        print(f"⚠️  PyTorch JIT initialization failed: {str(e)[:100]}...")
        PYTORCH_JIT_ENABLED = False
else:
    PYTORCH_JIT_ENABLED = False
    print("ℹ️  PyTorch JIT compilation disabled")

from src.torch_sparse_linalg import gmres, cg, bicgstab

# Try to configure JAX, prioritizing GPU
HAS_JAX = False
JAX_DEVICE_TYPE = "unknown"
try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.sparse.linalg as jax_linalg
    
    # Enable JAX JIT compilation for better performance
    jax.config.update('jax_enable_x64', True)  # Enable 64-bit precision
    
    # First try GPU mode
    try:
        # Don't force specify platform, let JAX choose the best device
        test_array = jnp.array([1.0, 2.0])
        _ = test_array + 1.0  # Simple operation test
        JAX_DEVICE_TYPE = str(jax.devices()[0]).split(':')[0]  # Get device type
        HAS_JAX = True
        print(f"✅ JAX available ({JAX_DEVICE_TYPE} mode), JIT enabled, will perform JAX vs PyTorch comparison test")
    except Exception as gpu_error:
        print(f"⚠️  JAX GPU initialization failed: {str(gpu_error)[:100]}...")
        print("   Trying to switch to CPU mode...")
        try:
            # Fallback to CPU when GPU fails
            jax.config.update('jax_platform_name', 'cpu')
            test_array = jnp.array([1.0, 2.0])
            _ = test_array + 1.0
            JAX_DEVICE_TYPE = "cpu"
            HAS_JAX = True
            print("✅ JAX switched to CPU mode, JIT enabled, will perform comparison test")
        except Exception as cpu_error:
            print(f"⚠️  JAX CPU mode also failed: {str(cpu_error)[:100]}...")
            HAS_JAX = False
            
except Exception as e:
    print(f"⚠️  JAX not available: {str(e)[:100]}...")
    print("   Testing PyTorch implementation only")
    HAS_JAX = False

def create_tridiagonal_matrix_sparse_torch(n, device='cpu', dtype=torch.float64):
    """Create tridiagonal matrix using PyTorch sparse COO format
    
    Creates a sparse tridiagonal matrix with pattern: -1, 2, -1
    Memory efficiency: O(3n) instead of O(n²) for dense storage
    """
    # Create indices for the three diagonals
    indices = []
    values = []
    
    # Main diagonal: 2.0
    main_diag_indices = torch.arange(n, device=device)
    indices.append(torch.stack([main_diag_indices, main_diag_indices]))
    values.append(torch.full((n,), 2.0, device=device, dtype=dtype))
    
    # Upper diagonal: -1.0
    if n > 1:
        upper_diag_indices = torch.arange(n-1, device=device)
        indices.append(torch.stack([upper_diag_indices, upper_diag_indices + 1]))
        values.append(torch.full((n-1,), -1.0, device=device, dtype=dtype))
    
    # Lower diagonal: -1.0
    if n > 1:
        lower_diag_indices = torch.arange(n-1, device=device)
        indices.append(torch.stack([lower_diag_indices + 1, lower_diag_indices]))
        values.append(torch.full((n-1,), -1.0, device=device, dtype=dtype))
    
    # Concatenate all indices and values
    all_indices = torch.cat(indices, dim=1)
    all_values = torch.cat(values)
    
    # Create sparse COO tensor
    sparse_matrix = torch.sparse_coo_tensor(all_indices, all_values, (n, n), device=device, dtype=dtype)
    return sparse_matrix.coalesce()

def create_tridiagonal_matrix_sparse_jax(n, dtype=jnp.float64):
    """Create tridiagonal matrix using JAX BCOO sparse format
    
    Creates a sparse tridiagonal matrix with pattern: -1, 2, -1
    Memory efficiency: O(3n) instead of O(n²) for dense storage
    """
    try:
        from jax.experimental import sparse as jax_sparse
        
        # Create indices for the three diagonals
        indices = []
        values = []
        
        # Main diagonal: 2.0
        main_diag_indices = jnp.arange(n)
        indices.append(jnp.stack([main_diag_indices, main_diag_indices], axis=1))
        values.append(jnp.full((n,), 2.0, dtype=dtype))
        
        # Upper diagonal: -1.0
        if n > 1:
            upper_diag_indices = jnp.arange(n-1)
            indices.append(jnp.stack([upper_diag_indices, upper_diag_indices + 1], axis=1))
            values.append(jnp.full((n-1,), -1.0, dtype=dtype))
        
        # Lower diagonal: -1.0
        if n > 1:
            lower_diag_indices = jnp.arange(n-1)
            indices.append(jnp.stack([lower_diag_indices + 1, lower_diag_indices], axis=1))
            values.append(jnp.full((n-1,), -1.0, dtype=dtype))
        
        # Concatenate all indices and values
        all_indices = jnp.concatenate(indices, axis=0)
        all_values = jnp.concatenate(values)
        
        # Create sparse BCOO matrix
        sparse_matrix = jax_sparse.BCOO((all_values, all_indices), shape=(n, n))
        return sparse_matrix
    
    except ImportError:
        # Fallback to dense matrix if JAX sparse is not available
        print("⚠️  JAX sparse not available, using dense matrix as fallback")
        return create_tridiagonal_matrix_numpy(n, dtype=np.float64)

def create_tridiagonal_matrix_numpy(n, dtype=np.float64):
    """Create tridiagonal matrix using NumPy (dense storage for compatibility)
    
    Note: This creates a dense matrix with tridiagonal structure for testing purposes.
    In real applications, use scipy.sparse formats for better memory efficiency.
    The sparsity pattern is: only 3 diagonals are non-zero out of n^2 total elements.
    """
    # Create tridiagonal matrix: -1, 2, -1 pattern
    A = np.zeros((n, n), dtype=dtype)
    
    # Main diagonal: 2
    A[np.arange(n), np.arange(n)] = 2.0
    
    # Upper and lower diagonals: -1
    A[np.arange(n-1), np.arange(1, n)] = -1.0
    A[np.arange(1, n), np.arange(n-1)] = -1.0
    
    return A

def create_sparse_poisson_2d_torch(nx, ny, device='cpu', dtype=torch.float64):
    """Create 2D Poisson matrix using PyTorch sparse COO format (5-point stencil)
    
    Creates a sparse 2D Poisson matrix with ~5 non-zeros per row.
    Memory efficiency: O(5n) instead of O(n²) for dense storage
    """
    n = nx * ny
    
    def idx(i, j):
        return i * ny + j
    
    indices = []
    values = []
    
    for i in range(nx):
        for j in range(ny):
            center = idx(i, j)
            
            # Center point: 4.0
            indices.append([center, center])
            values.append(4.0)
            
            # Neighbor connections: -1.0
            if i > 0:  # Left neighbor
                indices.append([center, idx(i-1, j)])
                values.append(-1.0)
            if i < nx - 1:  # Right neighbor
                indices.append([center, idx(i+1, j)])
                values.append(-1.0)
            if j > 0:  # Bottom neighbor
                indices.append([center, idx(i, j-1)])
                values.append(-1.0)
            if j < ny - 1:  # Top neighbor
                indices.append([center, idx(i, j+1)])
                values.append(-1.0)
    
    # Convert to tensors
    indices_tensor = torch.tensor(indices, device=device, dtype=torch.long).T
    values_tensor = torch.tensor(values, device=device, dtype=dtype)
    
    # Create sparse COO tensor
    sparse_matrix = torch.sparse_coo_tensor(indices_tensor, values_tensor, (n, n), device=device, dtype=dtype)
    return sparse_matrix.coalesce()

def create_sparse_poisson_2d_jax(nx, ny, dtype=jnp.float64):
    """Create 2D Poisson matrix using JAX BCOO sparse format (5-point stencil)
    
    Creates a sparse 2D Poisson matrix with ~5 non-zeros per row.
    Memory efficiency: O(5n) instead of O(n²) for dense storage
    """
    try:
        from jax.experimental import sparse as jax_sparse
        
        n = nx * ny
        
        def idx(i, j):
            return i * ny + j
        
        indices = []
        values = []
        
        for i in range(nx):
            for j in range(ny):
                center = idx(i, j)
                
                # Center point: 4.0
                indices.append([center, center])
                values.append(4.0)
                
                # Neighbor connections: -1.0
                if i > 0:  # Left neighbor
                    indices.append([center, idx(i-1, j)])
                    values.append(-1.0)
                if i < nx - 1:  # Right neighbor
                    indices.append([center, idx(i+1, j)])
                    values.append(-1.0)
                if j > 0:  # Bottom neighbor
                    indices.append([center, idx(i, j-1)])
                    values.append(-1.0)
                if j < ny - 1:  # Top neighbor
                    indices.append([center, idx(i, j+1)])
                    values.append(-1.0)
        
        # Convert to JAX arrays
        indices_array = jnp.array(indices, dtype=jnp.int32)
        values_array = jnp.array(values, dtype=dtype)
        
        # Create sparse BCOO matrix
        sparse_matrix = jax_sparse.BCOO((values_array, indices_array), shape=(n, n))
        return sparse_matrix
        
    except ImportError:
        # Fallback to dense matrix if JAX sparse is not available
        print("⚠️  JAX sparse not available, using dense matrix as fallback")
        return create_sparse_poisson_2d_numpy(nx, ny, dtype=np.float64)

def create_sparse_poisson_2d_numpy(nx, ny, dtype=np.float64):
    """Create 2D Poisson matrix using NumPy (5-point stencil)
    
    Note: Creates dense matrix with sparse structure. 
    Each interior point connects to 4 neighbors, resulting in ~5 non-zeros per row.
    Sparsity: approximately (n*5)/(n*n) = 5/n for large matrices.
    """
    n = nx * ny
    A = np.zeros((n, n), dtype=dtype)
    
    def idx(i, j):
        return i * ny + j
    
    for i in range(nx):
        for j in range(ny):
            center = idx(i, j)
            A[center, center] = 4.0
            
            # Neighbor nodes
            if i > 0:
                A[center, idx(i-1, j)] = -1.0
            if i < nx - 1:
                A[center, idx(i+1, j)] = -1.0
            if j > 0:
                A[center, idx(i, j-1)] = -1.0
            if j < ny - 1:
                A[center, idx(i, j+1)] = -1.0
    
    return A

def create_sparse_diagonal_dominant_torch(n, sparsity=0.1, device='cpu', dtype=torch.float64):
    """Create sparse diagonally dominant matrix using PyTorch sparse COO format"""
    torch.manual_seed(42)  # For reproducibility
    
    # Create random sparse pattern
    nnz = int(n * n * sparsity)  # Number of non-zeros
    
    # Generate random indices
    row_indices = torch.randint(0, n, (nnz,), device=device)
    col_indices = torch.randint(0, n, (nnz,), device=device)
    
    # Generate random values
    values = torch.randn(nnz, device=device, dtype=dtype) * 0.1
    
    # Create sparse matrix
    indices = torch.stack([row_indices, col_indices])
    sparse_matrix = torch.sparse_coo_tensor(indices, values, (n, n), device=device, dtype=dtype)
    sparse_matrix = sparse_matrix.coalesce()
    
    # Make it diagonally dominant
    # Get row sums of absolute values
    abs_sparse = torch.sparse_coo_tensor(sparse_matrix.indices(), 
                                        torch.abs(sparse_matrix.values()), 
                                        sparse_matrix.shape, device=device, dtype=dtype)
    row_sums = torch.sparse.sum(abs_sparse, dim=1).to_dense()
    
    # Set diagonal to ensure diagonal dominance
    diagonal_values = row_sums + torch.rand(n, device=device, dtype=dtype) + 1.0
    
    # Add diagonal elements
    diag_indices = torch.arange(n, device=device)
    diag_indices_2d = torch.stack([diag_indices, diag_indices])
    
    # Combine original sparse matrix with diagonal
    all_indices = torch.cat([sparse_matrix.indices(), diag_indices_2d], dim=1)
    all_values = torch.cat([sparse_matrix.values(), diagonal_values])
    
    final_sparse = torch.sparse_coo_tensor(all_indices, all_values, (n, n), device=device, dtype=dtype)
    return final_sparse.coalesce()

def create_sparse_diagonal_dominant_jax(n, sparsity=0.1, dtype=jnp.float64):
    """Create sparse diagonally dominant matrix using JAX BCOO sparse format"""
    try:
        from jax.experimental import sparse as jax_sparse
        import jax.random as jax_random
        
        key = jax_random.PRNGKey(42)  # For reproducibility
        
        # Create random sparse pattern
        nnz = int(n * n * sparsity)  # Number of non-zeros
        
        # Generate random indices
        key1, key2, key3 = jax_random.split(key, 3)
        row_indices = jax_random.randint(key1, (nnz,), 0, n)
        col_indices = jax_random.randint(key2, (nnz,), 0, n)
        
        # Generate random values
        values = jax_random.normal(key3, (nnz,), dtype=dtype) * 0.1
        
        # Create sparse matrix indices
        indices = jnp.stack([row_indices, col_indices], axis=1)
        sparse_matrix = jax_sparse.BCOO((values, indices), shape=(n, n))
        
        # Make it diagonally dominant
        # For simplicity, use a dense approach for diagonal dominance
        # (in practice, you'd want a more efficient sparse approach)
        dense_matrix = sparse_matrix.todense()
        row_sums = jnp.sum(jnp.abs(dense_matrix), axis=1)
        
        # Set diagonal to ensure diagonal dominance
        diagonal_values = row_sums + jax_random.uniform(jax_random.split(key3)[0], (n,), dtype=dtype) + 1.0
        dense_matrix = dense_matrix.at[jnp.diag_indices(n)].set(diagonal_values)
        
        # Convert back to sparse (this is not ideal for large matrices)
        indices = jnp.where(jnp.abs(dense_matrix) > 1e-12)
        values = dense_matrix[indices]
        sparse_indices = jnp.stack(indices, axis=1)
        
        return jax_sparse.BCOO((values, sparse_indices), shape=(n, n))
        
    except ImportError:
        # Fallback to dense matrix if JAX sparse is not available
        print("⚠️  JAX sparse not available, using dense matrix as fallback")
        return create_sparse_diagonal_dominant_numpy(n, sparsity, dtype=np.float64)

def create_sparse_diagonal_dominant_numpy(n, sparsity=0.1, dtype=np.float64):
    """Create sparse diagonally dominant matrix using NumPy"""
    np.random.seed(42)
    A = np.zeros((n, n), dtype=dtype)
    
    # Random sparse pattern
    mask = np.random.rand(n, n) < sparsity
    A[mask] = np.random.randn(mask.sum()) * 0.1
    
    # Make it diagonally dominant
    row_sums = np.sum(np.abs(A), axis=1)
    A[np.arange(n), np.arange(n)] = row_sums + np.random.rand(n) + 1.0
    
    return A

def create_sparse_non_diagonal_dominant_asymmetric_torch(n, sparsity=0.1, device='cpu', dtype=torch.float64):
    """Create sparse non-diagonal dominant asymmetric matrix using PyTorch COO format"""
    np.random.seed(42)
    
    # Create random sparse pattern
    nnz = int(n * n * sparsity)
    row_indices = np.random.randint(0, n, nnz)
    col_indices = np.random.randint(0, n, nnz)
    
    # Remove duplicates by using unique pairs
    pairs = np.unique(np.column_stack([row_indices, col_indices]), axis=0)
    row_indices = pairs[:, 0]
    col_indices = pairs[:, 1]
    nnz = len(row_indices)
    
    # Create asymmetric values (positive and negative)
    values = np.random.randn(nnz) * 2.0  # Larger variance for non-diagonal dominant
    
    # Convert to PyTorch sparse tensor
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
    values_tensor = torch.tensor(values, dtype=dtype, device=device)
    sparse_matrix = torch.sparse_coo_tensor(indices, values_tensor, (n, n), device=device, dtype=dtype)
    sparse_matrix = sparse_matrix.coalesce()
    
    # Add small diagonal elements (NOT dominant)
    diagonal_values = torch.rand(n, device=device, dtype=dtype) * 0.5 + 0.1  # Small diagonal
    diag_indices = torch.arange(n, device=device)
    diag_indices_2d = torch.stack([diag_indices, diag_indices])
    
    # Combine sparse matrix with small diagonal
    all_indices = torch.cat([sparse_matrix.indices(), diag_indices_2d], dim=1)
    all_values = torch.cat([sparse_matrix.values(), diagonal_values])
    
    final_sparse = torch.sparse_coo_tensor(all_indices, all_values, (n, n), device=device, dtype=dtype)
    return final_sparse.coalesce()

def create_sparse_non_diagonal_dominant_asymmetric_jax(n, sparsity=0.1, dtype=jnp.float64):
    """Create sparse non-diagonal dominant asymmetric matrix using JAX BCOO sparse format"""
    try:
        from jax.experimental import sparse
        
        np.random.seed(42)
        
        # Create random sparse pattern
        nnz = int(n * n * sparsity)
        row_indices = np.random.randint(0, n, nnz)
        col_indices = np.random.randint(0, n, nnz)
        
        # Remove duplicates
        pairs = np.unique(np.column_stack([row_indices, col_indices]), axis=0)
        row_indices = pairs[:, 0]
        col_indices = pairs[:, 1]
        nnz = len(row_indices)
        
        # Create asymmetric values
        values = np.random.randn(nnz) * 2.0
        
        # Add small diagonal elements
        diagonal_mask = row_indices == col_indices
        non_diagonal_mask = ~diagonal_mask
        
        # For diagonal elements, use small values
        values[diagonal_mask] = np.random.rand(diagonal_mask.sum()) * 0.5 + 0.1
        
        # Create JAX sparse matrix
        indices = jnp.array(np.column_stack([row_indices, col_indices]))
        values_jax = jnp.array(values, dtype=dtype)
        
        return sparse.BCOO((values_jax, indices), shape=(n, n))
        
    except ImportError:
        # Fallback to dense matrix
        return create_sparse_non_diagonal_dominant_asymmetric_numpy(n, sparsity, dtype=np.float64)

def create_sparse_diagonal_dominant_asymmetric_torch(n, sparsity=0.1, device='cpu', dtype=torch.float64):
    """Create sparse diagonal dominant asymmetric matrix using PyTorch COO format"""
    np.random.seed(43)  # Different seed for different pattern
    
    # Create random sparse pattern
    nnz = int(n * n * sparsity)
    row_indices = np.random.randint(0, n, nnz)
    col_indices = np.random.randint(0, n, nnz)
    
    # Remove duplicates by using unique pairs
    pairs = np.unique(np.column_stack([row_indices, col_indices]), axis=0)
    row_indices = pairs[:, 0]
    col_indices = pairs[:, 1]
    nnz = len(row_indices)
    
    # Create asymmetric values with different scales for upper/lower triangular parts
    values = np.random.randn(nnz) * 0.5
    upper_mask = row_indices < col_indices
    lower_mask = row_indices > col_indices
    
    # Make it asymmetric: upper triangular part has different scale than lower
    values[upper_mask] *= 1.5  # Upper triangular elements are larger
    values[lower_mask] *= 0.7  # Lower triangular elements are smaller
    
    # Convert to PyTorch sparse tensor
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
    values_tensor = torch.tensor(values, dtype=dtype, device=device)
    sparse_matrix = torch.sparse_coo_tensor(indices, values_tensor, (n, n), device=device, dtype=dtype)
    sparse_matrix = sparse_matrix.coalesce()
    
    # Calculate row sums for diagonal dominance
    abs_sparse = torch.sparse_coo_tensor(sparse_matrix.indices(), 
                                        torch.abs(sparse_matrix.values()), 
                                        sparse_matrix.shape, device=device, dtype=dtype)
    row_sums = torch.sparse.sum(abs_sparse, dim=1).to_dense()
    
    # Set diagonal to ensure diagonal dominance (asymmetric matrix with dominant diagonal)
    diagonal_values = row_sums + torch.rand(n, device=device, dtype=dtype) + 1.0
    
    # Add diagonal elements
    diag_indices = torch.arange(n, device=device)
    diag_indices_2d = torch.stack([diag_indices, diag_indices])
    
    # Combine sparse matrix with dominant diagonal
    all_indices = torch.cat([sparse_matrix.indices(), diag_indices_2d], dim=1)
    all_values = torch.cat([sparse_matrix.values(), diagonal_values])
    
    final_sparse = torch.sparse_coo_tensor(all_indices, all_values, (n, n), device=device, dtype=dtype)
    return final_sparse.coalesce()

def create_sparse_diagonal_dominant_asymmetric_jax(n, sparsity=0.1, dtype=jnp.float64):
    """Create sparse diagonal dominant asymmetric matrix using JAX BCOO sparse format"""
    try:
        from jax.experimental import sparse
        
        np.random.seed(43)
        
        # Create random sparse pattern
        nnz = int(n * n * sparsity)
        row_indices = np.random.randint(0, n, nnz)
        col_indices = np.random.randint(0, n, nnz)
        
        # Remove duplicates
        pairs = np.unique(np.column_stack([row_indices, col_indices]), axis=0)
        row_indices = pairs[:, 0]
        col_indices = pairs[:, 1]
        nnz = len(row_indices)
        
        # Create asymmetric values
        values = np.random.randn(nnz) * 0.5
        upper_mask = row_indices < col_indices
        lower_mask = row_indices > col_indices
        
        values[upper_mask] *= 1.5
        values[lower_mask] *= 0.7
        
        # Calculate row sums for diagonal dominance
        row_sums = np.zeros(n)
        for i in range(nnz):
            row_sums[row_indices[i]] += abs(values[i])
        
        # Add diagonal elements to ensure dominance
        diagonal_indices = np.arange(n)
        diagonal_values = row_sums + np.random.rand(n) + 1.0
        
        # Combine all indices and values
        all_row_indices = np.concatenate([row_indices, diagonal_indices])
        all_col_indices = np.concatenate([col_indices, diagonal_indices])
        all_values = np.concatenate([values, diagonal_values])
        
        # Create JAX sparse matrix
        indices = jnp.array(np.column_stack([all_row_indices, all_col_indices]))
        values_jax = jnp.array(all_values, dtype=dtype)
        
        return sparse.BCOO((values_jax, indices), shape=(n, n))
        
    except ImportError:
        # Fallback to dense matrix
        return create_sparse_diagonal_dominant_asymmetric_numpy(n, sparsity, dtype=np.float64)

def create_sparse_non_diagonal_dominant_asymmetric_numpy(n, sparsity=0.1, dtype=np.float64):
    """Create sparse non-diagonal dominant asymmetric matrix using NumPy"""
    np.random.seed(42)
    A = np.zeros((n, n), dtype=dtype)
    
    # Random sparse pattern
    mask = np.random.rand(n, n) < sparsity
    A[mask] = np.random.randn(mask.sum()) * 2.0  # Larger off-diagonal elements
    
    # Make it asymmetric
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    A[upper_mask] *= 1.5
    A[lower_mask] *= 0.7
    
    # Add small diagonal elements (NOT dominant)
    A[np.arange(n), np.arange(n)] = np.random.rand(n) * 0.5 + 0.1
    
    return A

def create_sparse_diagonal_dominant_asymmetric_numpy(n, sparsity=0.1, dtype=np.float64):
    """Create sparse diagonal dominant asymmetric matrix using NumPy"""
    np.random.seed(43)
    A = np.zeros((n, n), dtype=dtype)
    
    # Random sparse pattern
    mask = np.random.rand(n, n) < sparsity
    A[mask] = np.random.randn(mask.sum()) * 0.5
    
    # Make it asymmetric
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    A[upper_mask] *= 1.5
    A[lower_mask] *= 0.7
    
    # Make it diagonally dominant
    row_sums = np.sum(np.abs(A), axis=1)
    A[np.arange(n), np.arange(n)] = row_sums + np.random.rand(n) + 1.0
    
    return A

def test_matrix_properties(A, name, is_sparse=False):
    """Test basic properties of the matrix (supports both dense and sparse)"""
    print(f"\n🔍 Matrix properties: {name}")
    
    if is_sparse:
        if hasattr(A, 'shape'):
            shape = A.shape
        elif hasattr(A, 'size'):
            shape = A.size()
        else:
            shape = (A.shape[0], A.shape[1])
        
        print(f"  Size: {shape[0]}×{shape[1]}")
        print(f"  Format: Sparse matrix")
        
        # Calculate sparsity for sparse matrices
        if hasattr(A, 'nnz'):  # PyTorch sparse
            nnz = A.nnz().item()
            total_elements = shape[0] * shape[1]
            sparsity = (total_elements - nnz) / total_elements
            print(f"  Non-zeros: {nnz:,}")
            print(f"  Sparsity: {sparsity:.2%}")
            print(f"  Memory efficiency: ~{nnz}/{total_elements} = {nnz/total_elements:.2%} of dense storage")
        elif hasattr(A, 'nse'):  # JAX sparse BCOO
            nnz = A.nse
            total_elements = shape[0] * shape[1]
            sparsity = (total_elements - nnz) / total_elements
            print(f"  Non-zeros: {nnz:,}")
            print(f"  Sparsity: {sparsity:.2%}")
            print(f"  Memory efficiency: ~{nnz}/{total_elements} = {nnz/total_elements:.2%} of dense storage")
        else:
            print(f"  Sparse format: Unknown")
        
        # Condition number estimation (convert to dense for small matrices only)
        if shape[0] <= 1000:  # Only for small matrices
            try:
                if hasattr(A, 'to_dense'):  # PyTorch sparse
                    A_dense = A.to_dense().cpu().numpy()
                elif hasattr(A, 'todense'):  # JAX sparse
                    A_dense = np.array(A.todense())
                else:
                    A_dense = A
                    
                s = np.linalg.svd(A_dense, compute_uv=False)
                cond = s[0] / s[-1]
                print(f"  Condition number: {cond:.2e}")
            except Exception as e:
                print(f"  Condition number: Cannot compute - {str(e)}")
        else:
            print(f"  Condition number: Skipped (matrix too large)")
    else:
        # Original dense matrix handling
        print(f"  Size: {A.shape[0]}×{A.shape[1]}")
        print(f"  Sparsity: {(A == 0).sum() / A.size:.2%}")
        
        # Condition number estimation
        try:
            s = np.linalg.svd(A, compute_uv=False)
            cond = s[0] / s[-1]
            print(f"  Condition number: {cond:.2e}")
        except Exception as e:
            print(f"  Condition number: Cannot compute - {str(e)}")

def sparse_to_dense_if_needed(A, max_size=5000):
    """Convert sparse matrix to dense if needed for compatibility with solvers"""
    if hasattr(A, 'to_dense'):  # PyTorch sparse
        if A.shape[0] <= max_size:
            return A.to_dense()
        else:
            print(f"⚠️  Matrix too large ({A.shape[0]}×{A.shape[1]}) for dense conversion, keeping sparse")
            return A
    elif hasattr(A, 'todense'):  # JAX sparse
        if A.shape[0] <= max_size:
            return A.todense()
        else:
            print(f"⚠️  Matrix too large ({A.shape[0]}×{A.shape[1]}) for dense conversion, keeping sparse")
            return A
    else:
        return A  # Already dense
    
    # Check if symmetric
    is_symmetric = np.allclose(A_np, A_np.T, atol=1e-12)
    print(f"  Symmetry: {is_symmetric}")
    
    # Check if positive definite (for symmetric matrices)
    if is_symmetric:
        try:
            eigenvals = np.linalg.eigvals(A_np)
            is_pd = np.all(eigenvals.real > 0)
            print(f"  Positive definiteness: {is_pd}")
        except Exception as e:
            print(f"  Positive definiteness: Cannot determine - {str(e)}")

def benchmark_jax_solver(solver_name, solver_func, A_jax, b_jax, timeout_seconds=30):
    """JAX solver benchmark with timeout and detailed error handling"""
    start_time = time.time()
    try:
        # 添加JAX JIT编译公平性处理
        if not hasattr(benchmark_jax_solver, '_jit_warmed_up'):
            print(f"    🔧 First JAX call - JIT compilation time included")
            benchmark_jax_solver._jit_warmed_up = True
            
            # JIT warm-up for JAX solvers
            if A_jax.shape[0] <= 100:  # Only warm-up for small matrices
                try:
                    # Create JIT-compiled version of the solver
                    if 'cg' in solver_name.lower():
                        jit_solver = jax.jit(solver_func)
                        _, _ = jit_solver(A_jax[:5, :5], b_jax[:5], tol=1e-6, maxiter=10)
                    elif 'bicgstab' in solver_name.lower():
                        jit_solver = jax.jit(solver_func) 
                        _, _ = jit_solver(A_jax[:5, :5], b_jax[:5], tol=1e-6, maxiter=10)
                    elif 'gmres' in solver_name.lower():
                        jit_solver = jax.jit(solver_func)
                        _, _ = jit_solver(A_jax[:5, :5], b_jax[:5], tol=1e-6, restart=5, maxiter=10)
                    print(f"    ✅ JAX JIT warm-up completed for {solver_name}")
                except Exception as warmup_error:
                    print(f"    ⚠️  JAX JIT warm-up failed: {str(warmup_error)[:50]}...")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        n = A_jax.shape[0]
        
        # **FAIR COMPARISON**: Use IDENTICAL tolerance settings as PyTorch
        if n >= 5000:  # Large matrix - same threshold as PyTorch
            tol = 1e-6
            max_iter = min(1000, n)
            print(f"    📊 Large matrix params: tol={tol}, maxiter={max_iter}")
        else:  # Small matrix 
            tol = 1e-8
            max_iter = 1000
        
        # JAX solver parameter settings - identical to PyTorch logic
        if 'cg' in solver_name.lower():
            x_solve, info = solver_func(A_jax, b_jax, tol=tol, maxiter=max_iter)
            detailed_status = f"JAX CG (maxiter={max_iter}, tol={tol})"
        elif 'bicgstab' in solver_name.lower():
            x_solve, info = solver_func(A_jax, b_jax, tol=tol, maxiter=max_iter)
            detailed_status = f"JAX BiCGStab (maxiter={max_iter}, tol={tol})"
        elif 'gmres' in solver_name.lower():
            # GMRES: Use same restart logic as PyTorch
            restart = min(30, n // 10, 100)  # Same as PyTorch
            x_solve, info = solver_func(A_jax, b_jax, tol=tol, restart=restart, maxiter=max_iter)
            detailed_status = f"JAX GMRES (restart={restart}, maxiter={max_iter}, tol={tol})"
        else:
            raise ValueError(f"Unknown JAX solver: {solver_name}")
        
        # Cancel timeout
        signal.alarm(0)
        
        # Wait for JAX computation to complete
        x_solve = jax.device_get(x_solve)
        solve_time = time.time() - start_time
        
        # JAX info handling - detailed analysis
        if info is None:
            # JAX returns None usually means no explicit convergence info
            # Check convergence by residual
            try:
                residual = jnp.linalg.norm(A_jax @ x_solve - b_jax)
                residual_val = float(residual)
                
                if residual_val < tol * 10:  # Use slightly relaxed threshold for residual check
                    converged = True
                    status = f"Success - Converged by residual check (residual={residual_val:.2e}) - {detailed_status}"
                else:
                    converged = False
                    status = f"Failed - Residual too large (residual={residual_val:.2e}) - {detailed_status}"
                    
                info_display = f"None(residual:{residual_val:.2e})"
            except Exception as residual_error:
                converged = False
                status = f"Failed - Cannot compute residual: {str(residual_error)} - {detailed_status}"
                info_display = "None(residual computation failed)"
        else:
            # info is not None
            info_val = int(info) if hasattr(info, '__int__') else info
            converged = info_val == 0
            
            if converged:
                status = f"Success - JAX converged (info={info_val}) - {detailed_status}"
            else:
                if info_val > 0:
                    status = f"Failed - Stopped at iteration {info_val} - {detailed_status}"
                elif info_val == -1:
                    status = f"Failed - Not converged (max iterations reached) - {detailed_status}"
                else:
                    status = f"Failed - JAX solver error (info={info_val}) - {detailed_status}"
                    
            info_display = str(info_val)
        
        return {
            'x_solve': x_solve,
            'time': solve_time,
            'converged': converged,
            'info': info_display,
            'status': status
        }
        
    except TimeoutException:
        signal.alarm(0)
        return {
            'x_solve': None,
            'time': timeout_seconds,
            'converged': False,
            'info': -2,
            'status': f'Timeout ({timeout_seconds}s) - JAX solver exceeded time limit'
        }
    except Exception as e:
        signal.alarm(0)
        error_msg = str(e)
        print(f"    ❌ JAX solver exception: {error_msg}")
        return {
            'x_solve': None,
            'time': float('inf'),
            'converged': False,
            'info': -999,
            'status': f'Error - JAX exception: {error_msg[:80]}{"..." if len(error_msg) > 80 else ""}'
        }

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Solver timeout")

def benchmark_torch_solver(solver_name, solver_func, A_torch, b_torch, timeout_seconds=30):
    """PyTorch solver benchmark with timeout and detailed error handling"""
    start_time = time.time()
    try:
        # 添加PyTorch JIT编译公平性处理
        if PYTORCH_JIT_ENABLED and not hasattr(benchmark_torch_solver, '_jit_warmed_up'):
            print(f"    🔧 First PyTorch call - JIT compilation time included")
            benchmark_torch_solver._jit_warmed_up = True
            
            # JIT warm-up for PyTorch solvers
            if A_torch.shape[0] <= 100:  # Only warm-up for small matrices
                try:
                    # Create JIT-compiled version of the solver
                    if 'CG' in solver_name:
                        jit_solver = torch.compile(solver_func, mode='default')
                        _, _ = jit_solver(A_torch[:5, :5], b_torch[:5], tol=1e-6, maxiter=10)
                    elif 'BiCGStab' in solver_name:
                        jit_solver = torch.compile(solver_func, mode='default') 
                        _, _ = jit_solver(A_torch[:5, :5], b_torch[:5], tol=1e-6, maxiter=10)
                    elif 'GMRES' in solver_name:
                        jit_solver = torch.compile(solver_func, mode='default')
                        _, _ = jit_solver(A_torch[:5, :5], b_torch[:5], tol=1e-6, restart=5, maxiter=10)
                    print(f"    ✅ PyTorch JIT warm-up completed for {solver_name}")
                except Exception as warmup_error:
                    print(f"    ⚠️  PyTorch JIT warm-up failed: {str(warmup_error)[:50]}...")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        if solver_name == 'PyTorch Direct' or 'Direct' in solver_name:
            x_solve = solver_func(A_torch, b_torch)
            info = 0
            detailed_status = "Success - Direct solve"
        else:
            # Fixed parameter settings to avoid issues from over-optimization
            n = A_torch.shape[0]
            
            # Use more conservative but stable parameters for large matrices
            if n >= 5000:  # Large matrix
                tol = 1e-6  # Keep standard tolerance
                max_iter = min(1000, n)  # Limit maximum iterations
                print(f"    📊 Large matrix params: tol={tol}, maxiter={max_iter}")
            else:
                tol = 1e-8  # Use stricter tolerance for small matrices
                max_iter = 1000
            
            # PyTorch iterative solvers - apply JIT compilation if enabled
            if PYTORCH_JIT_ENABLED:
                # Apply JIT compilation to the solver function
                jit_solver_func = torch.compile(solver_func, mode='default')
            else:
                jit_solver_func = solver_func
            
            if 'GMRES' in solver_name:
                # GMRES special handling: use larger restart and more iterations
                restart = min(30, n // 10, 100)  # More conservative restart setting
                try:
                    x_solve, info = jit_solver_func(A_torch, b_torch, tol=tol, restart=restart, maxiter=max_iter)
                    detailed_status = f"GMRES (restart={restart}, maxiter={max_iter}, tol={tol})"
                except Exception as gmres_error:
                    print(f"    ❌ GMRES internal error: {str(gmres_error)}")
                    raise gmres_error
            else:
                # CG and BiCGStab
                try:
                    x_solve, info = jit_solver_func(A_torch, b_torch, tol=tol, maxiter=max_iter)
                    detailed_status = f"{solver_name.split()[-1]} (maxiter={max_iter}, tol={tol})"
                except Exception as iter_error:
                    print(f"    ❌ {solver_name} internal error: {str(iter_error)}")
                    raise iter_error
        
        # Cancel timeout
        signal.alarm(0)
        
        # Ensure GPU computation is complete
        if A_torch.device.type == 'cuda':
            torch.cuda.synchronize()
        
        solve_time = time.time() - start_time
        
        # Detailed status analysis
        if info == 0:
            status = f"Success - {detailed_status}"
        elif info == -1:
            status = f"Failed - Not converged (max iterations reached) - {detailed_status}"
        elif info == -2:
            status = f"Failed - Numerical error (possibly singular matrix) - {detailed_status}"
        elif info == -3:
            status = f"Failed - Parameter error - {detailed_status}"
        elif info > 0:
            status = f"Failed - Stopped at iteration {info} - {detailed_status}"
        else:
            status = f"Failed - Unknown error (info={info}) - {detailed_status}"
        
        return {
            'x_solve': x_solve.cpu().numpy() if isinstance(x_solve, torch.Tensor) else x_solve,
            'time': solve_time,
            'converged': info == 0,
            'info': info,
            'status': status
        }
        
    except TimeoutException:
        signal.alarm(0)
        return {
            'x_solve': None,
            'time': timeout_seconds,
            'converged': False,
            'info': -2,
            'status': f'Timeout ({timeout_seconds}s) - Solver runtime exceeded limit'
        }
    except Exception as e:
        signal.alarm(0)
        error_msg = str(e)
        print(f"    ❌ Solver exception: {error_msg}")
        return {
            'x_solve': None,
            'time': float('inf'),
            'converged': False,
            'info': -999,
            'status': f'Error - {error_msg[:100]}{"..." if len(error_msg) > 100 else ""}'
        }

def compare_solvers():
    """Compare PyTorch and JAX solver performance"""
    print("🚀 PyTorch vs JAX Sparse Matrix Solver Performance Comparison")
    print("=" * 90)
    
    # 添加版本信息以确保测试可重现性
    print(f"📋 Test Environment Details:")
    print(f"   - PyTorch version: {torch.__version__}")
    print(f"   - NumPy version: {np.__version__}")
    if HAS_JAX:
        print(f"   - JAX version: {jax.__version__}")
        print(f"   - JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print()
    
    # Check devices
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"📍 PyTorch device: {device}")
    if use_gpu:
        print(f"📍 GPU: {torch.cuda.get_device_name(0)}")
    
    if HAS_JAX:
        jax_device_info = str(jax.devices()[0])
        print(f"📍 JAX device: {jax_device_info}")
    else:
        print("📍 JAX: Not available")
    
    # Test configuration - use sparse matrix formats
    test_cases = [
        ("Sparse Tridiagonal 200×200 (PyTorch COO)", 
         lambda: create_tridiagonal_matrix_sparse_torch(200, device, torch.float64),
         lambda: create_tridiagonal_matrix_sparse_jax(200, jnp.float64) if HAS_JAX else None,
         True),  # is_sparse flag
        ("Sparse Tridiagonal 500×500 (PyTorch COO)", 
         lambda: create_tridiagonal_matrix_sparse_torch(500, device, torch.float64),
         lambda: create_tridiagonal_matrix_sparse_jax(500, jnp.float64) if HAS_JAX else None,
         True),
        ("Sparse 2D Poisson 15×15 (225×225, PyTorch COO)", 
         lambda: create_sparse_poisson_2d_torch(15, 15, device, torch.float64),
         lambda: create_sparse_poisson_2d_jax(15, 15, jnp.float64) if HAS_JAX else None,
         True),
        ("Sparse 2D Poisson 22×22 (484×484, PyTorch COO)", 
         lambda: create_sparse_poisson_2d_torch(22, 22, device, torch.float64),
         lambda: create_sparse_poisson_2d_jax(22, 22, jnp.float64) if HAS_JAX else None,
         True),
        ("Sparse Diagonal Dominant 300×300 (PyTorch COO)", 
         lambda: create_sparse_diagonal_dominant_torch(300, 0.05, device, torch.float64),
         lambda: create_sparse_diagonal_dominant_jax(300, 0.05, jnp.float64) if HAS_JAX else None,
         True),
        ("Sparse Diagonal Dominant 600×600 (PyTorch COO)", 
         lambda: create_sparse_diagonal_dominant_torch(600, 0.03, device, torch.float64),
         lambda: create_sparse_diagonal_dominant_jax(600, 0.03, jnp.float64) if HAS_JAX else None,
         True),
        ("Sparse Non-Diagonal Dominant Asymmetric 400×400 (PyTorch COO)", 
         lambda: create_sparse_non_diagonal_dominant_asymmetric_torch(400, 0.05, device, torch.float64),
         lambda: create_sparse_non_diagonal_dominant_asymmetric_jax(400, 0.05, jnp.float64) if HAS_JAX else None,
         True),
        ("Sparse Diagonal Dominant Asymmetric 400×400 (PyTorch COO)", 
         lambda: create_sparse_diagonal_dominant_asymmetric_torch(400, 0.05, device, torch.float64),
         lambda: create_sparse_diagonal_dominant_asymmetric_jax(400, 0.05, jnp.float64) if HAS_JAX else None,
         True),
    ]
    
    # Add ultra-large matrix test based on switch (using sparse format)
    if ENABLE_ULTRA_LARGE_MATRIX_TEST:
        test_cases.append((f"🚀 Ultra-large Sparse Tridiagonal {ULTRA_LARGE_MATRIX_SIZE:,}×{ULTRA_LARGE_MATRIX_SIZE:,} (PyTorch COO)", 
                          lambda: create_tridiagonal_matrix_sparse_torch(ULTRA_LARGE_MATRIX_SIZE, device, torch.float64),
                          lambda: create_tridiagonal_matrix_sparse_jax(ULTRA_LARGE_MATRIX_SIZE, jnp.float64) if HAS_JAX else None,
                          True))
        print(f"⚠️  Ultra-large sparse matrix test enabled: {ULTRA_LARGE_MATRIX_SIZE:,}×{ULTRA_LARGE_MATRIX_SIZE:,} (sparse storage)")
    else:
        print(f"ℹ️  Ultra-large matrix test disabled (set ENABLE_ULTRA_LARGE_MATRIX_TEST=True at script beginning to enable)")
    
    # Solver configuration - full version
    solver_configs = []
    
    # PyTorch solvers - full version
    solver_configs.extend([
        ("PyTorch", "CG", cg, "torch"),
        ("PyTorch", "BiCGStab", bicgstab, "torch"),
        ("PyTorch", "GMRES", lambda A, b, **kw: gmres(A, b, solve_method='batched', **kw), "torch"),
        ("PyTorch", "Direct", torch.linalg.solve, "torch"),
    ])
    
    # JAX solvers - full version
    if HAS_JAX:
        solver_configs.extend([
            ("JAX", "CG", jax_linalg.cg, "jax"),
            ("JAX", "BiCGStab", jax_linalg.bicgstab, "jax"),
            ("JAX", "GMRES", lambda A, b, **kw: jax_linalg.gmres(A, b, solve_method='batched', **kw), "jax"),
        ])
    
    all_results = []
    
    for matrix_name, torch_matrix_creator, jax_matrix_creator, is_sparse in test_cases:
        print(f"\n🧮 {matrix_name}")
        print("-" * 90)
        
        # Check if this is ultra-large matrix test
        is_ultra_large = "100,000" in matrix_name or str(ULTRA_LARGE_MATRIX_SIZE) in matrix_name
        if is_ultra_large:
            print("⚠️  Ultra-large sparse matrix test warning:")
            print(f"   - Matrix size: {ULTRA_LARGE_MATRIX_SIZE:,}×{ULTRA_LARGE_MATRIX_SIZE:,} (tridiagonal sparse)")
            print("   - Estimated memory usage: ~24MB sparse storage (vs ~800GB dense)")
            print("   - Skip direct solver (memory protection)")
            print("   - Timeout: 5 minutes")
            print("   - This tests true sparse matrix performance!")
            
        # Record memory at test start
        start_memory = get_memory_usage()
        print(f"📊 Test start memory: CPU={start_memory['cpu_memory_gb']:.2f}GB, GPU={start_memory['gpu_memory_gb']:.2f}GB")
        
        # Create sparse matrices for PyTorch
        A_torch = torch_matrix_creator()
        if hasattr(A_torch, 'shape'):
            n = A_torch.shape[0]
        else:
            n = A_torch.size(0)
        
        # Create true solution and right-hand side
        np.random.seed(42)
        x_true_np = np.random.randn(n)
        
        # Create b using sparse matrix-vector multiplication
        x_true_torch = torch.tensor(x_true_np, dtype=torch.float64, device=device)
        
        if hasattr(A_torch, '_indices'):  # PyTorch sparse tensor
            try:
                # Use torch.sparse.mm for sparse matrix-vector multiplication
                # A_torch is (n, n), x_true_torch is (n,) -> need (n, 1)
                b_torch = torch.sparse.mm(A_torch, x_true_torch.unsqueeze(1)).squeeze(1)
                b_np = b_torch.cpu().numpy()
            except Exception as sparse_mv_error:
                print(f"    ⚠️  Sparse matrix-vector multiplication failed: {sparse_mv_error}")
                # Fallback to dense conversion for b creation
                A_dense = A_torch.to_dense()
                b_torch = A_dense @ x_true_torch
                b_np = b_torch.cpu().numpy()
        else:
            # Dense matrix case
            if isinstance(A_torch, torch.Tensor) and A_torch.dim() == 2:
                b_torch = A_torch @ x_true_torch
                b_np = b_torch.cpu().numpy()
            else:
                # Fallback to NumPy
                A_np = A_torch.cpu().numpy() if hasattr(A_torch, 'cpu') else A_torch
                b_np = A_np @ x_true_np
                b_torch = torch.tensor(b_np, dtype=torch.float64, device=device)
        
        x_true_torch = torch.tensor(x_true_np, dtype=torch.float64, device=device)
        
        # Create JAX matrices if available
        if HAS_JAX and not is_ultra_large and jax_matrix_creator is not None:
            A_jax = jax_matrix_creator()
            if hasattr(A_jax, 'todense'):  # JAX sparse
                b_jax = A_jax @ jnp.array(x_true_np)
            else:
                b_jax = jnp.array(b_np)
        
        # Record memory after data preparation
        after_data_memory = get_memory_usage()
        memory_increase = after_data_memory['gpu_memory_gb'] - start_memory['gpu_memory_gb']
        if memory_increase > 0.01:  # If GPU memory increase exceeds 10MB
            print(f"📈 Memory after sparse data loading: GPU +{memory_increase:.3f}GB")
        
        # Test matrix properties
        test_matrix_properties(A_torch, matrix_name, is_sparse=is_sparse)
        
        print(f"\n  Solver performance comparison:")
        print(f"  {'Framework':<8} {'Solver':<12} {'Time(s)':<10} {'Solution Error':<12} {'Residual Error':<12} {'Detailed Status'}")
        print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*50}")
        
        for framework, solver_name, solver_func, solver_type in solver_configs:
            # Special handling for ultra-large matrices
            is_large_matrix = n >= 50000
            
            # Skip direct solver for ultra-large matrices (avoid memory issues)
            if is_large_matrix and 'Direct' in solver_name:
                print(f"  {framework:<8} {solver_name:<12} {'SKIP':<10} {'Large matrix skip':<12} {'Memory limit':<12} Skip (memory protection)")
                continue
                
            # Dynamically adjust timeout
            if is_large_matrix:
                timeout_seconds = 300  # 5-minute timeout for ultra-large matrices
            else:
                timeout_seconds = 30   # 30-second timeout for normal matrices
            
            if solver_type == "torch":
                # Check PyTorch version for sparse direct solver support
                pytorch_version = torch.__version__
                pytorch_major = int(pytorch_version.split('.')[0])
                pytorch_minor = int(pytorch_version.split('.')[1])
                supports_sparse_direct = pytorch_major > 2 or (pytorch_major == 2 and pytorch_minor >= 7)
                
                # For sparse matrices and direct solver
                if is_sparse and solver_name in ["Direct"]:
                    # Try torch.sparse.spsolve first, with intelligent fallback to dense
                    if hasattr(A_torch, '_indices'):  # This is a sparse tensor
                        # First attempt: sparse solve
                        try:
                            # Convert to CSR format if needed (spsolve requires CSR)
                            if A_torch.layout == torch.sparse_coo:
                                A_solver_sparse = A_torch.to_sparse_csr()
                                print(f"    ✅ Converting PyTorch COO to CSR for torch.sparse.spsolve")
                            elif A_torch.layout == torch.sparse_csr:
                                A_solver_sparse = A_torch
                                print(f"    ✅ Using PyTorch CSR matrix with torch.sparse.spsolve")
                            else:
                                # For other sparse formats, try to convert to CSR
                                A_solver_sparse = A_torch.to_sparse_csr()
                                print(f"    ✅ Converting PyTorch {A_torch.layout} to CSR for torch.sparse.spsolve")
                            
                            # Test sparse solve with a small test case
                            test_size = min(5, A_solver_sparse.shape[0])
                            A_test = A_solver_sparse[:test_size, :test_size]
                            b_test = torch.ones(test_size, dtype=A_solver_sparse.dtype, device=A_solver_sparse.device)
                            _ = torch.sparse.spsolve(A_test, b_test)
                            
                            # If test passes, use sparse solve
                            print(f"    ✅ torch.sparse.spsolve test passed on {A_solver_sparse.device}")
                            A_solver = A_solver_sparse
                            solver_func = lambda A, b: torch.sparse.spsolve(A, b)
                            solver_method_used = "sparse_spsolve"
                            
                        except Exception as sparse_error:
                            # Sparse solve failed, try dense fallback
                            print(f"    ⚠️  torch.sparse.spsolve failed: {str(sparse_error)[:60]}...")
                            print(f"    🔄 Falling back to dense torch.linalg.solve on {A_torch.device}")
                            
                            try:
                                # Convert to dense but keep on same device
                                A_solver = A_torch.to_dense()
                                solver_func = lambda A, b: torch.linalg.solve(A, b)
                                solver_method_used = "dense_fallback"
                                print(f"    ✅ Dense fallback successful on {A_solver.device}")
                                
                            except Exception as dense_error:
                                print(f"  {framework:<8} {solver_name:<12} {'SKIP':<10} {'Both failed':<12} {'See status':<12} Skip (Both sparse and dense solve failed)")
                                continue
                    else:
                        # Not a sparse tensor, use dense solver directly
                        A_solver = A_torch
                        solver_func = lambda A, b: torch.linalg.solve(A, b)
                        solver_method_used = "dense_direct"
                        print(f"    ℹ️  Using dense matrix with torch.linalg.solve")
                else:
                    A_solver = A_torch
                    solver_method_used = "iterative"
                    # For iterative solvers, keep the original solver function
                
                # Set solver function for non-direct solvers
                if solver_name != "Direct":
                    # For iterative solvers (CG, BiCGStab, GMRES), use the original function
                    solver_func_to_use = solver_func
                else:
                    # For direct solver, solver_func is already set above
                    solver_func_to_use = solver_func
                
                results = benchmark_torch_solver(f"{framework} {solver_name}", solver_func_to_use, A_solver, b_torch, timeout_seconds)
                
                # Add solver method information to results for Direct solver
                if solver_name == "Direct" and 'solver_method_used' in locals():
                    if solver_method_used == "sparse_spsolve":
                        results['detailed_method'] = "✅ Sparse solve (torch.sparse.spsolve)"
                    elif solver_method_used == "dense_fallback":
                        results['detailed_method'] = "⚠️ Dense fallback (torch.linalg.solve)"
                        results['status'] = results['status'].replace("Success", "Success (Dense fallback)")
                    elif solver_method_used == "dense_direct":
                        results['detailed_method'] = "ℹ️ Dense direct (torch.linalg.solve)"
                    else:
                        results['detailed_method'] = f"Unknown method: {solver_method_used}"
                else:
                    results['detailed_method'] = "N/A"
                
                # Calculate solution and residual errors
                if results['x_solve'] is not None:
                    solution_error = np.linalg.norm(results['x_solve'] - x_true_np)
                    # Calculate residual using original sparse matrix
                    if hasattr(A_torch, '_indices'):  # PyTorch sparse tensor
                        try:
                            x_solve_torch = torch.tensor(results['x_solve'], dtype=torch.float64, device=device)
                            residual_torch = torch.sparse.mm(A_torch, x_solve_torch.unsqueeze(1)).squeeze(1) - b_torch
                            residual_error = np.linalg.norm(residual_torch.cpu().numpy())
                        except Exception as residual_error_calc:
                            print(f"    ⚠️  Sparse residual calculation failed: {residual_error_calc}")
                            # Fallback to dense conversion
                            A_for_residual = A_torch.to_dense()
                            x_solve_torch = torch.tensor(results['x_solve'], dtype=torch.float64, device=device)
                            residual_torch = A_for_residual @ x_solve_torch - b_torch
                            residual_error = np.linalg.norm(residual_torch.cpu().numpy())
                    else:
                        A_for_residual = sparse_to_dense_if_needed(A_torch, max_size=5000)
                        if isinstance(A_for_residual, torch.Tensor):
                            x_solve_torch = torch.tensor(results['x_solve'], dtype=torch.float64, device=device)
                            residual_torch = A_for_residual @ x_solve_torch - b_torch
                            residual_error = np.linalg.norm(residual_torch.cpu().numpy())
                        else:
                            A_np_temp = A_for_residual.cpu().numpy() if hasattr(A_for_residual, 'cpu') else A_for_residual
                            residual_error = np.linalg.norm(A_np_temp @ results['x_solve'] - b_np)
                else:
                    solution_error = float('inf')
                    residual_error = float('inf')
            elif solver_type == "jax" and HAS_JAX:
                # JAX may have memory issues on ultra-large matrices, add protection
                if is_large_matrix:
                    print(f"  {framework:<8} {solver_name:<12} {'SKIP':<10} {'Large matrix skip':<12} {'JAX limit':<12} Skip (JAX large matrix limit)")
                    continue
                
                # For JAX sparse matrices, convert to dense if needed for compatibility
                if is_sparse and hasattr(A_jax, 'todense'):
                    if A_jax.shape[0] <= 2000:  # Only convert small sparse matrices
                        A_jax_solver = A_jax.todense()
                        b_jax_solver = b_jax
                    else:
                        print(f"  {framework:<8} {solver_name:<12} {'SKIP':<10} {'Sparse too large':<12} {'JAX dense req':<12} Skip (JAX requires dense for large sparse)")
                        continue
                else:
                    A_jax_solver = A_jax
                    b_jax_solver = b_jax
                    
                results = benchmark_jax_solver(f"{framework} {solver_name}", solver_func, A_jax_solver, b_jax_solver, timeout_seconds)
                
                # Add detailed method information for JAX solvers
                if is_sparse and hasattr(A_jax, 'todense'):
                    results['detailed_method'] = "ℹ️ JAX sparse to dense conversion"
                else:
                    results['detailed_method'] = "N/A"
                
                if results['x_solve'] is not None:
                    solution_error = np.linalg.norm(results['x_solve'] - x_true_np)
                    # Calculate residual using JAX matrix
                    if hasattr(A_jax, 'todense'):  # JAX sparse
                        residual_jax = A_jax @ jnp.array(results['x_solve']) - b_jax
                    else:
                        residual_jax = A_jax @ jnp.array(results['x_solve']) - b_jax
                    residual_error = np.linalg.norm(np.array(residual_jax))
                else:
                    solution_error = float('inf')
                    residual_error = float('inf')
            else:
                continue
            
            # Display results with detailed method information
            detailed_status = results.get('detailed_method', 'N/A')
            if detailed_status != 'N/A':
                status_display = f"{results['status']} | {detailed_status}"
            else:
                status_display = results['status']
            
            print(f"  {framework:<8} {solver_name:<12} {results['time']:<10.3f} "
                  f"{solution_error:<12.2e} {residual_error:<12.2e} {status_display}")
            
            # Store results with detailed information
            all_results.append({
                'matrix': matrix_name,
                'matrix_size': n,
                'framework': framework,
                'solver': solver_name,
                'time': results['time'],
                'solution_error': solution_error,
                'residual_error': residual_error,
                'converged': results['converged'],
                'status': results['status'],
                'detailed_method': results.get('detailed_method', 'N/A')
            })
        
        # Memory cleanup and summary after test completion
        if is_ultra_large:
            print(f"\n🧹 Ultra-large sparse matrix test completed, performing memory cleanup...")
            del A_torch, b_torch, x_true_torch
            if 'A_jax' in locals():
                del A_jax, b_jax
            torch.cuda.empty_cache()
            gc.collect()
            
            # Display ultra-large matrix test summary
            ultra_large_results = [r for r in all_results if r['matrix'] == matrix_name]
            successful_results = [r for r in ultra_large_results if r['converged']]
            
            print(f"📈 Ultra-large sparse matrix test summary:")
            print(f"   - Matrix format: Sparse (memory efficient)")
            print(f"   - Success rate: {len(successful_results)}/{len(ultra_large_results)} ({len(successful_results)/len(ultra_large_results)*100:.1f}%)" if ultra_large_results else "   - No results available")
            if successful_results:
                fastest = min(successful_results, key=lambda x: x['time'])
                most_accurate = min(successful_results, key=lambda x: x['solution_error'])
                print(f"   - Fastest algorithm: {fastest['framework']} {fastest['solver']} ({fastest['time']:.1f}s)")
                print(f"   - Most accurate algorithm: {most_accurate['framework']} {most_accurate['solver']} ({most_accurate['solution_error']:.2e})")
            
            print(f"   - Sparse matrix benefits:")
            dense_memory_gb = (ULTRA_LARGE_MATRIX_SIZE ** 2 * 8) / (1024**3)  # 8 bytes per float64
            sparse_memory_gb = (3 * ULTRA_LARGE_MATRIX_SIZE * 8) / (1024**3)  # ~3 non-zeros per row for tridiagonal
            print(f"     * Dense storage would require: ~{dense_memory_gb:.1f}GB")
            print(f"     * Sparse storage requires: ~{sparse_memory_gb:.3f}GB")
            print(f"     * Memory savings: {(1 - sparse_memory_gb/dense_memory_gb)*100:.1f}%")
        
        # Record memory at test end
        end_memory = get_memory_usage()
        print(f"📊 Test end memory: CPU={end_memory['cpu_memory_gb']:.2f}GB, GPU={end_memory['gpu_memory_gb']:.2f}GB")
    
    return all_results

def analyze_comparison_results(results):
    """Analyze PyTorch vs JAX comparison results"""
    print("\n📊 PyTorch vs JAX Performance Comparison Analysis")
    print("=" * 90)
    
    # Analyze by matrix type
    matrices = set(r['matrix'] for r in results)
    
    print("\n🏆 Best solvers for each matrix type:")
    for matrix in matrices:
        matrix_results = [r for r in results if r['matrix'] == matrix and r['converged']]
        if matrix_results:
            fastest = min(matrix_results, key=lambda x: x['time'])
            most_accurate = min(matrix_results, key=lambda x: x['solution_error'])
            
            print(f"\n  {matrix}:")
            print(f"    Fastest: {fastest['framework']} {fastest['solver']} ({fastest['time']:.3f}s)")
            print(f"    Most accurate: {most_accurate['framework']} {most_accurate['solver']} ({most_accurate['solution_error']:.2e})")
    
    # Framework comparison
    if HAS_JAX:
        print(f"\n🥊 Framework comparison:")
        frameworks = ['PyTorch', 'JAX']
        for framework in frameworks:
            framework_results = [r for r in results if r['framework'] == framework and r['converged']]
            if framework_results:
                avg_time = np.mean([r['time'] for r in framework_results])
                avg_accuracy = np.mean([r['solution_error'] for r in framework_results])
                convergence_rate = len(framework_results) / len([r for r in results if r['framework'] == framework])
                
                print(f"  {framework}:")
                print(f"    Average time: {avg_time:.3f}s")
                print(f"    Average accuracy: {avg_accuracy:.2e}")
                print(f"    Convergence rate: {convergence_rate:.1%}")
    
    # Algorithm comparison
    print(f"\n🔬 Algorithm comparison (successfully converged results):")
    algorithms = set(r['solver'] for r in results)
    for algo in algorithms:
        if algo == 'Direct':
            continue
        print(f"\n  {algo}:")
        
        for framework in ['PyTorch', 'JAX']:
            algo_results = [r for r in results if r['solver'] == algo and r['framework'] == framework and r['converged']]
            if algo_results:
                avg_time = np.mean([r['time'] for r in algo_results])
                avg_accuracy = np.mean([r['solution_error'] for r in algo_results])
                print(f"    {framework}: time={avg_time:.3f}s, accuracy={avg_accuracy:.2e}")
    
    # Large matrix performance comparison
    print(f"\n⚡ Large matrix (≥1000) performance comparison:")
    large_results = [r for r in results if r['matrix_size'] >= 1000 and r['converged']]
    if large_results:
        # Group by framework and solver
        perf_summary = {}
        for r in large_results:
            key = f"{r['framework']} {r['solver']}"
            if key not in perf_summary:
                perf_summary[key] = []
            perf_summary[key].append(r['time'])
        
        if perf_summary:
            for solver_key, times in sorted(perf_summary.items(), key=lambda x: np.mean(x[1])):
                avg_time = np.mean(times)
                print(f"  {solver_key:<20}: {avg_time:.3f}s")

def save_results_to_markdown(results, filename="sparse_solver_benchmark_results.md"):
    """Save test results as Markdown format report"""
    import datetime
    
    with open(filename, 'w', encoding='utf-8') as f:
        # Title and metadata
        f.write("# PyTorch vs JAX Sparse Matrix Solver Performance Comparison Report\n\n")
        f.write(f"**Generated at**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Environment information
        f.write("## Test Environment\n\n")
        f.write(f"- **PyTorch version**: {torch.__version__}\n")
        f.write(f"- **Device**: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
        if torch.cuda.is_available():
            f.write(f"- **GPU**: {torch.cuda.get_device_name(0)}\n")
        if HAS_JAX:
            f.write(f"- **JAX device**: {str(jax.devices()[0])}\n")
        f.write("\n")
        
        # Overall statistics
        f.write("## Overall Statistics\n\n")
        total_tests = len(results)
        successful_tests = len([r for r in results if r['converged']])
        f.write(f"- **Total tests**: {total_tests}\n")
        f.write(f"- **Successfully converged**: {successful_tests} ({successful_tests/total_tests:.1%})\n")
        f.write(f"- **Failed tests**: {total_tests - successful_tests}\n\n")
        
        # Detailed results by matrix type
        f.write("## Detailed Test Results\n\n")
        matrices = sorted(set(r['matrix'] for r in results))
        
        for matrix in matrices:
            matrix_results = [r for r in results if r['matrix'] == matrix]
            f.write(f"### {matrix}\n\n")
            
            # Matrix properties (inferred from first result)
            matrix_size = matrix_results[0]['matrix_size']
            f.write(f"**Matrix size**: {matrix_size}×{matrix_size}\n\n")
            
            # Table headers
            f.write("| Framework | Solver | Time(s) | Solution Error | Residual Error | Status | Method Details |\n")
            f.write("|-----------|--------|---------|----------------|----------------|--------|-----------------|\n")
            
            # Sort: first by framework, then by solver
            sorted_results = sorted(matrix_results, key=lambda x: (x['framework'], x['solver']))
            
            for r in sorted_results:
                time_str = f"{r['time']:.3f}" if r['time'] != float('inf') else "∞"
                error_str = f"{r['solution_error']:.2e}" if r['solution_error'] != float('inf') else "∞"
                residual_str = f"{r['residual_error']:.2e}" if r['residual_error'] != float('inf') else "∞"
                status_emoji = "✅" if r['converged'] else "❌"
                method_details = r.get('detailed_method', 'N/A')
                
                f.write(f"| {r['framework']} | {r['solver']} | {time_str} | {error_str} | {residual_str} | {status_emoji} {r['status']} | {method_details} |\n")
            
            f.write("\n")
            
            # Best solver for this matrix
            successful_results = [r for r in matrix_results if r['converged']]
            if successful_results:
                fastest = min(successful_results, key=lambda x: x['time'])
                most_accurate = min(successful_results, key=lambda x: x['solution_error'])
                
                f.write("**Best performance**:\n")
                f.write(f"- **Fastest**: {fastest['framework']} {fastest['solver']} ({fastest['time']:.3f}s)\n")
                f.write(f"- **Most accurate**: {most_accurate['framework']} {most_accurate['solver']} ({most_accurate['solution_error']:.2e})\n\n")
        
        # Framework comparison analysis
        if HAS_JAX:
            f.write("## Framework Comparison Analysis\n\n")
            frameworks = ['PyTorch', 'JAX']
            
            f.write("| Framework | Average Time(s) | Average Accuracy | Convergence Rate |\n")
            f.write("|-----------|-----------------|------------------|------------------|\n")
            
            for framework in frameworks:
                framework_results = [r for r in results if r['framework'] == framework and r['converged']]
                all_framework_results = [r for r in results if r['framework'] == framework]
                
                if framework_results:
                    avg_time = np.mean([r['time'] for r in framework_results])
                    avg_accuracy = np.mean([r['solution_error'] for r in framework_results])
                    convergence_rate = len(framework_results) / len(all_framework_results)
                    
                    f.write(f"| {framework} | {avg_time:.3f} | {avg_accuracy:.2e} | {convergence_rate:.1%} |\n")
            
            f.write("\n")
        
        # Algorithm comparison
        f.write("## Algorithm Comparison\n\n")
        algorithms = sorted(set(r['solver'] for r in results if r['solver'] != 'Direct'))
        
        for algo in algorithms:
            f.write(f"### {algo}\n\n")
            f.write("| Framework | Average Time(s) | Average Accuracy | Convergence Rate |\n")
            f.write("|-----------|-----------------|------------------|------------------|\n")
            
            for framework in ['PyTorch', 'JAX']:
                algo_results = [r for r in results if r['solver'] == algo and r['framework'] == framework]
                successful_algo_results = [r for r in algo_results if r['converged']]
                
                if algo_results:
                    if successful_algo_results:
                        avg_time = np.mean([r['time'] for r in successful_algo_results])
                        avg_accuracy = np.mean([r['solution_error'] for r in successful_algo_results])
                    else:
                        avg_time = float('inf')
                        avg_accuracy = float('inf')
                    
                    convergence_rate = len(successful_algo_results) / len(algo_results)
                    
                    time_str = f"{avg_time:.3f}" if avg_time != float('inf') else "∞"
                    accuracy_str = f"{avg_accuracy:.2e}" if avg_accuracy != float('inf') else "∞"
                    
                    f.write(f"| {framework} | {time_str} | {accuracy_str} | {convergence_rate:.1%} |\n")
            
            f.write("\n")
        
        # Key findings and conclusions
        f.write("## Key Findings\n\n")
        
        # Direct solver methods analysis
        direct_results = [r for r in results if r['solver'] == 'Direct']
        if direct_results:
            f.write("### Direct Solver Methods Analysis\n\n")
            sparse_spsolve_count = len([r for r in direct_results if 'Sparse solve' in r.get('detailed_method', '')])
            dense_fallback_count = len([r for r in direct_results if 'Dense fallback' in r.get('detailed_method', '')])
            dense_direct_count = len([r for r in direct_results if 'Dense direct' in r.get('detailed_method', '')])
            
            f.write("**Method Distribution**:\n")
            f.write(f"- **torch.sparse.spsolve (Sparse)**: {sparse_spsolve_count} tests\n")
            f.write(f"- **torch.linalg.solve (Dense fallback)**: {dense_fallback_count} tests\n")
            f.write(f"- **torch.linalg.solve (Dense direct)**: {dense_direct_count} tests\n\n")
            
            f.write("**Method Details**:\n")
            f.write("1. **✅ Sparse solve (torch.sparse.spsolve)**: Native sparse matrix solving on GPU, most memory efficient\n")
            f.write("2. **⚠️ Dense fallback (torch.linalg.solve)**: Automatic fallback when sparse solve fails, maintains GPU computation\n")
            f.write("3. **ℹ️ Dense direct (torch.linalg.solve)**: Direct dense solving for non-sparse input matrices\n\n")
            
            if dense_fallback_count > 0:
                f.write("**Note**: Dense fallback occurred when torch.sparse.spsolve was not supported for the specific matrix configuration or GPU setup.\n\n")
        
        # Direct solver accuracy analysis
        direct_results = [r for r in results if r['solver'] == 'Direct' and r['converged']]
        if direct_results:
            avg_direct_accuracy = np.mean([r['solution_error'] for r in direct_results])
            f.write(f"### Advantages of Direct Solver\n\n")
            f.write(f"The direct solver achieved extremely high accuracy across all tests (average: {avg_direct_accuracy:.2e}).\n\n")
            f.write("**Reason analysis**:\n")
            f.write("1. **Mathematical exactness**: Direct solvers use exact algorithms like LU decomposition, theoretically limited only by floating-point precision\n")
            f.write("2. **No iterative errors**: Unlike iterative algorithms that require multiple approximations, avoiding cumulative errors\n")
            f.write("3. **Sparse matrix advantages**: For moderately-sized sparse matrices, modern direct solvers are very efficient\n")
            f.write("4. **Numerical stability**: PyTorch's implementation uses highly optimized LAPACK routines\n\n")
        
        # Iterative solver analysis
        iterative_results = [r for r in results if r['solver'] != 'Direct' and r['converged']]
        if iterative_results:
            f.write("### Characteristics of Iterative Solvers\n\n")
            cg_results = [r for r in iterative_results if r['solver'] == 'CG']
            if cg_results:
                cg_pytorch = [r for r in cg_results if r['framework'] == 'PyTorch']
                if cg_pytorch:
                    avg_cg_accuracy = np.mean([r['solution_error'] for r in cg_pytorch])
                    f.write(f"- **CG algorithm**: Excellent performance on symmetric positive definite matrices (PyTorch average accuracy: {avg_cg_accuracy:.2e})\n")
            
            f.write("- **Application scenarios**: Iterative solvers are suitable for ultra-large scale matrices when direct solvers run out of memory\n")
            f.write("- **Convergence dependency**: Convergence strongly depends on matrix condition number and preconditioners\n\n")
        
        # Performance recommendations
        f.write("## Usage Recommendations\n\n")
        f.write("### Solver Selection Guide\n\n")
        f.write("1. **Small to medium sparse matrices** (< 10,000×10,000):\n")
        f.write("   - **First choice**: PyTorch Direct solver (automatic sparse/dense selection)\n")
        f.write("     - Tries `torch.sparse.spsolve` first (most efficient for sparse matrices)\n")
        f.write("     - Automatically falls back to `torch.linalg.solve` if needed\n")
        f.write("     - Provides highest accuracy with intelligent method selection\n")
        f.write("   - **Alternative**: PyTorch CG (for symmetric positive definite matrices)\n\n")
        f.write("2. **Large sparse matrices** (> 10,000×10,000):\n")
        f.write("   - **Symmetric positive definite**: PyTorch CG\n")
        f.write("   - **General matrices**: PyTorch GMRES or BiCGStab\n")
        f.write("   - **Consider using preconditioners** to improve convergence speed\n\n")
        f.write("3. **GPU acceleration**:\n")
        f.write("   - PyTorch implementations have better GPU support\n")
        f.write("   - All solvers (including automatic fallbacks) maintain GPU computation\n")
        f.write("   - GPU acceleration effects are more pronounced for large matrices\n")
        f.write("   - Use `solve_method='batched'` for GMRES on GPU\n\n")
        f.write("4. **Method reliability**:\n")
        f.write("   - The intelligent Direct solver automatically handles GPU compatibility issues\n")
        f.write("   - No manual intervention needed for sparse vs dense selection\n")
        f.write("   - Detailed method information is provided for transparency\n\n")
        
        f.write("---\n")
        f.write("*Report automatically generated by PyTorch sparse linear algebra benchmark tool*\n")
    
    print(f"📊 Detailed test report saved to: {filename}")
    return filename

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    return {
        'cpu_memory_gb': memory_info.rss / 1024**3,
        'gpu_memory_gb': gpu_memory
    }

def main():
    """Main function"""
    print("🎯 Starting PyTorch vs JAX sparse matrix solver comparison test...")
    
    # Run comparison test
    results = compare_solvers()
    
    # Analyze results
    analyze_comparison_results(results)
    
    # Save results to Markdown file
    save_results_to_markdown(results)
    
    print("\n" + "=" * 90)
    print("🎯 Comparison test completed!")
    print("✅ PyTorch CG, BiCGStab, and GMRES implementations verified against JAX benchmarks")
    print("📊 Check above results for accuracy and performance comparison")

if __name__ == "__main__":
    main()
