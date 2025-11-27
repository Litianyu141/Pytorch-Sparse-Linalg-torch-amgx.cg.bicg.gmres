"""
Module availability detection for pytorch_sparse_solver.

This module provides utilities to check which solver backends are available
on the current system. Each module (A, B, C) can be installed independently.
"""

import warnings
from typing import Dict, List, Optional
from functools import lru_cache


@lru_cache(maxsize=1)
def check_module_a_available() -> bool:
    """
    Check if Module A (JAX-style iterative solvers) is available.

    Module A requires:
    - PyTorch >= 2.0

    Returns:
        bool: True if Module A is available
    """
    try:
        import torch
        # Module A is pure PyTorch, so it should always be available if torch is installed
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def check_module_b_available() -> bool:
    """
    Check if Module B (PyAMGX integration) is available.

    Module B requires:
    - PyTorch >= 2.0
    - NVIDIA GPU with CUDA support
    - pyamgx library installed (with AMGX backend)

    Returns:
        bool: True if Module B is available
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False

        import pyamgx
        return True
    except ImportError:
        return False
    except Exception as e:
        warnings.warn(f"Module B check failed: {e}")
        return False


@lru_cache(maxsize=1)
def check_module_c_available() -> bool:
    """
    Check if Module C (cuDSS direct solver) is available.

    Module C requires:
    - PyTorch >= 2.7 compiled with cuDSS support
    - NVIDIA GPU with CUDA support
    - cuDSS library installed

    Returns:
        bool: True if Module C is available
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False

        # Check if torch.sparse.spsolve is available (requires cuDSS-enabled PyTorch)
        if not hasattr(torch.sparse, 'spsolve'):
            return False

        # Try a simple test to verify cuDSS is actually working
        try:
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
            if "cuDSS" in str(e) or "spsolve" in str(e).lower():
                return False
            return False
        except Exception:
            return False

    except ImportError:
        return False
    except Exception as e:
        warnings.warn(f"Module C check failed: {e}")
        return False


def get_available_backends() -> Dict[str, bool]:
    """
    Get a dictionary of all available solver backends.

    Returns:
        Dict[str, bool]: Dictionary mapping backend names to availability status
    """
    return {
        'module_a': check_module_a_available(),  # JAX-style iterative solvers
        'module_b': check_module_b_available(),  # PyAMGX integration
        'module_c': check_module_c_available(),  # cuDSS direct solver
    }


def get_available_backend_list() -> List[str]:
    """
    Get a list of available backend names.

    Returns:
        List[str]: List of available backend names
    """
    backends = get_available_backends()
    return [name for name, available in backends.items() if available]


def print_availability_report() -> None:
    """Print a detailed availability report for all modules."""
    print("=" * 60)
    print("PyTorch Sparse Solver - Module Availability Report")
    print("=" * 60)

    # Module A
    module_a = check_module_a_available()
    status_a = "✅ Available" if module_a else "❌ Not Available"
    print(f"\nModule A (JAX-style Iterative Solvers): {status_a}")
    print("  - CG (Conjugate Gradient)")
    print("  - BiCGStab (Bi-Conjugate Gradient Stabilized)")
    print("  - GMRES (Generalized Minimal Residual)")
    if not module_a:
        print("  ⚠️  Requires: PyTorch >= 2.0")

    # Module B
    module_b = check_module_b_available()
    status_b = "✅ Available" if module_b else "❌ Not Available"
    print(f"\nModule B (PyAMGX GPU Solver): {status_b}")
    print("  - AMGX CG with AMG preconditioner")
    print("  - AMGX BiCGStab with AMG preconditioner")
    print("  - AMGX GMRES with AMG preconditioner")
    print("  - Automatic differentiation support")
    if not module_b:
        print("  ⚠️  Requires: NVIDIA GPU + CUDA + pyamgx library")
        print("  📖 Installation: https://github.com/NVIDIA/AMGX")

    # Module C
    module_c = check_module_c_available()
    status_c = "✅ Available" if module_c else "❌ Not Available"
    print(f"\nModule C (cuDSS Direct Solver): {status_c}")
    print("  - torch.sparse.spsolve (LU factorization)")
    print("  - GPU-accelerated direct method")
    if not module_c:
        print("  ⚠️  Requires: PyTorch >= 2.7 compiled with USE_CUDSS=1")
        print("  📖 See README.md for compilation instructions")

    print("\n" + "=" * 60)

    available_count = sum([module_a, module_b, module_c])
    print(f"Total: {available_count}/3 modules available")
    print("=" * 60)


if __name__ == "__main__":
    print_availability_report()
