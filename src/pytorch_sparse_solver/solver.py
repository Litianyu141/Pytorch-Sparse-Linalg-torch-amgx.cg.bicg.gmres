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
Unified Sparse Linear System Solver Interface

This module provides a unified interface for solving sparse linear systems Ax = b
using various backends:
- Module A: JAX-style iterative solvers (CG, BiCGStab, GMRES) - pure PyTorch
- Module B: PyAMGX GPU-accelerated solvers with AMG preconditioner
- Module C: cuDSS direct solver via torch.sparse.spsolve

The interface automatically detects which backends are available and provides
a seamless way to switch between them.

Example:
    >>> from pytorch_sparse_solver import SparseSolver
    >>>
    >>> # Create solver with automatic backend selection
    >>> solver = SparseSolver()
    >>> print(f"Available backends: {solver.available_backends}")
    >>>
    >>> # Solve using default backend
    >>> x, info = solver.solve(A, b)
    >>>
    >>> # Or specify a particular backend
    >>> x, info = solver.solve(A, b, method='cg', backend='module_a')
"""

import torch
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass

from .utils.availability import (
    check_module_a_available,
    check_module_b_available,
    check_module_c_available,
    get_available_backends,
)


class SolverMethod(Enum):
    """Available solver methods across all backends."""
    CG = "cg"
    BICGSTAB = "bicgstab"
    GMRES = "gmres"
    DIRECT = "direct"  # Only available with Module C


class SolverBackend(Enum):
    """Available solver backends."""
    MODULE_A = "module_a"  # JAX-style iterative solvers
    MODULE_B = "module_b"  # PyAMGX GPU solvers
    MODULE_C = "module_c"  # cuDSS direct solver
    AUTO = "auto"          # Automatic selection


@dataclass
class SolverResult:
    """Result from sparse solver."""
    x: torch.Tensor           # Solution vector
    converged: bool           # Whether solver converged
    iterations: Optional[int] # Number of iterations (None for direct solvers)
    residual: Optional[float] # Final residual norm
    backend: str              # Backend used
    method: str               # Method used


class SparseSolver:
    """
    Unified sparse linear system solver with multiple backends.

    This class provides a unified interface for solving sparse linear systems Ax = b
    using various backends. It automatically detects available backends and allows
    users to choose the most appropriate solver for their problem.

    Attributes:
        available_backends: List of available backend names
        default_backend: Default backend to use when not specified
        default_method: Default solver method

    Example:
        >>> solver = SparseSolver()
        >>> x, info = solver.solve(A, b)
        >>>
        >>> # Specify method and backend
        >>> x, info = solver.solve(A, b, method='gmres', backend='module_a')
        >>>
        >>> # Use context manager for batch solving
        >>> with solver.session() as s:
        ...     x1, _ = s.solve(A1, b1)
        ...     x2, _ = s.solve(A2, b2)
    """

    def __init__(
        self,
        default_backend: str = "auto",
        default_method: str = "cg",
        verbose: bool = False
    ):
        """
        Initialize the sparse solver.

        Args:
            default_backend: Default backend to use ('auto', 'module_a', 'module_b', 'module_c')
            default_method: Default solver method ('cg', 'bicgstab', 'gmres', 'direct')
            verbose: Whether to print debug information
        """
        self.verbose = verbose
        self._backends_checked = False
        self._available_backends: Dict[str, bool] = {}

        self.default_backend = default_backend
        self.default_method = default_method

        # Lazy load modules
        self._module_a = None
        self._module_b = None
        self._module_c = None

    @property
    def available_backends(self) -> List[str]:
        """Get list of available backend names."""
        if not self._backends_checked:
            self._check_backends()
        return [k for k, v in self._available_backends.items() if v]

    def _check_backends(self) -> None:
        """Check which backends are available."""
        self._available_backends = get_available_backends()
        self._backends_checked = True

        if self.verbose:
            for name, available in self._available_backends.items():
                status = "✅" if available else "❌"
                print(f"  {status} {name}")

    def _load_module_a(self):
        """Lazy load Module A."""
        if self._module_a is None:
            try:
                from .module_a import cg, bicgstab, gmres
                self._module_a = {
                    'cg': cg,
                    'bicgstab': bicgstab,
                    'gmres': gmres,
                }
            except ImportError as e:
                raise RuntimeError(f"Failed to load Module A: {e}")
        return self._module_a

    def _load_module_b(self):
        """Lazy load Module B."""
        if self._module_b is None:
            try:
                from .module_b import amgx_cg, amgx_bicgstab, amgx_gmres
                self._module_b = {
                    'cg': amgx_cg,
                    'bicgstab': amgx_bicgstab,
                    'gmres': amgx_gmres,
                }
            except ImportError as e:
                raise RuntimeError(f"Failed to load Module B: {e}")
        return self._module_b

    def _load_module_c(self):
        """Lazy load Module C."""
        if self._module_c is None:
            try:
                from .module_c import cudss_spsolve
                self._module_c = {
                    'direct': cudss_spsolve,
                }
            except ImportError as e:
                raise RuntimeError(f"Failed to load Module C: {e}")
        return self._module_c

    def _select_backend(
        self,
        backend: str,
        method: str,
        A: torch.Tensor
    ) -> Tuple[str, str]:
        """
        Select the appropriate backend and method.

        Args:
            backend: Requested backend ('auto' or specific backend)
            method: Requested method
            A: Input matrix

        Returns:
            Tuple of (selected_backend, selected_method)
        """
        if not self._backends_checked:
            self._check_backends()

        available = self.available_backends

        if not available:
            raise RuntimeError("No sparse solver backends are available!")

        if backend != "auto":
            if backend not in available:
                raise ValueError(
                    f"Backend '{backend}' is not available. "
                    f"Available backends: {available}"
                )
            return backend, method

        # Auto selection logic
        # Priority: Module C (direct) > Module B (GPU AMG) > Module A (iterative)

        # For direct method, must use Module C
        if method == "direct":
            if "module_c" in available:
                return "module_c", "direct"
            else:
                raise ValueError(
                    "Direct solver requires Module C (cuDSS), which is not available. "
                    "Use an iterative method (cg, bicgstab, gmres) instead."
                )

        # For iterative methods, prefer Module B (GPU) if available and matrix is on GPU
        if A.is_cuda and "module_b" in available:
            return "module_b", method

        # Default to Module A
        if "module_a" in available:
            return "module_a", method

        # Fallback to whatever is available
        return available[0], method

    def solve(
        self,
        A: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
        b: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        method: Optional[str] = None,
        backend: Optional[str] = None,
        tol: float = 1e-5,
        atol: float = 0.0,
        maxiter: Optional[int] = None,
        M: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, SolverResult]:
        """
        Solve the sparse linear system Ax = b.

        Args:
            A: Coefficient matrix (sparse, dense, or callable for matrix-free)
            b: Right-hand side vector
            x0: Initial guess (optional)
            method: Solver method ('cg', 'bicgstab', 'gmres', 'direct')
            backend: Backend to use ('auto', 'module_a', 'module_b', 'module_c')
            tol: Relative convergence tolerance
            atol: Absolute convergence tolerance
            maxiter: Maximum number of iterations
            M: Preconditioner (callable)
            **kwargs: Additional method-specific arguments

        Returns:
            Tuple of (solution tensor, SolverResult with details)

        Example:
            >>> solver = SparseSolver()
            >>> x, result = solver.solve(A, b, method='cg', tol=1e-6)
            >>> print(f"Converged: {result.converged}, Iterations: {result.iterations}")
        """
        if method is None:
            method = self.default_method
        if backend is None:
            backend = self.default_backend

        # Handle tensor input for backend selection
        A_tensor = A if isinstance(A, torch.Tensor) else b  # Use b's device as fallback

        # Select backend
        selected_backend, selected_method = self._select_backend(backend, method, A_tensor)

        if self.verbose:
            print(f"Using backend: {selected_backend}, method: {selected_method}")

        # Dispatch to appropriate backend
        if selected_backend == "module_a":
            return self._solve_module_a(
                A, b, x0, selected_method, tol, atol, maxiter, M, **kwargs
            )
        elif selected_backend == "module_b":
            return self._solve_module_b(
                A, b, selected_method, tol, maxiter, **kwargs
            )
        elif selected_backend == "module_c":
            return self._solve_module_c(A, b, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {selected_backend}")

    def _solve_module_a(
        self,
        A: Union[torch.Tensor, Callable],
        b: torch.Tensor,
        x0: Optional[torch.Tensor],
        method: str,
        tol: float,
        atol: float,
        maxiter: Optional[int],
        M: Optional[Callable],
        **kwargs
    ) -> Tuple[torch.Tensor, SolverResult]:
        """Solve using Module A (JAX-style iterative solvers)."""
        module = self._load_module_a()

        if method not in module:
            raise ValueError(f"Method '{method}' not available in Module A. Use: {list(module.keys())}")

        solver_func = module[method]

        # Build kwargs
        solve_kwargs = {'tol': tol, 'atol': atol}
        if maxiter is not None:
            solve_kwargs['maxiter'] = maxiter
        if M is not None:
            solve_kwargs['M'] = M
        if x0 is not None:
            solve_kwargs['x0'] = x0

        # Add GMRES-specific arguments
        if method == 'gmres':
            if 'restart' in kwargs:
                solve_kwargs['restart'] = kwargs['restart']
            if 'solve_method' in kwargs:
                solve_kwargs['solve_method'] = kwargs['solve_method']

        # Solve
        x, info = solver_func(A, b, **solve_kwargs)

        converged = (info == 0)

        # Compute residual
        if callable(A):
            Ax = A(x)
        elif A.is_sparse:
            Ax = torch.sparse.mm(A, x.unsqueeze(-1)).squeeze(-1)
        else:
            Ax = torch.mv(A, x)
        residual = torch.norm(b - Ax).item() / torch.norm(b).item()

        result = SolverResult(
            x=x,
            converged=converged,
            iterations=None,  # Module A doesn't return iteration count
            residual=residual,
            backend="module_a",
            method=method
        )

        return x, result

    def _solve_module_b(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        method: str,
        tol: float,
        maxiter: Optional[int],
        **kwargs
    ) -> Tuple[torch.Tensor, SolverResult]:
        """Solve using Module B (PyAMGX)."""
        module = self._load_module_b()

        if method not in module:
            raise ValueError(f"Method '{method}' not available in Module B. Use: {list(module.keys())}")

        solver_func = module[method]

        # Build kwargs
        solve_kwargs = {'tol': tol}
        if maxiter is not None:
            solve_kwargs['maxiter'] = maxiter

        # Solve
        x = solver_func(A, b, **solve_kwargs)

        # Compute residual
        if A.is_sparse:
            Ax = torch.sparse.mm(A, x.unsqueeze(-1)).squeeze(-1)
        else:
            Ax = torch.mv(A, x)
        residual = torch.norm(b - Ax).item() / torch.norm(b).item()

        result = SolverResult(
            x=x,
            converged=True,  # Assume converged if no exception
            iterations=None,
            residual=residual,
            backend="module_b",
            method=method
        )

        return x, result

    def _solve_module_c(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, SolverResult]:
        """Solve using Module C (cuDSS direct solver)."""
        module = self._load_module_c()
        solver_func = module['direct']

        # Solve
        x = solver_func(A, b)

        # Compute residual
        if A.is_sparse:
            Ax = torch.sparse.mm(A, x.unsqueeze(-1)).squeeze(-1)
        else:
            Ax = torch.mv(A, x)
        residual = torch.norm(b - Ax).item() / torch.norm(b).item()

        result = SolverResult(
            x=x,
            converged=True,  # Direct solver always "converges"
            iterations=None,  # Direct solver has no iterations
            residual=residual,
            backend="module_c",
            method="direct"
        )

        return x, result

    def cg(
        self,
        A: Union[torch.Tensor, Callable],
        b: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, SolverResult]:
        """Shortcut for CG solver."""
        return self.solve(A, b, method='cg', **kwargs)

    def bicgstab(
        self,
        A: Union[torch.Tensor, Callable],
        b: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, SolverResult]:
        """Shortcut for BiCGStab solver."""
        return self.solve(A, b, method='bicgstab', **kwargs)

    def gmres(
        self,
        A: Union[torch.Tensor, Callable],
        b: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, SolverResult]:
        """Shortcut for GMRES solver."""
        return self.solve(A, b, method='gmres', **kwargs)

    def direct(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, SolverResult]:
        """Shortcut for direct solver (cuDSS)."""
        return self.solve(A, b, method='direct', backend='module_c', **kwargs)

    def __repr__(self) -> str:
        backends = self.available_backends
        return (
            f"SparseSolver(\n"
            f"  available_backends={backends},\n"
            f"  default_backend='{self.default_backend}',\n"
            f"  default_method='{self.default_method}'\n"
            f")"
        )


# Convenience functions for direct use without creating a SparseSolver instance

_default_solver: Optional[SparseSolver] = None


def _get_default_solver() -> SparseSolver:
    """Get or create the default solver instance."""
    global _default_solver
    if _default_solver is None:
        _default_solver = SparseSolver()
    return _default_solver


def solve(
    A: Union[torch.Tensor, Callable],
    b: torch.Tensor,
    method: str = "cg",
    backend: str = "auto",
    **kwargs
) -> Tuple[torch.Tensor, SolverResult]:
    """
    Solve sparse linear system Ax = b using the default solver.

    This is a convenience function that uses a shared SparseSolver instance.

    Args:
        A: Coefficient matrix or callable
        b: Right-hand side vector
        method: Solver method ('cg', 'bicgstab', 'gmres', 'direct')
        backend: Backend to use ('auto', 'module_a', 'module_b', 'module_c')
        **kwargs: Additional solver arguments

    Returns:
        Tuple of (solution tensor, SolverResult)

    Example:
        >>> from pytorch_sparse_solver import solve
        >>> x, result = solve(A, b, method='cg')
    """
    solver = _get_default_solver()
    return solver.solve(A, b, method=method, backend=backend, **kwargs)


def cg(A, b, **kwargs):
    """Solve Ax = b using Conjugate Gradient."""
    return solve(A, b, method='cg', **kwargs)


def bicgstab(A, b, **kwargs):
    """Solve Ax = b using BiCGStab."""
    return solve(A, b, method='bicgstab', **kwargs)


def gmres(A, b, **kwargs):
    """Solve Ax = b using GMRES."""
    return solve(A, b, method='gmres', **kwargs)


def direct_solve(A, b, **kwargs):
    """Solve Ax = b using direct solver (cuDSS)."""
    return solve(A, b, method='direct', backend='module_c', **kwargs)
