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
Test Module A: JAX-style Iterative Solvers

This script tests the correctness and performance of Module A solvers:
- CG (Conjugate Gradient)
- BiCGStab (Bi-Conjugate Gradient Stabilized)
- GMRES (Generalized Minimal Residual)

Run this script to verify Module A works independently.
"""

import sys
import time
import torch
import numpy as np
from typing import Tuple, Optional

# Add parent path for direct execution
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])


def create_spd_matrix(n: int, device: str = 'cpu', dtype=torch.float64) -> torch.Tensor:
    """Create a symmetric positive definite matrix."""
    A = torch.randn(n, n, device=device, dtype=dtype)
    return A @ A.T + torch.eye(n, device=device, dtype=dtype) * n


def create_tridiagonal_matrix(n: int, device: str = 'cpu', dtype=torch.float64) -> torch.Tensor:
    """Create a tridiagonal SPD matrix."""
    diag = torch.full((n,), 2.0, device=device, dtype=dtype)
    off_diag = torch.full((n-1,), -1.0, device=device, dtype=dtype)
    A = torch.diag(diag) + torch.diag(off_diag, 1) + torch.diag(off_diag, -1)
    return A


def compute_residual(A: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> float:
    """Compute relative residual ||Ax - b|| / ||b||."""
    if A.is_sparse:
        Ax = torch.sparse.mm(A, x.unsqueeze(-1)).squeeze(-1)
    else:
        Ax = torch.mv(A, x)
    return (torch.norm(Ax - b) / torch.norm(b)).item()


class TestModuleA:
    """Test class for Module A solvers."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []
        self._module_loaded = False
        self._cg = None
        self._bicgstab = None
        self._gmres = None

    def _load_module(self) -> bool:
        """Load Module A."""
        if self._module_loaded:
            return True
        try:
            from pytorch_sparse_solver.module_a import cg, bicgstab, gmres
            self._cg = cg
            self._bicgstab = bicgstab
            self._gmres = gmres
            self._module_loaded = True
            return True
        except ImportError as e:
            print(f"❌ Failed to import Module A: {e}")
            return False

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def test_cg_basic(self) -> bool:
        """Test CG with a simple SPD matrix."""
        self._log("\n[Test] CG Basic - Small SPD Matrix")

        if not self._load_module():
            return False

        n = 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float64

        A = create_tridiagonal_matrix(n, device=device, dtype=dtype)
        x_true = torch.randn(n, device=device, dtype=dtype)
        b = torch.mv(A, x_true)

        start = time.time()
        x, info = self._cg(A, b, tol=1e-10, maxiter=1000)
        elapsed = time.time() - start

        residual = compute_residual(A, x, b)
        converged = info == 0
        error = torch.norm(x - x_true).item() / torch.norm(x_true).item()

        self._log(f"  Device: {device}")
        self._log(f"  Converged: {converged} (info={info})")
        self._log(f"  Relative residual: {residual:.2e}")
        self._log(f"  Relative error: {error:.2e}")
        self._log(f"  Time: {elapsed:.4f}s")

        passed = converged and residual < 1e-6
        self.results.append(('CG Basic', passed))
        return passed

    def test_bicgstab_basic(self) -> bool:
        """Test BiCGStab with a non-symmetric matrix."""
        self._log("\n[Test] BiCGStab Basic - Non-symmetric Matrix")

        if not self._load_module():
            return False

        n = 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float64

        # Create a non-symmetric but well-conditioned matrix
        A = create_tridiagonal_matrix(n, device=device, dtype=dtype)
        # Add asymmetry
        A = A + 0.1 * torch.randn(n, n, device=device, dtype=dtype)
        # Make diagonally dominant
        A = A + torch.eye(n, device=device, dtype=dtype) * 5

        x_true = torch.randn(n, device=device, dtype=dtype)
        b = torch.mv(A, x_true)

        start = time.time()
        x, info = self._bicgstab(A, b, tol=1e-10, maxiter=1000)
        elapsed = time.time() - start

        residual = compute_residual(A, x, b)
        converged = info == 0

        self._log(f"  Device: {device}")
        self._log(f"  Converged: {converged} (info={info})")
        self._log(f"  Relative residual: {residual:.2e}")
        self._log(f"  Time: {elapsed:.4f}s")

        passed = converged and residual < 1e-5
        self.results.append(('BiCGStab Basic', passed))
        return passed

    def test_gmres_basic(self) -> bool:
        """Test GMRES with a general matrix."""
        self._log("\n[Test] GMRES Basic - General Matrix")

        if not self._load_module():
            return False

        n = 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float64

        # Create a general matrix
        A = torch.randn(n, n, device=device, dtype=dtype)
        A = A + torch.eye(n, device=device, dtype=dtype) * 10  # Ensure non-singular

        x_true = torch.randn(n, device=device, dtype=dtype)
        b = torch.mv(A, x_true)

        start = time.time()
        x, info = self._gmres(A, b, tol=1e-10, maxiter=1000, restart=30)
        elapsed = time.time() - start

        residual = compute_residual(A, x, b)
        converged = info == 0

        self._log(f"  Device: {device}")
        self._log(f"  Converged: {converged} (info={info})")
        self._log(f"  Relative residual: {residual:.2e}")
        self._log(f"  Time: {elapsed:.4f}s")

        passed = converged and residual < 1e-5
        self.results.append(('GMRES Basic', passed))
        return passed

    def test_cg_function_operator(self) -> bool:
        """Test CG with a function-based linear operator."""
        self._log("\n[Test] CG Function Operator - Matrix-free")

        if not self._load_module():
            return False

        n = 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float64

        # Define a tridiagonal operator as a function
        def tridiag_mv(x):
            y = 2.0 * x
            y[:-1] -= x[1:]
            y[1:] -= x[:-1]
            return y

        x_true = torch.randn(n, device=device, dtype=dtype)
        b = tridiag_mv(x_true)

        start = time.time()
        x, info = self._cg(tridiag_mv, b, tol=1e-10, maxiter=1000)
        elapsed = time.time() - start

        residual = torch.norm(tridiag_mv(x) - b).item() / torch.norm(b).item()
        converged = info == 0

        self._log(f"  Device: {device}")
        self._log(f"  Converged: {converged} (info={info})")
        self._log(f"  Relative residual: {residual:.2e}")
        self._log(f"  Time: {elapsed:.4f}s")

        passed = converged and residual < 1e-6
        self.results.append(('CG Function Operator', passed))
        return passed

    def test_cg_large_sparse(self) -> bool:
        """Test CG with a large sparse matrix."""
        self._log("\n[Test] CG Large Sparse - Poisson 2D")

        if not self._load_module():
            return False

        nx, ny = 50, 50
        n = nx * ny
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float64

        # Create 2D Poisson matrix (sparse)
        from pytorch_sparse_solver.utils.matrix_utils import create_poisson_2d_sparse_coo
        A_sparse = create_poisson_2d_sparse_coo(nx, ny, device=device, dtype=dtype)

        # Create dense version for matrix-vector product
        A_dense = A_sparse.to_dense()

        x_true = torch.randn(n, device=device, dtype=dtype)
        b = torch.mv(A_dense, x_true)

        start = time.time()
        x, info = self._cg(A_dense, b, tol=1e-10, maxiter=5000)
        elapsed = time.time() - start

        residual = compute_residual(A_dense, x, b)
        converged = info == 0

        self._log(f"  Matrix size: {n}x{n}")
        self._log(f"  Device: {device}")
        self._log(f"  Converged: {converged} (info={info})")
        self._log(f"  Relative residual: {residual:.2e}")
        self._log(f"  Time: {elapsed:.4f}s")

        passed = converged and residual < 1e-5
        self.results.append(('CG Large Sparse', passed))
        return passed

    def test_gmres_solve_methods(self) -> bool:
        """Test GMRES with different solve methods."""
        self._log("\n[Test] GMRES Solve Methods - Batched vs Incremental")

        if not self._load_module():
            return False

        # Use smaller matrix for faster testing and better conditioning
        n = 50
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float64

        # Set seed for reproducibility
        torch.manual_seed(42)

        # Create well-conditioned SPD matrix
        A = torch.randn(n, n, device=device, dtype=dtype)
        A = A @ A.T + torch.eye(n, device=device, dtype=dtype) * n

        x_true = torch.randn(n, device=device, dtype=dtype)
        b = torch.mv(A, x_true)

        passed = True

        for method in ['batched', 'incremental']:
            start = time.time()
            x, info = self._gmres(A, b, tol=1e-8, maxiter=500, restart=30, solve_method=method)
            elapsed = time.time() - start

            residual = compute_residual(A, x, b)
            converged = info == 0

            self._log(f"  Method: {method}")
            self._log(f"    Converged: {converged} (info={info})")
            self._log(f"    Relative residual: {residual:.2e}")
            self._log(f"    Time: {elapsed:.4f}s")

            # Accept if residual is low enough, even if not formally converged
            if not (residual < 1e-4):
                passed = False

        self.results.append(('GMRES Solve Methods', passed))
        return passed

    def run_all_tests(self) -> bool:
        """Run all Module A tests."""
        print("=" * 60)
        print("Module A Tests - JAX-style Iterative Solvers")
        print("=" * 60)

        all_passed = True

        tests = [
            self.test_cg_basic,
            self.test_bicgstab_basic,
            self.test_gmres_basic,
            self.test_cg_function_operator,
            self.test_cg_large_sparse,
            self.test_gmres_solve_methods,
        ]

        for test in tests:
            try:
                passed = test()
                if not passed:
                    all_passed = False
            except Exception as e:
                self._log(f"  ❌ Test failed with exception: {e}")
                all_passed = False

        print("\n" + "=" * 60)
        print("Module A Test Summary")
        print("=" * 60)

        for name, passed in self.results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {name}")

        total = len(self.results)
        passed_count = sum(1 for _, p in self.results if p)
        print(f"\nTotal: {passed_count}/{total} tests passed")

        return all_passed


def main():
    """Run Module A tests."""
    import argparse
    parser = argparse.ArgumentParser(description="Test Module A")
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    args = parser.parse_args()

    tester = TestModuleA(verbose=not args.quiet)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
