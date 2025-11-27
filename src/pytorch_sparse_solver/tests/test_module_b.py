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
Test Module B: PyAMGX GPU Solvers

This script tests the correctness and performance of Module B solvers:
- AMGX CG
- AMGX BiCGStab
- AMGX GMRES

Run this script to verify Module B works independently.
"""

import sys
import time
import torch
import numpy as np
from typing import Tuple, Optional

# Add parent path for direct execution
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])


def compute_residual(A: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> float:
    """Compute relative residual ||Ax - b|| / ||b||."""
    if A.is_sparse:
        Ax = torch.sparse.mm(A, x.unsqueeze(-1)).squeeze(-1)
    else:
        Ax = torch.mv(A, x)
    return (torch.norm(Ax - b) / torch.norm(b)).item()


class TestModuleB:
    """Test class for Module B (PyAMGX) solvers."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []
        self._module_loaded = False
        self._amgx_cg = None
        self._amgx_bicgstab = None
        self._amgx_gmres = None

    def _load_module(self) -> bool:
        """Load Module B."""
        if self._module_loaded:
            return True
        try:
            # First check if pyamgx is available at all
            import pyamgx
            from pytorch_sparse_solver.module_b import amgx_cg, amgx_bicgstab, amgx_gmres
            self._amgx_cg = amgx_cg
            self._amgx_bicgstab = amgx_bicgstab
            self._amgx_gmres = amgx_gmres
            self._module_loaded = True
            return True
        except ImportError as e:
            self._log(f"  ⚠️ Module B not available: {e}")
            self._log("     This is expected if pyamgx is not installed.")
            return False

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        if not torch.cuda.is_available():
            self._log("  ⚠️ CUDA not available, skipping test")
            return False
        return True

    def test_amgx_cg_basic(self) -> bool:
        """Test AMGX CG with a simple SPD matrix."""
        self._log("\n[Test] AMGX CG Basic - Small SPD Matrix")

        if not self._load_module():
            self.results.append(('AMGX CG Basic', None))  # Skip
            return True

        if not self._check_cuda():
            self.results.append(('AMGX CG Basic', None))
            return True

        n = 100
        device = 'cuda'
        dtype = torch.float64

        # Create SPD matrix
        A = torch.randn(n, n, device=device, dtype=dtype)
        A = A @ A.T + torch.eye(n, device=device, dtype=dtype) * n

        x_true = torch.randn(n, device=device, dtype=dtype)
        b = torch.mv(A, x_true)

        start = time.time()
        x = self._amgx_cg(A, b, tol=1e-10, maxiter=1000)
        elapsed = time.time() - start

        residual = compute_residual(A, x, b)

        self._log(f"  Relative residual: {residual:.2e}")
        self._log(f"  Time: {elapsed:.4f}s")

        passed = residual < 1e-5
        self.results.append(('AMGX CG Basic', passed))
        return passed

    def test_amgx_bicgstab_basic(self) -> bool:
        """Test AMGX BiCGStab with a non-symmetric matrix."""
        self._log("\n[Test] AMGX BiCGStab Basic - Non-symmetric Matrix")

        if not self._load_module():
            self.results.append(('AMGX BiCGStab Basic', None))
            return True

        if not self._check_cuda():
            self.results.append(('AMGX BiCGStab Basic', None))
            return True

        n = 100
        device = 'cuda'
        dtype = torch.float64

        # Create non-symmetric matrix
        A = torch.randn(n, n, device=device, dtype=dtype)
        A = A + torch.eye(n, device=device, dtype=dtype) * 10

        x_true = torch.randn(n, device=device, dtype=dtype)
        b = torch.mv(A, x_true)

        start = time.time()
        x = self._amgx_bicgstab(A, b, tol=1e-10, maxiter=1000)
        elapsed = time.time() - start

        residual = compute_residual(A, x, b)

        self._log(f"  Relative residual: {residual:.2e}")
        self._log(f"  Time: {elapsed:.4f}s")

        passed = residual < 1e-5
        self.results.append(('AMGX BiCGStab Basic', passed))
        return passed

    def test_amgx_gmres_basic(self) -> bool:
        """Test AMGX GMRES with a general matrix."""
        self._log("\n[Test] AMGX GMRES Basic - General Matrix")

        if not self._load_module():
            self.results.append(('AMGX GMRES Basic', None))
            return True

        if not self._check_cuda():
            self.results.append(('AMGX GMRES Basic', None))
            return True

        n = 100
        device = 'cuda'
        dtype = torch.float64

        A = torch.randn(n, n, device=device, dtype=dtype)
        A = A + torch.eye(n, device=device, dtype=dtype) * 10

        x_true = torch.randn(n, device=device, dtype=dtype)
        b = torch.mv(A, x_true)

        start = time.time()
        x = self._amgx_gmres(A, b, tol=1e-10, maxiter=1000)
        elapsed = time.time() - start

        residual = compute_residual(A, x, b)

        self._log(f"  Relative residual: {residual:.2e}")
        self._log(f"  Time: {elapsed:.4f}s")

        passed = residual < 1e-5
        self.results.append(('AMGX GMRES Basic', passed))
        return passed

    def test_amgx_autodiff(self) -> bool:
        """Test automatic differentiation through AMGX solver."""
        self._log("\n[Test] AMGX Autodiff - Gradient Computation")

        if not self._load_module():
            self.results.append(('AMGX Autodiff', None))
            return True

        if not self._check_cuda():
            self.results.append(('AMGX Autodiff', None))
            return True

        n = 50
        device = 'cuda'
        dtype = torch.float64

        # Create SPD matrix
        A = torch.randn(n, n, device=device, dtype=dtype)
        A = A @ A.T + torch.eye(n, device=device, dtype=dtype) * n

        b = torch.randn(n, device=device, dtype=dtype, requires_grad=True)

        # Solve and compute gradient
        x = self._amgx_cg(A, b, tol=1e-10, maxiter=1000)
        loss = torch.sum(x ** 2)

        try:
            loss.backward()
            grad_b = b.grad

            self._log(f"  Solution norm: {torch.norm(x).item():.4f}")
            self._log(f"  Gradient norm: {torch.norm(grad_b).item():.4f}")
            self._log(f"  Gradient has NaN: {torch.isnan(grad_b).any().item()}")

            passed = grad_b is not None and not torch.isnan(grad_b).any()
        except Exception as e:
            self._log(f"  ❌ Autodiff failed: {e}")
            passed = False

        self.results.append(('AMGX Autodiff', passed))
        return passed

    def test_amgx_sparse_matrix(self) -> bool:
        """Test AMGX with sparse matrix input."""
        self._log("\n[Test] AMGX Sparse Matrix - COO Format")

        if not self._load_module():
            self.results.append(('AMGX Sparse Matrix', None))
            return True

        if not self._check_cuda():
            self.results.append(('AMGX Sparse Matrix', None))
            return True

        n = 100
        device = 'cuda'
        dtype = torch.float64

        # Create sparse tridiagonal matrix
        from pytorch_sparse_solver.utils.matrix_utils import create_tridiagonal_sparse_coo
        A_sparse = create_tridiagonal_sparse_coo(n, device=device, dtype=dtype)
        A_dense = A_sparse.to_dense()

        x_true = torch.randn(n, device=device, dtype=dtype)
        b = torch.mv(A_dense, x_true)

        start = time.time()
        x = self._amgx_cg(A_sparse, b, tol=1e-10, maxiter=1000)
        elapsed = time.time() - start

        residual = compute_residual(A_dense, x, b)

        self._log(f"  Relative residual: {residual:.2e}")
        self._log(f"  Time: {elapsed:.4f}s")

        passed = residual < 1e-5
        self.results.append(('AMGX Sparse Matrix', passed))
        return passed

    def run_all_tests(self) -> bool:
        """Run all Module B tests."""
        print("=" * 60)
        print("Module B Tests - PyAMGX GPU Solvers")
        print("=" * 60)

        all_passed = True

        tests = [
            self.test_amgx_cg_basic,
            self.test_amgx_bicgstab_basic,
            self.test_amgx_gmres_basic,
            self.test_amgx_autodiff,
            self.test_amgx_sparse_matrix,
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
        print("Module B Test Summary")
        print("=" * 60)

        for name, passed in self.results:
            if passed is None:
                status = "⏭️ SKIP"
            elif passed:
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            print(f"  {status}: {name}")

        total = len(self.results)
        passed_count = sum(1 for _, p in self.results if p is True)
        skipped_count = sum(1 for _, p in self.results if p is None)
        print(f"\nTotal: {passed_count}/{total - skipped_count} tests passed ({skipped_count} skipped)")

        return all_passed


def main():
    """Run Module B tests."""
    import argparse
    parser = argparse.ArgumentParser(description="Test Module B")
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    args = parser.parse_args()

    tester = TestModuleB(verbose=not args.quiet)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
