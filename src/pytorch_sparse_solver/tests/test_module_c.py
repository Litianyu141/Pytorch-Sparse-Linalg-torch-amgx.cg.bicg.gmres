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
Test Module C: cuDSS Direct Solver

This script tests the correctness of Module C (cuDSS direct solver).
Run this script to verify Module C works independently.
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
    if A.is_sparse_csr:
        Ax = torch.sparse.mm(A, x.unsqueeze(-1)).squeeze(-1)
    elif A.is_sparse:
        Ax = torch.sparse.mm(A, x.unsqueeze(-1)).squeeze(-1)
    else:
        Ax = torch.mv(A, x)
    return (torch.norm(Ax - b) / torch.norm(b)).item()


class TestModuleC:
    """Test class for Module C (cuDSS) solver."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []
        self._module_loaded = False
        self._cudss_spsolve = None
        self._cudss_available = None

    def _load_module(self) -> bool:
        """Load Module C."""
        if self._module_loaded:
            return True
        try:
            from pytorch_sparse_solver.module_c import cudss_spsolve, cudss_available
            self._cudss_spsolve = cudss_spsolve
            self._cudss_available = cudss_available
            self._module_loaded = True
            return True
        except ImportError as e:
            print(f"❌ Module C not available: {e}")
            return False

    def _check_available(self) -> bool:
        """Check if cuDSS is available."""
        if not self._load_module():
            return False
        if not self._cudss_available():
            self._log("  ⚠️ cuDSS not available (requires PyTorch compiled with USE_CUDSS=1)")
            return False
        return True

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def test_cudss_basic(self) -> bool:
        """Test cuDSS with a simple sparse matrix."""
        self._log("\n[Test] cuDSS Basic - Small Sparse Matrix")

        if not self._check_available():
            self.results.append(('cuDSS Basic', None))
            return True

        n = 100
        device = 'cuda'
        dtype = torch.float64

        # Create a sparse CSR matrix (tridiagonal)
        diag = torch.full((n,), 2.0, device=device, dtype=dtype)
        off_diag = torch.full((n-1,), -1.0, device=device, dtype=dtype)

        # Build CSR components
        values_list = []
        col_indices_list = []
        row_ptr = [0]

        for i in range(n):
            row_nnz = 0
            if i > 0:  # Lower diagonal
                values_list.append(-1.0)
                col_indices_list.append(i - 1)
                row_nnz += 1
            # Main diagonal
            values_list.append(2.0)
            col_indices_list.append(i)
            row_nnz += 1
            if i < n - 1:  # Upper diagonal
                values_list.append(-1.0)
                col_indices_list.append(i + 1)
                row_nnz += 1
            row_ptr.append(row_ptr[-1] + row_nnz)

        crow_indices = torch.tensor(row_ptr, device=device, dtype=torch.int64)
        col_indices = torch.tensor(col_indices_list, device=device, dtype=torch.int64)
        values = torch.tensor(values_list, device=device, dtype=dtype)

        A_csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(n, n))

        # Create RHS
        x_true = torch.randn(n, device=device, dtype=dtype)
        A_dense = A_csr.to_dense()
        b = torch.mv(A_dense, x_true)

        start = time.time()
        x = self._cudss_spsolve(A_csr, b)
        elapsed = time.time() - start

        residual = compute_residual(A_csr, x, b)

        self._log(f"  Relative residual: {residual:.2e}")
        self._log(f"  Time: {elapsed:.4f}s")

        passed = residual < 1e-10  # Direct solver should be very accurate
        self.results.append(('cuDSS Basic', passed))
        return passed

    def test_cudss_large_matrix(self) -> bool:
        """Test cuDSS with a larger sparse matrix."""
        self._log("\n[Test] cuDSS Large - Poisson 2D")

        if not self._check_available():
            self.results.append(('cuDSS Large', None))
            return True

        nx, ny = 30, 30
        n = nx * ny
        device = 'cuda'
        dtype = torch.float64

        # Create 2D Poisson matrix in CSR format
        from pytorch_sparse_solver.utils.matrix_utils import create_poisson_2d_sparse_coo
        A_coo = create_poisson_2d_sparse_coo(nx, ny, device=device, dtype=dtype)
        A_csr = A_coo.to_sparse_csr()
        A_dense = A_coo.to_dense()

        x_true = torch.randn(n, device=device, dtype=dtype)
        b = torch.mv(A_dense, x_true)

        start = time.time()
        x = self._cudss_spsolve(A_csr, b)
        elapsed = time.time() - start

        residual = compute_residual(A_dense, x, b)

        self._log(f"  Matrix size: {n}x{n}")
        self._log(f"  Relative residual: {residual:.2e}")
        self._log(f"  Time: {elapsed:.4f}s")

        passed = residual < 1e-10
        self.results.append(('cuDSS Large', passed))
        return passed

    def test_cudss_autodiff(self) -> bool:
        """Test automatic differentiation through cuDSS solver."""
        self._log("\n[Test] cuDSS Autodiff - Gradient Computation")

        if not self._check_available():
            self.results.append(('cuDSS Autodiff', None))
            return True

        n = 50
        device = 'cuda'
        dtype = torch.float64

        # Create a simple sparse CSR matrix
        values_list = []
        col_indices_list = []
        row_ptr = [0]

        for i in range(n):
            row_nnz = 0
            if i > 0:
                values_list.append(-1.0)
                col_indices_list.append(i - 1)
                row_nnz += 1
            values_list.append(3.0)  # Make diagonally dominant
            col_indices_list.append(i)
            row_nnz += 1
            if i < n - 1:
                values_list.append(-1.0)
                col_indices_list.append(i + 1)
                row_nnz += 1
            row_ptr.append(row_ptr[-1] + row_nnz)

        crow_indices = torch.tensor(row_ptr, device=device, dtype=torch.int64)
        col_indices = torch.tensor(col_indices_list, device=device, dtype=torch.int64)
        values = torch.tensor(values_list, device=device, dtype=dtype)

        A_csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(n, n))

        b = torch.randn(n, device=device, dtype=dtype, requires_grad=True)

        x = self._cudss_spsolve(A_csr, b)
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

        self.results.append(('cuDSS Autodiff', passed))
        return passed

    def run_all_tests(self) -> bool:
        """Run all Module C tests."""
        print("=" * 60)
        print("Module C Tests - cuDSS Direct Solver")
        print("=" * 60)

        all_passed = True

        tests = [
            self.test_cudss_basic,
            self.test_cudss_large_matrix,
            self.test_cudss_autodiff,
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
        print("Module C Test Summary")
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
    """Run Module C tests."""
    import argparse
    parser = argparse.ArgumentParser(description="Test Module C")
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    args = parser.parse_args()

    tester = TestModuleC(verbose=not args.quiet)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
