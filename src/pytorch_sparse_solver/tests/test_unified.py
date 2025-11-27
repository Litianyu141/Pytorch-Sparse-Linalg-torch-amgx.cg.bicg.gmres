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
Test Unified Interface: SparseSolver

This script tests the unified SparseSolver interface which provides
seamless access to all available backends (Module A, B, C).

Tests include:
- Backend detection and selection
- Method dispatching
- Fallback behavior when backends are unavailable
- Module independence (each combination should work)
"""

import sys
import time
import torch
import numpy as np
from typing import Tuple, Optional, List

# Add parent path for direct execution
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])


class TestUnifiedInterface:
    """Test class for unified SparseSolver interface."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def test_availability_check(self) -> bool:
        """Test backend availability checking."""
        self._log("\n[Test] Availability Check")

        from pytorch_sparse_solver import get_available_backends, print_availability_report

        backends = get_available_backends()
        self._log(f"  Available backends: {backends}")

        # Module A should always be available if PyTorch is installed
        passed = 'module_a' in backends and backends['module_a'] == True

        if self.verbose:
            print_availability_report()

        self.results.append(('Availability Check', passed))
        return passed

    def test_solver_creation(self) -> bool:
        """Test SparseSolver creation."""
        self._log("\n[Test] SparseSolver Creation")

        from pytorch_sparse_solver import SparseSolver

        try:
            solver = SparseSolver()
            self._log(f"  Created solver: {solver}")
            self._log(f"  Available backends: {solver.available_backends}")

            passed = len(solver.available_backends) > 0
        except Exception as e:
            self._log(f"  ❌ Failed to create solver: {e}")
            passed = False

        self.results.append(('Solver Creation', passed))
        return passed

    def test_module_a_via_unified(self) -> bool:
        """Test solving via unified interface using Module A."""
        self._log("\n[Test] Unified Interface - Module A")

        from pytorch_sparse_solver import SparseSolver, check_module_a_available

        if not check_module_a_available():
            self._log("  ⚠️ Module A not available, skipping")
            self.results.append(('Unified - Module A', None))
            return True

        n = 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float64

        # Create SPD matrix
        A = torch.randn(n, n, device=device, dtype=dtype)
        A = A @ A.T + torch.eye(n, device=device, dtype=dtype) * n

        x_true = torch.randn(n, device=device, dtype=dtype)
        b = torch.mv(A, x_true)

        solver = SparseSolver(default_backend='module_a')

        # Test CG
        start = time.time()
        x, result = solver.solve(A, b, method='cg', backend='module_a', tol=1e-8)
        elapsed = time.time() - start

        self._log(f"  Method: CG")
        self._log(f"  Backend: {result.backend}")
        self._log(f"  Converged: {result.converged}")
        self._log(f"  Residual: {result.residual:.2e}")
        self._log(f"  Time: {elapsed:.4f}s")

        passed = result.converged and result.residual < 1e-5
        self.results.append(('Unified - Module A', passed))
        return passed

    def test_auto_backend_selection(self) -> bool:
        """Test automatic backend selection."""
        self._log("\n[Test] Auto Backend Selection")

        from pytorch_sparse_solver import SparseSolver

        n = 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float64

        A = torch.randn(n, n, device=device, dtype=dtype)
        A = A @ A.T + torch.eye(n, device=device, dtype=dtype) * n

        b = torch.randn(n, device=device, dtype=dtype)

        solver = SparseSolver(default_backend='auto')

        x, result = solver.solve(A, b, method='cg', tol=1e-8)

        self._log(f"  Auto-selected backend: {result.backend}")
        self._log(f"  Converged: {result.converged}")
        self._log(f"  Residual: {result.residual:.2e}")

        passed = result.converged and result.residual < 1e-5
        self.results.append(('Auto Backend Selection', passed))
        return passed

    def test_shortcut_methods(self) -> bool:
        """Test shortcut methods (cg, bicgstab, gmres)."""
        self._log("\n[Test] Shortcut Methods")

        from pytorch_sparse_solver import SparseSolver

        n = 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float64

        A = torch.randn(n, n, device=device, dtype=dtype)
        A = A @ A.T + torch.eye(n, device=device, dtype=dtype) * n

        b = torch.randn(n, device=device, dtype=dtype)

        solver = SparseSolver()

        passed = True
        for method_name in ['cg', 'bicgstab', 'gmres']:
            method = getattr(solver, method_name)
            x, result = method(A, b, tol=1e-6)

            self._log(f"  {method_name.upper()}: converged={result.converged}, residual={result.residual:.2e}")

            if not result.converged or result.residual > 1e-4:
                passed = False

        self.results.append(('Shortcut Methods', passed))
        return passed

    def test_convenience_functions(self) -> bool:
        """Test module-level convenience functions."""
        self._log("\n[Test] Convenience Functions")

        from pytorch_sparse_solver import solve, cg, bicgstab, gmres

        n = 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float64

        A = torch.randn(n, n, device=device, dtype=dtype)
        A = A @ A.T + torch.eye(n, device=device, dtype=dtype) * n

        b = torch.randn(n, device=device, dtype=dtype)

        passed = True

        # Test solve()
        x, result = solve(A, b, method='cg')
        self._log(f"  solve(): residual={result.residual:.2e}")
        if result.residual > 1e-4:
            passed = False

        # Test cg()
        x, result = cg(A, b)
        self._log(f"  cg(): residual={result.residual:.2e}")
        if result.residual > 1e-4:
            passed = False

        # Test bicgstab()
        x, result = bicgstab(A, b)
        self._log(f"  bicgstab(): residual={result.residual:.2e}")
        if result.residual > 1e-4:
            passed = False

        # Test gmres()
        x, result = gmres(A, b)
        self._log(f"  gmres(): residual={result.residual:.2e}")
        if result.residual > 1e-4:
            passed = False

        self.results.append(('Convenience Functions', passed))
        return passed

    def test_module_b_via_unified(self) -> bool:
        """Test solving via unified interface using Module B."""
        self._log("\n[Test] Unified Interface - Module B")

        from pytorch_sparse_solver import SparseSolver, check_module_b_available

        if not check_module_b_available():
            self._log("  ⚠️ Module B not available, skipping")
            self.results.append(('Unified - Module B', None))
            return True

        n = 100
        device = 'cuda'
        dtype = torch.float64

        A = torch.randn(n, n, device=device, dtype=dtype)
        A = A @ A.T + torch.eye(n, device=device, dtype=dtype) * n

        b = torch.randn(n, device=device, dtype=dtype)

        solver = SparseSolver()

        x, result = solver.solve(A, b, method='cg', backend='module_b', tol=1e-8)

        self._log(f"  Backend: {result.backend}")
        self._log(f"  Residual: {result.residual:.2e}")

        passed = result.residual < 1e-5
        self.results.append(('Unified - Module B', passed))
        return passed

    def test_module_c_via_unified(self) -> bool:
        """Test solving via unified interface using Module C."""
        self._log("\n[Test] Unified Interface - Module C")

        from pytorch_sparse_solver import SparseSolver, check_module_c_available

        if not check_module_c_available():
            self._log("  ⚠️ Module C not available, skipping")
            self.results.append(('Unified - Module C', None))
            return True

        n = 100
        device = 'cuda'
        dtype = torch.float64

        # Create sparse CSR matrix
        values_list = []
        col_indices_list = []
        row_ptr = [0]

        for i in range(n):
            row_nnz = 0
            if i > 0:
                values_list.append(-1.0)
                col_indices_list.append(i - 1)
                row_nnz += 1
            values_list.append(3.0)
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
        b = torch.randn(n, device=device, dtype=dtype)

        solver = SparseSolver()

        x, result = solver.solve(A_csr, b, method='direct', backend='module_c')

        self._log(f"  Backend: {result.backend}")
        self._log(f"  Residual: {result.residual:.2e}")

        passed = result.residual < 1e-10  # Direct solver should be very accurate
        self.results.append(('Unified - Module C', passed))
        return passed

    def test_error_handling(self) -> bool:
        """Test error handling for invalid inputs."""
        self._log("\n[Test] Error Handling")

        from pytorch_sparse_solver import SparseSolver

        solver = SparseSolver()
        passed = True

        # Test invalid backend
        try:
            n = 10
            A = torch.randn(n, n)
            A = A @ A.T + torch.eye(n) * n
            b = torch.randn(n)
            solver.solve(A, b, backend='invalid_backend')
            self._log("  ❌ Should have raised error for invalid backend")
            passed = False
        except ValueError as e:
            self._log(f"  ✅ Correctly raised ValueError: {str(e)[:50]}...")

        # Test invalid method
        try:
            solver.solve(A, b, method='invalid_method')
            # This might not raise an error for auto backend
            self._log("  ⚠️ Invalid method may be handled differently by backends")
        except ValueError as e:
            self._log(f"  ✅ Correctly raised ValueError: {str(e)[:50]}...")

        self.results.append(('Error Handling', passed))
        return passed

    def run_all_tests(self) -> bool:
        """Run all unified interface tests."""
        print("=" * 60)
        print("Unified Interface Tests - SparseSolver")
        print("=" * 60)

        all_passed = True

        tests = [
            self.test_availability_check,
            self.test_solver_creation,
            self.test_module_a_via_unified,
            self.test_auto_backend_selection,
            self.test_shortcut_methods,
            self.test_convenience_functions,
            self.test_module_b_via_unified,
            self.test_module_c_via_unified,
            self.test_error_handling,
        ]

        for test in tests:
            try:
                passed = test()
                if not passed:
                    all_passed = False
            except Exception as e:
                self._log(f"  ❌ Test failed with exception: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False

        print("\n" + "=" * 60)
        print("Unified Interface Test Summary")
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
    """Run unified interface tests."""
    import argparse
    parser = argparse.ArgumentParser(description="Test Unified Interface")
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    args = parser.parse_args()

    tester = TestUnifiedInterface(verbose=not args.quiet)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
