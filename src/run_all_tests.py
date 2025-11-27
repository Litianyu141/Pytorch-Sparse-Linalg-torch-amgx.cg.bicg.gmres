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
Run all tests for pytorch_sparse_solver.

This script runs tests for all available modules:
- Module A: JAX-style iterative solvers (always available)
- Module B: PyAMGX GPU solvers (if pyamgx is installed)
- Module C: cuDSS direct solver (if PyTorch has cuDSS support)
- Unified interface tests
"""

import sys
import os

# Add current directory to path for direct execution
sys.path.insert(0, os.path.dirname(__file__))


def run_all_tests():
    """Run all module tests."""
    print("=" * 70)
    print("PyTorch Sparse Solver - Complete Test Suite")
    print("=" * 70)

    results = {}

    # Test Module A (always available)
    print("\n" + "=" * 70)
    print("Running Module A Tests...")
    print("=" * 70)
    try:
        from pytorch_sparse_solver.tests.test_module_a import TestModuleA
        tester = TestModuleA(verbose=True)
        results['Module A'] = tester.run_all_tests()
    except Exception as e:
        print(f"Module A tests failed with exception: {e}")
        results['Module A'] = False

    # Test Module B (if available)
    print("\n" + "=" * 70)
    print("Running Module B Tests...")
    print("=" * 70)
    try:
        from pytorch_sparse_solver.utils.availability import check_module_b_available
        if check_module_b_available():
            from pytorch_sparse_solver.tests.test_module_b import TestModuleB
            tester = TestModuleB(verbose=True)
            results['Module B'] = tester.run_all_tests()
        else:
            print("Module B not available (pyamgx not installed), skipping...")
            results['Module B'] = None
    except Exception as e:
        print(f"Module B tests failed with exception: {e}")
        results['Module B'] = False

    # Test Module C (if available)
    print("\n" + "=" * 70)
    print("Running Module C Tests...")
    print("=" * 70)
    try:
        from pytorch_sparse_solver.utils.availability import check_module_c_available
        if check_module_c_available():
            from pytorch_sparse_solver.tests.test_module_c import TestModuleC
            tester = TestModuleC(verbose=True)
            results['Module C'] = tester.run_all_tests()
        else:
            print("Module C not available (cuDSS not enabled), skipping...")
            results['Module C'] = None
    except Exception as e:
        print(f"Module C tests failed with exception: {e}")
        results['Module C'] = False

    # Test Unified Interface
    print("\n" + "=" * 70)
    print("Running Unified Interface Tests...")
    print("=" * 70)
    try:
        from pytorch_sparse_solver.tests.test_unified import TestUnifiedInterface
        tester = TestUnifiedInterface(verbose=True)
        results['Unified Interface'] = tester.run_all_tests()
    except Exception as e:
        print(f"Unified interface tests failed with exception: {e}")
        results['Unified Interface'] = False

    # Print summary
    print("\n" + "=" * 70)
    print("Complete Test Suite Summary")
    print("=" * 70)

    all_passed = True
    for module, passed in results.items():
        if passed is None:
            status = "SKIPPED (not available)"
        elif passed:
            status = "PASSED"
        else:
            status = "FAILED"
            all_passed = False
        print(f"  {module}: {status}")

    print("=" * 70)

    return all_passed


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Run all pytorch_sparse_solver tests")
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    args = parser.parse_args()

    success = run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
