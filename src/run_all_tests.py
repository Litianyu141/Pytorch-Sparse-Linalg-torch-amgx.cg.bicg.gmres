#!/usr/bin/env python3
"""
Run all tests for pytorch_sparse_solver package.

Usage:
    python run_all_tests.py              # Run all tests
    python run_all_tests.py --module a   # Run only Module A tests
    python run_all_tests.py --quick      # Run quick tests only
"""

import sys
import argparse
from typing import List, Tuple

def run_tests(modules: List[str], verbose: bool = True) -> Tuple[int, int]:
    """Run tests for specified modules."""
    passed = 0
    failed = 0

    # Import here to avoid issues if package not installed
    sys.path.insert(0, str(__file__).rsplit('/', 1)[0])

    if 'a' in modules or 'all' in modules:
        print("\n" + "=" * 60)
        print("Running Module A Tests")
        print("=" * 60)
        try:
            from pytorch_sparse_solver.tests.test_module_a import TestModuleA
            tester = TestModuleA(verbose=verbose)
            if tester.run_all_tests():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Module A tests failed: {e}")
            failed += 1

    if 'b' in modules or 'all' in modules:
        print("\n" + "=" * 60)
        print("Running Module B Tests")
        print("=" * 60)
        try:
            from pytorch_sparse_solver.tests.test_module_b import TestModuleB
            tester = TestModuleB(verbose=verbose)
            if tester.run_all_tests():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Module B tests failed: {e}")
            failed += 1

    if 'c' in modules or 'all' in modules:
        print("\n" + "=" * 60)
        print("Running Module C Tests")
        print("=" * 60)
        try:
            from pytorch_sparse_solver.tests.test_module_c import TestModuleC
            tester = TestModuleC(verbose=verbose)
            if tester.run_all_tests():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Module C tests failed: {e}")
            failed += 1

    if 'unified' in modules or 'all' in modules:
        print("\n" + "=" * 60)
        print("Running Unified Interface Tests")
        print("=" * 60)
        try:
            from pytorch_sparse_solver.tests.test_unified import TestUnifiedInterface
            tester = TestUnifiedInterface(verbose=verbose)
            if tester.run_all_tests():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Unified interface tests failed: {e}")
            failed += 1

    return passed, failed


def main():
    parser = argparse.ArgumentParser(description="Run pytorch_sparse_solver tests")
    parser.add_argument('--module', '-m', type=str, choices=['a', 'b', 'c', 'unified', 'all'],
                       default='all', help='Which module to test')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    args = parser.parse_args()

    modules = [args.module] if args.module != 'all' else ['a', 'b', 'c', 'unified']

    print("=" * 60)
    print("PyTorch Sparse Solver - Test Suite")
    print("=" * 60)

    # Check availability
    try:
        from pytorch_sparse_solver.utils.availability import print_availability_report
        print_availability_report()
    except Exception as e:
        print(f"Could not check availability: {e}")

    passed, failed = run_tests(modules, verbose=not args.quiet)

    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    print(f"Test suites passed: {passed}")
    print(f"Test suites failed: {failed}")

    if failed > 0:
        print("\n❌ Some tests failed!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
