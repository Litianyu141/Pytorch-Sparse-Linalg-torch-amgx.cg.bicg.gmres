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
Unified test and benchmark runner for pytorch_sparse_solver.

Usage:
    # Run correctness tests only
    python src/run.py --test
    python src/run.py -t

    # Run performance benchmarks only
    python src/run.py --benchmark
    python src/run.py -b

    # Run both tests and benchmarks
    python src/run.py --all
    python src/run.py -a

    # Quick benchmark (smaller matrices, fewer runs)
    python src/run.py --benchmark --quick

    # Custom benchmark settings
    python src/run.py --benchmark --sizes 100,500,1000 --runs 5

    # Quiet mode (less output)
    python src/run.py --test --quiet

Examples:
    python src/run.py --test                    # Run all correctness tests
    python src/run.py --benchmark --quick       # Quick performance benchmark
    python src/run.py --all                     # Run everything
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path for direct execution
sys.path.insert(0, os.path.dirname(__file__))


# =============================================================================
# Correctness Tests
# =============================================================================

def run_all_tests(verbose: bool = True) -> bool:
    """
    Run all module correctness tests.

    Tests:
    - Module A: JAX-style iterative solvers (CG, BiCGStab, GMRES)
    - Module B: PyAMGX GPU solvers (if pyamgx installed)
    - Module C: cuDSS direct solver (if PyTorch has cuDSS support)
    - Unified interface tests

    Returns:
        bool: True if all tests passed, False otherwise
    """
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
        tester = TestModuleA(verbose=verbose)
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
            tester = TestModuleB(verbose=verbose)
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
            tester = TestModuleC(verbose=verbose)
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
        tester = TestUnifiedInterface(verbose=verbose)
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


# =============================================================================
# Performance Benchmarks
# =============================================================================

def run_benchmarks(quick: bool = False, sizes: list = None, num_runs: int = 3) -> bool:
    """
    Run performance benchmarks for all available modules.

    Benchmarks:
    - Module A: CG, BiCGStab, GMRES
    - Module B: CG, BiCGStab (GMRES has AMG config issues)
    - Module C: Direct (cuDSS)

    Args:
        quick: If True, use smaller matrices and fewer runs
        sizes: List of matrix sizes to test
        num_runs: Number of runs per configuration

    Returns:
        bool: True if benchmarks completed successfully
    """
    try:
        from pytorch_sparse_solver.tests.benchmark import SparseSolverBenchmark, BenchmarkConfig, BenchmarkResult
    except ImportError as e:
        print(f"Failed to import benchmark module: {e}")
        return False

    # Custom benchmark class that handles each backend appropriately
    class CustomBenchmark(SparseSolverBenchmark):
        def run_all_benchmarks(self):
            """Run benchmarks with backend-specific method selection."""
            print("=" * 80)
            print("PyTorch Sparse Solver - Comprehensive Performance Benchmark")
            print("=" * 80)
            print(f"Device: {self.config.device}")
            print(f"Matrix sizes: {self.config.matrix_sizes}")
            print(f"Matrix types: {self.config.matrix_types}")
            print(f"Runs per config: {self.config.num_runs}")
            print("=" * 80)
            print("Testing Strategy:")
            print("  - Module A: CG, BiCGStab, GMRES")
            print("  - Module B: CG, BiCGStab (GMRES has AMG config issues)")
            print("  - Module C: Direct (cuDSS)")
            print("=" * 80)

            test_configs = [
                ('module_a', ['cg', 'bicgstab', 'gmres']),
                ('module_b', ['cg', 'bicgstab']),
                ('module_c', ['direct']),
            ]

            current_test = 0
            total_tests = sum(
                len(methods) * len(self.config.matrix_sizes) * len(self.config.matrix_types)
                for _, methods in test_configs
            )

            for matrix_type in self.config.matrix_types:
                print(f"\n[Matrix Type: {matrix_type}]")

                for size in self.config.matrix_sizes:
                    print(f"\n  Size: {size}x{size}")

                    for backend, methods in test_configs:
                        for method in methods:
                            current_test += 1
                            print(f"    [{current_test}/{total_tests}] {backend}/{method}...", end=" ", flush=True)

                            result = self.run_single_benchmark(backend, method, size, matrix_type)
                            self.results.append(result)

                            if result.error_message:
                                print(f"SKIP ({result.error_message[:40]}...)")
                            elif result.converged:
                                print(f"OK (time={result.solve_time*1000:.2f}ms, residual={result.residual:.2e})")
                            else:
                                print(f"FAIL (residual={result.residual:.2e})")

            return self.results

    # Configure benchmark settings
    if quick:
        matrix_sizes = [100, 200, 500]
        matrix_types = ['poisson2d']
        num_runs = 2
    else:
        matrix_sizes = sizes if sizes else [100, 200, 500, 1000]
        matrix_types = ['poisson2d', 'tridiagonal']

    config = BenchmarkConfig(
        matrix_sizes=matrix_sizes,
        backends=['module_a', 'module_b', 'module_c'],
        methods=['cg', 'bicgstab', 'gmres', 'direct'],  # Will be filtered per backend
        matrix_types=matrix_types,
        num_runs=num_runs,
        warmup_runs=1,
    )

    benchmark = CustomBenchmark(config)
    results = benchmark.run_all_benchmarks()
    benchmark.print_summary()
    report_path = benchmark.generate_markdown_report()

    print(f"\n{'='*80}")
    print(f"Benchmark Complete!")
    print(f"Report saved to: {report_path}")
    print(f"{'='*80}")

    return True


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Unified test and benchmark runner for pytorch_sparse_solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/run.py --test                    Run correctness tests
  python src/run.py --benchmark               Run performance benchmarks
  python src/run.py --benchmark --quick       Quick benchmark (smaller matrices)
  python src/run.py --all                     Run both tests and benchmarks
  python src/run.py -t -q                     Run tests in quiet mode
        """
    )

    # Mode selection
    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument('--test', '-t', action='store_true',
                           help='Run correctness tests')
    mode_group.add_argument('--benchmark', '-b', action='store_true',
                           help='Run performance benchmarks')
    mode_group.add_argument('--all', '-a', action='store_true',
                           help='Run both tests and benchmarks')

    # Benchmark options
    bench_group = parser.add_argument_group('Benchmark Options')
    bench_group.add_argument('--quick', action='store_true',
                            help='Quick benchmark with smaller matrices and fewer runs')
    bench_group.add_argument('--sizes', type=str, default=None,
                            help='Comma-separated list of matrix sizes (e.g., "100,500,1000")')
    bench_group.add_argument('--runs', type=int, default=3,
                            help='Number of runs per benchmark configuration (default: 3)')

    # General options
    general_group = parser.add_argument_group('General Options')
    general_group.add_argument('--quiet', '-q', action='store_true',
                              help='Quiet mode (less verbose output)')

    args = parser.parse_args()

    # Default to --test if no mode specified
    if not (args.test or args.benchmark or args.all):
        args.test = True

    # Parse sizes if provided
    sizes = None
    if args.sizes:
        sizes = [int(s.strip()) for s in args.sizes.split(',')]

    success = True

    # Run tests if requested
    if args.test or args.all:
        print("\n" + "#" * 80)
        print("#  RUNNING CORRECTNESS TESTS")
        print("#" * 80 + "\n")
        test_success = run_all_tests(verbose=not args.quiet)
        success = success and test_success

    # Run benchmarks if requested
    if args.benchmark or args.all:
        print("\n" + "#" * 80)
        print("#  RUNNING PERFORMANCE BENCHMARKS")
        print("#" * 80 + "\n")
        bench_success = run_benchmarks(quick=args.quick, sizes=sizes, num_runs=args.runs)
        success = success and bench_success

    # Final summary
    print("\n" + "=" * 80)
    if args.all:
        print("ALL TASKS COMPLETED" if success else "SOME TASKS FAILED")
    elif args.test:
        print("TESTS COMPLETED" if success else "TESTS FAILED")
    elif args.benchmark:
        print("BENCHMARKS COMPLETED" if success else "BENCHMARKS FAILED")
    print("=" * 80)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
