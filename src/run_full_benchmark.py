#!/usr/bin/env python3
"""
Comprehensive benchmark script that tests all modules appropriately:
- Module A: CG, BiCGStab, GMRES
- Module B: CG, BiCGStab (GMRES has config issues)
- Module C: Direct solve
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pytorch_sparse_solver.tests.benchmark import SparseSolverBenchmark, BenchmarkConfig, BenchmarkResult
import torch

# Custom benchmark that handles each backend appropriately
class CustomBenchmark(SparseSolverBenchmark):
    def run_all_benchmarks(self):
        """Run benchmarks with backend-specific method selection."""
        print("=" * 80)
        print("PyTorch Sparse Solver - Comprehensive Performance Benchmark")
        print("=" * 80)
        print(f"Device: {self.config.device}")
        print(f"Matrix sizes: {self.config.matrix_sizes}")
        print(f"Matrix types: {self.config.matrix_types}")
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

if __name__ == '__main__':
    config = BenchmarkConfig(
        matrix_sizes=[100, 200, 500],
        backends=['module_a', 'module_b', 'module_c'],
        methods=['cg', 'bicgstab', 'gmres', 'direct'],  # Will be filtered per backend
        matrix_types=['poisson2d', 'tridiagonal'],
        num_runs=3,
        warmup_runs=1,
    )

    benchmark = CustomBenchmark(config)
    results = benchmark.run_all_benchmarks()
    benchmark.print_summary()
    report_path = benchmark.generate_markdown_report()

    print(f"\n{'='*80}")
    print(f"✅ Benchmark Complete!")
    print(f"📄 Report saved to: {report_path}")
    print(f"{'='*80}")
