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
Performance Benchmark for pytorch_sparse_solver

This script benchmarks the performance of different solver backends across
various matrix sizes and types.

Metrics measured:
- Solve time
- Memory usage
- Accuracy (residual)
- Scalability

Features:
- Auto-generates markdown report
- Saves reports to Logger/ directory with date naming
- Supports all three modules (A, B, C)

Run with:
    python -m pytorch_sparse_solver.tests.benchmark
    python -m pytorch_sparse_solver.tests.benchmark --sizes 100,500,1000
    python -m pytorch_sparse_solver.tests.benchmark --quick
"""

import sys
import time
import gc
import os
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# Add parent path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    backend: str
    method: str
    matrix_size: int
    matrix_type: str
    solve_time: float
    residual: float
    memory_used_mb: float
    converged: bool
    error_message: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark."""
    matrix_sizes: List[int] = field(default_factory=lambda: [100, 500, 1000])
    methods: List[str] = field(default_factory=lambda: ['cg', 'bicgstab', 'gmres'])
    backends: List[str] = field(default_factory=lambda: ['module_a', 'module_b', 'module_c'])
    matrix_types: List[str] = field(default_factory=lambda: ['tridiagonal', 'poisson2d', 'dense_spd'])
    num_runs: int = 3
    warmup_runs: int = 1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.float64
    tol: float = 1e-8
    maxiter: int = 1000


class SparseSolverBenchmark:
    """Benchmark suite for sparse solver performance."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self._solver = None

    def _get_solver(self):
        """Get or create solver instance."""
        if self._solver is None:
            # Import from local package
            import sys
            src_path = str(Path(__file__).parent.parent.parent)
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            from pytorch_sparse_solver import SparseSolver
            self._solver = SparseSolver(verbose=False)
        return self._solver

    def _create_matrix(self, n: int, matrix_type: str, device: str, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create test matrix and right-hand side."""
        if matrix_type == 'tridiagonal':
            # Create tridiagonal matrix directly
            diag = 2.0 * torch.ones(n, device=device, dtype=dtype)
            off_diag = -torch.ones(n - 1, device=device, dtype=dtype)
            A_dense = torch.diag(diag) + torch.diag(off_diag, 1) + torch.diag(off_diag, -1)
        elif matrix_type == 'poisson2d':
            # Create 2D Poisson matrix
            nx = int(np.sqrt(n))
            ny = nx
            n = nx * ny

            diag = 4.0 * torch.ones(n, device=device, dtype=dtype)
            A_dense = torch.diag(diag)

            # Off-diagonals for x-direction
            for i in range(n - 1):
                if (i + 1) % nx != 0:  # Not at right boundary
                    A_dense[i, i+1] = -1.0
                    A_dense[i+1, i] = -1.0

            # Off-diagonals for y-direction
            for i in range(n - nx):
                A_dense[i, i+nx] = -1.0
                A_dense[i+nx, i] = -1.0

        elif matrix_type == 'dense_spd':
            A = torch.randn(n, n, device=device, dtype=dtype)
            A_dense = A @ A.T + torch.eye(n, device=device, dtype=dtype) * n
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")

        # Create RHS
        x_true = torch.randn(A_dense.shape[0], device=device, dtype=dtype)
        b = torch.mv(A_dense, x_true)

        return A_dense, b

    def _get_memory_usage_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

    def run_single_benchmark(
        self,
        backend: str,
        method: str,
        matrix_size: int,
        matrix_type: str
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        device = self.config.device
        dtype = self.config.dtype

        # Create matrix
        try:
            A, b = self._create_matrix(matrix_size, matrix_type, device, dtype)
        except Exception as e:
            return BenchmarkResult(
                backend=backend,
                method=method,
                matrix_size=matrix_size,
                matrix_type=matrix_type,
                solve_time=0.0,
                residual=float('inf'),
                memory_used_mb=0.0,
                converged=False,
                error_message=f"Matrix creation failed: {e}"
            )

        solver = self._get_solver()

        # Check if backend is available
        if backend not in solver.available_backends:
            return BenchmarkResult(
                backend=backend,
                method=method,
                matrix_size=matrix_size,
                matrix_type=matrix_type,
                solve_time=0.0,
                residual=float('inf'),
                memory_used_mb=0.0,
                converged=False,
                error_message=f"Backend {backend} not available"
            )

        # Module C only supports direct method
        if backend == 'module_c' and method != 'direct':
            return BenchmarkResult(
                backend=backend,
                method=method,
                matrix_size=matrix_size,
                matrix_type=matrix_type,
                solve_time=0.0,
                residual=float('inf'),
                memory_used_mb=0.0,
                converged=False,
                error_message="Module C only supports 'direct' method"
            )

        # For Module C, convert to CSR
        if backend == 'module_c':
            if not A.is_sparse:
                A = A.to_sparse_coo().to_sparse_csr()
            elif not A.is_sparse_csr:
                A = A.to_sparse_csr()
            method = 'direct'

        # Warmup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        try:
            for _ in range(self.config.warmup_runs):
                _, _ = solver.solve(A, b, method=method, backend=backend, tol=1e-8, maxiter=1000)
        except Exception:
            pass  # Ignore warmup errors

        # Actual benchmark
        times = []
        residuals = []
        converged_all = True

        memory_before = self._get_memory_usage_mb()

        for _ in range(self.config.num_runs):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            try:
                x, result = solver.solve(A, b, method=method, backend=backend, tol=1e-8, maxiter=1000)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time

                times.append(elapsed)
                residuals.append(result.residual)
                if not result.converged:
                    converged_all = False
            except Exception as e:
                return BenchmarkResult(
                    backend=backend,
                    method=method,
                    matrix_size=matrix_size,
                    matrix_type=matrix_type,
                    solve_time=0.0,
                    residual=float('inf'),
                    memory_used_mb=0.0,
                    converged=False,
                    error_message=str(e)
                )

        memory_after = self._get_memory_usage_mb()

        return BenchmarkResult(
            backend=backend,
            method=method,
            matrix_size=matrix_size,
            matrix_type=matrix_type,
            solve_time=np.mean(times),
            residual=np.mean(residuals),
            memory_used_mb=memory_after - memory_before,
            converged=converged_all
        )

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all configured benchmarks."""
        print("=" * 80)
        print("PyTorch Sparse Solver - Performance Benchmark")
        print("=" * 80)
        print(f"Device: {self.config.device}")
        print(f"Matrix sizes: {self.config.matrix_sizes}")
        print(f"Backends: {self.config.backends}")
        print(f"Methods: {self.config.methods}")
        print(f"Matrix types: {self.config.matrix_types}")
        print(f"Runs per config: {self.config.num_runs}")
        print("=" * 80)

        total_tests = (
            len(self.config.backends) *
            len(self.config.methods) *
            len(self.config.matrix_sizes) *
            len(self.config.matrix_types)
        )
        current_test = 0

        for matrix_type in self.config.matrix_types:
            print(f"\n[Matrix Type: {matrix_type}]")

            for size in self.config.matrix_sizes:
                print(f"\n  Size: {size}x{size}")

                for backend in self.config.backends:
                    methods = ['direct'] if backend == 'module_c' else self.config.methods

                    for method in methods:
                        current_test += 1
                        print(f"    [{current_test}/{total_tests}] {backend}/{method}...", end=" ", flush=True)

                        result = self.run_single_benchmark(backend, method, size, matrix_type)
                        self.results.append(result)

                        if result.error_message:
                            print(f"SKIP ({result.error_message[:30]}...)")
                        elif result.converged:
                            print(f"OK (time={result.solve_time:.4f}s, residual={result.residual:.2e})")
                        else:
                            print(f"FAIL (residual={result.residual:.2e})")

        return self.results

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("Benchmark Summary")
        print("=" * 80)

        # Group results by backend
        backends = set(r.backend for r in self.results)

        for backend in sorted(backends):
            backend_results = [r for r in self.results if r.backend == backend and r.error_message is None]

            if not backend_results:
                print(f"\n{backend}: No successful results")
                continue

            print(f"\n{backend}:")

            # Group by method
            methods = set(r.method for r in backend_results)
            for method in sorted(methods):
                method_results = [r for r in backend_results if r.method == method]

                avg_time = np.mean([r.solve_time for r in method_results])
                avg_residual = np.mean([r.residual for r in method_results])
                success_rate = sum(1 for r in method_results if r.converged) / len(method_results) * 100

                print(f"  {method}: avg_time={avg_time:.4f}s, avg_residual={avg_residual:.2e}, success={success_rate:.1f}%")

    def export_csv(self, filename: str):
        """Export results to CSV."""
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'backend', 'method', 'matrix_size', 'matrix_type',
                'solve_time', 'residual', 'memory_mb', 'converged', 'error'
            ])

            for r in self.results:
                writer.writerow([
                    r.backend, r.method, r.matrix_size, r.matrix_type,
                    r.solve_time, r.residual, r.memory_used_mb, r.converged, r.error_message or ''
                ])

        print(f"\nResults exported to {filename}")

    def generate_markdown_report(self, output_dir: Path = None) -> str:
        """Generate comprehensive markdown report with tables."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent.parent / "Logger"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with date
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_filename = f"benchmark_report_{timestamp}.md"
        report_path = output_dir / report_filename

        # Generate report content
        report = self._generate_markdown_content(timestamp)

        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📄 Markdown report saved to: {report_path}")

        return str(report_path)

    def _generate_markdown_content(self, timestamp: str) -> str:
        """Generate markdown report content with tables."""
        lines = []

        # Header
        lines.append("# PyTorch Sparse Solver - Performance Benchmark Report")
        lines.append("")
        lines.append(f"**Generated:** {timestamp.replace('_', ' ').replace('-', ':')}")
        lines.append(f"**Device:** {self.config.device}")
        lines.append(f"**Precision:** {self.config.dtype}")
        lines.append(f"**Tolerance:** {self.config.tol:.2e}")
        lines.append(f"**Max Iterations:** {self.config.maxiter}")
        lines.append(f"**Number of runs:** {self.config.num_runs}")
        lines.append("")

        # System Info
        lines.append("## System Information")
        lines.append("")
        if torch.cuda.is_available():
            lines.append(f"- **GPU:** {torch.cuda.get_device_name(0)}")
            lines.append(f"- **CUDA Version:** {torch.version.cuda}")
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            lines.append(f"- **GPU Memory:** {mem_total:.1f} GB")
        else:
            lines.append("- **Device:** CPU only")
        lines.append(f"- **PyTorch Version:** {torch.__version__}")
        lines.append("")

        # Module availability
        lines.append("## Module Availability")
        lines.append("")
        lines.append("| Module | Description | Status |")
        lines.append("|--------|-------------|--------|")

        solver = self._get_solver()
        module_a = 'module_a' in solver.available_backends
        module_b = 'module_b' in solver.available_backends
        module_c = 'module_c' in solver.available_backends

        lines.append(f"| Module A | JAX-style Iterative (CG, BiCGStab, GMRES) | {'✅ Available' if module_a else '❌ Not Available'} |")
        lines.append(f"| Module B | PyAMGX GPU Solvers | {'✅ Available' if module_b else '❌ Not Available'} |")
        lines.append(f"| Module C | cuDSS Direct Solver | {'✅ Available' if module_c else '❌ Not Available'} |")
        lines.append("")

        # Detailed Results by Matrix Type
        lines.append("## Detailed Results")
        lines.append("")

        matrix_types = sorted(set(r.matrix_type for r in self.results))
        sizes = sorted(set(r.matrix_size for r in self.results))

        for matrix_type in matrix_types:
            lines.append(f"### Matrix Type: {matrix_type}")
            lines.append("")

            for size in sizes:
                filtered = [r for r in self.results
                           if r.matrix_type == matrix_type and r.matrix_size == size]

                if not filtered:
                    continue

                lines.append(f"#### Size: {size}x{size}")
                lines.append("")
                lines.append("| Backend | Method | Time (ms) | Memory (MB) | Residual | Status |")
                lines.append("|---------|--------|-----------|-------------|----------|--------|")

                for r in sorted(filtered, key=lambda x: (x.backend, x.method)):
                    if r.error_message:
                        status = f"❌ {r.error_message[:30]}..."
                        time_str = "N/A"
                        mem_str = "N/A"
                        res_str = "N/A"
                    else:
                        status = "✅" if r.converged else "⚠️ Not converged"
                        time_str = f"{r.solve_time*1000:.3f}"
                        mem_str = f"{r.memory_used_mb:.2f}" if r.memory_used_mb > 0 else "N/A"
                        res_str = f"{r.residual:.2e}"

                    lines.append(f"| {r.backend} | {r.method} | {time_str} | {mem_str} | {res_str} | {status} |")

                lines.append("")

        # Performance Comparison Tables
        lines.append("## Performance Comparison")
        lines.append("")

        # Time comparison table
        lines.append("### Solve Time Comparison (ms)")
        lines.append("")

        # Get unique methods
        methods = sorted(set(r.method for r in self.results if r.error_message is None))
        backends = sorted(set(r.backend for r in self.results if r.error_message is None))

        for matrix_type in matrix_types:
            lines.append(f"#### {matrix_type}")
            lines.append("")

            # Header
            header = "| Size |"
            separator = "|------|"
            for backend in backends:
                for method in methods:
                    header += f" {backend}/{method} |"
                    separator += "----------------|"
            lines.append(header)
            lines.append(separator)

            # Data rows
            for size in sizes:
                row = f"| {size} |"
                for backend in backends:
                    for method in methods:
                        matching = [r for r in self.results
                                   if r.matrix_type == matrix_type
                                   and r.matrix_size == size
                                   and r.backend == backend
                                   and r.method == method
                                   and r.error_message is None]
                        if matching:
                            row += f" {matching[0].solve_time*1000:.3f} |"
                        else:
                            row += " N/A |"
                lines.append(row)

            lines.append("")

        # Best performers summary
        lines.append("## Best Performers Summary")
        lines.append("")
        lines.append("### Fastest Solver by Configuration")
        lines.append("")
        lines.append("| Matrix Type | Size | Best Method | Time (ms) | Backend |")
        lines.append("|-------------|------|-------------|-----------|---------|")

        for matrix_type in matrix_types:
            for size in sizes:
                valid_results = [r for r in self.results
                                if r.matrix_type == matrix_type
                                and r.matrix_size == size
                                and r.error_message is None
                                and r.converged]
                if valid_results:
                    best = min(valid_results, key=lambda x: x.solve_time)
                    lines.append(f"| {matrix_type} | {size} | {best.method} | {best.solve_time*1000:.3f} | {best.backend} |")

        lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Report generated by PyTorch Sparse Solver Benchmark Suite*")
        lines.append(f"*Configuration: sizes={self.config.matrix_sizes}, runs={self.config.num_runs}*")

        return "\n".join(lines)


def main():
    """Run benchmark suite."""
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark sparse solvers")
    parser.add_argument('--sizes', type=str, default='100,500,1000',
                       help='Comma-separated matrix sizes')
    parser.add_argument('--backends', type=str, default='module_a,module_b,module_c',
                       help='Comma-separated backends')
    parser.add_argument('--methods', type=str, default='cg,bicgstab,gmres',
                       help='Comma-separated methods')
    parser.add_argument('--types', type=str, default='tridiagonal,poisson2d',
                       help='Comma-separated matrix types')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per config')
    parser.add_argument('--quick', action='store_true', help='Quick benchmark with small sizes')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for reports (default: Logger/)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Output CSV file (optional)')
    parser.add_argument('--tol', type=float, default=1e-8, help='Solver tolerance')
    parser.add_argument('--maxiter', type=int, default=1000, help='Maximum iterations')
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        sizes = [100, 200, 500]
        runs = 2
        types = ['poisson2d']
    else:
        sizes = [int(s) for s in args.sizes.split(',')]
        runs = args.runs
        types = [t.strip() for t in args.types.split(',')]

    config = BenchmarkConfig(
        matrix_sizes=sizes,
        backends=[b.strip() for b in args.backends.split(',')],
        methods=[m.strip() for m in args.methods.split(',')],
        matrix_types=types,
        num_runs=runs,
        tol=args.tol,
        maxiter=args.maxiter,
    )

    benchmark = SparseSolverBenchmark(config)
    benchmark.run_all_benchmarks()
    benchmark.print_summary()

    # Generate markdown report
    output_dir = Path(args.output) if args.output else None
    report_path = benchmark.generate_markdown_report(output_dir)

    # Export CSV if requested
    if args.csv:
        benchmark.export_csv(args.csv)

    print(f"\n✅ Benchmark complete!")
    print(f"📄 Report: {report_path}")

    return benchmark.results


if __name__ == '__main__':
    main()
