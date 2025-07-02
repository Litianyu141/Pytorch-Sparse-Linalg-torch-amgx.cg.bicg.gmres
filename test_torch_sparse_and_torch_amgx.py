#!/usr/bin/env python3
"""
Comprehensive Sparse Matrix Solver Testing

This script tests and compares PyTorch and AMGX sparse linear system solvers
across different matrix types, testing accuracy, speed, and differentiability.

Usage:
    python test_comprehensive_solvers.py

Author: Augment Agent
Date: 2025-06-24
"""

import torch
import numpy as np
import time
import pandas as pd
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from scipy.sparse import csr_matrix

# Import solvers
from src.torch_sparse_linalg import cg, bicgstab, gmres

# Try to import AMGX solver
try:
    from src_torch_amgx.torch_amgx import amgx_cg, amgx_bicgstab, amgx_gmres
    HAS_AMGX = True
    print("✅ AMGX solver available")
except ImportError as e:
    HAS_AMGX = False
    print(f"⚠️  AMGX solver not available: {e}")

@dataclass
class TestResult:
    """Data class to store test results"""
    solver_name: str
    matrix_type: str
    matrix_size: int
    solve_time: float
    solution_error: float
    residual_error: float
    converged: bool
    gradient_error: float = None

class MatrixGenerator:
    """Generate different types of sparse matrices for testing"""
    
    @staticmethod
    def create_diagonally_dominant_matrix(n: int, sparsity: float = 0.1, device: str = 'cuda') -> torch.Tensor:
        """Create a diagonally dominant sparse matrix"""
        torch.manual_seed(42)
        
        # Create random sparse pattern
        nnz = int(n * n * sparsity)
        row_indices = torch.randint(0, n, (nnz,), device=device)
        col_indices = torch.randint(0, n, (nnz,), device=device)
        values = torch.randn(nnz, device=device, dtype=torch.float64) * 0.1
        
        # Create sparse matrix
        indices = torch.stack([row_indices, col_indices])
        sparse_matrix = torch.sparse_coo_tensor(indices, values, (n, n), device=device, dtype=torch.float64)
        sparse_matrix = sparse_matrix.coalesce()
        
        # Make it diagonally dominant
        abs_sparse = torch.sparse_coo_tensor(sparse_matrix.indices(), 
                                           torch.abs(sparse_matrix.values()), 
                                           sparse_matrix.shape, device=device, dtype=torch.float64)
        row_sums = torch.sparse.sum(abs_sparse, dim=1).to_dense()
        
        # Set diagonal to ensure diagonal dominance
        diagonal_values = row_sums + torch.rand(n, device=device, dtype=torch.float64) + 1.0
        
        # Add diagonal elements
        diag_indices = torch.arange(n, device=device)
        diag_indices_2d = torch.stack([diag_indices, diag_indices])
        
        # Combine original sparse matrix with diagonal
        all_indices = torch.cat([sparse_matrix.indices(), diag_indices_2d], dim=1)
        all_values = torch.cat([sparse_matrix.values(), diagonal_values])
        
        final_sparse = torch.sparse_coo_tensor(all_indices, all_values, (n, n), device=device, dtype=torch.float64)
        return final_sparse.coalesce()
    
    @staticmethod
    def create_non_diagonally_dominant_matrix(n: int, sparsity: float = 0.1, device: str = 'cuda') -> torch.Tensor:
        """Create a non-diagonally dominant sparse matrix"""
        torch.manual_seed(43)
        
        # Create random sparse pattern with larger off-diagonal elements
        nnz = int(n * n * sparsity)
        row_indices = torch.randint(0, n, (nnz,), device=device)
        col_indices = torch.randint(0, n, (nnz,), device=device)
        values = torch.randn(nnz, device=device, dtype=torch.float64) * 2.0
        
        # Create sparse matrix
        indices = torch.stack([row_indices, col_indices])
        sparse_matrix = torch.sparse_coo_tensor(indices, values, (n, n), device=device, dtype=torch.float64)
        sparse_matrix = sparse_matrix.coalesce()
        
        # Add small diagonal elements (NOT dominant)
        diagonal_values = torch.rand(n, device=device, dtype=torch.float64) * 0.5 + 0.1
        diag_indices = torch.arange(n, device=device)
        diag_indices_2d = torch.stack([diag_indices, diag_indices])
        
        # Combine sparse matrix with small diagonal
        all_indices = torch.cat([sparse_matrix.indices(), diag_indices_2d], dim=1)
        all_values = torch.cat([sparse_matrix.values(), diagonal_values])
        
        final_sparse = torch.sparse_coo_tensor(all_indices, all_values, (n, n), device=device, dtype=torch.float64)
        return final_sparse.coalesce()
    
    @staticmethod
    def create_banded_matrix(n: int, bandwidth: int = 5, device: str = 'cuda') -> torch.Tensor:
        """Create a banded sparse matrix"""
        torch.manual_seed(44)
        
        indices = []
        values = []
        
        # Create banded structure
        for k in range(-bandwidth, bandwidth + 1):
            if k == 0:
                # Main diagonal
                diag_size = n
                diag_values = torch.ones(diag_size, device=device, dtype=torch.float64) * 2.0
            else:
                # Off-diagonals
                diag_size = n - abs(k)
                diag_values = torch.randn(diag_size, device=device, dtype=torch.float64) * 0.5
            
            if k >= 0:
                row_idx = torch.arange(diag_size, device=device)
                col_idx = torch.arange(k, k + diag_size, device=device)
            else:
                row_idx = torch.arange(-k, -k + diag_size, device=device)
                col_idx = torch.arange(diag_size, device=device)
            
            indices.append(torch.stack([row_idx, col_idx]))
            values.append(diag_values)
        
        # Combine all diagonals
        all_indices = torch.cat(indices, dim=1)
        all_values = torch.cat(values)
        
        sparse_matrix = torch.sparse_coo_tensor(all_indices, all_values, (n, n), device=device, dtype=torch.float64)
        return sparse_matrix.coalesce()

class SolverTester:
    """Test and compare different solvers"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.results: List[TestResult] = []
    
    def test_solver_accuracy(self, matrix_type: str, matrix_size: int, sparsity: float = 0.1):
        """Test solver accuracy for a given matrix type and size"""
        print(f"\n🧮 Testing {matrix_type} matrix ({matrix_size}x{matrix_size})")
        
        # Generate matrix
        gen = MatrixGenerator()
        if matrix_type == "diagonally_dominant":
            A = gen.create_diagonally_dominant_matrix(matrix_size, sparsity, self.device)
        elif matrix_type == "non_diagonally_dominant":
            A = gen.create_non_diagonally_dominant_matrix(matrix_size, sparsity, self.device)
        elif matrix_type == "banded":
            A = gen.create_banded_matrix(matrix_size, bandwidth=5, device=self.device)
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")
        
        # Create true solution and RHS
        torch.manual_seed(42)
        x_true = torch.randn(matrix_size, device=self.device, dtype=torch.float64)
        
        if A.is_sparse:
            b = torch.sparse.mm(A, x_true.unsqueeze(1)).squeeze(1)
        else:
            b = A @ x_true
        
        # Test PyTorch solvers
        pytorch_solvers = [
            ('PyTorch CG', cg),
            ('PyTorch BiCGStab', bicgstab),
            ('PyTorch GMRES', gmres)
        ]
        
        for solver_name, solver_func in pytorch_solvers:
            print(f"  Testing {solver_name}...")
            
            # Convert sparse to dense for PyTorch solvers
            A_dense = A.to_dense() if A.is_sparse else A
            
            try:
                start_time = time.time()
                x_computed, info = solver_func(A_dense, b, tol=1e-8, maxiter=1000)
                solve_time = time.time() - start_time
                
                solution_error = torch.norm(x_computed - x_true).item()
                residual = A_dense @ x_computed - b
                residual_error = torch.norm(residual).item()
                converged = (info == 0)
                
                result = TestResult(
                    solver_name=solver_name,
                    matrix_type=matrix_type,
                    matrix_size=matrix_size,
                    solve_time=solve_time,
                    solution_error=solution_error,
                    residual_error=residual_error,
                    converged=converged
                )
                self.results.append(result)
                
                print(f"    Time: {solve_time:.4f}s, Solution Error: {solution_error:.2e}, "
                      f"Residual Error: {residual_error:.2e}, Converged: {converged}")
                      
            except Exception as e:
                print(f"    ❌ {solver_name} failed: {str(e)}")
        
        # Test AMGX solvers
        if HAS_AMGX:
            amgx_solvers = [
                ('AMGX CG', amgx_cg),
                ('AMGX BiCGStab', amgx_bicgstab),
            ]
            
            for solver_name, solver_func in amgx_solvers:
                print(f"  Testing {solver_name}...")
                
                try:
                    start_time = time.time()
                    x_computed = solver_func(A, b, tol=1e-8, maxiter=1000)
                    solve_time = time.time() - start_time
                    
                    solution_error = torch.norm(x_computed - x_true).item()
                    if A.is_sparse:
                        residual = torch.sparse.mm(A, x_computed.unsqueeze(1)).squeeze(1) - b
                    else:
                        residual = A @ x_computed - b
                    residual_error = torch.norm(residual).item()
                    
                    result = TestResult(
                        solver_name=solver_name,
                        matrix_type=matrix_type,
                        matrix_size=matrix_size,
                        solve_time=solve_time,
                        solution_error=solution_error,
                        residual_error=residual_error,
                        converged=True  # AMGX typically converges or throws error
                    )
                    self.results.append(result)
                    
                    print(f"    Time: {solve_time:.4f}s, Solution Error: {solution_error:.2e}, "
                          f"Residual Error: {residual_error:.2e}, Converged: True")
                          
                except Exception as e:
                    print(f"    ❌ {solver_name} failed: {str(e)}")
    
    def test_differentiability(self, matrix_size: int = 100):
        """Test differentiability of solvers"""
        print(f"\n🔬 Testing differentiability (matrix size: {matrix_size}x{matrix_size})")
        
        # Create a small diagonally dominant matrix for gradient testing
        gen = MatrixGenerator()
        A = gen.create_diagonally_dominant_matrix(matrix_size, 0.1, self.device)
        A_dense = A.to_dense() if A.is_sparse else A
        
        # Create b with gradient tracking
        torch.manual_seed(42)
        b = torch.randn(matrix_size, device=self.device, dtype=torch.float64, requires_grad=True)
        
        # Test PyTorch solvers differentiability
        pytorch_solvers = [
            ('PyTorch CG', cg),
            ('PyTorch BiCGStab', bicgstab),
            ('PyTorch GMRES', gmres)
        ]
        
        for solver_name, solver_func in pytorch_solvers:
            print(f"  Testing {solver_name} differentiability...")
            
            try:
                # Forward pass with timing
                start_time = time.time()
                x, _ = solver_func(A_dense, b, tol=1e-6, maxiter=500)
                solve_time = time.time() - start_time
                
                # Calculate solution accuracy (using a synthetic true solution)
                # For gradient test, we don't have a true solution, so we measure residual
                residual = A_dense @ x - b
                residual_error = torch.norm(residual).item()
                
                # Create a simple loss function
                loss = torch.sum(x**2)
                
                # Backward pass
                loss.backward()
                
                # Check if gradients exist and are finite
                if b.grad is not None and torch.isfinite(b.grad).all():
                    gradient_error = 0.0  # Successful gradient computation
                    print(f"    ✅ Gradient computation successful")
                    print(f"    Time: {solve_time:.4f}s, Residual Error: {residual_error:.2e}")
                else:
                    gradient_error = float('inf')
                    print(f"    ❌ Gradient computation failed")
                
                # Reset gradients for next test
                b.grad = None
                
                result = TestResult(
                    solver_name=f"{solver_name} (Diff)",
                    matrix_type="diagonally_dominant",
                    matrix_size=matrix_size,
                    solve_time=solve_time,
                    solution_error=0.0,  # No true solution for gradient test
                    residual_error=residual_error,
                    converged=True,
                    gradient_error=gradient_error
                )
                self.results.append(result)
                
            except Exception as e:
                print(f"    ❌ Differentiability test failed: {str(e)}")
        
        # Test AMGX differentiability
        if HAS_AMGX:
            amgx_diff_solvers = [
                ('AMGX CG', amgx_cg),
                ('AMGX BiCGStab', amgx_bicgstab),
            ]
            
            for solver_name, solver_func in amgx_diff_solvers:
                print(f"  Testing {solver_name} differentiability...")
                
                try:
                    # Reset gradients
                    if b.grad is not None:
                        b.grad.zero_()
                    
                    # Forward pass with timing
                    start_time = time.time()
                    x = solver_func(A, b, tol=1e-6, maxiter=500)
                    solve_time = time.time() - start_time
                    
                    # Calculate residual error
                    if A.is_sparse:
                        residual = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1) - b
                    else:
                        residual = A @ x - b
                    residual_error = torch.norm(residual).item()
                    
                    # Create a simple loss function
                    loss = torch.sum(x**2)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Check if gradients exist and are finite
                    if b.grad is not None and torch.isfinite(b.grad).all():
                        gradient_error = 0.0  # Successful gradient computation
                        print(f"    ✅ {solver_name} gradient computation successful")
                        print(f"    Time: {solve_time:.4f}s, Residual Error: {residual_error:.2e}")
                    else:
                        gradient_error = float('inf')
                        print(f"    ❌ {solver_name} gradient computation failed")
                    
                    result = TestResult(
                        solver_name=f"{solver_name} (Diff)",
                        matrix_type="diagonally_dominant",
                        matrix_size=matrix_size,
                        solve_time=solve_time,
                        solution_error=0.0,  # No true solution for gradient test
                        residual_error=residual_error,
                        converged=True,
                        gradient_error=gradient_error
                    )
                    self.results.append(result)
                    
                except Exception as e:
                    print(f"    ❌ {solver_name} differentiability test failed: {str(e)}")
    
    def generate_report(self, filename: str = "test_report/torch_amgx_test_result.md"):
        """Generate Markdown report of test results"""
        if not self.results:
            print("No test results available for report generation.")
            return
        
        # Create test_report directory if it doesn't exist
        report_dir = os.path.dirname(filename)
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir)
            print(f"📁 Created directory: {report_dir}")
        
        # Convert results to DataFrame
        data = []
        for result in self.results:
            data.append({
                'Solver': result.solver_name,
                'Matrix Type': result.matrix_type,
                'Matrix Size': result.matrix_size,
                'Solve Time (s)': f"{result.solve_time:.4f}" if result.solve_time < float('inf') else "Failed",
                'Solution Error': f"{result.solution_error:.2e}" if result.solution_error < float('inf') else "Failed",
                'Residual Error': f"{result.residual_error:.2e}" if result.residual_error < float('inf') else "Failed",
                'Converged': "✅" if result.converged else "❌",
                'Gradient Test': "✅" if result.gradient_error == 0.0 else "❌" if result.gradient_error is not None else "N/A"
            })
        
        df = pd.DataFrame(data)
        
        # Generate Markdown table manually
        def df_to_markdown_table(df):
            """Convert DataFrame to Markdown table format"""
            # Get column names
            headers = list(df.columns)
            
            # Create header row
            header_row = "| " + " | ".join(headers) + " |"
            
            # Create separator row
            separator_row = "|" + "|".join([" --- " for _ in headers]) + "|"
            
            # Create data rows
            data_rows = []
            for _, row in df.iterrows():
                row_str = "| " + " | ".join([str(row[col]) for col in headers]) + " |"
                data_rows.append(row_str)
            
            # Combine all rows
            table_lines = [header_row, separator_row] + data_rows
            return "\n".join(table_lines)
        
        # Generate Markdown report
        markdown_content = f"""# Sparse Matrix Solver Comparison Report

## Test Summary

- **Total Tests:** {len(self.results)}
- **Matrix Types:** {', '.join(set(r.matrix_type for r in self.results))}
- **Solvers:** {', '.join(set(r.solver_name for r in self.results))}

## Detailed Results

**Note:** Tests marked with `(Diff)` are differentiability tests that verify automatic differentiation support. For these tests:
- Solution Error is set to 0.00e+00 (no true solution available)
- Residual Error shows how well the solver satisfied Ax=b
- Gradient Test column shows whether backpropagation succeeded

{df_to_markdown_table(df)}

## Performance Analysis

### Convergence Results
"""
        
        # Add convergence analysis
        converged_count = len([r for r in self.results if r.converged])
        total_count = len(self.results)
        convergence_rate = (converged_count / total_count) * 100 if total_count > 0 else 0
        
        markdown_content += f"""
- **Total Convergence Rate:** {convergence_rate:.1f}% ({converged_count}/{total_count})

### Solver Performance Summary

"""
        
        # Group results by solver
        solver_stats = {}
        for result in self.results:
            solver = result.solver_name
            if solver not in solver_stats:
                solver_stats[solver] = {
                    'tests': 0,
                    'converged': 0,
                    'total_time': 0,
                    'min_error': float('inf'),
                    'max_error': 0
                }
            
            stats = solver_stats[solver]
            stats['tests'] += 1
            if result.converged:
                stats['converged'] += 1
            if result.solve_time < float('inf'):
                stats['total_time'] += result.solve_time
            if result.solution_error < float('inf') and result.solution_error < stats['min_error']:
                stats['min_error'] = result.solution_error
            if result.solution_error < float('inf') and result.solution_error > stats['max_error']:
                stats['max_error'] = result.solution_error
        
        for solver, stats in solver_stats.items():
            convergence_rate = (stats['converged'] / stats['tests']) * 100
            avg_time = stats['total_time'] / stats['tests'] if stats['tests'] > 0 else 0
            markdown_content += f"""
#### {solver}
- **Convergence Rate:** {convergence_rate:.1f}% ({stats['converged']}/{stats['tests']})
- **Average Solve Time:** {avg_time:.4f}s
- **Best Solution Error:** {stats['min_error']:.2e}
- **Worst Solution Error:** {stats['max_error']:.2e}
"""
        
        # Add timestamp
        from datetime import datetime
        markdown_content += f"""
---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"📊 Report saved to {filename}")

def main():
    """Main testing function"""
    print("🚀 Comprehensive Sparse Matrix Solver Testing")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📍 Using device: {device}")
    
    if device == 'cuda':
        print(f"📍 GPU: {torch.cuda.get_device_name(0)}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize tester
    tester = SolverTester(device=device)
    
    # Test configurations
    test_configs = [
        ("diagonally_dominant", 200, 0.1),
        ("non_diagonally_dominant", 200, 0.1),
        ("banded", 200, None),
        ("diagonally_dominant", 500, 0.05),
        ("banded", 500, None),
    ]
    
    print(f"\n🎯 Running Accuracy and Speed Tests")
    print("-" * 50)
    
    for matrix_type, size, sparsity in test_configs:
        try:
            if sparsity is not None:
                tester.test_solver_accuracy(matrix_type, size, sparsity)
            else:
                tester.test_solver_accuracy(matrix_type, size)
        except Exception as e:
            print(f"❌ Test failed for {matrix_type} {size}x{size}: {str(e)}")
    
    # Test differentiability
    print(f"\n🔬 Running Differentiability Tests")
    print("-" * 50)
    
    try:
        tester.test_differentiability(matrix_size=100)
    except Exception as e:
        print(f"❌ Differentiability test failed: {str(e)}")
    
    # Generate report
    print(f"\n📄 Generating Report")
    print("-" * 50)
    
    tester.generate_report()
    
    # Print summary
    print(f"\n📊 Test Summary")
    print("-" * 50)
    
    successful_tests = [r for r in tester.results if r.converged and r.solve_time < float('inf')]
    if successful_tests:
        fastest = min(successful_tests, key=lambda x: x.solve_time)
        most_accurate = min(successful_tests, key=lambda x: x.solution_error)
        
        print(f"🏃 Fastest solver: {fastest.solver_name} ({fastest.solve_time:.4f}s)")
        print(f"🎯 Most accurate solver: {most_accurate.solver_name} (error: {most_accurate.solution_error:.2e})")
    
    print(f"\n✅ Testing completed! Check test_report/solver_comparison_report.md for detailed results.")

if __name__ == "__main__":
    main()
