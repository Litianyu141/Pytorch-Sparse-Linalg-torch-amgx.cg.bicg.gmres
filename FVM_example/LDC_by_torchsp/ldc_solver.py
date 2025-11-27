#!/usr/bin/env python3
"""
Lid-Driven Cavity Flow Solver using pytorch_sparse_solver

This is the main entry point for the LDC simulation that demonstrates
the usage of all three solver modules (A, B, C) in pytorch_sparse_solver.

Features:
- Supports all solver backends: Module A (CG, BiCGStab, GMRES), Module B (PyAMGX), Module C (cuDSS)
- Auto-detects available modules and uses the best available solver
- Benchmarks different solvers and generates performance reports

Usage:
    python ldc_solver.py                    # Run with default settings
    python ldc_solver.py --solver cg        # Use specific solver
    python ldc_solver.py --backend module_c # Use specific backend
    python ldc_solver.py --benchmark        # Benchmark all solvers
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from enum import Enum
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import pytorch_sparse_solver
try:
    from pytorch_sparse_solver import SparseSolver, SolverResult
    from pytorch_sparse_solver.utils.availability import (
        check_module_a_available,
        check_module_b_available,
        check_module_c_available,
        print_availability_report
    )
    SOLVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pytorch_sparse_solver not available: {e}")
    print("Falling back to direct PyTorch solver...")
    SOLVER_AVAILABLE = False


class SolverConfig:
    """Configuration for the pressure solver"""
    def __init__(self,
                 backend: str = 'auto',
                 method: str = 'cg',
                 tol: float = 1e-10,
                 maxiter: int = 1000,
                 restart: int = 30):
        self.backend = backend
        self.method = method
        self.tol = tol
        self.maxiter = maxiter
        self.restart = restart


class LDCSimulation:
    """
    Lid-Driven Cavity Flow Simulation using Finite Volume Method
    with pytorch_sparse_solver for linear system solving
    """

    def __init__(self,
                 nx: int = 100,
                 ny: int = 100,
                 Re: float = 100.0,
                 Ut: float = 1.0,
                 solver_config: SolverConfig = None,
                 device: str = None,
                 dtype: torch.dtype = torch.float64):
        """
        Initialize the LDC simulation

        Args:
            nx, ny: Grid resolution
            Re: Reynolds number
            Ut: Top wall velocity
            solver_config: Configuration for the linear solver
            device: 'cuda' or 'cpu' (auto-detect if None)
            dtype: Floating point precision
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        torch.set_default_dtype(dtype)

        # Grid parameters
        self.nx, self.ny = nx, ny
        self.lx, self.ly = 1.0, 1.0
        self.dx = torch.tensor(self.lx / nx, dtype=dtype, device=self.device)
        self.dy = torch.tensor(self.ly / ny, dtype=dtype, device=self.device)

        # Physical parameters
        self.Ut = torch.tensor(Ut, dtype=dtype, device=self.device)
        self.miu = torch.tensor(Ut * self.lx / Re, dtype=dtype, device=self.device)
        self.Re = Re

        # Time step (CFL condition)
        dt_diffusion = 0.25 * (self.lx/nx)**2 / (Ut * self.lx / Re)
        dt_convection = 4.0 * (Ut * self.lx / Re) / Ut**2
        self.dt = torch.tensor(min(dt_diffusion, dt_convection), dtype=dtype, device=self.device)

        # Solver configuration
        self.solver_config = solver_config or SolverConfig()

        # Initialize solver
        self._init_solver()

        # Initialize grids and fields
        self._setup_grids()
        self._init_fields()
        self._setup_pressure_matrix()

        # Statistics
        self.stats = {
            'momentum_time': 0.0,
            'solver_time': 0.0,
            'total_steps': 0
        }

        print(f"LDC Simulation Initialized:")
        print(f"  Grid: {nx}x{ny}")
        print(f"  Re: {Re}")
        print(f"  Device: {self.device}")
        print(f"  dt: {self.dt.item():.6f}")

    def _init_solver(self):
        """Initialize the sparse solver"""
        if SOLVER_AVAILABLE:
            self.solver = SparseSolver(verbose=False)
            print(f"  Solver Backend: {self.solver_config.backend}")
            print(f"  Solver Method: {self.solver_config.method}")
            print(f"  Available backends: {self.solver.available_backends}")
        else:
            self.solver = None
            print("  Solver: PyTorch Direct (fallback)")

    def _setup_grids(self):
        """Setup staggered grid coordinates"""
        # Cell center coordinates
        xx = torch.linspace(self.dx/2, self.lx - self.dx/2, self.nx, device=self.device)
        yy = torch.linspace(self.dy/2, self.ly - self.dy/2, self.ny, device=self.device)
        self.xcc, self.ycc = torch.meshgrid(xx, yy, indexing='ij')
        self.xcc, self.ycc = self.xcc.t(), self.ycc.t()

    def _init_fields(self):
        """Initialize velocity and pressure fields"""
        self.u = torch.zeros((self.ny+2, self.nx+2), device=self.device, dtype=self.dtype)
        self.v = torch.zeros((self.ny+2, self.nx+2), device=self.device, dtype=self.dtype)
        self.ut = torch.zeros_like(self.u)
        self.vt = torch.zeros_like(self.v)
        self.p = torch.zeros((self.ny+2, self.nx+2), device=self.device, dtype=self.dtype)

    def _setup_pressure_matrix(self):
        """Setup pressure Poisson equation matrix"""
        n = self.nx * self.ny
        dx_sq = (self.dx * self.dx).item()
        dy_sq = (self.dy * self.dy).item()

        # Build sparse matrix
        indices_list = []
        values_list = []

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                diag_val = 0.0

                # East neighbor
                if i < self.nx - 1:
                    indices_list.append([idx, idx + 1])
                    values_list.append(1.0 / dx_sq)
                    diag_val -= 1.0 / dx_sq

                # West neighbor
                if i > 0:
                    indices_list.append([idx, idx - 1])
                    values_list.append(1.0 / dx_sq)
                    diag_val -= 1.0 / dx_sq

                # North neighbor
                if j < self.ny - 1:
                    indices_list.append([idx, idx + self.nx])
                    values_list.append(1.0 / dy_sq)
                    diag_val -= 1.0 / dy_sq

                # South neighbor
                if j > 0:
                    indices_list.append([idx, idx - self.nx])
                    values_list.append(1.0 / dy_sq)
                    diag_val -= 1.0 / dy_sq

                # Diagonal
                indices_list.append([idx, idx])
                values_list.append(diag_val)

        indices = torch.tensor(indices_list, dtype=torch.long, device=self.device).t()
        values = torch.tensor(values_list, dtype=self.dtype, device=self.device)

        self.A_sparse = torch.sparse_coo_tensor(
            indices, values, (n, n), device=self.device, dtype=self.dtype
        ).coalesce()

        # Also create CSR version for Module C
        self.A_csr = self.A_sparse.to_sparse_csr()

        # Dense version for iterative solvers
        self.A_dense = self.A_sparse.to_dense()

    def apply_boundary_conditions(self):
        """Apply velocity boundary conditions"""
        # u-velocity
        self.u[1:-1, 1] = 0.0      # Left
        self.u[1:-1, -1] = 0.0     # Right
        self.u[-1, 1:] = 2.0 * self.Ut - self.u[-2, 1:]  # Top (moving)
        self.u[0, 1:] = -self.u[1, 1:]   # Bottom

        # v-velocity
        self.v[1:, 0] = -self.v[1:, 1]   # Left
        self.v[1:, -1] = -self.v[1:, -2] # Right
        self.v[1, 1:-1] = 0.0     # Bottom
        self.v[-1, 1:-1] = 0.0    # Top

    def compute_momentum(self):
        """Compute momentum predictor step"""
        # U-momentum
        ue = 0.5 * (self.u[1:-1, 3:] + self.u[1:-1, 2:-1])
        uw = 0.5 * (self.u[1:-1, 2:-1] + self.u[1:-1, 1:-2])
        un = 0.5 * (self.u[2:, 2:-1] + self.u[1:-1, 2:-1])
        us = 0.5 * (self.u[1:-1, 2:-1] + self.u[0:-2, 2:-1])
        vn = 0.5 * (self.v[2:, 2:-1] + self.v[2:, 1:-2])
        vs = 0.5 * (self.v[1:-1, 2:-1] + self.v[1:-1, 1:-2])

        conv_u = -(ue*ue - uw*uw)/self.dx - (un*vn - us*vs)/self.dy
        diff_u = self.miu * (
            (self.u[1:-1, 3:] - 2*self.u[1:-1, 2:-1] + self.u[1:-1, 1:-2])/self.dx**2 +
            (self.u[2:, 2:-1] - 2*self.u[1:-1, 2:-1] + self.u[0:-2, 2:-1])/self.dy**2
        )
        self.ut[1:-1, 2:-1] = self.u[1:-1, 2:-1] + self.dt * (conv_u + diff_u)

        # V-momentum
        ve = 0.5 * (self.v[2:-1, 2:] + self.v[2:-1, 1:-1])
        vw = 0.5 * (self.v[2:-1, 1:-1] + self.v[2:-1, :-2])
        ue_v = 0.5 * (self.u[2:-1, 2:] + self.u[1:-2, 2:])
        uw_v = 0.5 * (self.u[2:-1, 1:-1] + self.u[1:-2, 1:-1])
        vn = 0.5 * (self.v[3:, 1:-1] + self.v[2:-1, 1:-1])
        vs = 0.5 * (self.v[2:-1, 1:-1] + self.v[1:-2, 1:-1])

        conv_v = -(ue_v*ve - uw_v*vw)/self.dx - (vn*vn - vs*vs)/self.dy
        diff_v = self.miu * (
            (self.v[2:-1, 2:] - 2*self.v[2:-1, 1:-1] + self.v[2:-1, :-2])/self.dx**2 +
            (self.v[3:, 1:-1] - 2*self.v[2:-1, 1:-1] + self.v[1:-2, 1:-1])/self.dy**2
        )
        self.vt[2:-1, 1:-1] = self.v[2:-1, 1:-1] + self.dt * (conv_v + diff_v)

    def solve_pressure(self) -> Tuple[float, str]:
        """Solve pressure Poisson equation using pytorch_sparse_solver"""
        # Compute divergence
        div = (self.ut[1:-1, 2:] - self.ut[1:-1, 1:-1])/self.dx + \
              (self.vt[2:, 1:-1] - self.vt[1:-1, 1:-1])/self.dy
        rhs = (1.0 / self.dt) * div.reshape(-1)

        tic = time.perf_counter()

        if self.solver is not None:
            # Use pytorch_sparse_solver
            cfg = self.solver_config

            # Select appropriate matrix format based on backend
            if cfg.backend == 'module_c':
                A = self.A_csr
            else:
                A = self.A_dense

            try:
                x, result = self.solver.solve(
                    A, rhs,
                    method=cfg.method,
                    backend=cfg.backend,
                    tol=cfg.tol,
                    maxiter=cfg.maxiter
                )
                status = f"{cfg.method}/{result.backend} - Residual: {result.residual:.2e}"
            except Exception as e:
                # Fallback to direct solver
                x = torch.linalg.solve(self.A_dense, rhs)
                status = f"Fallback to direct: {str(e)[:50]}"
        else:
            # Fallback to PyTorch direct solver
            x = torch.linalg.solve(self.A_dense, rhs)
            status = "PyTorch direct solver"

        solve_time = time.perf_counter() - tic

        # Update pressure field
        self.p.fill_(0.0)
        self.p[1:-1, 1:-1] = x.reshape(self.ny, self.nx)

        return solve_time, status

    def pressure_correction(self):
        """Apply pressure correction to velocities"""
        self.u[1:-1, 2:-1] = self.ut[1:-1, 2:-1] - \
            self.dt * (self.p[1:-1, 2:-1] - self.p[1:-1, 1:-2]) / self.dx
        self.v[2:-1, 1:-1] = self.vt[2:-1, 1:-1] - \
            self.dt * (self.p[2:-1, 1:-1] - self.p[1:-2, 1:-1]) / self.dy

    def check_mass_conservation(self) -> float:
        """Check continuity residual"""
        div = (self.u[1:-1, 2:] - self.u[1:-1, 1:-1])/self.dx + \
              (self.v[2:, 1:-1] - self.v[1:-1, 1:-1])/self.dy
        return torch.norm(div).item()

    def step(self) -> Dict[str, Any]:
        """Perform one time step"""
        self.apply_boundary_conditions()

        tic = time.perf_counter()
        self.compute_momentum()
        mom_time = time.perf_counter() - tic

        solve_time, status = self.solve_pressure()
        self.pressure_correction()

        mass_res = self.check_mass_conservation()

        # Update stats
        self.stats['momentum_time'] += mom_time
        self.stats['solver_time'] += solve_time
        self.stats['total_steps'] += 1

        return {
            'momentum_time': mom_time,
            'solver_time': solve_time,
            'solver_status': status,
            'mass_residual': mass_res
        }

    def get_velocities(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get cell-centered velocities"""
        ucc = 0.5 * (self.u[1:-1, 2:] + self.u[1:-1, 1:-1])
        vcc = 0.5 * (self.v[2:, 1:-1] + self.v[1:-1, 1:-1])
        speed = torch.sqrt(ucc**2 + vcc**2)
        return ucc, vcc, speed

    def run(self, nsteps: int, plot_interval: int = 0, save_dir: str = None) -> Dict[str, Any]:
        """Run simulation for specified number of steps"""
        print(f"\nRunning {nsteps} time steps...")

        for step in range(nsteps):
            info = self.step()

            if step % 10 == 0 or step == nsteps - 1:
                print(f"Step {step:4d}: Mass residual: {info['mass_residual']:.2e}, "
                      f"Solver: {info['solver_status'][:50]}")

            if plot_interval > 0 and (step % plot_interval == 0 or step == nsteps - 1):
                self.plot(step, save_dir)

        # Summary
        print(f"\n=== Simulation Complete ===")
        print(f"Total steps: {self.stats['total_steps']}")
        print(f"Momentum time: {self.stats['momentum_time']:.3f}s")
        print(f"Solver time: {self.stats['solver_time']:.3f}s")
        print(f"Avg solver time: {self.stats['solver_time']/nsteps*1000:.2f} ms/step")

        return self.stats

    def plot(self, step: int, save_dir: str = None):
        """Plot current state"""
        ucc, vcc, speed = self.get_velocities()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Velocity magnitude
        im = axes[0].contourf(self.xcc.cpu().numpy(), self.ycc.cpu().numpy(),
                              speed.cpu().numpy(), levels=20, cmap='RdBu_r')
        plt.colorbar(im, ax=axes[0])
        axes[0].set_title(f'Velocity Magnitude (Step {step})')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_aspect('equal')

        # Streamlines
        x = np.linspace(0, 1, self.nx)
        y = np.linspace(0, 1, self.ny)
        xx, yy = np.meshgrid(x, y)
        axes[1].streamplot(xx, yy, ucc.cpu().numpy(), vcc.cpu().numpy(),
                          color=speed.cpu().numpy(), cmap='autumn', density=1.5)
        axes[1].set_title(f'Streamlines (Step {step})')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_aspect('equal')

        plt.tight_layout()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{save_dir}/ldc_step_{step:06d}.png", dpi=150)
            plt.close()
        else:
            plt.pause(0.01)


def benchmark_solvers(nx: int = 50, nsteps: int = 50):
    """Benchmark all available solvers"""
    print("\n" + "=" * 70)
    print("LDC Solver Benchmark")
    print("=" * 70)

    if SOLVER_AVAILABLE:
        print_availability_report()

    configs = []

    # Module A configurations
    for method in ['cg', 'bicgstab', 'gmres']:
        configs.append(('module_a', method, f'Module A - {method.upper()}'))

    # Module C (direct)
    configs.append(('module_c', 'direct', 'Module C - cuDSS Direct'))

    # Module B (if available)
    if SOLVER_AVAILABLE and check_module_b_available():
        for method in ['cg', 'bicgstab', 'gmres']:
            configs.append(('module_b', method, f'Module B - AMGX {method.upper()}'))

    results = []

    for backend, method, name in configs:
        print(f"\n[Testing] {name}")

        try:
            solver_cfg = SolverConfig(backend=backend, method=method)
            sim = LDCSimulation(nx=nx, ny=nx, Re=100.0, solver_config=solver_cfg)

            start = time.perf_counter()
            stats = sim.run(nsteps, plot_interval=0)
            total_time = time.perf_counter() - start

            results.append({
                'name': name,
                'backend': backend,
                'method': method,
                'total_time': total_time,
                'solver_time': stats['solver_time'],
                'success': True
            })
            print(f"  ✅ Success: {total_time:.3f}s total, {stats['solver_time']:.3f}s solver")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({
                'name': name,
                'backend': backend,
                'method': method,
                'success': False,
                'error': str(e)
            })

    # Print summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)

    successful = [r for r in results if r['success']]
    if successful:
        sorted_results = sorted(successful, key=lambda x: x['total_time'])
        print("\nPerformance Ranking (fastest first):")
        for i, r in enumerate(sorted_results, 1):
            print(f"  {i}. {r['name']:30s}: {r['total_time']:.3f}s "
                  f"(solver: {r['solver_time']:.3f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(description='LDC Flow Solver using pytorch_sparse_solver')
    parser.add_argument('--nx', type=int, default=100, help='Grid size')
    parser.add_argument('--steps', type=int, default=100, help='Number of time steps')
    parser.add_argument('--Re', type=float, default=100.0, help='Reynolds number')
    parser.add_argument('--backend', type=str, default='auto',
                       choices=['auto', 'module_a', 'module_b', 'module_c'],
                       help='Solver backend')
    parser.add_argument('--method', type=str, default='cg',
                       choices=['cg', 'bicgstab', 'gmres', 'direct'],
                       help='Solver method')
    parser.add_argument('--plot', type=int, default=0, help='Plot interval (0=no plots)')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save plots')

    args = parser.parse_args()

    if args.benchmark:
        benchmark_solvers(nx=min(args.nx, 50), nsteps=min(args.steps, 50))
    else:
        solver_cfg = SolverConfig(
            backend=args.backend,
            method=args.method
        )

        sim = LDCSimulation(
            nx=args.nx,
            ny=args.nx,
            Re=args.Re,
            solver_config=solver_cfg
        )

        sim.run(args.steps, plot_interval=args.plot, save_dir=args.save_dir)


if __name__ == '__main__':
    main()
