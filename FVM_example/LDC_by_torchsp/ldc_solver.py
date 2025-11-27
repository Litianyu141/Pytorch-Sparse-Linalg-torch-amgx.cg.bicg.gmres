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
Lid-Driven Cavity (LDC) Flow Solver using pytorch_sparse_solver

This script solves the classic lid-driven cavity flow problem using the
Finite Volume Method (FVM) with staggered grid and fractional step method.
The pressure Poisson equation is solved using Module A's iterative solvers
(BiCGStab or GMRES) with JIT compilation enabled for performance.

Usage:
    # Run with default settings (Re=400, 100x100 grid, 1000 steps, BiCGStab)
    python ldc_solver.py

    # Use GMRES solver instead of BiCGStab
    python ldc_solver.py --method gmres

    # Customize grid size and Reynolds number
    python ldc_solver.py --nx 64 --Re 100 --steps 500

    # Run quick test
    python ldc_solver.py --quick

    # Save results to specific directory
    python ldc_solver.py --save-dir ./my_results

    # Disable display (for headless servers)
    python ldc_solver.py --no-plot

    # Show help
    python ldc_solver.py --help

Parameters:
    --nx        : Grid resolution (default: 100)
    --Re        : Reynolds number (default: 400)
    --steps     : Number of time steps (default: 1000)
    --method    : Solver method: 'bicgstab' or 'gmres' (default: bicgstab)
    --quick     : Quick test with smaller grid (32x32, 200 steps)
    --save-dir  : Directory to save results (default: auto-generated)
    --no-plot   : Disable plotting (useful for headless servers)

Output:
    - Velocity magnitude contour (left subplot)
    - Streamlines colored by speed (right subplot)
    - Saved as PNG in the specified directory

Reference:
    Based on the staggered grid FVM implementation from:
    Tony Saad's Lid-Driven Cavity tutorial
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Suppress matplotlib warnings on headless systems
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend by default
import matplotlib.pyplot as plt

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import pytorch_sparse_solver Module A
try:
    from pytorch_sparse_solver.module_a import bicgstab, gmres
    from pytorch_sparse_solver.module_a.torch_sparse_linalg import _JIT_ENABLED
    SOLVER_AVAILABLE = True
    jit_status = "enabled" if _JIT_ENABLED else "disabled"
    print(f"pytorch_sparse_solver loaded successfully (JIT {jit_status})")
except ImportError as e:
    print(f"Warning: pytorch_sparse_solver not found: {e}")
    print("Using fallback PyTorch direct solver")
    SOLVER_AVAILABLE = False
    _JIT_ENABLED = False
    bicgstab = None
    gmres = None


class LDCSolver:
    """
    Lid-Driven Cavity Flow Solver using FVM with Staggered Grid

    Uses BiCGStab or GMRES from pytorch_sparse_solver Module A for pressure solve.
    JIT compilation is enabled for better performance.
    """

    def __init__(self, nx: int = 100, Re: float = 400.0,
                 method: str = 'bicgstab',
                 device: str = None, dtype: torch.dtype = torch.float64):
        """
        Initialize the LDC solver.

        Args:
            nx: Grid resolution (nx x nx cells)
            Re: Reynolds number
            method: Solver method ('bicgstab' or 'gmres')
            device: 'cuda' or 'cpu' (auto-detect if None)
            dtype: Floating point precision
        """
        # Device selection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.dtype = dtype

        # Solver method
        self.method = method.lower()
        if self.method not in ['bicgstab', 'gmres']:
            raise ValueError(f"Unknown method: {method}. Use 'bicgstab' or 'gmres'")

        # Grid parameters
        self.nx = self.ny = nx
        self.lx = self.ly = 1.0
        self.dx = self.lx / nx
        self.dy = self.ly / nx

        # Physical parameters
        self.Ut = 1.0  # Top lid velocity
        self.miu = self.Ut * self.lx / Re  # Dynamic viscosity
        self.Re = Re

        # Time step (CFL condition)
        dt_diffusion = 0.25 * self.dx**2 / self.miu
        dt_convection = 4.0 * self.miu / self.Ut**2
        self.dt = min(dt_diffusion, dt_convection)

        # Initialize fields
        self._init_fields()
        self._setup_pressure_matrix()

        # Statistics
        self.momentum_time = 0.0
        self.solver_time = 0.0
        self.total_solver_iterations = 0

        print(f"LDC Solver Initialized:")
        print(f"  Grid: {nx}x{nx}")
        print(f"  Reynolds Number: {Re}")
        print(f"  Device: {self.device}")
        print(f"  Solver: {self.method.upper()}")
        print(f"  dt: {self.dt:.6f}")
        print(f"  JIT: {'Enabled' if _JIT_ENABLED else 'Disabled'}")

    def _init_fields(self):
        """Initialize velocity and pressure fields."""
        ny, nx = self.ny, self.nx

        # Velocity fields with ghost cells
        self.u = torch.zeros((ny+2, nx+2), device=self.device, dtype=self.dtype)
        self.v = torch.zeros((ny+2, nx+2), device=self.device, dtype=self.dtype)
        self.ut = torch.zeros_like(self.u)
        self.vt = torch.zeros_like(self.v)

        # Pressure field
        self.p = torch.zeros((ny+2, nx+2), device=self.device, dtype=self.dtype)

        # Cell-centered coordinates for plotting
        xx = np.linspace(self.dx/2, self.lx - self.dx/2, nx)
        yy = np.linspace(self.dy/2, self.ly - self.dy/2, ny)
        self.xcc, self.ycc = np.meshgrid(xx, yy)

    def _setup_pressure_matrix(self):
        """Build pressure Poisson equation coefficient matrix."""
        nx, ny = self.nx, self.ny
        dx2, dy2 = self.dx**2, self.dy**2

        # Build coefficient arrays
        Ap = np.zeros((ny, nx))
        Ae = (1.0/dx2) * np.ones((ny, nx))
        Aw = (1.0/dx2) * np.ones((ny, nx))
        An = (1.0/dy2) * np.ones((ny, nx))
        As = (1.0/dy2) * np.ones((ny, nx))

        # Apply boundary conditions (zero gradient)
        Aw[:, 0] = 0.0   # Left wall
        Ae[:, -1] = 0.0  # Right wall
        An[-1, :] = 0.0  # Top wall
        As[0, :] = 0.0   # Bottom wall

        # Diagonal coefficients
        Ap = -(Aw + Ae + An + As)

        # Build sparse matrix using diagonals
        n = nx * ny
        d0 = Ap.reshape(n)
        de = Ae.reshape(n)[:-1]
        dw = Aw.reshape(n)[1:]
        dn = An.reshape(n)[:-nx]
        ds = As.reshape(n)[nx:]

        # Create sparse COO tensor
        indices_list = []
        values_list = []

        for i in range(n):
            # Diagonal
            indices_list.append([i, i])
            values_list.append(d0[i])

            # East (i, i+1)
            if i < n-1 and (i+1) % nx != 0:
                indices_list.append([i, i+1])
                values_list.append(de[i])

            # West (i, i-1)
            if i > 0 and i % nx != 0:
                indices_list.append([i, i-1])
                values_list.append(dw[i-1])

            # North (i, i+nx)
            if i < n - nx:
                indices_list.append([i, i+nx])
                values_list.append(dn[i])

            # South (i, i-nx)
            if i >= nx:
                indices_list.append([i, i-nx])
                values_list.append(ds[i-nx])

        indices = torch.tensor(indices_list, dtype=torch.long, device=self.device).t()
        values = torch.tensor(values_list, dtype=self.dtype, device=self.device)

        self.A_sparse = torch.sparse_coo_tensor(
            indices, values, (n, n), device=self.device, dtype=self.dtype
        ).coalesce()

        # Dense version for iterative solver
        self.A_dense = self.A_sparse.to_dense()

    def apply_boundary_conditions(self):
        """Apply velocity boundary conditions."""
        Ut, Ub = self.Ut, 0.0
        Vl, Vr = 0.0, 0.0

        # u-velocity BCs
        self.u[1:-1, 1] = 0.0                          # Left wall
        self.u[1:-1, -1] = 0.0                         # Right wall
        self.u[-1, 1:] = 2.0 * Ut - self.u[-2, 1:]    # Top wall (moving lid)
        self.u[0, 1:] = 2.0 * Ub - self.u[1, 1:]      # Bottom wall

        # v-velocity BCs
        self.v[1:, 0] = 2.0 * Vl - self.v[1:, 1]      # Left wall
        self.v[1:, -1] = 2.0 * Vr - self.v[1:, -2]    # Right wall
        self.v[1, 1:-1] = 0.0                          # Bottom wall
        self.v[-1, 1:-1] = 0.0                         # Top wall

    def compute_momentum(self):
        """Compute momentum predictor step (vectorized)."""
        dx, dy, dt, miu = self.dx, self.dy, self.dt, self.miu

        # U-momentum (interior points: u[1:-1, 2:-1])
        ue = 0.5 * (self.u[1:-1, 3:] + self.u[1:-1, 2:-1])
        uw = 0.5 * (self.u[1:-1, 2:-1] + self.u[1:-1, 1:-2])
        un = 0.5 * (self.u[2:, 2:-1] + self.u[1:-1, 2:-1])
        us = 0.5 * (self.u[1:-1, 2:-1] + self.u[:-2, 2:-1])
        vn = 0.5 * (self.v[2:, 2:-1] + self.v[2:, 1:-2])
        vs = 0.5 * (self.v[1:-1, 2:-1] + self.v[1:-1, 1:-2])

        convection_u = -(ue*ue - uw*uw)/dx - (un*vn - us*vs)/dy
        diffusion_u = miu * (
            (self.u[1:-1, 3:] - 2*self.u[1:-1, 2:-1] + self.u[1:-1, 1:-2])/dx**2 +
            (self.u[2:, 2:-1] - 2*self.u[1:-1, 2:-1] + self.u[:-2, 2:-1])/dy**2
        )
        self.ut[1:-1, 2:-1] = self.u[1:-1, 2:-1] + dt * (convection_u + diffusion_u)

        # V-momentum (interior points: v[2:-1, 1:-1])
        ve = 0.5 * (self.v[2:-1, 2:] + self.v[2:-1, 1:-1])
        vw = 0.5 * (self.v[2:-1, 1:-1] + self.v[2:-1, :-2])
        ue_v = 0.5 * (self.u[2:-1, 2:] + self.u[1:-2, 2:])
        uw_v = 0.5 * (self.u[2:-1, 1:-1] + self.u[1:-2, 1:-1])
        vn = 0.5 * (self.v[3:, 1:-1] + self.v[2:-1, 1:-1])
        vs = 0.5 * (self.v[2:-1, 1:-1] + self.v[1:-2, 1:-1])

        convection_v = -(ue_v*ve - uw_v*vw)/dx - (vn*vn - vs*vs)/dy
        diffusion_v = miu * (
            (self.v[2:-1, 2:] - 2*self.v[2:-1, 1:-1] + self.v[2:-1, :-2])/dx**2 +
            (self.v[3:, 1:-1] - 2*self.v[2:-1, 1:-1] + self.v[1:-2, 1:-1])/dy**2
        )
        self.vt[2:-1, 1:-1] = self.v[2:-1, 1:-1] + dt * (convection_v + diffusion_v)

    def solve_pressure(self):
        """Solve pressure Poisson equation using BiCGStab or GMRES."""
        dx, dy, dt = self.dx, self.dy, self.dt

        # Compute divergence of predicted velocity
        div_ut = torch.zeros_like(self.p)
        div_ut[1:-1, 1:-1] = (
            (self.ut[1:-1, 2:] - self.ut[1:-1, 1:-1])/dx +
            (self.vt[2:, 1:-1] - self.vt[1:-1, 1:-1])/dy
        )

        # RHS of pressure equation
        prhs = (1.0/dt) * div_ut[1:-1, 1:-1].reshape(-1)

        tic = time.perf_counter()
        solver_info = 0

        if SOLVER_AVAILABLE:
            try:
                if self.method == 'bicgstab':
                    pt, info = bicgstab(self.A_dense, prhs, tol=1e-10, maxiter=1000)
                else:  # gmres
                    pt, info = gmres(self.A_dense, prhs, tol=1e-10, maxiter=1000, restart=30)
                solver_info = info
            except Exception as e:
                # Fallback to direct solver on error
                warnings.warn(f"Iterative solver failed: {e}. Using direct solver.")
                pt = torch.linalg.solve(self.A_dense, prhs)
                solver_info = -1
        else:
            # Fallback to PyTorch direct solver
            pt = torch.linalg.solve(self.A_dense, prhs)

        solve_time = time.perf_counter() - tic
        self.solver_time += solve_time

        # Update pressure field
        self.p.fill_(0.0)
        self.p[1:-1, 1:-1] = pt.reshape(self.ny, self.nx)

        return solver_info

    def pressure_correction(self):
        """Apply pressure correction to velocities."""
        dx, dy, dt = self.dx, self.dy, self.dt

        self.u[1:-1, 2:-1] = self.ut[1:-1, 2:-1] - dt * (
            self.p[1:-1, 2:-1] - self.p[1:-1, 1:-2]
        ) / dx

        self.v[2:-1, 1:-1] = self.vt[2:-1, 1:-1] - dt * (
            self.p[2:-1, 1:-1] - self.p[1:-2, 1:-1]
        ) / dy

    def compute_mass_residual(self):
        """Compute mass conservation residual."""
        dx, dy = self.dx, self.dy
        div = torch.zeros((self.ny+2, self.nx+2), device=self.device, dtype=self.dtype)
        div[1:-1, 1:-1] = (
            (self.u[1:-1, 2:] - self.u[1:-1, 1:-1])/dx +
            (self.v[2:, 1:-1] - self.v[1:-1, 1:-1])/dy
        )
        return torch.norm(div).item()

    def step(self):
        """Perform one time step."""
        self.apply_boundary_conditions()

        tic = time.perf_counter()
        self.compute_momentum()
        self.momentum_time += time.perf_counter() - tic

        solver_info = self.solve_pressure()
        self.pressure_correction()

        mass_res = self.compute_mass_residual()
        return mass_res, solver_info

    def get_cell_centered_velocities(self):
        """Get cell-centered velocities for plotting."""
        ucc = 0.5 * (self.u[1:-1, 2:] + self.u[1:-1, 1:-1])
        vcc = 0.5 * (self.v[2:, 1:-1] + self.v[1:-1, 1:-1])
        speed = torch.sqrt(ucc**2 + vcc**2)
        return ucc.cpu().numpy(), vcc.cpu().numpy(), speed.cpu().numpy()

    def run(self, nsteps: int, print_interval: int = 100):
        """
        Run simulation for specified number of steps.

        Args:
            nsteps: Number of time steps
            print_interval: Print progress every N steps
        """
        print(f"\nRunning {nsteps} time steps...")
        start_time = time.perf_counter()

        for step in range(nsteps):
            mass_res, solver_info = self.step()

            if step % print_interval == 0 or step == nsteps - 1:
                ucc, vcc, speed = self.get_cell_centered_velocities()
                max_vel = np.max(speed)
                print(f"Step {step:5d}: Mass residual={mass_res:.2e}, "
                      f"Max velocity={max_vel:.4f}, Solver info={solver_info}")

        total_time = time.perf_counter() - start_time

        print(f"\n{'='*60}")
        print(f"Simulation Complete")
        print(f"{'='*60}")
        print(f"Total steps: {nsteps}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Momentum time: {self.momentum_time:.3f}s ({self.momentum_time/total_time*100:.1f}%)")
        print(f"Solver time: {self.solver_time:.3f}s ({self.solver_time/total_time*100:.1f}%)")
        print(f"Avg solver time: {self.solver_time/nsteps*1000:.2f} ms/step")
        print(f"{'='*60}")

    def plot_results(self, save_path: str = None, show: bool = False):
        """
        Plot velocity magnitude and streamlines side by side.

        Args:
            save_path: Path to save the figure (optional)
            show: Whether to display the plot interactively
        """
        ucc, vcc, speed = self.get_cell_centered_velocities()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Lid-Driven Cavity Flow (Re={self.Re:.0f}, Grid={self.nx}x{self.ny}, Solver={self.method.upper()})',
                     fontsize=14)

        # Left: Velocity magnitude contour
        levels = np.linspace(speed.min(), speed.max(), 50)
        im = axes[0].contourf(self.xcc, self.ycc, speed, levels=levels, cmap='RdBu_r')
        plt.colorbar(im, ax=axes[0], label='Velocity Magnitude')
        axes[0].set_xlabel('x', fontsize=12)
        axes[0].set_ylabel('y', fontsize=12)
        axes[0].set_title('Velocity Magnitude', fontsize=12)
        axes[0].set_aspect('equal')

        # Right: Streamlines
        x = np.linspace(0, self.lx, self.nx)
        y = np.linspace(0, self.ly, self.ny)
        xx, yy = np.meshgrid(x, y)
        strm = axes[1].streamplot(xx, yy, ucc, vcc,
                                   color=speed, density=1.5,
                                   cmap='autumn', linewidth=1.5)
        plt.colorbar(strm.lines, ax=axes[1], label='Speed')
        axes[1].set_xlim([0, self.lx])
        axes[1].set_ylim([0, self.ly])
        axes[1].set_xlabel('x', fontsize=12)
        axes[1].set_ylabel('y', fontsize=12)
        axes[1].set_title('Streamlines', fontsize=12)
        axes[1].set_aspect('equal')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Lid-Driven Cavity Flow Solver using pytorch_sparse_solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ldc_solver.py                              # Default: Re=400, 100x100, 1000 steps, BiCGStab
  python ldc_solver.py --method gmres               # Use GMRES instead of BiCGStab
  python ldc_solver.py --nx 64 --Re 100             # Re=100, 64x64 grid
  python ldc_solver.py --quick                      # Quick test: 32x32, 200 steps
  python ldc_solver.py --save-dir ./results         # Save to specific directory
        """
    )

    parser.add_argument('--nx', type=int, default=100,
                        help='Grid resolution (default: 100)')
    parser.add_argument('--Re', type=float, default=400.0,
                        help='Reynolds number (default: 400)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of time steps (default: 1000)')
    parser.add_argument('--method', type=str, default='bicgstab',
                        choices=['bicgstab', 'gmres'],
                        help='Solver method (default: bicgstab)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with smaller grid (32x32, 200 steps)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')

    args = parser.parse_args()

    # Quick test mode
    if args.quick:
        args.nx = 32
        args.steps = 200
        args.Re = 100.0

    # Create solver and run
    solver = LDCSolver(nx=args.nx, Re=args.Re, method=args.method)
    solver.run(nsteps=args.steps, print_interval=max(1, args.steps // 10))

    # Generate save path
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(__file__).parent.parent / "torch_results" / f"Re{args.Re:.0f}_nx{args.nx}_{timestamp}"

    save_path = save_dir / f"ldc_Re{args.Re:.0f}_nx{args.nx}_{args.method}_steps{args.steps}.png"

    # Plot and save results
    if not args.no_plot:
        solver.plot_results(save_path=str(save_path), show=False)
        print(f"\nTo view the results, open: {save_path}")
    else:
        solver.plot_results(save_path=str(save_path), show=False)


if __name__ == '__main__':
    main()
