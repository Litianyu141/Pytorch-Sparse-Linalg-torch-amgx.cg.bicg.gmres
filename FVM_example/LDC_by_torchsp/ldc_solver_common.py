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
Shared Lid-Driven Cavity (LDC) FVM implementation for backend-specific solver variants.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


class BaseLDCSolver:
    """Shared FVM / fractional-step implementation with pluggable pressure solves."""

    def __init__(
        self,
        nx: int = 100,
        Re: float = 400.0,
        method: str = "bicgstab",
        device: str = None,
        dtype: torch.dtype = torch.float64,
        solver_label: str = "Unknown",
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.dtype = dtype
        self.method = method.lower()
        self.solver_label = solver_label

        self.nx = self.ny = nx
        self.lx = self.ly = 1.0
        self.dx = self.lx / nx
        self.dy = self.ly / nx

        self.Ut = 1.0
        self.miu = self.Ut * self.lx / Re
        self.Re = Re

        dt_diffusion = 0.25 * self.dx**2 / self.miu
        dt_convection = 4.0 * self.miu / self.Ut**2
        self.dt = min(dt_diffusion, dt_convection)

        self._init_fields()
        self._setup_pressure_matrix()

        self.momentum_time = 0.0
        self.solver_time = 0.0
        self.last_solver_info = None

        print("LDC Solver Initialized:")
        print(f"  Grid: {nx}x{nx}")
        print(f"  Reynolds Number: {Re}")
        print(f"  Device: {self.device}")
        print(f"  Backend: {self.solver_label}")
        print(f"  Method: {self.method.upper()}")
        print(f"  dt: {self.dt:.6f}")

    def _init_fields(self):
        ny, nx = self.ny, self.nx
        self.u = torch.zeros((ny + 2, nx + 2), device=self.device, dtype=self.dtype)
        self.v = torch.zeros((ny + 2, nx + 2), device=self.device, dtype=self.dtype)
        self.ut = torch.zeros_like(self.u)
        self.vt = torch.zeros_like(self.v)
        self.p = torch.zeros((ny + 2, nx + 2), device=self.device, dtype=self.dtype)

        xx = np.linspace(self.dx / 2, self.lx - self.dx / 2, nx)
        yy = np.linspace(self.dy / 2, self.ly - self.dy / 2, ny)
        self.xcc, self.ycc = np.meshgrid(xx, yy)

    def _setup_pressure_matrix(self):
        nx, ny = self.nx, self.ny
        dx2, dy2 = self.dx**2, self.dy**2

        Ap = np.zeros((ny, nx))
        Ae = (1.0 / dx2) * np.ones((ny, nx))
        Aw = (1.0 / dx2) * np.ones((ny, nx))
        An = (1.0 / dy2) * np.ones((ny, nx))
        As = (1.0 / dy2) * np.ones((ny, nx))

        Aw[:, 0] = 0.0
        Ae[:, -1] = 0.0
        An[-1, :] = 0.0
        As[0, :] = 0.0
        Ap = -(Aw + Ae + An + As)

        n = nx * ny
        d0 = Ap.reshape(n)
        de = Ae.reshape(n)[:-1]
        dw = Aw.reshape(n)[1:]
        dn = An.reshape(n)[:-nx]
        ds = As.reshape(n)[nx:]

        indices_list = []
        values_list = []
        for i in range(n):
            indices_list.append([i, i])
            values_list.append(d0[i])
            if i < n - 1 and (i + 1) % nx != 0:
                indices_list.append([i, i + 1])
                values_list.append(de[i])
            if i > 0 and i % nx != 0:
                indices_list.append([i, i - 1])
                values_list.append(dw[i - 1])
            if i < n - nx:
                indices_list.append([i, i + nx])
                values_list.append(dn[i])
            if i >= nx:
                indices_list.append([i, i - nx])
                values_list.append(ds[i - nx])

        indices = torch.tensor(indices_list, dtype=torch.long, device=self.device).t()
        values = torch.tensor(values_list, dtype=self.dtype, device=self.device)
        self.A_sparse = torch.sparse_coo_tensor(indices, values, (n, n), device=self.device, dtype=self.dtype).coalesce()
        self.A_dense = self.A_sparse.to_dense()
        self.A_csr = self.A_sparse.to_sparse_csr()

    def apply_boundary_conditions(self):
        Ut, Ub = self.Ut, 0.0
        Vl, Vr = 0.0, 0.0

        self.u[1:-1, 1] = 0.0
        self.u[1:-1, -1] = 0.0
        self.u[-1, 1:] = 2.0 * Ut - self.u[-2, 1:]
        self.u[0, 1:] = 2.0 * Ub - self.u[1, 1:]

        self.v[1:, 0] = 2.0 * Vl - self.v[1:, 1]
        self.v[1:, -1] = 2.0 * Vr - self.v[1:, -2]
        self.v[1, 1:-1] = 0.0
        self.v[-1, 1:-1] = 0.0

    def compute_momentum(self):
        dx, dy, dt, miu = self.dx, self.dy, self.dt, self.miu

        ue = 0.5 * (self.u[1:-1, 3:] + self.u[1:-1, 2:-1])
        uw = 0.5 * (self.u[1:-1, 2:-1] + self.u[1:-1, 1:-2])
        un = 0.5 * (self.u[2:, 2:-1] + self.u[1:-1, 2:-1])
        us = 0.5 * (self.u[1:-1, 2:-1] + self.u[:-2, 2:-1])
        vn = 0.5 * (self.v[2:, 2:-1] + self.v[2:, 1:-2])
        vs = 0.5 * (self.v[1:-1, 2:-1] + self.v[1:-1, 1:-2])

        convection_u = -(ue * ue - uw * uw) / dx - (un * vn - us * vs) / dy
        diffusion_u = miu * (
            (self.u[1:-1, 3:] - 2 * self.u[1:-1, 2:-1] + self.u[1:-1, 1:-2]) / dx**2 +
            (self.u[2:, 2:-1] - 2 * self.u[1:-1, 2:-1] + self.u[:-2, 2:-1]) / dy**2
        )
        self.ut[1:-1, 2:-1] = self.u[1:-1, 2:-1] + dt * (convection_u + diffusion_u)

        ve = 0.5 * (self.v[2:-1, 2:] + self.v[2:-1, 1:-1])
        vw = 0.5 * (self.v[2:-1, 1:-1] + self.v[2:-1, :-2])
        ue_v = 0.5 * (self.u[2:-1, 2:] + self.u[1:-2, 2:])
        uw_v = 0.5 * (self.u[2:-1, 1:-1] + self.u[1:-2, 1:-1])
        vn = 0.5 * (self.v[3:, 1:-1] + self.v[2:-1, 1:-1])
        vs = 0.5 * (self.v[2:-1, 1:-1] + self.v[1:-2, 1:-1])

        convection_v = -(ue_v * ve - uw_v * vw) / dx - (vn * vn - vs * vs) / dy
        diffusion_v = miu * (
            (self.v[2:-1, 2:] - 2 * self.v[2:-1, 1:-1] + self.v[2:-1, :-2]) / dx**2 +
            (self.v[3:, 1:-1] - 2 * self.v[2:-1, 1:-1] + self.v[1:-2, 1:-1]) / dy**2
        )
        self.vt[2:-1, 1:-1] = self.v[2:-1, 1:-1] + dt * (convection_v + diffusion_v)

    def _solve_linear_system(self, prhs: torch.Tensor):
        raise NotImplementedError

    def solve_pressure(self):
        dx, dy, dt = self.dx, self.dy, self.dt
        div_ut = torch.zeros_like(self.p)
        div_ut[1:-1, 1:-1] = (
            (self.ut[1:-1, 2:] - self.ut[1:-1, 1:-1]) / dx +
            (self.vt[2:, 1:-1] - self.vt[1:-1, 1:-1]) / dy
        )
        prhs = (1.0 / dt) * div_ut[1:-1, 1:-1].reshape(-1)

        tic = time.perf_counter()
        pt, solver_info = self._solve_linear_system(prhs)
        self.solver_time += time.perf_counter() - tic
        self.last_solver_info = solver_info

        self.p.fill_(0.0)
        self.p[1:-1, 1:-1] = pt.reshape(self.ny, self.nx)
        return solver_info

    def pressure_correction(self):
        dx, dy, dt = self.dx, self.dy, self.dt
        self.u[1:-1, 2:-1] = self.ut[1:-1, 2:-1] - dt * (self.p[1:-1, 2:-1] - self.p[1:-1, 1:-2]) / dx
        self.v[2:-1, 1:-1] = self.vt[2:-1, 1:-1] - dt * (self.p[2:-1, 1:-1] - self.p[1:-2, 1:-1]) / dy

    def compute_mass_residual(self):
        dx, dy = self.dx, self.dy
        div = torch.zeros((self.ny + 2, self.nx + 2), device=self.device, dtype=self.dtype)
        div[1:-1, 1:-1] = (
            (self.u[1:-1, 2:] - self.u[1:-1, 1:-1]) / dx +
            (self.v[2:, 1:-1] - self.v[1:-1, 1:-1]) / dy
        )
        return torch.norm(div).item()

    def step(self):
        self.apply_boundary_conditions()
        tic = time.perf_counter()
        self.compute_momentum()
        self.momentum_time += time.perf_counter() - tic
        solver_info = self.solve_pressure()
        self.pressure_correction()
        return self.compute_mass_residual(), solver_info

    def get_cell_centered_velocities(self):
        ucc = 0.5 * (self.u[1:-1, 2:] + self.u[1:-1, 1:-1])
        vcc = 0.5 * (self.v[2:, 1:-1] + self.v[1:-1, 1:-1])
        speed = torch.sqrt(ucc**2 + vcc**2)
        return ucc.cpu().numpy(), vcc.cpu().numpy(), speed.cpu().numpy()

    def run(self, nsteps: int, print_interval: int = 100):
        print(f"\nRunning {nsteps} time steps...")
        start_time = time.perf_counter()
        for step in range(nsteps):
            mass_res, solver_info = self.step()
            if step % print_interval == 0 or step == nsteps - 1:
                _, _, speed = self.get_cell_centered_velocities()
                print(
                    f"Step {step:5d}: Mass residual={mass_res:.2e}, "
                    f"Max velocity={np.max(speed):.4f}, Solver info={solver_info}"
                )

        total_time = time.perf_counter() - start_time
        print(f"\n{'=' * 60}")
        print("Simulation Complete")
        print(f"{'=' * 60}")
        print(f"Total steps: {nsteps}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Momentum time: {self.momentum_time:.3f}s ({self.momentum_time / total_time * 100:.1f}%)")
        print(f"Solver time: {self.solver_time:.3f}s ({self.solver_time / total_time * 100:.1f}%)")
        print(f"Avg solver time: {self.solver_time / nsteps * 1000:.2f} ms/step")
        print(f"{'=' * 60}")

    def plot_results(self, save_path: str = None, show: bool = False):
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        ucc, vcc, speed = self.get_cell_centered_velocities()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"Lid-Driven Cavity Flow (Re={self.Re:.0f}, Grid={self.nx}x{self.ny}, "
            f"Backend={self.solver_label}, Method={self.method.upper()})",
            fontsize=14,
        )

        levels = np.linspace(speed.min(), speed.max(), 50)
        im = axes[0].contourf(self.xcc, self.ycc, speed, levels=levels, cmap="RdBu_r")
        plt.colorbar(im, ax=axes[0], label="Velocity Magnitude")
        axes[0].set_xlabel("x", fontsize=12)
        axes[0].set_ylabel("y", fontsize=12)
        axes[0].set_title("Velocity Magnitude", fontsize=12)
        axes[0].set_aspect("equal")

        x = np.linspace(0, self.lx, self.nx)
        y = np.linspace(0, self.ly, self.ny)
        xx, yy = np.meshgrid(x, y)
        strm = axes[1].streamplot(xx, yy, ucc, vcc, color=speed, density=1.5, cmap="autumn", linewidth=1.5)
        plt.colorbar(strm.lines, ax=axes[1], label="Speed")
        axes[1].set_xlim([0, self.lx])
        axes[1].set_ylim([0, self.ly])
        axes[1].set_xlabel("x", fontsize=12)
        axes[1].set_ylabel("y", fontsize=12)
        axes[1].set_title("Streamlines", fontsize=12)
        axes[1].set_aspect("equal")

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()


def build_parser(description: str, method_choices: Sequence[str], default_method: str):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nx", type=int, default=100, help="Grid resolution (default: 100)")
    parser.add_argument("--Re", type=float, default=400.0, help="Reynolds number (default: 400)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of time steps (default: 1000)")
    parser.add_argument("--method", type=str, default=default_method, choices=list(method_choices), help="Pressure solver method")
    parser.add_argument("--quick", action="store_true", help="Quick test with smaller grid (32x32, 200 steps)")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    return parser


def run_solver_cli(
    solver_cls,
    description: str,
    method_choices: Sequence[str],
    default_method: str,
):
    parser = build_parser(description, method_choices, default_method)
    args = parser.parse_args()

    if args.quick:
        args.nx = 32
        args.steps = 200
        args.Re = 100.0

    solver = solver_cls(nx=args.nx, Re=args.Re, method=args.method)
    solver.run(nsteps=args.steps, print_interval=max(1, args.steps // 10))

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(__file__).parent.parent / "torch_results" / f"{solver.solver_label.lower()}_Re{args.Re:.0f}_nx{args.nx}_{timestamp}"

    save_path = save_dir / f"ldc_Re{args.Re:.0f}_nx{args.nx}_{solver.solver_label.lower()}_{args.method}_steps{args.steps}.png"
    if not args.no_plot:
        solver.plot_results(save_path=str(save_path), show=False)
        print(f"\nTo view the results, open: {save_path}")
