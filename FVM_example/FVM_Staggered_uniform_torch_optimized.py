#!/usr/bin/env python3
"""
Optimized Lid-Driven Cavity Flow Solver using Finite Volume Method (FVM)
with Staggered Uniform Grid in PyTorch

Optimizations:
1. Modular code structure with separate functions
2. Fully vectorized operations (no for loops)
3. Three solver options: Direct, BiCGStab, GMRES (with batched option)
4. Improved memory management and computational efficiency
5. Better error handling and convergence monitoring
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import time
import numpy as np
from src.torch_sparse_linalg import gmres, bicgstab
from typing import Tuple, Dict, Any, Optional
from enum import Enum

class SolverType(Enum):
    """Enumeration for different pressure solver types"""
    DIRECT = "direct"
    BICGSTAB = "bicgstab" 
    GMRES = "gmres"
    GMRES_BATCHED = "gmres_batched"

class FVMSolver:
    """
    Finite Volume Method solver for lid-driven cavity flow
    using staggered uniform grid and PyTorch tensors
    """
    
    def __init__(self, 
                 nx: int = 100, 
                 ny: int = 100,
                 lx: float = 1.0,
                 ly: float = 1.0,
                 miu: float = 0.01,
                 Ut: float = 1.0,
                 solver_type: SolverType = SolverType.GMRES,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float64):
        """
        Initialize the FVM solver
        
        Args:
            nx, ny: Grid resolution in x and y directions
            lx, ly: Domain length in x and y directions
            miu: Dynamic viscosity
            Ut: Top wall velocity
            solver_type: Type of pressure solver to use
            device: PyTorch device (auto-detect if None)
            dtype: Floating point precision
        """
        
        # Set device and precision
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        torch.set_default_dtype(dtype)
        
        print(f"Using device: {self.device}")
        print(f"Using precision: {dtype}")
        
        # Grid parameters - ensure all scalars use correct dtype
        self.nx, self.ny = nx, ny
        self.lx, self.ly = float(lx), float(ly)
        self.dx = torch.tensor(lx / nx, dtype=dtype, device=self.device)
        self.dy = torch.tensor(ly / ny, dtype=dtype, device=self.device)
        self.miu = torch.tensor(miu, dtype=dtype, device=self.device)
        self.solver_type = solver_type
        
        # Boundary conditions
        self.Ut = torch.tensor(Ut, dtype=dtype, device=self.device)  # Top wall velocity
        self.Ub = torch.tensor(0.0, dtype=dtype, device=self.device)  # Bottom wall velocity
        self.Vl = torch.tensor(0.0, dtype=dtype, device=self.device)  # Left wall velocity
        self.Vr = torch.tensor(0.0, dtype=dtype, device=self.device)  # Right wall velocity
        
        print(f'Reynolds Number: {(Ut * lx / miu):.2f}')
        
        # Time step based on CFL condition
        dt_diffusion = 0.25 * (lx/nx)**2 / miu
        dt_convection = 4.0 * miu / Ut**2
        self.dt = torch.tensor(min(dt_diffusion, dt_convection), dtype=dtype, device=self.device)
        print(f'dt = {self.dt.item():.6f}')
        
        # Initialize grids and fields
        self._setup_grids()
        self._initialize_fields()
        self._setup_pressure_matrix()
        
        # Solver configuration
        self._setup_solver_params()
        
    def _setup_grids(self):
        """Setup staggered grid coordinates"""
        # Cell center coordinates
        xx = torch.linspace(self.dx/2.0, self.lx - self.dx/2.0, self.nx, device=self.device)
        yy = torch.linspace(self.dy/2.0, self.ly - self.dy/2.0, self.ny, device=self.device)
        self.xcc, self.ycc = torch.meshgrid(xx, yy, indexing='ij')
        self.xcc = self.xcc.t()  # Transpose for consistency
        self.ycc = self.ycc.t()
        
        # x-direction staggered grid (u-velocity points)
        xxs = torch.linspace(0, self.lx, self.nx+1, device=self.device)
        self.xu, self.yu = torch.meshgrid(xxs, yy, indexing='ij')
        self.xu = self.xu.t()
        self.yu = self.yu.t()
        
        # y-direction staggered grid (v-velocity points)
        yys = torch.linspace(0, self.ly, self.ny+1, device=self.device)
        self.xv, self.yv = torch.meshgrid(xx, yys, indexing='ij')
        self.xv = self.xv.t()
        self.yv = self.yv.t()
        
    def _initialize_fields(self):
        """Initialize velocity and pressure fields with ghost cells"""
        # Velocity fields with ghost cells
        self.u = torch.ones((self.ny+2, self.nx+2), device=self.device, dtype=self.dtype)
        self.v = torch.zeros((self.ny+2, self.nx+2), device=self.device, dtype=self.dtype)
        self.ut = torch.zeros_like(self.u)  # u_tilde (predictor)
        self.vt = torch.zeros_like(self.v)  # v_tilde (predictor)
        
        # Pressure field with ghost cells
        self.p = torch.zeros((self.ny+2, self.nx+2), device=self.device, dtype=self.dtype)
        
    def _setup_pressure_matrix(self):
        """Setup the pressure Poisson equation coefficient matrix"""
        # Coefficient matrices for pressure equation with explicit dtype
        dx_sq = self.dx * self.dx  # Keep as tensor
        dy_sq = self.dy * self.dy  # Keep as tensor
        
        Ae = (torch.tensor(1.0, dtype=self.dtype, device=self.device)/dx_sq) * torch.ones((self.ny, self.nx), device=self.device, dtype=self.dtype)
        Aw = (torch.tensor(1.0, dtype=self.dtype, device=self.device)/dx_sq) * torch.ones((self.ny, self.nx), device=self.device, dtype=self.dtype)
        An = (torch.tensor(1.0, dtype=self.dtype, device=self.device)/dy_sq) * torch.ones((self.ny, self.nx), device=self.device, dtype=self.dtype)
        As = (torch.tensor(1.0, dtype=self.dtype, device=self.device)/dy_sq) * torch.ones((self.ny, self.nx), device=self.device, dtype=self.dtype)
        
        # Apply boundary conditions (no-slip walls)
        Aw[:, 0] = 0.0    # Left wall
        Ae[:, -1] = 0.0   # Right wall
        An[-1, :] = 0.0   # Top wall
        As[0, :] = 0.0    # Bottom wall
        
        # Central coefficient
        Ap = -(Aw + Ae + An + As)
        
        # Create sparse matrix in COO format
        self._create_sparse_pressure_matrix(Ap, Ae, Aw, An, As)
        
    def _create_sparse_pressure_matrix(self, Ap, Ae, Aw, An, As):
        """Create sparse pressure matrix in COO format for efficient solving"""
        n = self.nx * self.ny
        
        # Flatten coefficient matrices with explicit dtype
        d0 = Ap.reshape(n).to(dtype=self.dtype)
        de = Ae.reshape(n).to(dtype=self.dtype)
        dw = Aw.reshape(n).to(dtype=self.dtype)
        dn = An.reshape(n).to(dtype=self.dtype)
        ds = As.reshape(n).to(dtype=self.dtype)
        
        # Build sparse matrix indices and values
        indices_list = []
        values_list = []
        
        # Main diagonal
        i_main = torch.arange(n, device=self.device, dtype=torch.long)
        indices_list.append(torch.stack([i_main, i_main]))
        values_list.append(d0)
        
        # East neighbors (i, i+1) - exclude right boundary points
        mask_east = ((torch.arange(n, device=self.device, dtype=torch.long) + 1) % self.nx != 0)
        i_east = torch.arange(n, device=self.device, dtype=torch.long)[mask_east]
        j_east = i_east + 1
        indices_list.extend([
            torch.stack([i_east, j_east]),  # (i, i+1)
            torch.stack([j_east, i_east])   # (i+1, i) - symmetric
        ])
        values_list.extend([de[mask_east], dw[j_east]])
        
        # North neighbors (i, i+nx) - exclude top boundary points
        i_north = torch.arange(n - self.nx, device=self.device, dtype=torch.long)
        j_north = i_north + self.nx
        indices_list.extend([
            torch.stack([i_north, j_north]),  # (i, i+nx)
            torch.stack([j_north, i_north])   # (i+nx, i) - symmetric
        ])
        values_list.extend([dn[i_north], ds[j_north]])
        
        # Concatenate all indices and values with consistent dtypes
        self.A_indices = torch.cat(indices_list, dim=1)
        self.A_values = torch.cat(values_list, dim=0).to(dtype=self.dtype)
        self.A_sparse = torch.sparse_coo_tensor(
            self.A_indices, self.A_values, (n, n), device=self.device, dtype=self.dtype
        ).coalesce()  # Optimize sparse matrix
        
    def _setup_solver_params(self):
        """Setup parameters for different solvers"""
        self.solver_params = {
            SolverType.DIRECT: {},
            SolverType.BICGSTAB: {
                'tol': 1e-10,
                'maxiter': 1000,
                'atol': 1e-12
            },
            SolverType.GMRES: {
                'tol': 1e-10,
                'maxiter': 1000,
                'restart': 20,
                'atol': 1e-12,
                'solve_method': 'incremental'
            },
            SolverType.GMRES_BATCHED: {
                'tol': 1e-10,
                'maxiter': 1000,
                'restart': 20,
                'atol': 1e-12,
                'solve_method': 'batched'
            }
        }
        
    def apply_boundary_conditions(self):
        """Apply boundary conditions to velocity fields (vectorized)"""
        # u-velocity boundary conditions
        self.u[1:-1, 1] = 0.0              # Left wall
        self.u[1:-1, -1] = 0.0             # Right wall
        self.u[-1, 1:] = 2.0*self.Ut - self.u[-2, 1:]  # Top wall (moving)
        self.u[0, 1:] = 2.0*self.Ub - self.u[1, 1:]    # Bottom wall
        
        # v-velocity boundary conditions
        self.v[1:, 0] = 2.0*self.Vl - self.v[1:, 1]     # Left wall
        self.v[1:, -1] = 2.0*self.Vr - self.v[1:, -2]   # Right wall
        self.v[1, 1:-1] = 0.0              # Bottom wall
        self.v[-1, 1:-1] = 0.0             # Top wall
        
    def compute_momentum_predictor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute momentum predictor step (fully vectorized)
        Returns convection and diffusion timing
        """
        tic = time.perf_counter()
        
        # U-momentum equation
        self._compute_u_momentum()
        
        # V-momentum equation  
        self._compute_v_momentum()
        
        toc = time.perf_counter()
        return toc - tic, 0.0  # Return timing info
        
    def _compute_u_momentum(self):
        """Compute u-momentum equation (vectorized)"""
        # Interpolate velocities to face centers
        ue = 0.5 * (self.u[1:-1, 3:] + self.u[1:-1, 2:-1])
        uw = 0.5 * (self.u[1:-1, 2:-1] + self.u[1:-1, 1:-2])
        un = 0.5 * (self.u[2:, 2:-1] + self.u[1:-1, 2:-1])
        us = 0.5 * (self.u[1:-1, 2:-1] + self.u[0:-2, 2:-1])
        vn = 0.5 * (self.v[2:, 2:-1] + self.v[2:, 1:-2])
        vs = 0.5 * (self.v[1:-1, 2:-1] + self.v[1:-1, 1:-2])
        
        # Convection terms (conservative form)
        convection = -(ue*ue - uw*uw)/self.dx - (un*vn - us*vs)/self.dy
        
        # Diffusion terms (central difference)
        diffusion = self.miu * (
            (self.u[1:-1, 3:] - 2.0*self.u[1:-1, 2:-1] + self.u[1:-1, 1:-2])/self.dx**2 + 
            (self.u[2:, 2:-1] - 2.0*self.u[1:-1, 2:-1] + self.u[0:-2, 2:-1])/self.dy**2
        )
        
        # Update predictor
        self.ut[1:-1, 2:-1] = self.u[1:-1, 2:-1] + self.dt * (convection + diffusion)
        
    def _compute_v_momentum(self):
        """Compute v-momentum equation (vectorized)"""
        # Interpolate velocities to face centers
        ve = 0.5 * (self.v[2:-1, 2:] + self.v[2:-1, 1:-1])
        vw = 0.5 * (self.v[2:-1, 1:-1] + self.v[2:-1, :-2])
        ue = 0.5 * (self.u[2:-1, 2:] + self.u[1:-2, 2:])
        uw = 0.5 * (self.u[2:-1, 1:-1] + self.u[1:-2, 1:-1])
        vn = 0.5 * (self.v[3:, 1:-1] + self.v[2:-1, 1:-1])
        vs = 0.5 * (self.v[2:-1, 1:-1] + self.v[1:-2, 1:-1])
        
        # Convection terms (conservative form)
        convection = -(ue*ve - uw*vw)/self.dx - (vn*vn - vs*vs)/self.dy
        
        # Diffusion terms (central difference)
        diffusion = self.miu * (
            (self.v[2:-1, 2:] - 2.0*self.v[2:-1, 1:-1] + self.v[2:-1, :-2])/self.dx**2 + 
            (self.v[3:, 1:-1] - 2.0*self.v[2:-1, 1:-1] + self.v[1:-2, 1:-1])/self.dy**2
        )
        
        # Update predictor
        self.vt[2:-1, 1:-1] = self.v[2:-1, 1:-1] + self.dt * (convection + diffusion)
        
    def solve_pressure_poisson(self) -> Tuple[float, str]:
        """
        Solve pressure Poisson equation with selected solver
        Returns (solve_time, status_message)
        """
        tic = time.perf_counter()
        
        # Compute divergence of predictor velocity (RHS)
        div_ut = torch.zeros((self.ny+2, self.nx+2), device=self.device, dtype=self.dtype)
        div_ut[1:-1, 1:-1] = (
            (self.ut[1:-1, 2:] - self.ut[1:-1, 1:-1])/self.dx + 
            (self.vt[2:, 1:-1] - self.vt[1:-1, 1:-1])/self.dy
        )
        
        # RHS of pressure equation - ensure correct dtype from the start
        prhs_2d = (torch.tensor(1.0, device=self.device, dtype=self.dtype) / self.dt) * div_ut[1:-1, 1:-1]
        prhs = prhs_2d.reshape(-1)
        
        # Ensure data types are consistent (debugging disabled)
        
        # Solve pressure system
        pt, status = self._solve_linear_system(prhs)
        
        # Reconstruct pressure field
        self.p.fill_(0.0)  # Clear pressure field
        self.p[1:-1, 1:-1] = pt.reshape(self.ny, self.nx)
        
        toc = time.perf_counter()
        return toc - tic, status
        
    def _solve_linear_system(self, rhs: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Solve linear system with selected solver method
        """
        if self.solver_type == SolverType.DIRECT:
            return self._solve_direct(rhs)
        elif self.solver_type == SolverType.BICGSTAB:
            return self._solve_bicgstab(rhs)
        elif self.solver_type in [SolverType.GMRES, SolverType.GMRES_BATCHED]:
            return self._solve_gmres(rhs)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
            
    def _solve_direct(self, rhs: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Solve using direct method (torch.linalg.solve)"""
        try:
            A_dense = self.A_sparse.to_dense()
            solution = torch.linalg.solve(A_dense, rhs)
            return solution, "Direct solver - Success"
        except RuntimeError as e:
            # Fallback to iterative solver
            print(f"Direct solver failed: {e}")
            print("Falling back to GMRES...")
            return self._solve_gmres(rhs)
            
    def _solve_bicgstab(self, rhs: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Solve using BiCGStab iterative solver"""
        params = self.solver_params[SolverType.BICGSTAB]
        try:
            solution, info = bicgstab(
                self.A_sparse,
                rhs,
                x0=None,
                tol=params['tol'],
                maxiter=params['maxiter'],
                atol=params['atol']
            )
            
            if info == 0:
                status = f"BiCGStab - Converged (tol={params['tol']:.2e})"
            else:
                status = f"BiCGStab - Warning: info={info}"
                
            return solution, status
            
        except Exception as e:
            print(f"BiCGStab failed: {e}")
            return self._solve_gmres(rhs)
            
    def _solve_gmres(self, rhs: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Solve using GMRES iterative solver"""
        solver_key = self.solver_type if self.solver_type in [SolverType.GMRES, SolverType.GMRES_BATCHED] else SolverType.GMRES
        params = self.solver_params[solver_key]
        
        # try:
        # Convert sparse matrix to dense to ensure proper dtype handling
        A_dense = self.A_sparse.to_dense().to(dtype=self.dtype)
        
        solution, info = gmres(
            A_dense,  # Use dense matrix with correct dtype
            rhs,
            x0=None,
            tol=params['tol'],
            maxiter=params['maxiter'],
            restart=params['restart'],
            atol=params['atol'],
            solve_method=params['solve_method']
        )
        
        method_name = "GMRES" if params['solve_method'] == 'incremental' else "GMRES-Batched"
        if info == 0:
            status = f"{method_name} - Converged (tol={params['tol']:.2e}, restart={params['restart']})"
        else:
            status = f"{method_name} - Warning: info={info}"
            
        return solution, status
            
        # except Exception as e:
        #     raise RuntimeError(f"All solvers failed. Last error: {e}")
            
    def pressure_correction(self):
        """Apply pressure correction to velocities (vectorized)"""
        # Correct u-velocity
        self.u[1:-1, 2:-1] = (
            self.ut[1:-1, 2:-1] - 
            self.dt * (self.p[1:-1, 2:-1] - self.p[1:-1, 1:-2])/self.dx
        )
        
        # Correct v-velocity
        self.v[2:-1, 1:-1] = (
            self.vt[2:-1, 1:-1] - 
            self.dt * (self.p[2:-1, 1:-1] - self.p[1:-2, 1:-1])/self.dy
        )
        
    def check_mass_conservation(self) -> float:
        """Check mass conservation (continuity equation residual)"""
        divunp1 = torch.zeros((self.ny+2, self.nx+2), device=self.device, dtype=self.dtype)
        divunp1[1:self.ny+1, 1:self.nx+1] = (
            (self.u[1:self.ny+1, 2:self.nx+2] - self.u[1:self.ny+1, 1:self.nx+1]) / self.dx +
            (self.v[2:self.ny+2, 1:self.nx+1] - self.v[1:self.ny+1, 1:self.nx+1]) / self.dy
        )
        return torch.norm(divunp1).item()
        
    def get_cell_centered_velocities(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get cell-centered velocities and speed magnitude"""
        ucc = 0.5 * (self.u[1:-1, 2:] + self.u[1:-1, 1:-1])
        vcc = 0.5 * (self.v[2:, 1:-1] + self.v[1:-1, 1:-1])
        speed = torch.sqrt(ucc**2 + vcc**2)
        return ucc, vcc, speed
        
    def time_step(self) -> Dict[str, Any]:
        """
        Perform one time step of the simulation
        Returns timing and status information
        """
        # Apply boundary conditions
        self.apply_boundary_conditions()
        
        # Momentum predictor step
        mom_time, _ = self.compute_momentum_predictor()
        
        # Solve pressure Poisson equation
        solver_time, solver_status = self.solve_pressure_poisson()
        
        # Pressure correction
        self.pressure_correction()
        
        # Check mass conservation
        mass_residual = self.check_mass_conservation()
        
        return {
            'momentum_time': mom_time,
            'solver_time': solver_time,
            'solver_status': solver_status,
            'mass_residual': mass_residual
        }


def run_simulation(nx: int = 100, 
                  nsteps: int = 100,
                  solver_type: SolverType = SolverType.GMRES,
                  plot_interval: int = 100,
                  save_results: bool = True):
    """
    Run the complete lid-driven cavity simulation
    
    Args:
        nx: Grid resolution (nx x nx grid)
        nsteps: Number of time steps
        solver_type: Pressure solver type
        plot_interval: Plotting interval
        save_results: Whether to save final results
    """
    
    print(f"\n=== Lid-Driven Cavity Flow Simulation ===")
    print(f"Grid: {nx}x{nx}")
    print(f"Time steps: {nsteps}")
    print(f"Solver: {solver_type.value}")
    print("=" * 50)
    
    # Initialize solver
    solver = FVMSolver(nx=nx, ny=nx, solver_type=solver_type)
    
    # Timing statistics
    total_time = time.perf_counter()
    momentum_time_total = 0.0
    solver_time_total = 0.0
    
    # Initialize plotting
    if plot_interval > 0:
        fig1 = plt.figure(figsize=[18, 8])
    
    # Main simulation loop
    for step in range(nsteps):
        # Perform time step
        step_info = solver.time_step()
        
        # Accumulate timing
        momentum_time_total += step_info['momentum_time']
        solver_time_total += step_info['solver_time']
        
        # Print progress
        if step % 10 == 0 or step == nsteps - 1:
            print(f"Step {step:4d}: "
                  f"Mass residual: {step_info['mass_residual']:.2e}, "
                  f"Solver: {step_info['solver_status']}")
        
        # Plotting
        if plot_interval > 0 and (step % plot_interval == 0 or step == nsteps - 1):
            plot_results(solver, step, fig1)
    
    # Final timing statistics
    total_time = time.perf_counter() - total_time
    
    print(f"\n=== Simulation Complete ===")
    print(f"Total momentum time: {momentum_time_total:.3f}s")
    print(f"Total solver time: {solver_time_total:.3f}s")
    print(f"Total simulation time: {total_time:.3f}s")
    print(f"Solver efficiency: {solver_time_total/total_time*100:.1f}% of total time")
    
    # Save results
    if save_results:
        save_path = f'FVM_example/results/Optimized_torch_{solver_type.value}_nx{nx}_steps{nsteps}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to: {save_path}")
    
    return solver


def plot_results(solver: FVMSolver, step: int, fig):
    """Plot velocity magnitude and streamlines"""
    ucc, vcc, speed = solver.get_cell_centered_velocities()
    
    # Convert to NumPy for plotting
    xx_np = solver.xcc.cpu().numpy()
    yy_np = solver.ycc.cpu().numpy()
    speed_np = speed.cpu().numpy()
    ucc_np = ucc.cpu().numpy()
    vcc_np = vcc.cpu().numpy()
    
    # Velocity magnitude contour
    plt.subplot(1, 2, 1)
    plt.cla()
    plt.contourf(xx_np, yy_np, speed_np, levels=20, cmap='RdBu_r')
    plt.colorbar(label='Velocity Magnitude')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(f'Velocity Magnitude (Step {step})', fontsize=14)
    plt.gca().set_aspect('equal')
    
    # Streamlines
    plt.subplot(1, 2, 2)
    plt.cla()
    x = torch.linspace(0, solver.lx, solver.nx).cpu().numpy()
    y = torch.linspace(0, solver.ly, solver.ny).cpu().numpy()
    xx_stream, yy_stream = np.meshgrid(x, y)
    
    plt.streamplot(xx_stream, yy_stream, ucc_np, vcc_np, 
                  color=speed_np, density=1.5, cmap='autumn', linewidth=1.5)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(f'Streamlines (Step {step})', fontsize=14)
    plt.gca().set_aspect('equal')
    plt.colorbar(label='Velocity Magnitude')
    
    plt.tight_layout()
    plt.pause(0.001)


def benchmark_solvers(nx: int = 100, nsteps: int = 50):
    """
    Benchmark different solver types
    """
    print(f"\n=== Solver Benchmark (nx={nx}, steps={nsteps}) ===")
    
    solvers_to_test = [
        SolverType.DIRECT,
        SolverType.BICGSTAB, 
        SolverType.GMRES,
        SolverType.GMRES_BATCHED
    ]
    
    results = {}
    
    for solver_type in solvers_to_test:
        print(f"\nTesting {solver_type.value}...")
        
        try:
            start_time = time.perf_counter()
            solver_instance = run_simulation(
                nx=nx, 
                nsteps=nsteps, 
                solver_type=solver_type,
                plot_interval=0,  # No plotting during benchmark
                save_results=False
            )
            end_time = time.perf_counter()
            
            results[solver_type.value] = {
                'total_time': end_time - start_time,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            results[solver_type.value] = {
                'total_time': float('inf'),
                'success': False,
                'error': str(e)
            }
            print(f"  ❌ Failed: {e}")
    
    # Print benchmark results
    print(f"\n=== Benchmark Results ===")
    for solver_name, result in results.items():
        if result['success']:
            print(f"{solver_name:15s}: {result['total_time']:7.3f}s ✅")
        else:
            print(f"{solver_name:15s}: Failed ❌ ({result['error'][:50]}...)")


''' Please refer to cavity_flow.py for the main function '''
# if __name__ == "__main__":
#     # Example usage
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Optimized FVM Lid-Driven Cavity Solver')
#     parser.add_argument('--nx', type=int, default=100, help='Grid resolution')
#     parser.add_argument('--steps', type=int, default=10, help='Number of time steps')
#     parser.add_argument('--solver', type=str, default='gmres', 
#                        choices=['direct', 'bicgstab', 'gmres', 'gmres_batched'],
#                        help='Pressure solver type')
#     parser.add_argument('--benchmark', action='store_true', help='Run solver benchmark')
#     parser.add_argument('--plot', type=int, default=10, help='Plot interval (0 to disable)')
    
#     args = parser.parse_args()
    
#     # Map string to enum
#     solver_map = {
#         'direct': SolverType.DIRECT,
#         'bicgstab': SolverType.BICGSTAB,
#         'gmres': SolverType.GMRES,
#         'gmres_batched': SolverType.GMRES_BATCHED
#     }
    
#     if args.benchmark:
#         benchmark_solvers(nx=args.nx, nsteps=args.steps)
#     else:
#         solver = run_simulation(
#             nx=args.nx,
#             nsteps=args.steps,
#             solver_type=solver_map[args.solver],
#             plot_interval=args.plot
#         )
