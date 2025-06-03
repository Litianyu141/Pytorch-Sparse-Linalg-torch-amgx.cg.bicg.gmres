#!/usr/bin/env python3
"""
Main script for Lid-Driven Cavity Flow Simulation
Complete Re=100 solver with configurable options

HOW TO USE:
1. Modify the parameters in the main() function (lines marked with # Configuration)
2. Choose solver type: BICGSTAB (default), GMRES, GMRES_BATCHED, or DIRECT
3. Set grid size (nx, ny), time steps (nsteps), and Reynolds number (Re)
4. Run: python main_cavity_flow.py

SOLVER OPTIONS:
- BICGSTAB: Best balance of speed and stability (default)
- GMRES: Good for difficult problems
- GMRES_BATCHED: GPU-optimized GMRES
- DIRECT: Most accurate but slowest for large grids
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from FVM_Staggered_uniform_torch_optimized import FVMSolver, SolverType
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

def create_output_directory(Re: float, nx: int, step: int) -> Path:
    """Create output directory structure: results/Re{Re}_nx{nx}/t{step:06d}/"""
    base_dir = Path(__file__).parent.parent / "FVM_example/torch_results"
    sim_dir = base_dir / f"Re{Re:.0f}_nx{nx}"
    step_dir = sim_dir / f"t{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir

def plot_results_at_step(solver: FVMSolver, step: int, Re: float, nx: int, 
                        save_plots: bool = True, show_plots: bool = False) -> str:
    """Plot results at a specific time step and save to organized directory"""
    # Get cell-centered velocities and pressure
    ucc, vcc, speed = solver.get_cell_centered_velocities()
    
    # Convert to numpy for plotting
    xx_np = solver.xcc.cpu().numpy()
    yy_np = solver.ycc.cpu().numpy()
    ucc_np = ucc.cpu().numpy()
    vcc_np = vcc.cpu().numpy()
    p_np = solver.p[1:-1, 1:-1].cpu().numpy()
    speed_np = speed.cpu().numpy()
    
    # Calculate physical time
    physical_time = step * solver.dt.item()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Lid-Driven Cavity Flow (Re={Re:.0f}, Grid={nx}x{nx}, t={physical_time:.4f})', fontsize=16)
    
    # 1. U-velocity contour
    levels = 20
    im1 = axes[0,0].contourf(xx_np, yy_np, ucc_np, levels=levels, cmap='RdBu_r')
    axes[0,0].set_title('U-velocity', fontsize=14)
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    axes[0,0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
    
    # 2. V-velocity contour
    im2 = axes[0,1].contourf(xx_np, yy_np, vcc_np, levels=levels, cmap='RdBu_r')
    axes[0,1].set_title('V-velocity', fontsize=14)
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    axes[0,1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
    
    # 3. Pressure contour
    im3 = axes[1,0].contourf(xx_np, yy_np, p_np, levels=levels, cmap='viridis')
    axes[1,0].set_title('Pressure', fontsize=14)
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    axes[1,0].set_aspect('equal')
    plt.colorbar(im3, ax=axes[1,0], shrink=0.8)
    
    # 4. Streamlines with speed magnitude
    # Create proper meshgrid for streamplot (needs strictly increasing coordinates)
    x_stream = np.linspace(0, 1, ucc_np.shape[1])
    y_stream = np.linspace(0, 1, ucc_np.shape[0])
    xx_stream, yy_stream = np.meshgrid(x_stream, y_stream)
    strm = axes[1,1].streamplot(xx_stream, yy_stream, ucc_np, vcc_np, 
                        color=speed_np, density=1.5, cmap='autumn', linewidth=1.5)
    axes[1,1].set_title('Streamlines (colored by speed)', fontsize=14)
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('y')
    axes[1,1].set_aspect('equal')
    plt.colorbar(strm.lines, ax=axes[1,1], shrink=0.8)
    
    plt.tight_layout()
    
    save_path = None
    if save_plots:
        # Create organized directory structure
        step_dir = create_output_directory(Re, nx, step)
        save_path = step_dir / f"cavity_flow_step_{step:06d}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show_plots:
        plt.show()
    else:
        plt.close(fig)  # Close figure to free memory if not showing
        
    return str(save_path) if save_path else "No file saved"

def benchmark_solvers(nx: int = 50, nsteps: int = 100):
    """Benchmark all available solvers"""
    print("\n=== Solver Benchmark ===")
    print(f"Grid: {nx}x{nx}, Steps: {nsteps}")
    
    solvers_to_test = [
        (SolverType.BICGSTAB, "BiCGStab"),
        (SolverType.GMRES, "GMRES"),
        (SolverType.GMRES_BATCHED, "GMRES Batched"),
        (SolverType.DIRECT, "Direct")
    ]
    
    results = {}
    
    for solver_type, name in solvers_to_test:
        print(f"\n  Testing {name}...")
        try:
            # Create solver
            solver = FVMSolver(
                nx=nx, ny=nx, 
                miu=0.01,  # Re = 100
                Ut=1.0,
                solver_type=solver_type,
                dtype=torch.float64
            )
            solver.dt = 0.01  # Fixed dt
            
            # Set stricter tolerances for iterative solvers
            if solver_type in [SolverType.BICGSTAB, SolverType.GMRES, SolverType.GMRES_BATCHED]:
                solver.solver_params[solver_type]['tol'] = 1e-12
                solver.solver_params[solver_type]['atol'] = 1e-14
                solver.solver_params[solver_type]['maxiter'] = 2000
            
            # Run simulation
            start_time = time.perf_counter()
            total_momentum_time = 0.0
            total_solver_time = 0.0
            
            for step in range(nsteps):
                step_info = solver.time_step()
                total_momentum_time += step_info['momentum_time']
                total_solver_time += step_info['solver_time']
            
            total_time = time.perf_counter() - start_time
            
            # Get final state
            mass_residual = solver.check_mass_conservation()
            
            results[name] = {
                'total_time': total_time,
                'momentum_time': total_momentum_time,
                'solver_time': total_solver_time,
                'mass_residual': mass_residual,
                'success': True
            }
            
            print(f"    ✅ Success: {total_time:.3f}s total, {total_solver_time:.3f}s solver")
            print(f"       Mass residual: {mass_residual:.2e}")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            results[name] = {
                'success': False,
                'error': str(e)
            }
    
    # Print summary
    print(f"\n=== Benchmark Summary ===")
    successful_solvers = {k: v for k, v in results.items() if v.get('success', False)}
    
    if successful_solvers:
        # Sort by total time
        sorted_solvers = sorted(successful_solvers.items(), key=lambda x: x[1]['total_time'])
        
        print("Performance ranking (fastest first):")
        for i, (name, result) in enumerate(sorted_solvers, 1):
            efficiency = result['solver_time'] / result['total_time'] * 100
            print(f"{i}. {name:15s}: {result['total_time']:7.3f}s total "
                  f"({result['solver_time']:6.3f}s solver, {efficiency:4.1f}%)")
    
    return results

def main():
    """Main function with configurable parameters"""
    print("=== Lid-Driven Cavity Flow Solver ===")
    print("Optimized FVM implementation with multiple pressure solvers")
    print()
    
    # ========== CONFIGURATION PARAMETERS - MODIFY AS NEEDED ==========
    
    # Grid parameters
    nx = 100                    # Grid points in x-direction
    ny = 100                    # Grid points in y-direction
    
    # Simulation parameters
    nsteps = 1000               # Number of time steps
    Re = 100.0                 # Reynolds number
    Ut = 1.0                   # Top wall velocity
    
    # Solver selection - Choose from:
    # SolverType.BICGSTAB, SolverType.GMRES, SolverType.GMRES_BATCHED, SolverType.DIRECT
    solver_type = SolverType.DIRECT  # Default: BiCGStab (best balance of speed and stability)
    
    # Solver tolerances (for iterative solvers only)
    solver_tolerance = 1e-12    # Residual tolerance
    solver_atol = 1e-14        # Absolute tolerance
    max_iterations = 2000      # Maximum iterations
    
    # Plotting parameters
    plot_by_step = 10          # Plot every N steps (0 = no intermediate plots, only final)
    save_plots = True          # Save plots to file
    show_plots = False         # Display plots on screen (set False for batch runs)
    
    # Optional settings
    run_benchmark = False      # Set to True to benchmark all solvers instead
    
    # ================================================================
    
    # Automatically calculate viscosity from Reynolds number
    miu = Ut * 1.0 / Re  # miu = Ut * lx / Re, with lx = 1.0
    
    print(f"Configuration:")
    print(f"  Grid: {nx}x{ny}")
    print(f"  Time steps: {nsteps}")
    print(f"  Reynolds number: {Re}")
    print(f"  Top wall velocity: {Ut}")
    print(f"  Viscosity: {miu:.6f}")
    print(f"  Solver: {solver_type.value}")
    print(f"  (dt will be calculated from CFL condition)")
    print()
    
    if run_benchmark:
        print("=== Running Solver Benchmark ===")
        benchmark_results = benchmark_solvers(nx=min(nx, 50), nsteps=min(nsteps, 100))
        return benchmark_results
    
    # Single solver simulation
    print(f"\n=== Running Simulation with {solver_type.value} ===")
    
    # Create solver (dt will be calculated automatically based on CFL)
    solver = FVMSolver(
        nx=nx, 
        ny=ny, 
        miu=miu,
        Ut=Ut,
        solver_type=solver_type,
        dtype=torch.float64  # Use double precision for better accuracy
    )
    
    print(f"Calculated dt from CFL condition: {solver.dt.item():.6f}")
    
    # Setup solver tolerances for iterative methods
    if solver_type in [SolverType.BICGSTAB, SolverType.GMRES, SolverType.GMRES_BATCHED]:
        # Set strict convergence criteria
        solver.solver_params[solver_type]['tol'] = solver_tolerance
        solver.solver_params[solver_type]['atol'] = solver_atol
        if 'maxiter' in solver.solver_params[solver_type]:
            solver.solver_params[solver_type]['maxiter'] = max_iterations
        print(f"Solver tolerance: {solver.solver_params[solver_type]['tol']:.2e}")
        print(f"Max iterations: {solver.solver_params[solver_type]['maxiter']}")
    
    print("=" * 50)
    
    # Run simulation
    start_time = time.perf_counter()
    total_momentum_time = 0.0
    total_solver_time = 0.0
    
    # Progress reporting
    report_interval = max(1, plot_by_step)  # Report every 'plot_by_step' steps

    print("Starting simulation...")
    for step in range(nsteps):
        step_info = solver.time_step()
        total_momentum_time += step_info['momentum_time']
        total_solver_time += step_info['solver_time']
        
        # Progress reporting
        if step % report_interval == 0 or step == nsteps - 1:
            print(f"Step {step:4d}/{nsteps}: Mass residual: {step_info['mass_residual']:.2e}")
            # Only print solver status occasionally to avoid spam
            if step % (report_interval * 4) == 0:
                print(f"         Solver: {step_info['solver_status']}")
        
        # Plot at specified intervals
        if plot_by_step > 0 and (step % plot_by_step == 0 or step == nsteps - 1):
            if save_plots or show_plots:
                save_path = plot_results_at_step(solver, step, Re, nx, save_plots, show_plots)
                if step % (plot_by_step * 2) == 0:  # Print less frequently
                    print(f"         Plot saved: {save_path}")
    
    total_time = time.perf_counter() - start_time
    
    print("\n=== Simulation Complete ===")
    print(f"Total simulation time: {total_time:.3f}s")
    print(f"Total momentum time: {total_momentum_time:.3f}s")
    print(f"Total solver time: {total_solver_time:.3f}s")
    print(f"Solver efficiency: {100*total_solver_time/total_time:.1f}% of total time")
    print(f"Average solver time per step: {total_solver_time/nsteps*1000:.2f}ms")
    
    # Get final state
    ucc, vcc, speed = solver.get_cell_centered_velocities()
    max_speed = torch.max(speed).item()
    final_mass_residual = solver.check_mass_conservation()
    
    print(f"Final maximum velocity: {max_speed:.6f}")
    print(f"Final mass residual: {final_mass_residual:.2e}")
    
    # Plot final results if not already plotted
    if plot_by_step == 0 and (save_plots or show_plots):
        print("\nGenerating final plot...")
        save_path = plot_results_at_step(solver, nsteps-1, Re, nx, save_plots, show_plots)
        print(f"Final plot saved: {save_path}")
    
    # Optional: save velocity profiles at centerlines for validation
    print("\nCenterline velocity profiles:")
    
    # U-velocity along vertical centerline (x = 0.5)
    mid_x = nx // 2
    u_centerline = ucc[:, mid_x].cpu().numpy()
    
    # V-velocity along horizontal centerline (y = 0.5)
    mid_y = ny // 2
    v_centerline = vcc[mid_y, :].cpu().numpy()
    
    print(f"  U-velocity at x=0.5: max={np.max(u_centerline):.6f}, min={np.min(u_centerline):.6f}")
    print(f"  V-velocity at y=0.5: max={np.max(v_centerline):.6f}, min={np.min(v_centerline):.6f}")
    
    # Show directory structure information
    if save_plots:
        results_dir = Path(__file__).parent.parent / "results" / f"Re{Re:.0f}_nx{nx}"
        print(f"\n=== Results Directory Structure ===")
        print(f"Results saved in: {results_dir}")
        if plot_by_step > 0:
            print(f"Time steps saved every {plot_by_step} steps")
            num_plots = len(list(results_dir.glob("*/cavity_flow_step_*.png"))) if results_dir.exists() else 0
            print(f"Total plots generated: {num_plots}")
    
    print(f"\n=== Simulation Successfully Completed ===")
    
    return solver

if __name__ == "__main__":
    main()
