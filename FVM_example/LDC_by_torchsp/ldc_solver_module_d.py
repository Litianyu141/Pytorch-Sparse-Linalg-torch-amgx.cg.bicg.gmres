#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ldc_solver_common import BaseLDCSolver
from pytorch_sparse_solver import SparseSolver


class LDCSolverModuleD(BaseLDCSolver):
    def __init__(self, *args, backend: str = "auto", **kwargs):
        self.backend = backend
        super().__init__(*args, solver_label="ModuleD", **kwargs)
        self.solver = SparseSolver(default_backend=backend, default_method=self.method)

    def _solve_linear_system(self, prhs):
        if self.method == "direct":
            A = self.A_csr
            kwargs = {}
        elif self.backend == "module_a":
            A = self.A_dense
            kwargs = {"restart": 30} if self.method == "gmres" else {}
        else:
            A = self.A_sparse
            kwargs = {"restart": 30} if self.method == "gmres" else {}

        pt, result = self.solver.solve(A, prhs, method=self.method, backend=self.backend, tol=1e-10, maxiter=1000, **kwargs)
        self.solver_label = f"ModuleD-{result.backend}"
        return pt, 0 if result.converged else 1


def main():
    parser = argparse.ArgumentParser(description="LDC solver using pytorch_sparse_solver Module D (unified interface)")
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--Re", type=float, default=400.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--method", type=str, default="bicgstab", choices=["bicgstab", "gmres", "direct"])
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "module_a", "module_b", "module_c"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.nx = 32
        args.steps = 200
        args.Re = 100.0

    solver = LDCSolverModuleD(nx=args.nx, Re=args.Re, method=args.method, backend=args.backend)
    solver.run(nsteps=args.steps, print_interval=max(1, args.steps // 10))

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(__file__).parent.parent / "torch_results" / f"moduled_{args.backend}_Re{args.Re:.0f}_nx{args.nx}_{timestamp}"

    save_path = save_dir / f"ldc_Re{args.Re:.0f}_nx{args.nx}_module_d_{args.backend}_{args.method}_steps{args.steps}.png"
    if not args.no_plot:
        solver.plot_results(save_path=str(save_path), show=False)
        print(f"\nTo view the results, open: {save_path}")


if __name__ == "__main__":
    main()
