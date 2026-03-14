#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ldc_solver_common import BaseLDCSolver, run_solver_cli
from pytorch_sparse_solver.module_a import bicgstab, gmres


class LDCSolverModuleA(BaseLDCSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, solver_label="ModuleA", **kwargs)

    def _solve_linear_system(self, prhs):
        if self.method == "bicgstab":
            pt, info = bicgstab(self.A_dense, prhs, tol=1e-10, maxiter=1000)
        else:
            pt, info = gmres(self.A_dense, prhs, tol=1e-10, maxiter=1000, restart=30)
        return pt, info


def main():
    run_solver_cli(
        LDCSolverModuleA,
        description="LDC solver using pytorch_sparse_solver Module A",
        method_choices=("bicgstab", "gmres"),
        default_method="bicgstab",
    )


if __name__ == "__main__":
    main()
