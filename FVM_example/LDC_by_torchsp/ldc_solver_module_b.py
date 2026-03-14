#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ldc_solver_common import BaseLDCSolver, run_solver_cli
from pytorch_sparse_solver.module_b import amgx_bicgstab, amgx_gmres


class LDCSolverModuleB(BaseLDCSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, solver_label="ModuleB", **kwargs)

    def _solve_linear_system(self, prhs):
        if self.method == "bicgstab":
            pt = amgx_bicgstab(self.A_sparse, prhs, tol=1e-10, maxiter=1000)
        else:
            pt = amgx_gmres(self.A_sparse, prhs, tol=1e-10, maxiter=1000)
        return pt, 0


def main():
    run_solver_cli(
        LDCSolverModuleB,
        description="LDC solver using pytorch_sparse_solver Module B (AMGX)",
        method_choices=("bicgstab", "gmres"),
        default_method="bicgstab",
    )


if __name__ == "__main__":
    main()
