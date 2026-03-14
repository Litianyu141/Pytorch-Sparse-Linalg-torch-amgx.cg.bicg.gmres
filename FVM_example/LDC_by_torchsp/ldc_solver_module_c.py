#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ldc_solver_common import BaseLDCSolver, run_solver_cli
from pytorch_sparse_solver.module_c import cudss_spsolve


class LDCSolverModuleC(BaseLDCSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, solver_label="ModuleC", **kwargs)

    def _solve_linear_system(self, prhs):
        pt = cudss_spsolve(self.A_csr, prhs)
        return pt, 0


def main():
    run_solver_cli(
        LDCSolverModuleC,
        description="LDC solver using pytorch_sparse_solver Module C (cuDSS direct)",
        method_choices=("direct",),
        default_method="direct",
    )


if __name__ == "__main__":
    main()
