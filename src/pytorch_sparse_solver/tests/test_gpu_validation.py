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
GPU integration validation for pytorch_sparse_solver.

This target is intended for end-to-end verification on CUDA systems. It covers:

- Module A forward solves and differentiable wrappers
- Module B forward solves and differentiability (when available)
- Module C forward direct solves and differentiability (when available)
- Unified interface routing on GPU
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Add parent path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _apply_matrix(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if A.is_sparse:
        return torch.sparse.mm(A, x.unsqueeze(-1)).squeeze(-1)
    return torch.mv(A, x)


def _relative_residual(A: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> float:
    residual = b - _apply_matrix(A, x)
    return (torch.norm(residual) / torch.norm(b)).item()


def _extract_grad_values(grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if grad is None:
        return None
    if getattr(grad, "is_sparse_csr", False):
        return grad.values()
    if grad.is_sparse:
        return grad.coalesce().values()
    return grad


def _check_grad_ok(grad: Optional[torch.Tensor], zero_tol: float = 1e-12) -> Tuple[bool, str]:
    values = _extract_grad_values(grad)
    if values is None:
        return False, "gradient is None"
    if values.numel() == 0:
        return False, "gradient is empty"
    if torch.isnan(values).any() or torch.isinf(values).any():
        return False, "gradient contains NaN or Inf"
    if torch.all(values.abs() <= zero_tol):
        return False, "gradient is numerically zero"
    return True, f"grad_norm={torch.norm(values).item():.3e}"


def _make_spd_dense(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    A = torch.randn(n, n, device=device, dtype=dtype)
    return A @ A.T + torch.eye(n, device=device, dtype=dtype) * n


def _make_general_dense(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    A = torch.randn(n, n, device=device, dtype=dtype) * 0.05
    A = A + torch.diag(torch.full((n,), 8.0, device=device, dtype=dtype))
    skew = torch.triu(torch.randn(n, n, device=device, dtype=dtype) * 0.02, diagonal=1)
    return A + skew


@dataclass
class ValidationRecord:
    name: str
    passed: Optional[bool]
    detail: str


class GPUValidationRunner:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.device = "cuda"
        self.dtype = torch.float64
        self.records: List[ValidationRecord] = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _record(self, name: str, passed: Optional[bool], detail: str) -> None:
        self.records.append(ValidationRecord(name, passed, detail))
        if self.verbose:
            if passed is None:
                prefix = "[SKIP]"
            elif passed:
                prefix = "[PASS]"
            else:
                prefix = "[FAIL]"
            print(f"{prefix} {name}: {detail}")

    def _run_check(self, name: str, fn):
        try:
            passed, detail = fn()
        except Exception as exc:
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        self._record(name, passed, detail)
        return passed

    def _require_cuda(self) -> bool:
        if not torch.cuda.is_available():
            self._record("CUDA availability", None, "CUDA is not available")
            return False
        torch.cuda.set_device(0)
        return True

    def run_module_a(self) -> bool:
        from pytorch_sparse_solver.module_a import (
            bicgstab,
            bicgstab_differentiable,
            cg,
            cg_differentiable,
            gmres,
            gmres_differentiable,
        )

        ok = True
        n = 96
        A_spd = _make_spd_dense(n, self.device, self.dtype)
        x_true = torch.randn(n, device=self.device, dtype=self.dtype)
        b_spd = torch.mv(A_spd, x_true)

        x_cg, info_cg = cg(A_spd, b_spd, tol=1e-10, maxiter=4000)
        res_cg = _relative_residual(A_spd, x_cg, b_spd)
        passed = info_cg == 0 and res_cg <= 1e-6
        self._record("Module A forward CG", passed, f"info={info_cg}, residual={res_cg:.3e}")
        ok &= passed

        A_gen = _make_general_dense(n, self.device, self.dtype)
        x_true = torch.randn(n, device=self.device, dtype=self.dtype)
        b_gen = torch.mv(A_gen, x_true)

        x_bi, info_bi = bicgstab(A_gen, b_gen, tol=1e-10, maxiter=4000)
        res_bi = _relative_residual(A_gen, x_bi, b_gen)
        passed = info_bi == 0 and res_bi <= 1e-6
        self._record("Module A forward BiCGStab", passed, f"info={info_bi}, residual={res_bi:.3e}")
        ok &= passed

        x_gm, info_gm = gmres(A_gen, b_gen, tol=1e-10, maxiter=200, restart=30)
        res_gm = _relative_residual(A_gen, x_gm, b_gen)
        passed = info_gm == 0 and res_gm <= 1e-6
        self._record("Module A forward GMRES", passed, f"info={info_gm}, residual={res_gm:.3e}")
        ok &= passed

        b = b_spd.clone().detach().requires_grad_(True)
        x = cg_differentiable(A_spd, b, tol=1e-10, maxiter=4000)
        loss = (x.square()).sum()
        loss.backward()
        passed, detail = _check_grad_ok(b.grad)
        self._record("Module A autodiff CG", passed, detail)
        ok &= passed

        b = b_gen.clone().detach().requires_grad_(True)
        x = bicgstab_differentiable(A_gen, b, tol=1e-10, maxiter=4000)
        loss = (x.square()).sum()
        loss.backward()
        passed, detail = _check_grad_ok(b.grad)
        self._record("Module A autodiff BiCGStab", passed, detail)
        ok &= passed

        b = b_gen.clone().detach().requires_grad_(True)
        x = gmres_differentiable(A_gen, b, tol=1e-10, maxiter=200, restart=30)
        loss = (x.square()).sum()
        loss.backward()
        passed, detail = _check_grad_ok(b.grad)
        self._record("Module A autodiff GMRES", passed, detail)
        ok &= passed

        b = b_spd.clone().detach().requires_grad_(True)
        x, info = cg(A_spd, b, tol=1e-10, maxiter=4000)
        loss = (x.square()).sum()
        loss.backward()
        passed, detail = _check_grad_ok(b.grad)
        passed = passed and info == 0
        self._record("Module A auto-adjoint CG API", passed, f"info={info}, {detail}")
        ok &= passed

        b = b_gen.clone().detach().requires_grad_(True)
        x, info = bicgstab(A_gen, b, tol=1e-10, maxiter=4000)
        loss = (x.square()).sum()
        loss.backward()
        passed, detail = _check_grad_ok(b.grad)
        passed = passed and info == 0
        self._record("Module A auto-adjoint BiCGStab API", passed, f"info={info}, {detail}")
        ok &= passed

        b = b_gen.clone().detach().requires_grad_(True)
        x, info = gmres(A_gen, b, tol=1e-10, maxiter=200, restart=30)
        loss = (x.square()).sum()
        loss.backward()
        passed, detail = _check_grad_ok(b.grad)
        passed = passed and info == 0
        self._record("Module A auto-adjoint GMRES API", passed, f"info={info}, {detail}")
        ok &= passed

        return ok

    def run_module_c(self) -> bool:
        from pytorch_sparse_solver.module_c import cudss_available, cudss_spsolve
        from pytorch_sparse_solver.utils.matrix_utils import create_tridiagonal_sparse_coo

        if not cudss_available():
            self._record("Module C availability", None, "cuDSS is not available")
            return True

        ok = True
        n = 128
        A_coo = create_tridiagonal_sparse_coo(
            n, diag_val=4.0, off_diag_val=-1.0, device=self.device, dtype=self.dtype
        )
        A_csr = A_coo.to_sparse_csr()
        A_dense = A_coo.to_dense()
        x_true = torch.randn(n, device=self.device, dtype=self.dtype)
        b = torch.mv(A_dense, x_true)

        x = cudss_spsolve(A_csr, b)
        residual = _relative_residual(A_dense, x, b)
        passed = residual <= 1e-10
        self._record("Module C forward direct", passed, f"residual={residual:.3e}")
        ok &= passed

        A_grad = A_csr.detach().requires_grad_(True)
        b_grad = b.clone().detach().requires_grad_(True)
        x = cudss_spsolve(A_grad, b_grad)
        loss = (x.square()).sum()
        loss.backward()

        passed_b, detail_b = _check_grad_ok(b_grad.grad)
        self._record("Module C autodiff rhs", passed_b, detail_b)
        ok &= passed_b

        passed_A, detail_A = _check_grad_ok(A_grad.grad)
        self._record("Module C autodiff matrix", passed_A, detail_A)
        ok &= passed_A

        return ok

    def run_module_d(self) -> bool:
        from pytorch_sparse_solver import SparseSolver, get_available_backends, solve
        from pytorch_sparse_solver.module_c import cudss_available

        ok = True
        backends = get_available_backends()
        solver = SparseSolver(default_backend="auto")

        n = 96
        A_spd = _make_spd_dense(n, self.device, self.dtype)
        x_true = torch.randn(n, device=self.device, dtype=self.dtype)
        b = torch.mv(A_spd, x_true)

        x, result = solver.solve(A_spd, b, method="cg", backend="module_a", tol=1e-10, maxiter=4000)
        residual = _relative_residual(A_spd, x, b)
        passed = result.backend == "module_a" and result.converged and residual <= 1e-6
        self._record("Unified explicit Module A", passed, f"backend={result.backend}, residual={residual:.3e}")
        ok &= passed

        x, result = solve(A_spd, b, method="cg", backend="auto", tol=1e-10, maxiter=4000)
        expected = "module_b" if backends.get("module_b", False) else "module_a"
        residual = _relative_residual(A_spd, x, b)
        passed = result.backend == expected and result.converged and residual <= 1e-5
        self._record("Unified auto iterative", passed, f"backend={result.backend}, expected={expected}, residual={residual:.3e}")
        ok &= passed

        if backends.get("module_b", False):
            A_sparse = A_spd.to_sparse_coo().coalesce()
            x, result = solve(A_sparse, b, method="amg", backend="module_b", tol=1e-10, maxiter=4000)
            residual = _relative_residual(A_spd, x, b)
            passed = result.backend == "module_b" and result.method == "amg" and residual <= 1e-5
            self._record("Unified explicit AMG", passed, f"backend={result.backend}, residual={residual:.3e}")
            ok &= passed

        if cudss_available():
            A_csr = A_spd.to_sparse_coo().to_sparse_csr()
            x, result = solve(A_csr, b, method="direct", backend="module_c")
            residual = _relative_residual(A_spd, x, b)
            passed = result.backend == "module_c" and residual <= 1e-10
            self._record("Unified explicit Module C", passed, f"backend={result.backend}, residual={residual:.3e}")
            ok &= passed

            A_grad = A_csr.detach().requires_grad_(True)
            b_grad = b.clone().detach().requires_grad_(True)
            x, result = solve(A_grad, b_grad, method="direct", backend="module_c")
            loss = (x.square()).sum()
            loss.backward()
            passed_b, detail_b = _check_grad_ok(b_grad.grad)
            self._record("Unified direct autodiff rhs", passed_b, detail_b)
            ok &= passed_b

            passed_A, detail_A = _check_grad_ok(A_grad.grad)
            self._record("Unified direct autodiff matrix", passed_A, detail_A)
            ok &= passed_A
        else:
            self._record("Unified explicit Module C", None, "cuDSS is not available")

        return ok

    def run_module_b(self) -> bool:
        from pytorch_sparse_solver import check_module_b_available

        if not check_module_b_available():
            self._record("Module B availability", None, "pyamgx / AMGX is not available")
            return True

        from pytorch_sparse_solver.module_b import amgx_amg, amgx_bicgstab, amgx_cg, amgx_gmres
        from pytorch_sparse_solver.utils.matrix_utils import create_tridiagonal_sparse_coo

        ok = True
        n = 128
        A_spd_sparse = create_tridiagonal_sparse_coo(
            n, diag_val=4.0, off_diag_val=-1.0, device=self.device, dtype=self.dtype
        )
        A_spd_dense = A_spd_sparse.to_dense()
        x_true = torch.randn(n, device=self.device, dtype=self.dtype)
        b_spd = torch.mv(A_spd_dense, x_true)

        x = amgx_cg(A_spd_sparse, b_spd, tol=1e-10, maxiter=4000)
        residual = _relative_residual(A_spd_dense, x, b_spd)
        passed = residual <= 1e-5
        self._record("Module B forward CG", passed, f"residual={residual:.3e}")
        ok &= passed

        x = amgx_amg(A_spd_sparse, b_spd, tol=1e-10, maxiter=4000)
        residual = _relative_residual(A_spd_dense, x, b_spd)
        passed = residual <= 1e-5
        self._record("Module B forward AMG", passed, f"residual={residual:.3e}")
        ok &= passed

        A_gen = _make_general_dense(n, self.device, self.dtype)
        x_true = torch.randn(n, device=self.device, dtype=self.dtype)
        b_gen = torch.mv(A_gen, x_true)

        def _forward_check(solver_fn, A, rhs):
            x_local = solver_fn(A, rhs, tol=1e-10, maxiter=4000)
            residual_local = _relative_residual(A, x_local, rhs)
            return residual_local <= 1e-5, f"residual={residual_local:.3e}"

        def _rhs_grad_check(solver_fn, A, rhs):
            rhs_var = rhs.clone().detach().requires_grad_(True)
            x_local = solver_fn(A, rhs_var, tol=1e-10, maxiter=4000)
            loss = (x_local.square()).sum()
            loss.backward()
            return _check_grad_ok(rhs_var.grad)

        def _matrix_grad_check():
            A_var = A_spd_dense.clone().detach().requires_grad_(True)
            b_var = b_spd.clone().detach().requires_grad_(True)
            x_local = amgx_cg(A_var, b_var, tol=1e-10, maxiter=4000)
            loss = (x_local.square()).sum()
            loss.backward()
            return _check_grad_ok(A_var.grad)

        ok &= self._run_check("Module B forward BiCGStab", lambda: _forward_check(amgx_bicgstab, A_gen, b_gen))
        ok &= self._run_check("Module B forward GMRES", lambda: _forward_check(amgx_gmres, A_gen, b_gen))
        ok &= self._run_check("Module B autodiff rhs CG", lambda: _rhs_grad_check(amgx_cg, A_spd_sparse, b_spd))
        ok &= self._run_check("Module B autodiff rhs AMG", lambda: _rhs_grad_check(amgx_amg, A_spd_sparse, b_spd))
        ok &= self._run_check("Module B autodiff rhs BiCGStab", lambda: _rhs_grad_check(amgx_bicgstab, A_gen, b_gen))
        ok &= self._run_check("Module B autodiff rhs GMRES", lambda: _rhs_grad_check(amgx_gmres, A_gen, b_gen))
        ok &= self._run_check("Module B autodiff matrix public wrapper", _matrix_grad_check)

        return ok

    def run_all(self) -> bool:
        print("=" * 72)
        print("PyTorch Sparse Solver - GPU Integration Validation")
        print("=" * 72)

        if not self._require_cuda():
            return True

        all_passed = True

        all_passed &= self.run_module_a()
        all_passed &= self.run_module_c()
        all_passed &= self.run_module_d()
        all_passed &= self.run_module_b()

        print("\n" + "=" * 72)
        print("GPU Validation Summary")
        print("=" * 72)

        passed_count = 0
        skipped_count = 0
        for record in self.records:
            if record.passed is None:
                status = "SKIP"
                skipped_count += 1
            elif record.passed:
                status = "PASS"
                passed_count += 1
            else:
                status = "FAIL"
            print(f"  {status}: {record.name} - {record.detail}")

        executed = len([r for r in self.records if r.passed is not None])
        print(f"\nTotal: {passed_count}/{executed} executed checks passed ({skipped_count} skipped)")
        return all_passed


def main() -> int:
    runner = GPUValidationRunner(verbose=True)
    success = runner.run_all()
    return 0 if success else 1


def test_gpu_validation() -> None:
    runner = GPUValidationRunner(verbose=False)
    assert runner.run_all()


if __name__ == "__main__":
    raise SystemExit(main())
