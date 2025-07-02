# Sparse Matrix Solver Comparison Report

## Test Summary

- **Total Tests:** 30
- **Matrix Types:** non_diagonally_dominant, diagonally_dominant, banded
- **Solvers:** AMGX CG (Diff), PyTorch GMRES, AMGX BiCGStab (Diff), PyTorch BiCGStab, PyTorch CG (Diff), PyTorch CG, AMGX BiCGStab, AMGX CG, PyTorch BiCGStab (Diff), PyTorch GMRES (Diff)

## Detailed Results

**Note:** Tests marked with `(Diff)` are differentiability tests that verify automatic differentiation support. For these tests:
- Solution Error is set to 0.00e+00 (no true solution available)
- Residual Error shows how well the solver satisfied Ax=b
- Gradient Test column shows whether backpropagation succeeded

| Solver | Matrix Type | Matrix Size | Solve Time (s) | Solution Error | Residual Error | Converged | Gradient Test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch CG | diagonally_dominant | 200 | 0.3360 | 3.27e-04 | 1.03e-03 | ❌ | N/A |
| PyTorch BiCGStab | diagonally_dominant | 200 | 0.0051 | 3.12e-08 | 8.39e-08 | ✅ | N/A |
| PyTorch GMRES | diagonally_dominant | 200 | 0.0598 | 6.18e-12 | 1.64e-11 | ✅ | N/A |
| AMGX CG | diagonally_dominant | 200 | 0.1644 | 3.27e-04 | 1.03e-03 | ✅ | N/A |
| AMGX BiCGStab | diagonally_dominant | 200 | 0.0514 | 3.12e-08 | 8.39e-08 | ✅ | N/A |
| PyTorch CG | non_diagonally_dominant | 200 | 0.1853 | Failed | Failed | ❌ | N/A |
| PyTorch BiCGStab | non_diagonally_dominant | 200 | 0.0281 | 9.43e+02 | 7.74e+03 | ❌ | N/A |
| PyTorch GMRES | non_diagonally_dominant | 200 | 16.0284 | 1.33e+01 | 1.07e+02 | ❌ | N/A |
| AMGX CG | non_diagonally_dominant | 200 | 0.1001 | Failed | Failed | ✅ | N/A |
| AMGX BiCGStab | non_diagonally_dominant | 200 | 0.1497 | Failed | Failed | ✅ | N/A |
| PyTorch CG | banded | 200 | 0.1786 | 2.81e+03 | 7.70e+03 | ❌ | N/A |
| PyTorch BiCGStab | banded | 200 | 0.0171 | 8.86e-07 | 3.27e-07 | ✅ | N/A |
| PyTorch GMRES | banded | 200 | 0.0655 | 1.14e-08 | 7.15e-09 | ✅ | N/A |
| AMGX CG | banded | 200 | 0.0976 | 2.81e+03 | 7.70e+03 | ✅ | N/A |
| AMGX BiCGStab | banded | 200 | 0.0503 | 8.86e-07 | 3.27e-07 | ✅ | N/A |
| PyTorch CG | diagonally_dominant | 500 | 0.1790 | 5.02e-04 | 1.73e-03 | ❌ | N/A |
| PyTorch BiCGStab | diagonally_dominant | 500 | 0.0042 | 2.96e-07 | 6.42e-07 | ✅ | N/A |
| PyTorch GMRES | diagonally_dominant | 500 | 0.0173 | 1.89e-11 | 5.60e-11 | ✅ | N/A |
| AMGX CG | diagonally_dominant | 500 | 0.1044 | 5.02e-04 | 1.73e-03 | ✅ | N/A |
| AMGX BiCGStab | diagonally_dominant | 500 | 0.0493 | 1.02e-07 | 2.18e-07 | ✅ | N/A |
| PyTorch CG | banded | 500 | 0.1785 | 5.04e+03 | 1.20e+04 | ❌ | N/A |
| PyTorch BiCGStab | banded | 500 | 0.0232 | 7.30e-07 | 1.95e-07 | ✅ | N/A |
| PyTorch GMRES | banded | 500 | 0.0810 | 2.85e-08 | 1.35e-08 | ✅ | N/A |
| AMGX CG | banded | 500 | 0.0957 | 5.04e+03 | 1.20e+04 | ✅ | N/A |
| AMGX BiCGStab | banded | 500 | 0.0522 | 2.09e-07 | 3.06e-07 | ✅ | N/A |
| PyTorch CG (Diff) | diagonally_dominant | 100 | 0.1029 | 0.00e+00 | 3.03e-03 | ✅ | ✅ |
| PyTorch BiCGStab (Diff) | diagonally_dominant | 100 | 0.0042 | 0.00e+00 | 8.31e-06 | ✅ | ✅ |
| PyTorch GMRES (Diff) | diagonally_dominant | 100 | 0.0205 | 0.00e+00 | 3.29e-12 | ✅ | ✅ |
| AMGX CG (Diff) | diagonally_dominant | 100 | 0.0726 | 0.00e+00 | 3.03e-03 | ✅ | ✅ |
| AMGX BiCGStab (Diff) | diagonally_dominant | 100 | 0.0479 | 0.00e+00 | 8.31e-06 | ✅ | ✅ |

## Performance Analysis

### Convergence Results

- **Total Convergence Rate:** 76.7% (23/30)

### Solver Performance Summary


#### PyTorch CG
- **Convergence Rate:** 0.0% (0/5)
- **Average Solve Time:** 0.2115s
- **Best Solution Error:** 3.27e-04
- **Worst Solution Error:** 5.04e+03

#### PyTorch BiCGStab
- **Convergence Rate:** 80.0% (4/5)
- **Average Solve Time:** 0.0155s
- **Best Solution Error:** 3.12e-08
- **Worst Solution Error:** 9.43e+02

#### PyTorch GMRES
- **Convergence Rate:** 80.0% (4/5)
- **Average Solve Time:** 3.2504s
- **Best Solution Error:** 6.18e-12
- **Worst Solution Error:** 1.33e+01

#### AMGX CG
- **Convergence Rate:** 100.0% (5/5)
- **Average Solve Time:** 0.1125s
- **Best Solution Error:** 3.27e-04
- **Worst Solution Error:** 5.04e+03

#### AMGX BiCGStab
- **Convergence Rate:** 100.0% (5/5)
- **Average Solve Time:** 0.0706s
- **Best Solution Error:** 3.12e-08
- **Worst Solution Error:** 8.86e-07

#### PyTorch CG (Diff)
- **Convergence Rate:** 100.0% (1/1)
- **Average Solve Time:** 0.1029s
- **Best Solution Error:** 0.00e+00
- **Worst Solution Error:** 0.00e+00

#### PyTorch BiCGStab (Diff)
- **Convergence Rate:** 100.0% (1/1)
- **Average Solve Time:** 0.0042s
- **Best Solution Error:** 0.00e+00
- **Worst Solution Error:** 0.00e+00

#### PyTorch GMRES (Diff)
- **Convergence Rate:** 100.0% (1/1)
- **Average Solve Time:** 0.0205s
- **Best Solution Error:** 0.00e+00
- **Worst Solution Error:** 0.00e+00

#### AMGX CG (Diff)
- **Convergence Rate:** 100.0% (1/1)
- **Average Solve Time:** 0.0726s
- **Best Solution Error:** 0.00e+00
- **Worst Solution Error:** 0.00e+00

#### AMGX BiCGStab (Diff)
- **Convergence Rate:** 100.0% (1/1)
- **Average Solve Time:** 0.0479s
- **Best Solution Error:** 0.00e+00
- **Worst Solution Error:** 0.00e+00

---
*Report generated on 2025-07-02 17:41:55*
