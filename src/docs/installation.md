# Installation Guide

This document provides detailed instructions for installing `pytorch_sparse_solver` and its modules.

## Table of Contents

- [Basic Installation (Module A)](#basic-installation-module-a)
- [Installing Module B (PyAMGX)](#installing-module-b-pyamgx)
- [Installing Module C (cuDSS)](#installing-module-c-cudss)
- [Full Installation](#full-installation)
- [Verifying Installation](#verifying-installation)

---

## Basic Installation (Module A)

Module A is a pure PyTorch implementation with no additional dependencies.

### Method 1: pip Installation

```bash
# Create conda environment
conda create -n pytorch_sparse python=3.11 -y
conda activate pytorch_sparse

# Install PyTorch
pip install torch>=2.0.0

# Install the package
cd /path/to/Pytorch_Sparse_Linalg-torch.cg-bicg-gmres-
pip install -e .
```

### Method 2: Direct Usage

```bash
# No installation needed, just add to Python path
import sys
sys.path.insert(0, '/path/to/Pytorch_Sparse_Linalg-torch.cg-bicg-gmres-')

from pytorch_sparse_solver import solve
```

---

## Installing Module B (PyAMGX)

Module B requires the NVIDIA AMGX library and pyamgx Python bindings.

### Step 1: Install Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake git
```

### Step 2: Compile AMGX

```bash
# Clone AMGX repository
git clone --recursive https://github.com/NVIDIA/AMGX.git
cd AMGX

# Create build directory
mkdir build && cd build

# Configure CMake (adjust CUDA_ARCH for your GPU)
# RTX 4090: 89, RTX 3090: 86, RTX 2080: 75, V100: 70
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCH=89 \
    -DCMAKE_INSTALL_PREFIX=/usr/local

# Compile (use multiple cores)
make -j$(nproc)

# Install
sudo make install

# Update library path
echo '/usr/local/lib' | sudo tee -a /etc/ld.so.conf.d/amgx.conf
sudo ldconfig
```

### Step 3: Install pyamgx

```bash
# Clone pyamgx
git clone https://github.com/shwina/pyamgx.git
cd pyamgx

# Set environment variables
export AMGX_DIR=/usr/local
export CUDA_HOME=/usr/local/cuda

# Install
pip install .
```

### Step 4: Verify Installation

```bash
python -c "import pyamgx; print('PyAMGX installed successfully!')"
```

---

## Installing Module C (cuDSS)

Module C requires PyTorch compiled from source with cuDSS support enabled.

### Step 1: Install cuDSS Library

```bash
# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-repo-ubuntu2204-0.6.0_0.6.0-1_amd64.deb
sudo dpkg -i cudss-local-repo-ubuntu2204-0.6.0_0.6.0-1_amd64.deb
sudo cp /var/cudss-local-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudss
```

### Step 2: Create New Conda Environment

```bash
conda create -n torch_sparse2.7 python=3.11 -y
conda activate torch_sparse2.7
```

### Step 3: Install Build Dependencies

```bash
# Install via conda
conda install cmake ninja -y

# Install Python dependencies
pip install pyyaml numpy typing-extensions sympy filelock networkx jinja2 fsspec six
pip install mkl-static mkl-include
```

### Step 4: Download PyTorch Source

```bash
cd ~
git clone --recursive https://github.com/pytorch/pytorch pytorch_source
cd pytorch_source

# Checkout v2.7.0
git checkout v2.7.0
git submodule sync
git submodule update --init --recursive
```

### Step 5: Compile PyTorch

```bash
# Set environment variables
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH}"
export USE_CUDSS=1
export USE_CUDA=1
export MAX_JOBS=12  # Use nproc to check CPU cores, recommend using half

# Compile (may take 1-2 hours)
python -m pip install --no-build-isolation -v -e .
```

### Step 6: Verify Installation

```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

# Test cuDSS
if torch.cuda.is_available():
    try:
        device = 'cuda'
        row_indices = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)
        col_indices = torch.tensor([0, 1, 1, 2, 0, 2], dtype=torch.long, device=device)
        values = torch.tensor([2.0, 1.0, 3.0, 1.0, 1.0, 4.0], dtype=torch.float32, device=device)
        indices = torch.stack([row_indices, col_indices])
        sparse_coo = torch.sparse_coo_tensor(indices, values, (3, 3), device=device)
        sparse_csr = sparse_coo.to_sparse_csr()
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        x = torch.sparse.spsolve(sparse_csr, b)
        print('cuDSS working!')
    except Exception as e:
        print('cuDSS not working:', e)
"
```

---

## Full Installation

To install all modules, follow this order:

1. **First complete Module C installation** (compile PyTorch with cuDSS)
2. **Then install AMGX and pyamgx** (Module B)
3. **Finally install this package**

```bash
# In the configured environment
cd /path/to/Pytorch_Sparse_Linalg-torch.cg-bicg-gmres-
pip install -e .
```

---

## Verifying Installation

Run the following command to verify all modules are correctly installed:

```bash
cd /path/to/Pytorch_Sparse_Linalg-torch.cg-bicg-gmres-
python -c "
from pytorch_sparse_solver.utils.availability import print_availability_report
print_availability_report()
"
```

Expected output (if all modules are installed correctly):

```
============================================================
PyTorch Sparse Solver - Module Availability Report
============================================================

Module A (JAX-style Iterative Solvers): Available
  - CG (Conjugate Gradient)
  - BiCGStab (Bi-Conjugate Gradient Stabilized)
  - GMRES (Generalized Minimal Residual)

Module B (PyAMGX GPU Solver): Available
  - AMGX CG with AMG preconditioner
  - AMGX BiCGStab with AMG preconditioner
  - AMGX GMRES with AMG preconditioner
  - Automatic differentiation support

Module C (cuDSS Direct Solver): Available
  - torch.sparse.spsolve (LU factorization)
  - GPU-accelerated direct method

============================================================
Total: 3/3 modules available
============================================================
```

### Run Complete Test Suite

```bash
python run_all_tests.py
```

### Run Performance Benchmarks

```bash
python -m pytorch_sparse_solver.tests.benchmark --sizes 100,500,1000 --backends module_a,module_b
```

---

## Troubleshooting

### Q: Module B shows as unavailable

A: Ensure:
- pyamgx is correctly installed
- AMGX library is in system path
- CUDA is available

### Q: Module C shows as unavailable

A: Ensure:
- Using PyTorch 2.7+ compiled from source
- `USE_CUDSS=1` was set during compilation
- cuDSS library is correctly installed

### Q: PyTorch compilation fails

A: Try:
- Reduce MAX_JOBS (memory may be insufficient)
- Ensure all submodules are properly updated
- Check CUDA version compatibility
