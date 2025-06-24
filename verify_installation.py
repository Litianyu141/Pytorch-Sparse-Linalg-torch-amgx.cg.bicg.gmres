#!/usr/bin/env python3
"""
Installation Verification Script

This script verifies that all components are properly installed and working:
- PyTorch with CUDA support
- PyTorch sparse solvers
- AMGX library and pyamgx bindings
- GPU availability and compatibility

Author: Augment Agent
Date: 2025-06-24
"""

import sys
import os
import torch
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_pytorch():
    """Check PyTorch installation and CUDA support"""
    print("🔧 Checking PyTorch Installation")
    print("-" * 40)
    
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"✅ GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("❌ CUDA not available")
        return False
    
    return True

def check_pytorch_solvers():
    """Check PyTorch sparse solvers"""
    print("\n🧮 Checking PyTorch Sparse Solvers")
    print("-" * 40)
    
    try:
        from src.torch_sparse_linalg import cg, bicgstab, gmres
        print("✅ PyTorch solvers imported successfully")
        
        # Quick test
        n = 50
        A = torch.eye(n, dtype=torch.float64, device='cuda') * 2
        b = torch.ones(n, dtype=torch.float64, device='cuda')
        
        x_cg, info_cg = cg(A, b, tol=1e-6, maxiter=100)
        x_bicg, info_bicg = bicgstab(A, b, tol=1e-6, maxiter=100)
        x_gmres, info_gmres = gmres(A, b, tol=1e-6, maxiter=100)
        
        print(f"✅ CG solver: converged={info_cg==0}")
        print(f"✅ BiCGStab solver: converged={info_bicg==0}")
        print(f"✅ GMRES solver: converged={info_gmres==0}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import PyTorch solvers: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch solvers test failed: {e}")
        return False

def check_amgx():
    """Check AMGX installation"""
    print("\n🚀 Checking AMGX Installation")
    print("-" * 40)
    
    try:
        import pyamgx
        print("✅ pyamgx imported successfully")
        print(f"✅ pyamgx version: {getattr(pyamgx, '__version__', 'unknown')}")
        
        # Test AMGX initialization
        pyamgx.initialize()
        print("✅ AMGX initialized successfully")
        
        # Test basic AMGX functionality
        config_dict = {
            "config_version": 2,
            "determinism_flag": 1,
            "exception_handling": 1,
            "solver": {
                "monitor_residual": 1,
                "solver": "BICGSTAB",
                "max_iters": 100,
                "tolerance": 1e-6,
                "convergence": "RELATIVE_INI_CORE"
            }
        }
        
        cfg = pyamgx.Config()
        cfg.create_from_dict(config_dict)
        print("✅ AMGX configuration created successfully")
        
        cfg.destroy()
        pyamgx.finalize()
        print("✅ AMGX basic functionality test passed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import pyamgx: {e}")
        print("   Install AMGX and pyamgx following the README instructions")
        return False
    except Exception as e:
        print(f"❌ AMGX test failed: {e}")
        return False

def check_torch_amgx():
    """Check torch_amgx integration"""
    print("\n🔗 Checking torch_amgx Integration")
    print("-" * 40)
    
    try:
        from src_torch_amgx.torch_amgx import amgx_cg, amgx_bicgstab
        print("✅ torch_amgx imported successfully")
        
        # Quick test
        n = 50
        A = torch.eye(n, dtype=torch.float64, device='cuda') * 2
        b = torch.ones(n, dtype=torch.float64, device='cuda')
        
        x_amgx_cg = amgx_cg(A, b, tol=1e-6, maxiter=100)
        x_amgx_bicg = amgx_bicgstab(A, b, tol=1e-6, maxiter=100)
        
        error_cg = torch.norm(A @ x_amgx_cg - b).item()
        error_bicg = torch.norm(A @ x_amgx_bicg - b).item()
        
        print(f"✅ AMGX CG solver: residual error={error_cg:.2e}")
        print(f"✅ AMGX BiCGStab solver: residual error={error_bicg:.2e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import torch_amgx: {e}")
        return False
    except Exception as e:
        print(f"❌ torch_amgx test failed: {e}")
        return False

def check_differentiability():
    """Check automatic differentiation"""
    print("\n🔬 Checking Automatic Differentiation")
    print("-" * 40)
    
    try:
        from src.torch_sparse_linalg import cg
        
        # Test PyTorch solver differentiability
        n = 30
        A = torch.eye(n, dtype=torch.float64, device='cuda') * 2
        b = torch.ones(n, dtype=torch.float64, device='cuda', requires_grad=True)
        
        x = cg(A, b, tol=1e-6, maxiter=100)[0]
        loss = torch.sum(x**2)
        loss.backward()
        
        if b.grad is not None and torch.isfinite(b.grad).all():
            print("✅ PyTorch solver differentiability: PASSED")
        else:
            print("❌ PyTorch solver differentiability: FAILED")
            return False
        
        # Test AMGX solver differentiability (if available)
        try:
            from src_torch_amgx.torch_amgx import amgx_cg
            
            b.grad = None
            x_amgx = amgx_cg(A, b, tol=1e-6, maxiter=100)
            loss_amgx = torch.sum(x_amgx**2)
            loss_amgx.backward()
            
            if b.grad is not None and torch.isfinite(b.grad).all():
                print("✅ AMGX solver differentiability: PASSED")
            else:
                print("❌ AMGX solver differentiability: FAILED")
                return False
                
        except ImportError:
            print("⚠️  AMGX solver not available, skipping differentiability test")
        
        return True
        
    except Exception as e:
        print(f"❌ Differentiability test failed: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🚀 PyTorch Sparse Linear Algebra Solvers - Installation Verification")
    print("=" * 70)
    
    checks = [
        ("PyTorch", check_pytorch),
        ("PyTorch Solvers", check_pytorch_solvers),
        ("AMGX", check_amgx),
        ("torch_amgx Integration", check_torch_amgx),
        ("Automatic Differentiation", check_differentiability),
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
            results[name] = False
    
    # Summary
    print("\n📊 Verification Summary")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:<30} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All checks passed! Your installation is ready to use.")
        print("\nNext steps:")
        print("- Run: python examples/basic_usage.py")
        print("- Run: python test_comprehensive_solvers.py")
    else:
        print("⚠️  Some checks failed. Please review the installation instructions.")
        print("See README.md for detailed installation steps.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
