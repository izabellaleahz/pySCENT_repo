#!/usr/bin/env python3
"""
Basic test to ensure all modules can be imported and basic functionality works
"""

import sys
import os

# Add pySCENT to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pySCENT'))

def test_imports():
    """Test that all core modules can be imported"""
    print("Testing imports...")

    try:
        from pySCENT import scent_algorithm_gpu, load_scent_data
        print("✓ Main functions imported")
    except ImportError as e:
        print(f"✗ Failed to import main functions: {e}")
        return False

    try:
        from pySCENT.poisson_gpu import fit_single_pair, poisson_glm_gpu
        print("✓ Poisson GLM functions imported")
    except ImportError as e:
        print(f"✗ Failed to import Poisson functions: {e}")
        return False

    try:
        from pySCENT.bootstrap_gpu import adaptive_bootstrap_gpu, basic_p_gpu
        print("✓ Bootstrap functions imported")
    except ImportError as e:
        print(f"✗ Failed to import bootstrap functions: {e}")
        return False

    try:
        from pySCENT.data_loader import load_scent_data, sparse_to_torch_coo
        print("✓ Data loader functions imported")
    except ImportError as e:
        print(f"✗ Failed to import data loader functions: {e}")
        return False

    try:
        from pySCENT.scent_gpu import calculate_fdr, save_scent_results
        print("✓ Utility functions imported")
    except ImportError as e:
        print(f"✗ Failed to import utility functions: {e}")
        return False

    return True

def test_torch_gpu():
    """Test PyTorch GPU availability"""
    print("\nTesting PyTorch GPU...")

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device count: {torch.cuda.device_count()}")
            print(f"✓ Current device: {torch.cuda.current_device()}")
        return True
    except ImportError as e:
        print(f"✗ PyTorch not available: {e}")
        return False

def test_basic_tensor_ops():
    """Test basic tensor operations work"""
    print("\nTesting basic tensor operations...")

    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create test tensors
        x = torch.randn(10, 5, device=device)
        y = torch.randn(10, 5, device=device)
        z = x + y

        print(f"✓ Tensor addition works on {device}")
        print(f"✓ Result shape: {z.shape}")
        return True
    except Exception as e:
        print(f"✗ Tensor operations failed: {e}")
        return False

def main():
    print("="*60)
    print("PYSCENT BASIC FUNCTIONALITY TEST")
    print("="*60)

    success = True

    success &= test_imports()
    success &= test_torch_gpu()
    success &= test_basic_tensor_ops()

    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED - pySCENT is ready to use!")
    else:
        print("❌ SOME TESTS FAILED - check dependencies and setup")
    print("="*60)

    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
