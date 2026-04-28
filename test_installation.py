"""
Simple test script to verify installation and basic functionality.

Run this script to check if all dependencies are installed correctly
and if the basic mesh generation and optimization work.
"""

import sys
import torch
import numpy as np
from pathlib import Path

print("=" * 70)
print("INSTALLATION AND FUNCTIONALITY TEST")
print("=" * 70)

# Check Python version
print("\n1. Python Version")
print(f"   Version: {sys.version}")
assert sys.version_info >= (3, 8), "Python 3.8+ required"
print("   ✓ OK")

# Check PyTorch
print("\n2. PyTorch")
try:
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    print("   ✓ OK")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

# Check NumPy
print("\n3. NumPy")
try:
    print(f"   Version: {np.__version__}")
    print("   ✓ OK")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

# Check Trimesh
print("\n4. Trimesh")
try:
    import trimesh
    print(f"   Version: {trimesh.__version__}")
    print("   ✓ OK")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

# Check Matplotlib
print("\n5. Matplotlib")
try:
    import matplotlib
    print(f"   Version: {matplotlib.__version__}")
    print("   ✓ OK")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

# Check local modules
print("\n6. Local Modules")
try:
    from src.create_mesh import create_icosphere, add_radial_noise
    print("   ✓ create_mesh")
    
    from src.curvature import compute_laplacian_curvature_proxy
    print("   ✓ curvature")
    
    from src.losses import total_loss
    print("   ✓ losses")
    
    from src.evaluate import evaluate_mesh
    print("   ✓ evaluate")
    
    from src.visualise import visualize_mesh_3d
    print("   ✓ visualise")
    
    print("   ✓ OK - All modules loaded")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    sys.exit(1)

# Run minimal test
print("\n7. Minimal Functionality Test")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create small sphere
    vertices, faces, _ = create_icosphere(subdivisions=2, radius=1.0, device=device)
    print(f"   ✓ Created mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    
    # Compute curvature
    curvature = compute_laplacian_curvature_proxy(vertices, faces)
    print(f"   ✓ Computed curvature: mean={curvature.mean():.6f}, std={curvature.std():.6f}")
    
    # Test evaluation
    metrics = evaluate_mesh(vertices, vertices, faces)
    print(f"   ✓ Evaluated mesh: {len(metrics)} metrics computed")
    
    print("   ✓ OK - All tests passed!")
    
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("INSTALLATION TEST COMPLETE")
print("=" * 70)
print("\n✓ All checks passed! You can now run:")
print("  python scripts/run_oracle_sphere.py")
print()
