"""
Configuration file for curvature optimization experiments.

This module contains default parameters and configuration constants
that can be easily modified for different experiments.
"""

# ============================================================================
# MESH GENERATION PARAMETERS
# ============================================================================

# Icosphere parameters
MESH_SUBDIVISIONS = 3           # Subdivision level: 1-5 recommended
MESH_RADIUS = 1.0               # Sphere radius

# Noise parameters
NOISE_STRENGTH = 0.10           # Epsilon: noise magnitude
NOISE_SEED = 42                 # Random seed for reproducibility
NOISE_TYPE = "radial"           # Type: "radial", "tangential", or "mixed"

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

# Gradient descent settings
NUM_ITERATIONS = 500            # Number of optimization iterations
LEARNING_RATE = 0.01            # Adam learning rate
GRADIENT_CLIP_NORM = 0.1        # Max gradient norm for stability

# Loss function weights
LAMBDA_EDGE = 0.1               # Edge length regularization weight
LAMBDA_POS = 0.01               # Position regularization weight
LAMBDA_CURV = 1.0               # Curvature loss weight (implicit)

# ============================================================================
# LAPLACIAN SMOOTHING BASELINE
# ============================================================================

SMOOTHING_ITERATIONS = 500      # Number of smoothing iterations
SMOOTHING_ALPHA = 0.1           # Smoothing parameter (0 to 1)

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

# Geometry parameters
TARGET_RADIUS = 1.0             # Expected sphere radius
COMPUTE_NORMAL_ERROR = True     # Whether to compute normal errors

# Metrics to report
METRICS_TO_REPORT = [
    'mean_vertex_error',
    'rmse_vertex_error',
    'max_vertex_error',
    'curvature_mse',
    'mean_radius_error',
    'normal_error'
]

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Figure sizes
FIGURE_SIZE_3D = (8, 8)         # Single mesh visualization
FIGURE_SIZE_COMPARISON = (20, 5) # Side-by-side comparison
FIGURE_SIZE_LOSS = (12, 4)      # Loss curves

# Plot settings
PLOT_DPI = 150                  # Resolution for saved figures
PLOT_ALPHA = 0.3                # Transparency for meshes
PLOT_EDGE_WIDTH = 0.5           # Edge line width

# Color settings
COLORMAP = 'viridis'            # Heatmap colormap
MESH_COLOR = 'cyan'             # Default mesh color

# ============================================================================
# OUTPUT PARAMETERS
# ============================================================================

# Directory structure
OUTPUT_BASE_DIR = "outputs"
OUTPUT_MESHES_DIR = "meshes"
OUTPUT_FIGURES_DIR = "figures"
OUTPUT_LOGS_DIR = "logs"

# File naming
SAVE_OPTIMIZATION_LOG = True
SAVE_SMOOTHING_LOG = True
SAVE_METRICS = True
SAVE_FIGURES = True

# ============================================================================
# DEVICE SETTINGS
# ============================================================================

# PyTorch device
USE_GPU = True                  # Whether to use GPU if available
DEVICE_TYPE = "cuda" if USE_GPU else "cpu"

# ============================================================================
# VERBOSITY
# ============================================================================

# Logging
VERBOSE_OPTIMIZATION = True     # Print optimization progress
VERBOSE_EVALUATION = True       # Print evaluation metrics
PRINT_FREQUENCY = 50            # Print every N iterations

# ============================================================================
# ADVANCED PARAMETERS
# ============================================================================

# Curvature computation
CURVATURE_EPSILON = 1e-8        # Small value to avoid division by zero
COMPUTE_ADJACENCY_CACHED = True # Cache adjacency list

# Edge detection
EDGE_DEDUPLICATE = True         # Remove duplicate edges
EDGE_MIN_LENGTH = 1e-6          # Minimum edge length threshold

# Optimization
USE_ADAM_OPTIMIZER = True       # Use Adam (vs SGD)
ADAM_BETAS = (0.9, 0.999)      # Adam beta parameters
ADAM_WEIGHT_DECAY = 0.0         # L2 regularization; usually 0 for mesh opt

# Numerical stability
CLAMP_CURVATURE = False         # Clamp curvature values
CURVATURE_MIN = -1.0            # Minimum curvature if clamped
CURVATURE_MAX = 1.0             # Maximum curvature if clamped

# ============================================================================
# EXPERIMENT PRESETS
# ============================================================================

# Presets for different experiment configurations

PRESETS = {
    "quick_test": {
        "mesh_subdivisions": 2,
        "noise_strength": 0.10,
        "num_iterations": 100,
        "learning_rate": 0.01,
        "lambda_edge": 0.1,
        "lambda_pos": 0.01,
    },
    "standard": {
        "mesh_subdivisions": 3,
        "noise_strength": 0.10,
        "num_iterations": 500,
        "learning_rate": 0.01,
        "lambda_edge": 0.1,
        "lambda_pos": 0.01,
    },
    "high_quality": {
        "mesh_subdivisions": 4,
        "noise_strength": 0.10,
        "num_iterations": 1000,
        "learning_rate": 0.005,
        "lambda_edge": 0.2,
        "lambda_pos": 0.01,
    },
    "low_noise": {
        "mesh_subdivisions": 3,
        "noise_strength": 0.05,
        "num_iterations": 300,
        "learning_rate": 0.01,
        "lambda_edge": 0.1,
        "lambda_pos": 0.01,
    },
    "high_noise": {
        "mesh_subdivisions": 3,
        "noise_strength": 0.20,
        "num_iterations": 1000,
        "learning_rate": 0.005,
        "lambda_edge": 0.15,
        "lambda_pos": 0.02,
    },
}


def get_preset(preset_name):
    """
    Get a preset configuration dictionary.
    
    Args:
        preset_name: Name of the preset ('quick_test', 'standard', etc.)
    
    Returns:
        dict: Configuration parameters for the preset
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. "
                        f"Available: {list(PRESETS.keys())}")
    return PRESETS[preset_name]


def print_config():
    """Print current configuration settings."""
    print("\n" + "=" * 70)
    print("CURRENT CONFIGURATION")
    print("=" * 70)
    print(f"\nMesh Generation:")
    print(f"  Subdivisions: {MESH_SUBDIVISIONS}")
    print(f"  Radius: {MESH_RADIUS}")
    print(f"\nNoise:")
    print(f"  Strength: {NOISE_STRENGTH}")
    print(f"  Seed: {NOISE_SEED}")
    print(f"  Type: {NOISE_TYPE}")
    print(f"\nOptimization:")
    print(f"  Iterations: {NUM_ITERATIONS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  λ_edge: {LAMBDA_EDGE}")
    print(f"  λ_pos: {LAMBDA_POS}")
    print(f"\nDevice: {DEVICE_TYPE}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_config()
    print("\nAvailable presets:")
    for name in PRESETS.keys():
        print(f"  - {name}")
