# =============================================================================
# DATA FILES CONFIGURATION
# =============================================================================

# CSV file names (files should be in 'data/' folder)
# Expected format: first column = age in kyr, second column = proxy values
# The CSV should be separated by commas
PROXY1_FILE = "file_1.csv"
PROXY2_FILE = "file_2.csv"

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# Rolling window size in kyr (default: 41 kyr - Milankovitch obliquity cycle)
WINDOW_SIZE = 41

# Interpolation resolution for common temporal grid in kyr (default: 1.0)
INTERPOLATION_RESOLUTION = 1.0

# Minimum periods for rolling correlation calculation
# If None, will be automatically set to max(10, window_size // 2)
MIN_PERIODS = None

# Correlation thresholds for period identification
THRESHOLD_HIGH = 0.7   # High correlation threshold (positive and negative)
THRESHOLD_LOW = 0.2    # Decoupling threshold

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

# Matplotlib and Seaborn styling
# See: https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
MATPLOTLIB_STYLE = 'seaborn-v0_8'

# See: https://seaborn.pydata.org/tutorial/color_palettes.html
SEABORN_PALETTE = "husl"

# Default figure parameters
DEFAULT_FIGURE_SIZE = (15, 10)
DEFAULT_FONT_SIZE = 12

# =============================================================================
# PLOT-SPECIFIC CONFIGURATIONS
# =============================================================================

# Comprehensive Analysis Plot
COMPREHENSIVE_ANALYSIS = {
    'figsize': (16, 12)
}

# Temporal Evolution Plot
TEMPORAL_EVOLUTION = {
    'figsize': (15, 8),
    'x_tick_interval': 40  # X-axis ticks every N kyr
}

# Window Comparison Plot
WINDOW_COMPARISON = {
    'window_sizes': [20, 41, 100, 400],  # Window sizes to compare (kyr)
    'figsize': (16, 10),
    'cycle_names': [
        'Precession (~20 kyr)',
        'Obliquity (~41 kyr)', 
        'Short Eccentricity (~100 kyr)',
        'Long Eccentricity (~400 kyr)'
    ]
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_configurations():
    """
    Validate configuration parameters and show warnings if needed.
    """
    import os
    
    warnings = []
    
    # Check if data files exist
    proxy1_path = f'data/{PROXY1_FILE}'
    proxy2_path = f'data/{PROXY2_FILE}'
    
    if not os.path.exists(proxy1_path):
        warnings.append(f"⚠️  Proxy 1 file not found: {proxy1_path}")
    
    if not os.path.exists(proxy2_path):
        warnings.append(f"⚠️  Proxy 2 file not found: {proxy2_path}")
    
    # Check window size
    if WINDOW_SIZE < 5:
        warnings.append(f"⚠️  Small window size ({WINDOW_SIZE} kyr) may generate unstable results")
    elif WINDOW_SIZE > 500:
        warnings.append(f"⚠️  Large window size ({WINDOW_SIZE} kyr) may lose temporal details")
    
    # Check thresholds
    if not (0 < THRESHOLD_HIGH <= 1):
        warnings.append(f"⚠️  THRESHOLD_HIGH should be between 0 and 1 (current: {THRESHOLD_HIGH})")
    
    if not (0 < THRESHOLD_LOW < THRESHOLD_HIGH):
        warnings.append(f"⚠️  THRESHOLD_LOW should be between 0 and THRESHOLD_HIGH (current: {THRESHOLD_LOW})")
    
    # Check window comparison sizes
    if len(WINDOW_COMPARISON['window_sizes']) != len(WINDOW_COMPARISON['cycle_names']):
        warnings.append(f"⚠️  Number of window_sizes and cycle_names should match")
    
    return warnings

def print_configuration_summary():
    """
    Print a summary of current configurations.
    """
    print("\n📋 CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"Proxy 1 file: {PROXY1_FILE}")
    print(f"Proxy 2 file: {PROXY2_FILE}")
    print(f"Window size: {WINDOW_SIZE} kyr")
    print(f"Interpolation resolution: {INTERPOLATION_RESOLUTION} kyr")
    print(f"Correlation thresholds: High={THRESHOLD_HIGH}, Low={THRESHOLD_LOW}")
    print(f"X-axis tick interval: {TEMPORAL_EVOLUTION['x_tick_interval']} kyr")
    print(f"Window comparison sizes: {WINDOW_COMPARISON['window_sizes']} kyr")
    print() 