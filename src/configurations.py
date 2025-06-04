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