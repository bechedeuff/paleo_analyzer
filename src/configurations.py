# =============================================================================
# DATA FILES CONFIGURATION
# =============================================================================

# CSV file names (files should be in 'data/' folder)
# Expected format: first column = age in kyr, second column = proxy values
# The CSV should be separated by commas
PROXY1_FILE = "d13C_age_cibicides.csv"
PROXY2_FILE = "lnmn-sue-TD-age.csv"

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
DEFAULT_DPI = 300

# =============================================================================
# ROLLING WINDOW ANALYSIS PARAMETERS
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
# ROLLING WINDOW PLOTS CONFIGURATIONS
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
# SPECTRAL ANALYSIS PARAMETERS
# =============================================================================

# Wavelet Analysis Parameters
WAVELET_TYPE = 'morlet'  # Type of wavelet ('morlet', 'paul', 'dog')
WAVELET_PARAM = 6        # Wavelet parameter (for Morlet: wavenumber k0)

# Frequency/Period Analysis
MIN_PERIOD = 2.0         # Minimum period to analyze (kyr)
MAX_PERIOD = 1000.0      # Maximum period to analyze (kyr)

# Statistical Analysis
COHERENCE_THRESHOLD = 0.7    # Coherence threshold for significant regions
CONFIDENCE_LEVEL = 0.95      # Confidence level for significance testing

# Milankovitch Cycles for Reference
MILANKOVITCH_CYCLES = {
    'precession': [19, 23],      # Precession cycles (kyr)
    'obliquity': [41],           # Obliquity cycles (kyr)
    'eccentricity_short': [100], # Short eccentricity cycles (kyr)
    'eccentricity_long': [400]   # Long eccentricity cycles (kyr)
}

# =============================================================================
# SPECTRAL PLOTS CONFIGURATIONS
# =============================================================================

# Wavelet Power Spectrum Plot
WAVELET_POWER_PLOT = {
    'figsize': (16, 10),
    'colormap': 'jet'
}

# Cross-Wavelet Plot
CROSS_WAVELET_PLOT = {
    'figsize': (16, 12),
    'colormap': 'RdBu_r',
}

# Global Wavelet Spectrum Plot
GLOBAL_SPECTRUM_PLOT = {
    'figsize': (12, 8),
    'show_milankovitch': True,
    'milankovitch_alpha': 0.3
}