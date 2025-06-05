# =============================================================================
# DATA FILES CONFIGURATION
# =============================================================================

# CSV file names (files should be in 'data/' folder)
# Expected format: first column = age in kyr, second column = proxy values
# The CSV should be separated by commas
PROXY1_FILE = "d13C_age_cibicides.csv"
PROXY2_FILE = "lnmn-sue-TD-age.csv"

# =============================================================================
# GENERAL VISUALIZATION CONFIGURATIONS
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

# =============================================================================
# LEAD-LAG ANALYSIS PARAMETERS
# =============================================================================

# Lead-Lag Analysis Configuration
LEADLAG_ANALYSIS = {
    'max_lag_kyr': 50,           # Maximum lag to test in both directions (kyr)
    'lag_step_kyr': 1.0,         # Step size for lag testing (kyr)  
    'methods': [                 # List of methods to compute
        'cross_correlation',      # Standard cross-correlation
        'ccf_auc',               # Cross-correlation AUC method
        'ccf_at_max_lag'         # Cross-correlation at maximum lag
    ],
    'correlation_types': [       # Types of correlation to test
        'pearson',               # Pearson correlation
        'spearman',              # Spearman rank correlation
        'kendall'                # Kendall tau correlation
    ],
    'significance_level': 0.05,  # Statistical significance level
    'confidence_level': 0.95,    # Confidence level for intervals
    'bootstrap_iterations': 1000, # Number of bootstrap iterations for confidence intervals
    'detrend_data': True,        # Whether to detrend data before analysis
    'normalize_data': True       # Whether to normalize data (z-score)
}

# Lead-Lag Plots Configuration
LEADLAG_PLOTS = {
    'comprehensive_figsize': (16, 10),
    'methods_comparison_figsize': (14, 10),
    'correlation_types_figsize': (14, 8),
    'confidence_intervals_figsize': (12, 8),
    'colormap': 'RdBu_r',
    'dpi': 300,
    'default_correlation_type': 'pearson',  # Default correlation type for contrast plot
    'contrast_line_color': 'darkred'        # Color for contrast line (avoid blue)
}