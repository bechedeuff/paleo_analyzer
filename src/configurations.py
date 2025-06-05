# =============================================================================
# GENERAL CONFIGURATIONS
# =============================================================================

# CSV file names (files should be in 'data/' folder)
# Expected format: first column = age in kyr, second column = proxy values
# The CSV should be separated by commas
PROXY1_FILE = "d13C_age_cibicides.csv"
PROXY2_FILE = "lnmn-sue-TD-age.csv"

# Interpolation resolution for common temporal grid in kyr
INTERPOLATION_RESOLUTION = 1.0

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

# Rolling Window Analysis Configuration
ROLLING_WINDOW_ANALYSIS = {
    'window_size': 41,               # Rolling window size in kyr (default: 41 kyr - Milankovitch obliquity cycle)
    'min_periods': None,             # Minimum periods for rolling correlation calculation (None = auto)
    'threshold_high': 0.7,           # High correlation threshold (positive and negative)
    'threshold_low': 0.2,            # Decoupling threshold
    'window_comparison_sizes': [20, 41, 100, 400],  # Window sizes to compare (kyr)
    'cycle_names': [
        'Precession (~20 kyr)',
        'Obliquity (~41 kyr)', 
        'Short Eccentricity (~100 kyr)',
        'Long Eccentricity (~400 kyr)'
    ]
}

# Rolling Window Plots Configuration
ROLLING_WINDOW_PLOTS = {
    'comprehensive_figsize': (16, 12),
    'temporal_evolution_figsize': (15, 8),
    'temporal_evolution_x_tick_interval': 40,  # X-axis ticks every N kyr
    'window_comparison_figsize': (16, 10),
}

# =============================================================================
# SPECTRAL ANALYSIS PARAMETERS
# =============================================================================

# Spectral Analysis Configuration
SPECTRAL_ANALYSIS = {
    'wavelet_type': 'morlet',        # Type of wavelet ('morlet', 'paul', 'dog')
    'wavelet_param': 6,              # Wavelet parameter (for Morlet: wavenumber k0)
    'min_period': 2.0,               # Minimum period to analyze (kyr)
    'max_period': 1000.0,            # Maximum period to analyze (kyr)
    'coherence_threshold': 0.7,      # Coherence threshold for significant regions
    'confidence_level': 0.95,        # Confidence level for significance testing
    'milankovitch_cycles': {
        'precession': [19, 23],      # Precession cycles (kyr)
        'obliquity': [41],           # Obliquity cycles (kyr)
        'eccentricity_short': [100], # Short eccentricity cycles (kyr)
        'eccentricity_long': [400]   # Long eccentricity cycles (kyr)
    }
}

# Spectral Plots Configuration
SPECTRAL_PLOTS = {
    'wavelet_power_figsize': (16, 10),
    'wavelet_power_colormap': 'jet',
    'cross_wavelet_figsize': (16, 12),
    'cross_wavelet_colormap': 'RdBu_r',
    'global_spectrum_figsize': (12, 8),
    'global_spectrum_show_milankovitch': True,
    'global_spectrum_milankovitch_alpha': 0.3,
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
    'default_correlation_type': 'pearson',  # Default correlation type for contrast plot
    'contrast_line_color': 'darkred'        # Color for contrast line (avoid blue because it's similar to the A's plot)
}