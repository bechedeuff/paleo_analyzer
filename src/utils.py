from typing import List, Dict, Any, Tuple, Optional
import os
import json
import pandas as pd
from datetime import datetime

# Import configurations
from . import configurations as config


def validate_configurations() -> List[str]:
    """
    Validate configuration parameters and show warnings if needed.
    
    Returns:
    --------
    List[str]
        List of warning messages for invalid configurations
    """
    warnings: List[str] = []
    
    # Check if data files exist
    proxy1_path: str = f'data/{config.PROXY1_FILE}'
    proxy2_path: str = f'data/{config.PROXY2_FILE}'
    
    if not os.path.exists(proxy1_path):
        warnings.append(f"⚠️  Proxy 1 file not found: {proxy1_path}")
    
    if not os.path.exists(proxy2_path):
        warnings.append(f"⚠️  Proxy 2 file not found: {proxy2_path}")
    
    # Check window size
    if config.WINDOW_SIZE < 5:
        warnings.append(f"⚠️  Small window size ({config.WINDOW_SIZE} kyr) may generate unstable results")
    elif config.WINDOW_SIZE > 500:
        warnings.append(f"⚠️  Large window size ({config.WINDOW_SIZE} kyr) may lose temporal details")
    
    # Check thresholds
    if not (0 < config.THRESHOLD_HIGH <= 1):
        warnings.append(f"⚠️  THRESHOLD_HIGH should be between 0 and 1 (current: {config.THRESHOLD_HIGH})")
    
    if not (0 < config.THRESHOLD_LOW < config.THRESHOLD_HIGH):
        warnings.append(f"⚠️  THRESHOLD_LOW should be between 0 and THRESHOLD_HIGH (current: {config.THRESHOLD_LOW})")
    
    # Check window comparison sizes
    if len(config.WINDOW_COMPARISON['window_sizes']) != len(config.WINDOW_COMPARISON['cycle_names']):
        warnings.append(f"⚠️  Number of window_sizes and cycle_names should match")
    
    # Check spectral analysis parameters
    if config.MIN_PERIOD <= 0:
        warnings.append(f"⚠️  MIN_PERIOD should be positive (current: {config.MIN_PERIOD})")
    
    if config.MAX_PERIOD <= config.MIN_PERIOD:
        warnings.append(f"⚠️  MAX_PERIOD should be greater than MIN_PERIOD")
    
    if not (0 < config.COHERENCE_THRESHOLD <= 1):
        warnings.append(f"⚠️  COHERENCE_THRESHOLD should be between 0 and 1 (current: {config.COHERENCE_THRESHOLD})")
    
    if not (0 < config.CONFIDENCE_LEVEL < 1):
        warnings.append(f"⚠️  CONFIDENCE_LEVEL should be between 0 and 1 (current: {config.CONFIDENCE_LEVEL})")
    
    if config.WAVELET_PARAM <= 0:
        warnings.append(f"⚠️  WAVELET_PARAM should be positive (current: {config.WAVELET_PARAM})")
    
    return warnings


def print_configuration_summary() -> None:
    """
    Print a summary of current configurations.
    """
    print("\n📋 CONFIGURATION SUMMARY")
    print("=" * 50)
    print("DATA FILES:")
    print(f"  Proxy 1 file: {config.PROXY1_FILE}")
    print(f"  Proxy 2 file: {config.PROXY2_FILE}")
    print("\nROLLING WINDOW ANALYSIS:")
    print(f"  Window size: {config.WINDOW_SIZE} kyr")
    print(f"  Interpolation resolution: {config.INTERPOLATION_RESOLUTION} kyr")
    print(f"  Correlation thresholds: High={config.THRESHOLD_HIGH}, Low={config.THRESHOLD_LOW}")
    print(f"  Window comparison sizes: {config.WINDOW_COMPARISON['window_sizes']} kyr")
    print("\nSPECTRAL ANALYSIS:")
    print(f"  Wavelet type: {config.WAVELET_TYPE} (param={config.WAVELET_PARAM})")
    print(f"  Period range: {config.MIN_PERIOD} - {config.MAX_PERIOD} kyr")
    print(f"  Coherence threshold: {config.COHERENCE_THRESHOLD}")
    print(f"  Confidence level: {config.CONFIDENCE_LEVEL}")
    print(f"  Milankovitch cycles: {list(config.MILANKOVITCH_CYCLES.keys())}")
    print()


def create_results_directory() -> str:
    """
    Create a new experiment directory with sequential numbering and subdirectories for both analyses.
    
    Returns:
    --------
    str
        Path to the created experiment directory
    """
    # Ensure main results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Find the next experiment number
    experiment_num: int = 1
    while True:
        experiment_dir: str = f'results/experiment_{experiment_num}'
        if not os.path.exists(experiment_dir):
            break
        experiment_num += 1
    
    # Create the experiment directory and subdirectories for both analyses
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Rolling window analysis subdirectories
    os.makedirs(f'{experiment_dir}/rolling_window', exist_ok=True)
    os.makedirs(f'{experiment_dir}/rolling_window/figures', exist_ok=True)
    
    # Spectral analysis subdirectories
    os.makedirs(f'{experiment_dir}/spectral', exist_ok=True)
    os.makedirs(f'{experiment_dir}/spectral/figures', exist_ok=True)
    
    # Lead-lag analysis subdirectories
    os.makedirs(f'{experiment_dir}/lead_lag', exist_ok=True)
    os.makedirs(f'{experiment_dir}/lead_lag/figures', exist_ok=True)
    
    print(f"📁 Created experiment directory: {experiment_dir}")
    print(f"   ├── rolling_window/")
    print(f"   │   └── figures/")
    print(f"   ├── spectral/")
    print(f"   │   └── figures/")
    print(f"   ├── lead_lag/")
    print(f"   │   └── figures/")
    print(f"   └── experiment_config.json (will be created)")
    return experiment_dir


def save_experiment_config(experiment_dir: str) -> str:
    """
    Save the current configuration to the experiment directory for reproducibility.
    
    Parameters:
    -----------
    experiment_dir : str
        Path to the experiment directory
        
    Returns:
    --------
    str
        Path to the saved configuration file
    """
    config_data: Dict[str, Any] = {
        "experiment_info": {
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "configuration_used": "src/configurations.py"
        },
        "data_files": {
            "proxy1_file": config.PROXY1_FILE,
            "proxy2_file": config.PROXY2_FILE
        },
        "analysis_parameters": {
            "window_size": config.WINDOW_SIZE,
            "interpolation_resolution": config.INTERPOLATION_RESOLUTION,
            "min_periods": config.MIN_PERIODS,
            "threshold_high": config.THRESHOLD_HIGH,
            "threshold_low": config.THRESHOLD_LOW
        },
        "visualization_settings": {
            "matplotlib_style": config.MATPLOTLIB_STYLE,
            "seaborn_palette": config.SEABORN_PALETTE,
            "default_figure_size": config.DEFAULT_FIGURE_SIZE,
            "default_font_size": config.DEFAULT_FONT_SIZE
        },
        "plot_configurations": {
            "comprehensive_analysis": config.COMPREHENSIVE_ANALYSIS,
            "temporal_evolution": config.TEMPORAL_EVOLUTION,
            "window_comparison": config.WINDOW_COMPARISON
        },
        "spectral_analysis": {
            "wavelet_type": config.WAVELET_TYPE,
            "wavelet_param": config.WAVELET_PARAM,
            "min_period": config.MIN_PERIOD,
            "max_period": config.MAX_PERIOD,
            "coherence_threshold": config.COHERENCE_THRESHOLD,
            "confidence_level": config.CONFIDENCE_LEVEL,
            "milankovitch_cycles": config.MILANKOVITCH_CYCLES
        },
        "spectral_plot_configurations": {
            "wavelet_power_plot": config.WAVELET_POWER_PLOT,
            "cross_wavelet_plot": config.CROSS_WAVELET_PLOT,
            "global_spectrum_plot": config.GLOBAL_SPECTRUM_PLOT
        },
        "lead_lag_analysis": {
            "leadlag_analysis": config.LEADLAG_ANALYSIS,
            "leadlag_plots": config.LEADLAG_PLOTS
        }
    }
    
    config_file: str = f'{experiment_dir}/experiment_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Experiment configuration saved: {config_file}")
    return config_file


def load_paleoclimate_data(proxy1_file: str, proxy2_file: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Load and clean paleoclimate data from CSV files
    
    Parameters:
    -----------
    proxy1_file : str
        Path to CSV file of the first proxy
    proxy2_file : str
        Path to CSV file of the second proxy
        
    Expected format of CSVs:
    - First column: age in kyr
    - Second column: proxy values
        
    Returns:
    --------
    tuple : (success: bool, data: dict or None)
        - success: True if loading successful, False otherwise
        - data: Dictionary with loaded data if successful, None otherwise
    """
    print("🔄 Loading paleoclimate data...")
    
    data = {}
    
    try:
        # Loading data from the first proxy
        proxy1_data = pd.read_csv(proxy1_file)
        # Detecting column names automatically
        original_cols1 = proxy1_data.columns.tolist()
        proxy1_name = original_cols1[1] if len(original_cols1) > 1 else 'proxy1'
        
        # Standardizing column names
        proxy1_data.columns = ['age_kyr', 'proxy1_values']
        print(f"✅ Data {proxy1_name} loaded: {len(proxy1_data)} points")
        
        data['proxy1_data'] = proxy1_data
        data['proxy1_name'] = proxy1_name
        data['proxy1_units'] = ''  # Can be expanded to detect units
        
    except Exception as e:
        print(f"❌ Error loading first proxy: {e}")
        return False, None
        
    try:
        # Loading data from the second proxy
        proxy2_data = pd.read_csv(proxy2_file)
        # Detecting column names automatically
        original_cols2 = proxy2_data.columns.tolist()
        proxy2_name = original_cols2[1] if len(original_cols2) > 1 else 'proxy2'
        
        # Standardizing column names
        proxy2_data.columns = ['age_kyr', 'proxy2_values']
        print(f"✅ Data {proxy2_name} loaded: {len(proxy2_data)} points")
        
        data['proxy2_data'] = proxy2_data
        data['proxy2_name'] = proxy2_name
        data['proxy2_units'] = ''  # Can be expanded to detect units
        
    except Exception as e:
        print(f"❌ Error loading second proxy: {e}")
        return False, None
    
    # Clean and sort data
    data['proxy1_data'] = data['proxy1_data'].dropna().sort_values('age_kyr').reset_index(drop=True)
    data['proxy2_data'] = data['proxy2_data'].dropna().sort_values('age_kyr').reset_index(drop=True)
    
    return True, data